"""
Baseline 1: text-based precedent retrieval on IL-PCSR
  - BM25 (sparse, fast, CPU)
  - Dense retrieval (optional, needs sentence-transformers; GPU recommended)
  - Hybrid via Reciprocal Rank Fusion (BM25 + dense)

Task: for each query judgment, rank all candidate precedents; the relevant
precedents are the ones the query case actually cites (relevant_precedent_ids).

Colab setup:
    !pip install -q datasets scikit-learn scipy numpy sentence-transformers

Run:  python baseline1_ilpcsr.py
"""
import json
from collections import defaultdict

import os

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import CountVectorizer

# Load HF_TOKEN from a .env file (next to this script) so gated datasets work
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("Tip: pip install python-dotenv to load HF_TOKEN from .env")
if not os.getenv("HF_TOKEN"):
    print("Warning: HF_TOKEN not set. Gated datasets like IL-PCSR will fail.")

# ----------------------------- SETTINGS ------------------------------------
DATASET = "Exploration-Lab/IL-PCSR"
QUERY_CONFIG = "queries"      # config name of the query split (from your URL)
POOL_CONFIG = None            # None = auto-detect the precedent-pool config
SPLIT = None                  # None = auto-pick "test" (else dev/validation)
MAX_QUERIES = 300             # use a subset first; set None for all
QUERY_MODE = "full"           # "full" = whole judgment, "facts" = only Facts sentences
FACT_KEYWORDS = ("fact",)     # matched against rhetorical_roles (lowercased)
KS = (5, 10, 20, 50)

# Pool column names (None = auto-detect). Fix these after you see the printout.
POOL_ID_COL = None
POOL_TEXT_COL = None
POOL_TITLE_COL = None

RUN_DENSE = True              # dense + hybrid (BM25 always runs)
DENSE_MODELS = [              # add "BAAI/bge-small-en-v1.5" to compare a second model
    "sentence-transformers/all-MiniLM-L6-v2",
]
CHUNK_WORDS = 150             # ~200 tokens, safe for 256-token models
MAX_CHUNKS = 10               # first ~1500 words of each judgment are embedded
USE_CACHE = True              # embeddings saved in ./emb_cache so reruns are instant
N_EXAMPLES = 3                # example queries printed for your slides
# ---------------------------------------------------------------------------


def to_text(x):
    if isinstance(x, (list, tuple)):
        return " ".join(str(s) for s in x)
    return "" if x is None else str(x)


def query_text(row, mode="full"):
    sents = row["text"]
    roles = row.get("rhetorical_roles")
    if mode == "facts" and roles and len(roles) == len(sents):
        picked = [s for s, r in zip(sents, roles)
                  if any(k in str(r).lower() for k in FACT_KEYWORDS)]
        if picked:
            return " ".join(picked)
    return to_text(sents)


# ------------------------------- BM25 --------------------------------------
class BM25:
    """Sparse BM25: fast because query-doc scoring is one sparse matrix product."""

    def __init__(self, docs, k1=1.2, b=0.75):
        self.vec = CountVectorizer(
            lowercase=True, stop_words="english",
            token_pattern=r"(?u)\b\w\w+\b",
        )
        tf = self.vec.fit_transform(docs).tocsr().astype(np.float32)
        n_docs = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(tf.sum(1)).ravel()
        avgdl = dl.mean()
        coo = tf.tocoo()
        denom = coo.data + k1 * (1 - b + b * dl[coo.row] / avgdl)
        data = coo.data * (k1 + 1) / denom * idf[coo.col]
        w = sp.csr_matrix((data, (coo.row, coo.col)), shape=tf.shape)
        self.W = w.T.tocsr()  # vocab x docs

    def scores(self, queries, binary=True):
        q = self.vec.transform(queries).astype(np.float32)
        if binary:  # long queries: count each term once
            q.data[:] = 1.0
        return (q @ self.W).toarray()  # n_queries x n_docs


# ------------------------------- Dense -------------------------------------
def dense_embed(texts, model, chunk_words=CHUNK_WORDS, max_chunks=MAX_CHUNKS):
    """Split into word chunks, embed each, average per text, L2-normalise."""
    chunks, owner = [], []
    for i, t in enumerate(texts):
        w = t.split()
        parts = [" ".join(w[j:j + chunk_words]) for j in range(0, len(w), chunk_words)]
        parts = parts[:max_chunks] or [""]
        chunks.extend(parts)
        owner.extend([i] * len(parts))
    emb = model.encode(chunks, batch_size=64, normalize_embeddings=True,
                       show_progress_bar=True)
    out = np.zeros((len(texts), emb.shape[1]), dtype=np.float32)
    np.add.at(out, np.array(owner), emb)
    out /= np.linalg.norm(out, axis=1, keepdims=True) + 1e-9
    return out


def cached_embed(texts, model, model_name, kind):
    """dense_embed with an on-disk cache keyed by model, settings and content."""
    import hashlib
    sig = hashlib.md5(("\n".join(t[:200] for t in texts) + str(len(texts))).encode("utf-8", "ignore")).hexdigest()[:10]
    safe = model_name.replace("/", "_")
    path = os.path.join("emb_cache", f"{safe}_{kind}_{len(texts)}_{CHUNK_WORDS}x{MAX_CHUNKS}_{sig}.npy")
    if USE_CACHE and os.path.exists(path):
        print(f"Loaded cached {kind} embeddings: {path}")
        return np.load(path)
    emb = dense_embed(texts, model)
    if USE_CACHE:
        os.makedirs("emb_cache", exist_ok=True)
        np.save(path, emb)
    return emb


def safe(s):
    """Windows consoles can choke on unicode in case titles."""
    return str(s).encode("ascii", "replace").decode()


def show_examples(score_matrix, q_titles, pool_titles, rel_sets, n=3, topk=5):
    order = np.argsort(-score_matrix, axis=1)
    for qi in range(min(n, len(q_titles))):
        print(f"\nQuery: {safe(q_titles[qi])}  ({len(rel_sets[qi])} relevant precedents)")
        for rank, d in enumerate(order[qi][:topk], 1):
            tag = "[HIT]" if d in rel_sets[qi] else "     "
            name = safe(pool_titles[d]) if pool_titles else f"doc {d}"
            print(f"  {rank}. {tag} {name[:90]}")


# ------------------------------- Fusion ------------------------------------
def rrf(score_mats, k=60):
    """Reciprocal Rank Fusion over several (n_queries x n_docs) score matrices."""
    fused = 0
    for s in score_mats:
        ranks = np.argsort(np.argsort(-s, axis=1), axis=1)  # 0-based rank
        fused = fused + 1.0 / (k + ranks + 1)
    return fused


# ------------------------------ Metrics ------------------------------------
def evaluate(score_matrix, rel_sets, ks=KS):
    """rel_sets[i] = set of relevant doc indices for query i (binary relevance)."""
    res = defaultdict(list)
    order = np.argsort(-score_matrix, axis=1)
    for qi, rel in enumerate(rel_sets):
        if not rel:
            continue
        hits = np.isin(order[qi], list(rel)).astype(float)
        pos = np.arange(len(hits))
        first = np.argmax(hits) if hits.any() else None
        res["MRR"].append(1.0 / (first + 1) if first is not None else 0.0)
        # average precision
        cum = np.cumsum(hits)
        res["MAP"].append(float((hits * cum / (pos + 1)).sum() / len(rel)))
        for k in ks:
            res[f"Recall@{k}"].append(hits[:k].sum() / len(rel))
            dcg = (hits[:k] / np.log2(pos[:k] + 2)).sum()
            ideal = (1.0 / np.log2(np.arange(min(len(rel), k)) + 2)).sum()
            res[f"nDCG@{k}"].append(dcg / ideal)
    return {m: float(np.mean(v)) for m, v in res.items()}


def print_table(results):
    metrics = list(next(iter(results.values())).keys())
    print("\n" + "System".ljust(16) + "".join(m.rjust(11) for m in metrics))
    for name, r in results.items():
        print(name.ljust(16) + "".join(f"{r[m]:11.4f}" for m in metrics))


# --------------------------- Data loading ----------------------------------
def pick(cols, preferred):
    for c in preferred:
        if c in cols:
            return c
    return None


def build_relevance(queries, pool_ids, pool_titles):
    """Map each query's relevant precedent ids (or titles as fallback) to pool indices."""
    id2idx = {str(i): k for k, i in enumerate(pool_ids)}
    title2idx = {str(t).strip().lower(): k for k, t in enumerate(pool_titles)} if pool_titles else {}

    def by_ids():
        return [{id2idx[str(i)] for i in (q["relevant_precedent_ids"] or []) if str(i) in id2idx}
                for q in queries]

    def by_titles():
        return [{title2idx[str(t).strip().lower()] for t in (q["relevant_precedents"] or [])
                 if str(t).strip().lower() in title2idx} for q in queries]

    total = sum(len(q["relevant_precedent_ids"] or []) for q in queries) or 1
    a = by_ids()
    cov_ids = sum(len(s) for s in a) / total
    print(f"Relevant precedents found in pool via ids: {cov_ids:.1%}")
    if cov_ids < 0.5 and title2idx:
        b = by_titles()
        cov_t = sum(len(s) for s in b) / total
        print(f"... via titles: {cov_t:.1%}")
        if cov_t > cov_ids:
            return b
    return a


def main():
    from datasets import get_dataset_config_names, load_dataset

    configs = get_dataset_config_names(DATASET)
    print("Available configs:", configs)

    pool_cfg = POOL_CONFIG or next(
        (c for c in configs if c != QUERY_CONFIG and ("prec" in c.lower() or "case" in c.lower())), None)
    if pool_cfg is None:
        raise SystemExit("Could not find the precedent pool config. Set POOL_CONFIG from the list above.")
    print("Query config:", QUERY_CONFIG, "| Pool config:", pool_cfg)

    qds = load_dataset(DATASET, QUERY_CONFIG)
    print("Query splits:", {k: len(v) for k, v in qds.items()})
    split = SPLIT or next((s for s in ("test", "validation", "dev") if s in qds), list(qds)[-1])
    queries = list(qds[split])
    if MAX_QUERIES:
        queries = queries[:MAX_QUERIES]
    print(f"Using split '{split}' with {len(queries)} queries")

    pds = load_dataset(DATASET, pool_cfg)
    rows, seen = [], set()
    id_col = POOL_ID_COL
    for sname in pds:  # use every split: the pool is the full candidate set
        cols = pds[sname].column_names
        print(f"Pool split '{sname}' columns:", cols)
        id_col = id_col or pick(cols, ["id", "precedent_id", "case_id", "doc_id"])
        for r in pds[sname]:
            key = str(r[id_col])
            if key not in seen:
                seen.add(key)
                rows.append(r)
    cols = list(rows[0].keys())
    text_col = POOL_TEXT_COL or pick(cols, ["text", "content", "judgment"])
    title_col = POOL_TITLE_COL or pick(cols, ["case_title", "title", "name"])
    print(f"Pool size: {len(rows)} | id={id_col} text={text_col} title={title_col}")

    pool_ids = [r[id_col] for r in rows]
    pool_titles = [r[title_col] for r in rows] if title_col else None
    pool_docs = [to_text(r[text_col]) for r in rows]

    # Role labels, so you can choose QUERY_MODE="facts" sensibly
    roles = {r for q in queries[:50] for r in (q.get("rhetorical_roles") or [])}
    print("Rhetorical role labels seen:", sorted(roles)[:20])

    q_texts = [query_text(q, QUERY_MODE) for q in queries]
    rel_sets = build_relevance(queries, pool_ids, pool_titles)
    keep = [i for i, s in enumerate(rel_sets) if s]
    print(f"Queries with >=1 relevant precedent in pool: {len(keep)}/{len(queries)}")
    q_texts = [q_texts[i] for i in keep]
    rel_sets = [rel_sets[i] for i in keep]

    results, mats = {}, {}

    bm25 = BM25(pool_docs)
    mats["BM25"] = bm25.scores(q_texts)
    results["BM25"] = evaluate(mats["BM25"], rel_sets)

    if RUN_DENSE:
        from sentence_transformers import SentenceTransformer
        import torch
        print("Dense device:", "GPU" if torch.cuda.is_available() else "CPU (slower, be patient)")
        for name in DENSE_MODELS:
            short = name.split("/")[-1]
            model = SentenceTransformer(name)
            d_emb = cached_embed(pool_docs, model, name, "pool")
            q_emb = cached_embed(q_texts, model, name, "queries")
            mats[f"Dense:{short}"] = q_emb @ d_emb.T
            results[f"Dense:{short}"] = evaluate(mats[f"Dense:{short}"], rel_sets)
            mats[f"Hybrid:{short}"] = rrf([mats["BM25"], mats[f"Dense:{short}"]])
            results[f"Hybrid:{short}"] = evaluate(mats[f"Hybrid:{short}"], rel_sets)

    print_table(results)
    best = max(results, key=lambda n: results[n]["nDCG@10"])
    print(f"\nBest system by nDCG@10: {best}. Example queries (top-5, [HIT] = cited by the case):")
    q_titles = [queries[i].get("case_title", "") for i in keep]
    show_examples(mats[best], q_titles, pool_titles, rel_sets, n=N_EXAMPLES)
    with open("baseline1_results.json", "w") as f:
        json.dump({"split": split, "n_queries": len(q_texts), "mode": QUERY_MODE,
                   "results": results}, f, indent=2)
    print("\nSaved baseline1_results.json")


if __name__ == "__main__":
    main()