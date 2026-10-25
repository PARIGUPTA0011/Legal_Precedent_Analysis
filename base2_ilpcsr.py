"""
Baseline 2: citation-aware precedent retrieval on IL-PCSR  (System 2 of the plan)

Uses ONLY the train split's citation links to build citation signals, tunes
hyperparameters on dev, and reports final numbers on test (no leakage).

Systems compared (all on the same test queries as Baseline 1):
  BM25               text-only baseline (System 1)
  BM25+Prior         BM25 + popularity prior (log of how often train cases cite a precedent)
  CiteKNN            find the most similar TRAIN cases, vote for the precedents they cite
  Doc+Anchor BM25    precedent text enriched with text of train cases that cite it
  RRF(BM25,CiteKNN)  fusion
  RRF(all three)     fusion of BM25, CiteKNN and Doc+Anchor

Needs base1_ilpcsr.py in the same folder. Run:  python base2_ilpcsr.py
"""
import json

import numpy as np
import scipy.sparse as sp

from base1_ilpcsr import (BM25, DATASET, DENSE_MODELS, POOL_CONFIG, POOL_ID_COL,
                          POOL_TEXT_COL, POOL_TITLE_COL, QUERY_CONFIG, QUERY_MODE,
                          build_relevance, cached_embed, evaluate, pick, print_table,
                          query_text, rrf, to_text)

# ----------------------------- SETTINGS ------------------------------------
MAX_TEST = 300          # same 300 test queries as Baseline 1 (first 300)
MAX_DEV = 200           # dev queries used only for tuning
LAMBDAS = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5)   # weight of the popularity prior
NEIGHBOURS = (5, 10, 30, 50)                  # K for CiteKNN
MAX_CITERS = 5          # citing train cases appended per precedent (anchor docs)
ANCHOR_WORDS = 800      # words kept from each citing case
TUNE_METRIC = "nDCG@10"
USE_DENSE = True        # also compare against the Baseline 1 hybrid (uses cached embeddings)
MAX_TRAIN = None        # e.g. 2000 if you run out of RAM (lowers citation coverage)
# ---------------------------------------------------------------------------


class LazyModel:
    """Loads the sentence-transformer only if embeddings are not already cached."""

    def __init__(self, name):
        self.name, self.m = name, None

    def encode(self, *args, **kwargs):
        if self.m is None:
            from sentence_transformers import SentenceTransformer
            self.m = SentenceTransformer(self.name)
        return self.m.encode(*args, **kwargs)


def minmax(s):
    mn, mx = s.min(1, keepdims=True), s.max(1, keepdims=True)
    return (s - mn) / (mx - mn + 1e-9)


def citation_matrix(rel_sets, n_pool):
    """Binary matrix: row = train case, column = precedent it cites."""
    rows, cols = [], []
    for i, rs in enumerate(rel_sets):
        for j in rs:
            rows.append(i)
            cols.append(j)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)),
                         shape=(len(rel_sets), n_pool))


def knn_cite_scores(q2q, cite, k):
    """Score precedents by the similarity-weighted votes of the k most similar train cases."""
    n_q, n_train = q2q.shape
    k = min(k, n_train)
    idx = np.argpartition(-q2q, k - 1, axis=1)[:, :k]
    w = np.take_along_axis(q2q, idx, 1)
    w = w / (w.max(1, keepdims=True) + 1e-9)
    rows = np.repeat(np.arange(n_q), k)
    wmat = sp.csr_matrix((w.ravel(), (rows, idx.ravel())), shape=(n_q, n_train))
    return (wmat @ cite).toarray()


def build_anchor_docs(train_texts, train_rel, n_pool, max_citers=MAX_CITERS,
                      max_words=ANCHOR_WORDS):
    """For each precedent: concatenated (truncated) text of train cases that cite it."""
    trunc = [" ".join(t.split()[:max_words]) for t in train_texts]
    citers = [[] for _ in range(n_pool)]
    for ti, rs in enumerate(train_rel):
        for p in rs:
            citers[p].append(ti)
    return [" ".join(trunc[t] for t in c[:max_citers]) for c in citers]


def run_experiment(pool_docs, train_texts, train_rel, dev_texts, dev_rel,
                   test_texts, test_rel, test_dense=None):
    n_pool = len(pool_docs)
    cite = citation_matrix(train_rel, n_pool)
    indeg = np.asarray(cite.sum(0)).ravel()
    prior = np.log1p(indeg)
    prior = prior / (prior.max() + 1e-9)

    # Diagnostic: how many relevant test precedents were ever cited in train?
    cited = indeg > 0
    pairs = [j for rs in test_rel for j in rs]
    cov = float(np.mean([cited[j] for j in pairs])) if pairs else 0.0
    print(f"Test relevant precedents cited >=1 time in train: {cov:.1%} "
          f"(ceiling for citation-only signals)")

    bm_pool = BM25(pool_docs)
    bm_train = BM25(train_texts)
    anchors = build_anchor_docs(train_texts, train_rel, n_pool)
    bm_anchor = BM25([d + " " + a for d, a in zip(pool_docs, anchors)])

    def all_scores(texts):
        bm = bm_pool.scores(texts)
        knn_sim = bm_train.scores(texts)
        anc = bm_anchor.scores(texts)
        return bm, knn_sim, anc

    dev_bm, dev_q2q, _ = all_scores(dev_texts)

    # ---- tune lambda (prior) and K (CiteKNN) on dev ----
    best_l = max(LAMBDAS, key=lambda l: evaluate(
        minmax(dev_bm) + l * prior[None, :], dev_rel)[TUNE_METRIC])
    best_k = max(NEIGHBOURS, key=lambda k: evaluate(
        knn_cite_scores(dev_q2q, cite, k) + 1e-6 * minmax(dev_bm), dev_rel)[TUNE_METRIC])
    print(f"Tuned on dev ({TUNE_METRIC}): lambda={best_l}, K={best_k}")

    # ---- final evaluation on test ----
    bm, q2q, anc = all_scores(test_texts)
    knn = knn_cite_scores(q2q, cite, best_k) + 1e-6 * minmax(bm)  # BM25 breaks ties
    prior_s = minmax(bm) + best_l * prior[None, :]

    systems = {
        "BM25": bm,
        f"BM25+Prior": prior_s,
        f"CiteKNN(K={best_k})": knn,
        "Doc+Anchor BM25": anc,
        "RRF(BM25,KNN)": rrf([bm, knn]),
        "RRF(BM25,KNN,Anc)": rrf([bm, knn, anc]),
    }
    if test_dense is not None:  # compare against the strongest text baseline
        hyb = rrf([bm, test_dense])
        systems["Hybrid (B1)"] = hyb
        systems["RRF(Hybrid,KNN)"] = rrf([hyb, knn])
        systems["RRF(Hybrid,KNN,Anc)"] = rrf([hyb, knn, anc])
    results = {name: evaluate(s, test_rel) for name, s in systems.items()}
    params = {"lambda": best_l, "K": best_k, "train_cases": len(train_texts),
              "cold_start_ceiling": cov}
    return results, params


# --------------------------- Data loading ----------------------------------
def find_split(names, *keys):
    return next((n for n in names if any(k in n.lower() for k in keys)), None)


def main():
    from datasets import get_dataset_config_names, load_dataset

    configs = get_dataset_config_names(DATASET)
    pool_cfg = POOL_CONFIG or next(
        (c for c in configs if c != QUERY_CONFIG and ("prec" in c.lower() or "case" in c.lower())), None)
    if pool_cfg is None:
        raise SystemExit(f"Set POOL_CONFIG in base1_ilpcsr.py. Configs: {configs}")

    pds = load_dataset(DATASET, pool_cfg)
    rows, seen, id_col = [], set(), POOL_ID_COL
    for sname in pds:
        cols = pds[sname].column_names
        id_col = id_col or pick(cols, ["id", "precedent_id", "case_id", "doc_id"])
        for r in pds[sname]:
            if str(r[id_col]) not in seen:
                seen.add(str(r[id_col]))
                rows.append(r)
    cols = list(rows[0].keys())
    text_col = POOL_TEXT_COL or pick(cols, ["text", "content", "judgment"])
    title_col = POOL_TITLE_COL or pick(cols, ["case_title", "title", "name"])
    pool_ids = [r[id_col] for r in rows]
    pool_titles = [r[title_col] for r in rows] if title_col else None
    pool_docs = [to_text(r[text_col]) for r in rows]
    print(f"Pool: {len(rows)} precedents")

    qds = load_dataset(DATASET, QUERY_CONFIG)
    names = list(qds)
    print("Query splits:", {k: len(v) for k, v in qds.items()})
    tr_n, dv_n, te_n = (find_split(names, "train"), find_split(names, "dev", "val"),
                        find_split(names, "test"))
    if not tr_n or not te_n:
        raise SystemExit(f"Could not find train/test splits in {names}")
    train, test = list(qds[tr_n]), list(qds[te_n])[:MAX_TEST]
    if MAX_TRAIN:
        train = train[:MAX_TRAIN]
    if dv_n:
        dev = list(qds[dv_n])[:MAX_DEV]
    else:  # no dev split: hold out the tail of train for tuning
        print("No dev split found: holding out the last cases of train for tuning")
        dev, train = train[-MAX_DEV:], train[:-MAX_DEV]

    # Safety: never let a dev/test case sit inside the train graph
    if "id" in train[0]:
        held = {str(r["id"]) for r in dev + test}
        before = len(train)
        train = [r for r in train if str(r["id"]) not in held]
        print(f"Removed {before - len(train)} train cases that overlap with dev/test")

    def prep(rs, drop_empty):
        texts = [query_text(r, QUERY_MODE) for r in rs]
        rel = build_relevance(rs, pool_ids, pool_titles)
        if drop_empty:
            keep = [i for i, s in enumerate(rel) if s]
            return [texts[i] for i in keep], [rel[i] for i in keep]
        return texts, rel

    print("--- train ---")
    tr_t, tr_r = prep(train, False)
    print("--- dev ---")
    dv_t, dv_r = prep(dev, True)
    print("--- test ---")
    te_t, te_r = prep(test, True)
    print(f"train={len(tr_t)} dev={len(dv_t)} test={len(te_t)}")

    test_dense = None
    if USE_DENSE:
        name = DENSE_MODELS[0]
        model = LazyModel(name)
        d_emb = cached_embed(pool_docs, model, name, "pool")
        q_emb = cached_embed(te_t, model, name, "queries")
        test_dense = q_emb @ d_emb.T

    results, params = run_experiment(pool_docs, tr_t, tr_r, dv_t, dv_r, te_t, te_r,
                                     test_dense=test_dense)
    print_table(results)
    with open("baseline2_results.json", "w") as f:
        json.dump({"params": params, "n_test": len(te_t), "results": results}, f, indent=2)
    print("\nSaved baseline2_results.json")


if __name__ == "__main__":
    main()