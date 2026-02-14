# src/pipeline/eval.py
import numpy as np

def _dcg(rels):
    rels = np.asarray(rels, dtype=float)
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))

def evaluate(indices, query_df, corpus_df, qrels, topk=(1,5,10)):
    """
    indices: np.ndarray [num_queries, K]  (row = ranked doc indices)
    qrels  : dict(query_id -> dict(doc_id -> score))   (BEIR-style)
    """
    doc_ids = corpus_df["doc_id"].astype(str).tolist()
    qids = query_df["query_id"].astype(str).tolist()

    K = indices.shape[1]
    topk = sorted(set(int(x) for x in topk))
    max_k = max(topk)
    assert max_k <= K, f"Need indices with at least {max_k} columns, got K={K}"

    # Recall@k (binary hit: any relevant in top-k)
    recalls = {k: [] for k in topk}
    ndcgs = []

    for qi, qid in enumerate(qids):
        rel_dict = qrels.get(qid, {})
        rel_set = set(rel_dict.keys())

        ranked_docids = [doc_ids[j] for j in indices[qi, :max_k]]
        for k in topk:
            hit = any(d in rel_set for d in ranked_docids[:k])
            recalls[k].append(1.0 if hit else 0.0)

        # nDCG@10 (or min(10, K))
        k_ndcg = min(10, K)
        rels = [rel_dict.get(d, 0.0) for d in ranked_docids[:k_ndcg]]
        dcg = _dcg(rels)

        ideal = sorted(rel_dict.values(), reverse=True)[:k_ndcg]
        idcg = _dcg(ideal)
        ndcgs.append(0.0 if idcg == 0.0 else dcg / idcg)

    results = {}
    for k in topk:
        results[f"recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
    results["ndcg@10"] = float(np.mean(ndcgs)) if ndcgs else 0.0
    return results
