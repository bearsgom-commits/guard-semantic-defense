import numpy as np


def _dcg_at_k(rels, k: int) -> float:
    """
    rels: list/array of relevance scores in rank order (0/1 here)
    """
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return float(np.sum(rels * discounts))


def _ndcg_single_relevant_at_10(retrieved_doc_ids, gt_doc_id, k: int = 10) -> float:
    """
    For a single relevant document (binary relevance).
    NDCG@k = DCG@k / IDCG@k, where IDCG@k is 1 (relevant doc at rank 1).
    """
    rels = [1 if d == gt_doc_id else 0 for d in retrieved_doc_ids[:k]]
    dcg = _dcg_at_k(rels, k)
    idcg = 1.0  # best case: relevant doc at rank 1
    return dcg / idcg


def evaluate(indices, query_df, corpus_df, topk_list):
    """
    indices: (n_queries, topk_max) array of row indices into corpus_df
    query_df must have: relevant_doc_id
    corpus_df must have: doc_id
    """
    results = {}

    # Recall@k
    for k in topk_list:
        hit = 0
        for i in range(len(query_df)):
            gt = query_df.iloc[i]["relevant_doc_id"]
            retrieved_ids = corpus_df.iloc[indices[i][:k]]["doc_id"].tolist()
            if gt in retrieved_ids:
                hit += 1
        results[f"recall@{k}"] = hit / len(query_df)

    # nDCG@10 (single relevant doc setting)
    ndcgs = []
    for i in range(len(query_df)):
        gt = query_df.iloc[i]["relevant_doc_id"]
        retrieved_ids_10 = corpus_df.iloc[indices[i][:10]]["doc_id"].tolist()
        ndcgs.append(_ndcg_single_relevant_at_10(retrieved_ids_10, gt, k=10))
    results["ndcg@10"] = float(np.mean(ndcgs))

    return results
