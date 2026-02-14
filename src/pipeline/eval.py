from __future__ import annotations
import numpy as np
import pandas as pd

def _pick_gold_column(query_df: pd.DataFrame) -> str:
    # 쿼리 정답(정답 문서 인덱스) 컬럼 후보
    candidates = ["gold_id", "pos_id", "positive_id", "answer_id", "target_id", "doc_id", "corpus_id"]
    for c in candidates:
        if c in query_df.columns:
            return c
    raise ValueError(
        "Query CSV must contain a gold/positive doc id column. "
        "Add one of: gold_id, pos_id, positive_id, answer_id, target_id, doc_id, corpus_id"
    )

def recall_at_k(indices: np.ndarray, gold: np.ndarray, k: int) -> float:
    hits = 0
    for i in range(len(gold)):
        if int(gold[i]) in set(map(int, indices[i, :k])):
            hits += 1
    return hits / max(1, len(gold))

def ndcg_at_k_binary(indices: np.ndarray, gold: np.ndarray, k: int = 10) -> float:
    """
    Binary relevance nDCG@k with exactly one relevant doc per query (gold).
    Stable + no sklearn shape issues.
    """
    dcgs = []
    for i in range(len(gold)):
        rel_doc = int(gold[i])
        ranking = list(map(int, indices[i, :k]))
        if rel_doc in ranking:
            rank = ranking.index(rel_doc) + 1  # 1-based
            dcg = 1.0 / np.log2(rank + 1.0)
        else:
            dcg = 0.0
        idcg = 1.0  # relevant doc at rank 1 => 1/log2(2)=1
        dcgs.append(dcg / idcg)
    return float(np.mean(dcgs)) if dcgs else 0.0

def evaluate(indices: np.ndarray, query_df: pd.DataFrame, corpus_df: pd.DataFrame, topk_list: list[int]):
    gold_col = _pick_gold_column(query_df)
    gold = query_df[gold_col].to_numpy()

    results = {}
    for k in topk_list:
        results[f"recall@{k}"] = recall_at_k(indices, gold, int(k))

    # 기본은 nDCG@10
    results["ndcg@10"] = ndcg_at_k_binary(indices, gold, k=10)

    return results
