import numpy as np

def semantic_search(query_emb: np.ndarray, corpus_emb: np.ndarray, topk: int = 10) -> np.ndarray:
    """
    Cosine similarity search with normalized embeddings.
    Returns indices: shape (num_queries, topk)
    """
    # (M,D) @ (D,N) -> (M,N)
    scores = query_emb @ corpus_emb.T
    # topk indices per row
    idx = np.argpartition(-scores, kth=topk - 1, axis=1)[:, :topk]
    # sort within topk by score descending
    top_scores = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-top_scores, axis=1)
    idx_sorted = np.take_along_axis(idx, order, axis=1)
    return idx_sorted
