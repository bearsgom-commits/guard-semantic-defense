import numpy as np

def semantic_search(query_emb, corpus_emb, topk=10):
    scores = np.dot(query_emb, corpus_emb.T)
    indices = np.argsort(-scores, axis=1)[:, :topk]
    return indices
