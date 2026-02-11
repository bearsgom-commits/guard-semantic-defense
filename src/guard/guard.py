import numpy as np

def apply_guard(embeddings: np.ndarray, eps: float):
    if eps == 0:
        return embeddings

    noise = np.random.normal(
        loc=0.0,
        scale=1.0/eps,
        size=embeddings.shape
    )
    return embeddings + noise
