import numpy as np

def apply_guard(embeddings: np.ndarray, eps: float, seed: int | None = None) -> np.ndarray:
    """
    Minimal guard: additive Gaussian noise.
    - eps == 0: no noise (baseline)
    - eps > 0: noise_std = 1/eps  (simple monotone control)
    """
    if eps <= 0:
        return embeddings

    rng = np.random.default_rng(seed)
    noise_std = 1.0 / float(eps)
    noise = rng.normal(loc=0.0, scale=noise_std, size=embeddings.shape).astype(embeddings.dtype)
    return embeddings + noise
