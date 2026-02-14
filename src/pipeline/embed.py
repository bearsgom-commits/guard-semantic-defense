from __future__ import annotations
import numpy as np
import pandas as pd

def _pick_text_column(df: pd.DataFrame) -> str:
    # 가장 흔한 텍스트 컬럼명 후보
    candidates = ["text", "sentence", "content", "document", "doc", "query"]
    for c in candidates:
        if c in df.columns:
            return c
    # fallback: 첫 번째 문자열 컬럼
    for c in df.columns:
        if df[c].dtype == object:
            return c
    raise ValueError("No text-like column found in CSV. Add a column like 'text'.")

def build_embeddings(model_name: str, corpus_path: str, query_path: str):
    """
    Returns:
      corpus_df, query_df, corpus_emb (N,D), query_emb (M,D)
    """
    from sentence_transformers import SentenceTransformer

    corpus_df = pd.read_csv(corpus_path)
    query_df = pd.read_csv(query_path)

    c_col = _pick_text_column(corpus_df)
    q_col = _pick_text_column(query_df)

    model = SentenceTransformer(model_name)

    corpus_texts = corpus_df[c_col].astype(str).tolist()
    query_texts = query_df[q_col].astype(str).tolist()

    corpus_emb = model.encode(
        corpus_texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )
    query_emb = model.encode(
        query_texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
    )

    corpus_emb = np.asarray(corpus_emb, dtype=np.float32)
    query_emb = np.asarray(query_emb, dtype=np.float32)

    return corpus_df, query_df, corpus_emb, query_emb
