# src/pipeline/embed.py
from __future__ import annotations

from typing import Tuple
import numpy as np
import pandas as pd

# sentence-transformers를 쓰는 버전 (권장: 안정적, 간단)
from sentence_transformers import SentenceTransformer


def build_embeddings(
    model_name: str,
    corpus_df: pd.DataFrame,
    query_df: pd.DataFrame,
    corpus_text_col: str = "text",
    query_text_col: str = "text",
    batch_size: int = 64,
    normalize: bool = True,
    show_progress_bar: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """
    Build embeddings for corpus and queries.

    Args:
        model_name: HF model name (e.g., "intfloat/multilingual-e5-base")
        corpus_df: DataFrame containing corpus texts
        query_df: DataFrame containing query texts
        corpus_text_col: column name for corpus text
        query_text_col: column name for query text
        batch_size: encoding batch size
        normalize: whether to L2-normalize embeddings
        show_progress_bar: show encoding progress bar

    Returns:
        (corpus_df, query_df, corpus_emb, query_emb)
          - corpus_emb: np.ndarray [N_corpus, dim]
          - query_emb : np.ndarray [N_query, dim]
    """
    # Basic validation
    if corpus_text_col not in corpus_df.columns:
        raise ValueError(f"corpus_df must contain column '{corpus_text_col}', got {list(corpus_df.columns)}")
    if query_text_col not in query_df.columns:
        raise ValueError(f"query_df must contain column '{query_text_col}', got {list(query_df.columns)}")

    corpus_texts = corpus_df[corpus_text_col].astype(str).fillna("").tolist()
    query_texts = query_df[query_text_col].astype(str).fillna("").tolist()

    # Load model
    model = SentenceTransformer(model_name)

    # Optional: E5-style prefixing (safe even if already prefixed)
    # multilingual-e5-base는 보통 "passage:" / "query:" prefix가 성능에 도움됨.
    def _maybe_prefix(texts, prefix):
        out = []
        for t in texts:
            t = t.strip()
            if t.lower().startswith(("query:", "passage:")):
                out.append(t)
            else:
                out.append(f"{prefix} {t}")
        return out

    # If user uses an E5 family model, apply prefixing automatically
    if "e5" in model_name.lower():
        corpus_texts = _maybe_prefix(corpus_texts, "passage:")
        query_texts = _maybe_prefix(query_texts, "query:")

    # Encode
    corpus_emb = model.encode(
        corpus_texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress_bar,
    )
    query_emb = model.encode(
        query_texts,
        batch_size=batch_size,
        normalize_embeddings=normalize,
        show_progress_bar=show_progress_bar,
    )

    corpus_emb = np.asarray(corpus_emb, dtype=np.float32)
    query_emb = np.asarray(query_emb, dtype=np.float32)

    return corpus_df, query_df, corpus_emb, query_emb
