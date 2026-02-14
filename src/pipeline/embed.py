# src/pipeline/embed.py
from __future__ import annotations
from typing import Tuple, Union
import numpy as np
import pandas as pd

def build_embeddings(
    model_name: str,
    corpus_source: Union[str, pd.DataFrame],
    query_source: Union[str, pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:

    if isinstance(corpus_source, str):
        corpus_df = pd.read_csv(corpus_source)
    else:
        corpus_df = corpus_source.copy()

    if isinstance(query_source, str):
        query_df = pd.read_csv(query_source)
    else:
        query_df = query_source.copy()

    # 필수 컬럼 보정
    if "doc_id" not in corpus_df.columns:
        corpus_df = corpus_df.reset_index(drop=True)
        corpus_df["doc_id"] = corpus_df.index.astype(str)

    if "qid" not in query_df.columns:
        query_df = query_df.reset_index(drop=True)
        query_df["qid"] = query_df.index.astype(str)

    # text 컬럼 이름 정규화
    if "text" not in corpus_df.columns:
        raise ValueError("corpus_df must have a 'text' column")
    if "text" not in query_df.columns:
        raise ValueError("query_df must have a 'text' column")

    # ---- 임베딩 생성 (기존 로직 유지) ----
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)

    corpus_texts = corpus_df["text"].astype(str).tolist()
    query_texts  = query_df["text"].astype(str).tolist()

    corpus_emb = model.encode(corpus_texts, normalize_embeddings=True, show_progress_bar=True)
    query_emb  = model.encode(query_texts,  normalize_embeddings=True, show_progress_bar=True)

    return corpus_df, query_df, np.asarray(corpus_emb), np.asarray(query_emb)
