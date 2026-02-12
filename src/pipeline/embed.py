import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer


def build_embeddings(model_name: str, corpus_path: str, query_path: str):
    """
    Load corpus/query CSVs and build normalized embeddings.
    corpus CSV columns: doc_id, sentence
    query  CSV columns: query_id, query_text, relevant_doc_id
    """
    model = SentenceTransformer(model_name)

    corpus_df = pd.read_csv(corpus_path)
    query_df = pd.read_csv(query_path)

    # basic validation (helps debugging in Actions)
    required_corpus_cols = {"doc_id", "sentence"}
    required_query_cols = {"query_id", "query_text", "relevant_doc_id"}

    if corpus_df.empty:
        raise ValueError(f"Corpus CSV is empty: {corpus_path}")
    if query_df.empty:
        raise ValueError(f"Query CSV is empty: {query_path}")

    if not required_corpus_cols.issubset(set(corpus_df.columns)):
        raise ValueError(f"Corpus CSV must contain columns {required_corpus_cols}, got {set(corpus_df.columns)}")
    if not required_query_cols.issubset(set(query_df.columns)):
        raise ValueError(f"Query CSV must contain columns {required_query_cols}, got {set(query_df.columns)}")

    corpus_embeddings = model.encode(
        corpus_df["sentence"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    )

    query_embeddings = model.encode(
        query_df["query_text"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    )

    return corpus_df, query_df, np.array(corpus_embeddings), np.array(query_embeddings)
