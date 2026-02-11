import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

def build_embeddings(model_name, corpus_path, query_path):
    model = SentenceTransformer(model_name)

    corpus_df = pd.read_csv(corpus_path)
    query_df = pd.read_csv(query_path)

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
