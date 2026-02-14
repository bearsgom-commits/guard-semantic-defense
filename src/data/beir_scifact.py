# src/data/beir_scifact.py
from __future__ import annotations
from typing import Tuple, Dict
import os
import pandas as pd

def load_scifact(root_dir: str = "data/beir_scifact") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    BEIR SciFact를 beir 공식 로더로 내려받아
    corpus_df: [doc_id, text]
    query_df : [qid, text, gold_id]
    반환
    """
    from beir.util import download_and_unzip
    from beir.datasets.data_loader import GenericDataLoader

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    os.makedirs(root_dir, exist_ok=True)

    # ✅ 중요: download_and_unzip은 '실제 데이터 폴더 경로'를 반환함
    data_dir = download_and_unzip(url, root_dir)
    # 예: data_dir == "data/beir_scifact/scifact"

    corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="test")

    # corpus_df
    corpus_rows = []
    for doc_id, doc in corpus.items():
        title = (doc.get("title") or "").strip()
        text = (doc.get("text") or "").strip()
        merged = (title + " " + text).strip()
        corpus_rows.append({"doc_id": str(doc_id), "text": merged})
    corpus_df = pd.DataFrame(corpus_rows)

    # query_df
    query_rows = [{"qid": str(qid), "text": str(qtext)} for qid, qtext in queries.items()]
    query_df = pd.DataFrame(query_rows)

    # 대표 gold_id (relevance 최대)
    gold_map: Dict[str, str] = {}
    for qid, rels in qrels.items():
        if not rels:
            continue
        best_doc = max(rels.items(), key=lambda x: x[1])[0]
        gold_map[str(qid)] = str(best_doc)

    query_df["gold_id"] = query_df["qid"].map(gold_map)
    query_df = query_df.dropna(subset=["gold_id"]).reset_index(drop=True)

    return corpus_df, query_df
