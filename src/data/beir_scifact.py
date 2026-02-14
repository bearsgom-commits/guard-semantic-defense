# src/data/beir_scifact.py
from __future__ import annotations
from typing import Tuple, Dict
import os
import pandas as pd

def load_scifact(data_dir: str = "data/beir_scifact") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    BEIR SciFact를 공식 beir 로더로 다운로드/로딩하여
    corpus_df: [doc_id, text]
    query_df : [qid, text, gold_id]  (대표 1개 정답)
    를 반환한다.
    """
    from beir.util import download_and_unzip
    from beir.datasets.data_loader import GenericDataLoader

    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
    os.makedirs(data_dir, exist_ok=True)

    # 다운로드/압축해제 (이미 있으면 재사용)
    zip_path = os.path.join(data_dir, "scifact.zip")
    if not (os.path.exists(os.path.join(data_dir, "corpus.jsonl")) and
            os.path.exists(os.path.join(data_dir, "queries.jsonl"))):
        download_and_unzip(url, data_dir)

    # 로딩
    corpus, queries, qrels = GenericDataLoader(data_folder=data_dir).load(split="test")

    # corpus_df
    # corpus: dict[doc_id] -> {"title":..., "text":...}
    corpus_rows = []
    for doc_id, doc in corpus.items():
        title = (doc.get("title") or "").strip()
        text = (doc.get("text") or "").strip()
        merged = (title + " " + text).strip()
        corpus_rows.append({"doc_id": str(doc_id), "text": merged})
    corpus_df = pd.DataFrame(corpus_rows)

    # query_df
    # queries: dict[qid] -> query_text
    query_rows = [{"qid": str(qid), "text": str(qtext)} for qid, qtext in queries.items()]
    query_df = pd.DataFrame(query_rows)

    # qrels: dict[qid] -> dict[doc_id] -> relevance
    # 대표 gold_id: relevance가 가장 큰 doc 하나
    gold_map: Dict[str, str] = {}
    for qid, rels in qrels.items():
        if not rels:
            continue
        # 가장 높은 relevance doc 선택
        best_doc = max(rels.items(), key=lambda x: x[1])[0]
        gold_map[str(qid)] = str(best_doc)

    query_df["gold_id"] = query_df["qid"].map(gold_map)
    query_df = query_df.dropna(subset=["gold_id"]).reset_index(drop=True)

    return corpus_df, query_df
