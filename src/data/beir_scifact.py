# src/data/beir_scifact.py
from __future__ import annotations
from typing import Tuple, Dict, List
import pandas as pd

def load_scifact() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns
    -------
    corpus_df: columns = ["doc_id", "text"]
    query_df : columns = ["qid", "text", "gold_id"]
      - gold_id는 SciFact의 qrels에서 relevance>0인 문서 중 1개를 대표로 사용
      - (구조연구 목적이라 1개 대표 정답으로도 충분)
    """
    # BEIR SciFact (via HF datasets)
    # dataset name may vary by mirror; the most common is "BeIR/scifact"
    from datasets import load_dataset

    # 1) corpus
    corpus = load_dataset("BeIR/scifact", "corpus", split="corpus")
    corpus_df = pd.DataFrame({
        "doc_id": corpus["doc_id"],
        "text": corpus["title"],  # title+text 붙일 수도 있음
    })
    # 문서 본문(text)까지 쓰고 싶으면:
    # corpus_df["text"] = corpus_df["text"].fillna("") + " " + pd.Series(corpus["text"]).fillna("")

    # 2) queries
    queries = load_dataset("BeIR/scifact", "queries", split="queries")
    query_df = pd.DataFrame({
        "qid": queries["query_id"],
        "text": queries["text"] if "text" in queries.column_names else queries["query"],
    })

    # 3) qrels: query->relevant doc(s)
    qrels = load_dataset("BeIR/scifact", "qrels", split="test")
    # qrels columns usually: query-id, corpus-id, score
    qrels_df = pd.DataFrame({
        "qid": qrels["query-id"],
        "doc_id": qrels["corpus-id"],
        "score": qrels["score"],
    })
    qrels_df = qrels_df[qrels_df["score"] > 0]

    # 대표 gold_id: 가장 먼저 매칭되는 doc_id 1개
    gold_map: Dict[str, str] = (
        qrels_df.sort_values(["qid", "score"], ascending=[True, False])
               .groupby("qid")["doc_id"]
               .first()
               .to_dict()
    )
    query_df["gold_id"] = query_df["qid"].map(gold_map)

    # gold_id 없는 query 제거 (일부 split/버전 차이 대비)
    query_df = query_df.dropna(subset=["gold_id"]).reset_index(drop=True)

    # doc_id, qid 타입을 문자열로 통일 (조인 안정성)
    corpus_df["doc_id"] = corpus_df["doc_id"].astype(str)
    query_df["qid"] = query_df["qid"].astype(str)
    query_df["gold_id"] = query_df["gold_id"].astype(str)

    return corpus_df, query_df
