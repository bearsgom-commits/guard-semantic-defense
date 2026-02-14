# src/data/beir_scifact.py
import os
import gzip
import shutil
import requests
import pandas as pd

HF_CORPUS_URL = "https://huggingface.co/datasets/BeIR/scifact/resolve/main/corpus.jsonl.gz"
HF_QUERIES_URL = "https://huggingface.co/datasets/BeIR/scifact/resolve/main/queries.jsonl.gz"
HF_QRELS_TEST_URL = "https://huggingface.co/datasets/BeIR/scifact-qrels/resolve/main/test.tsv"

def _download(url: str, out_path: str, timeout: int = 60):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return
    r = requests.get(url, stream=True, timeout=timeout)
    r.raise_for_status()
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=1024 * 1024):
            if chunk:
                f.write(chunk)

def _gunzip(gz_path: str, out_path: str):
    if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        return
    with gzip.open(gz_path, "rb") as f_in, open(out_path, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)

def load_scifact(data_dir: str = "data/beir_scifact", split: str = "test"):
    """
    Returns:
      corpus_df: columns = ["doc_id","text"]
      query_df : columns = ["query_id","text"]
      qrels    : dict(query_id -> dict(doc_id -> score))
    """
    os.makedirs(data_dir, exist_ok=True)

    corpus_gz = os.path.join(data_dir, "corpus.jsonl.gz")
    queries_gz = os.path.join(data_dir, "queries.jsonl.gz")
    qrels_tsv = os.path.join(data_dir, f"qrels_{split}.tsv")

    corpus_jsonl = os.path.join(data_dir, "corpus.jsonl")
    queries_jsonl = os.path.join(data_dir, "queries.jsonl")

    _download(HF_CORPUS_URL, corpus_gz)
    _download(HF_QUERIES_URL, queries_gz)
    _download(HF_QRELS_TEST_URL, qrels_tsv)

    _gunzip(corpus_gz, corpus_jsonl)
    _gunzip(queries_gz, queries_jsonl)

    corpus = []
    with open(corpus_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = pd.read_json(line, typ="series")
            doc_id = str(obj.get("_id"))
            title = obj.get("title") or ""
            text = obj.get("text") or ""
            merged = (str(title) + " " + str(text)).strip()
            corpus.append({"doc_id": doc_id, "text": merged})

    queries = []
    with open(queries_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            obj = pd.read_json(line, typ="series")
            qid = str(obj.get("_id"))
            qtext = str(obj.get("text") or "")
            queries.append({"query_id": qid, "text": qtext})

    # qrels: query-id, corpus-id, score (header 있음)
    qrels_df = pd.read_csv(qrels_tsv, sep="\t")
    qrels = {}
    for _, row in qrels_df.iterrows():
        qid = str(row.iloc[0])
        did = str(row.iloc[1])
        score = float(row.iloc[2])
        qrels.setdefault(qid, {})[did] = score

    corpus_df = pd.DataFrame(corpus)
    query_df = pd.DataFrame(queries)
    return corpus_df, query_df, qrels
