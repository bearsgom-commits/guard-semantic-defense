# src/run_eps_sweep.py
import os
import json
from datetime import datetime
import yaml
import pandas as pd
import numpy as np

from src.utils.seed import set_seed
from src.data.beir_scifact import load_scifact
from src.pipeline.embed import build_embeddings
from src.pipeline.search import semantic_search
from src.pipeline.eval import evaluate
from src.guard.guard import apply_guard

def main():
    config = yaml.safe_load(open("configs/eps_sweep_space.yaml", "r", encoding="utf-8"))

    seed = int(config["eval"]["seed"])
    set_seed(seed)

    # (1) BEIR SciFact 로드
    corpus_df, query_df, qrels = load_scifact(
        data_dir=config["data"].get("beir_dir", "data/beir_scifact"),
        split=config["data"].get("split", "test")
    )

    # (선택) 데모용 샘플링
    n_docs = int(config["data"].get("max_docs", 0))
    n_q = int(config["data"].get("max_queries", 0))
    if n_docs > 0 and n_docs < len(corpus_df):
        corpus_df = corpus_df.sample(n=n_docs, random_state=seed).reset_index(drop=True)
        # 샘플링 시 qrels에 없는 문서가 많아질 수 있어도 recall 정의상 동작은 함(성능은 떨어질 수 있음).
    if n_q > 0 and n_q < len(query_df):
        query_df = query_df.sample(n=n_q, random_state=seed).reset_index(drop=True)

    corpus_df, query_df, corpus_emb, query_emb = build_embeddings(
        config["model_name"], corpus_df, query_df
    )

    metrics = []
    eps_list = config["guard"]["eps_list"]

    # outputs
    metrics_path = config["output"]["metrics_path"]
    runlog_path = config["output"]["runlog_path"]
    rankings_dir = config["output"].get("rankings_dir", "results/retrieval/rankings")

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(runlog_path), exist_ok=True)
    os.makedirs(rankings_dir, exist_ok=True)

    for eps in eps_list:
        eps = float(eps)

        guarded_corpus = apply_guard(corpus_emb, eps)
        indices = semantic_search(query_emb, guarded_corpus, topk=int(config["eval"].get("search_topk", 10)))

        # (B 준비) 랭킹 저장: query_id별 topK doc_id
        topk_save = indices.shape[1]
        doc_ids = corpus_df["doc_id"].astype(str).tolist()
        qids = query_df["query_id"].astype(str).tolist()

        rows = []
        for i, qid in enumerate(qids):
            ranked = [doc_ids[j] for j in indices[i]]
            rows.append({"query_id": qid, "ranked_doc_ids": json.dumps(ranked)})
        pd.DataFrame(rows).to_csv(
            os.path.join(rankings_dir, f"rank_eps{eps:g}_seed{seed}_top{topk_save}.csv"),
            index=False
        )

        res = evaluate(indices, query_df, corpus_df, qrels, topk=config["eval"]["topk"])
        res["eps"] = eps
        res["seed"] = seed
        metrics.append(res)

    df = pd.DataFrame(metrics)
    df.to_csv(metrics_path, index=False)

    log = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "metrics_path": metrics_path,
        "rankings_dir": rankings_dir,
        "config": config,
    }
    with open(runlog_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

    print(f"Done. Saved: {metrics_path}")

if __name__ == "__main__":
    main()
