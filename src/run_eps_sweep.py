import json
import os
from datetime import datetime
import yaml
import pandas as pd

from src.utils.seed import set_seed
from src.pipeline.embed import build_embeddings
from src.pipeline.search import semantic_search
from src.pipeline.eval import evaluate
from src.guard.guard import apply_guard
from src.data.beir_scifact import load_scifact


def main():
    config = yaml.safe_load(open("configs/eps_sweep_space.yaml", "r", encoding="utf-8"))

    seed = int(config["eval"]["seed"])
    set_seed(seed)

    corpus_df, query_df = load_scifact
    query_df = query_df.sample(n=100, random_state=seed).reset_index(drop=True)
    
    corpus_df, query_df, corpus_emb, query_emb = build_embeddings(
        config["model_name"],
        #config["data"]["corpus_path"],
        #config["data"]["query_path"]
        corpus_df,
        query_df
    )

    metrics = []

    rankings_dir = config.get("output", {}).get("rankings_dir", "results/retrieval/rankings")
    os.makedirs(rankings_dir, exist_ok=True)

    for eps in config["guard"]["eps_list"]:
        eps_f = float(eps)

        guarded_corpus = apply_guard(corpus_emb, eps_f, seed=seed)
        indices = semantic_search(query_emb, guarded_corpus, topk=10)

        # ✅ eps별 랭킹 저장 (JSONL)
        eps_tag = str(eps).replace(".", "_")
        rank_path = os.path.join(rankings_dir, f"rankings_eps{eps_tag}_seed{seed}.jsonl")

        idx_list = indices.tolist() if hasattr(indices, "tolist") else indices
        with open(rank_path, "w", encoding="utf-8") as f:
            for qid, topk_ids in enumerate(idx_list):
                f.write(json.dumps({"qid": qid, "topk_doc_ids": topk_ids}, ensure_ascii=False) + "\n")

        res = evaluate(indices, query_df, corpus_df, config["eval"]["topk"])
        res["eps"] = eps_f
        res["seed"] = seed
        metrics.append(res)

    df = pd.DataFrame(metrics)

    metrics_path = config["output"]["metrics_path"]
    runlog_path = config["output"]["runlog_path"]

    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    os.makedirs(os.path.dirname(runlog_path), exist_ok=True)

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

    print(f"Done. Saved metrics: {metrics_path}")
    print(f"Done. Saved rankings: {rankings_dir}")


if __name__ == "__main__":
    main()
