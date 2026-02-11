import json
from datetime import datetime
import yaml
import pandas as pd

from src.utils.seed import set_seed
from src.pipeline.02_embed import build_embeddings
from src.pipeline.03_search import semantic_search
from src.pipeline.04_eval import evaluate
from src.guard.guard import apply_guard


def main():
    config = yaml.safe_load(open("configs/eps_sweep_space.yaml", "r", encoding="utf-8"))

    seed = int(config["eval"]["seed"])
    set_seed(seed)

    corpus_df, query_df, corpus_emb, query_emb = build_embeddings(
        config["model_name"],
        config["data"]["corpus_path"],
        config["data"]["query_path"]
    )

    metrics = []
    for eps in config["guard"]["eps_list"]:
        guarded_corpus = apply_guard(corpus_emb, float(eps))
        indices = semantic_search(query_emb, guarded_corpus, topk=10)

        res = evaluate(indices, query_df, corpus_df, config["eval"]["topk"])
        res["eps"] = float(eps)
        res["seed"] = seed
        metrics.append(res)

    df = pd.DataFrame(metrics)
    metrics_path = config["output"]["metrics_path"]
    runlog_path = config["output"]["runlog_path"]

    # ensure output dirs exist (Actions 환경)
    import os
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)

    df.to_csv(metrics_path, index=False)

    log = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "metrics_path": metrics_path,
        "config": config,
    }
    with open(runlog_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")

    print(f"Done. Saved: {metrics_path}")


if __name__ == "__main__":
    main()

