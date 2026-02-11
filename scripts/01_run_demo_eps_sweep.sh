#!/bin/bash

python - <<EOF
import yaml
import json
import pandas as pd
from datetime import datetime
from src.utils.seed import set_seed
from src.pipeline.02_embed import build_embeddings
from src.pipeline.03_search import semantic_search
from src.pipeline.04_eval import evaluate
from src.guard.guard import apply_guard

config = yaml.safe_load(open("configs/eps_sweep_space.yaml"))

set_seed(config["eval"]["seed"])

corpus_df, query_df, corpus_emb, query_emb = build_embeddings(
    config["model_name"],
    config["data"]["corpus_path"],
    config["data"]["query_path"]
)

metrics = []

for eps in config["guard"]["eps_list"]:
    guarded_corpus = apply_guard(corpus_emb, eps)
    indices = semantic_search(query_emb, guarded_corpus, topk=10)
    res = evaluate(indices, query_df, corpus_df, config["eval"]["topk"])
    res["eps"] = eps
    res["seed"] = config["eval"]["seed"]
    metrics.append(res)

df = pd.DataFrame(metrics)
df.to_csv(config["output"]["metrics_path"], index=False)

log = {
    "timestamp": str(datetime.now()),
    "config": config
}
with open(config["output"]["runlog_path"], "a") as f:
    f.write(json.dumps(log) + "\n")

print("Done. Results saved.")
EOF
