import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

def evaluate(indices, query_df, corpus_df, topk_list):
    results = {}

    for k in topk_list:
        correct = 0
        for i, row in query_df.iterrows():
            gt = row["relevant_doc_id"]
            retrieved = corpus_df.iloc[indices[i][:k]]["doc_id"].values
            if gt in retrieved:
                correct += 1
        results[f"recall@{k}"] = correct / len(query_df)

    # nDCG@10
    y_true = []
    y_score = []

    for i, row in query_df.iterrows():
        gt = row["relevant_doc_id"]
        scores = [1 if corpus_df.iloc[idx]["doc_id"] == gt else 0 for idx in indices[i][:10]]
        y_true.append(scores)
        y_score.append(list(range(10, 0, -1)))

    results["ndcg@10"] = np.mean([
        ndcg_score([y_true[i]], [y_score[i]])
        for i in range(len(y_true))
    ])

    return results
