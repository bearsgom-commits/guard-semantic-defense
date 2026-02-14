# src/run_rank_dynamics.py
from __future__ import annotations

import os
import re
import json
import glob
import yaml
import numpy as np
import pandas as pd
from scipy.stats import kendalltau


RANK_FILE_RE = re.compile(r"rank_eps(?P<eps>[0-9.]+)_seed(?P<seed>\d+)_top(?P<topk>\d+)\.csv$")


def _load_rank_csv(path: str) -> dict[str, list[str]]:
    """
    CSV format:
      query_id, ranked_doc_ids   (ranked_doc_ids is JSON list string)
    Returns:
      dict[query_id] -> list[doc_id] (ranked list)
    """
    df = pd.read_csv(path)
    if "query_id" not in df.columns or "ranked_doc_ids" not in df.columns:
        raise ValueError(f"Unexpected columns in {path}: {list(df.columns)}")
    out = {}
    for _, row in df.iterrows():
        qid = str(row["query_id"])
        ranked = json.loads(row["ranked_doc_ids"])
        out[qid] = [str(x) for x in ranked]
    return out


def _kendall_tau_for_query(rank_a: list[str], rank_b: list[str], k: int) -> float:
    """
    Compute Kendall's tau on the intersection set within top-k using induced ranks.
    If intersection < 2, return np.nan.
    """
    a = rank_a[:k]
    b = rank_b[:k]
    set_a = set(a)
    set_b = set(b)
    inter = list(set_a.intersection(set_b))
    if len(inter) < 2:
        return np.nan

    pos_a = {doc: i for i, doc in enumerate(a)}
    pos_b = {doc: i for i, doc in enumerate(b)}

    x = [pos_a[d] for d in inter]
    y = [pos_b[d] for d in inter]

    tau, _ = kendalltau(x, y)
    return float(tau) if tau is not None else np.nan


def _topk_overlap(rank_a: list[str], rank_b: list[str], k: int) -> float:
    a = set(rank_a[:k])
    b = set(rank_b[:k])
    if k <= 0:
        return np.nan
    return float(len(a.intersection(b)) / k)


def main():
    config = yaml.safe_load(open("configs/eps_sweep_space.yaml", "r", encoding="utf-8"))

    seed = int(config["eval"]["seed"])
    rankings_dir = config["output"].get("rankings_dir", "results/retrieval/rankings")
    out_path = config["output"].get("rank_dynamics_path", "results/retrieval/rank_tau_adjacent.csv")

    # Find ranking files
    files = sorted(glob.glob(os.path.join(rankings_dir, f"rank_eps*_seed{seed}_top*.csv")))
    if not files:
        raise FileNotFoundError(f"No ranking csv files found in: {rankings_dir}")

    # Parse eps/topk from filenames
    parsed = []
    for f in files:
        m = RANK_FILE_RE.search(os.path.basename(f))
        if not m:
            continue
        eps = float(m.group("eps"))
        topk = int(m.group("topk"))
        parsed.append((eps, topk, f))

    if not parsed:
        raise FileNotFoundError(f"Ranking files exist but none match expected pattern in: {rankings_dir}")

    # Use the smallest common topk across files (safe)
    common_topk = min(t for _, t, _ in parsed)

    # Sort by eps
    parsed.sort(key=lambda x: x[0])

    # Load all rankings into memory
    rank_by_eps = {}
    for eps, topk, fpath in parsed:
        ranks = _load_rank_csv(fpath)
        rank_by_eps[eps] = ranks

    eps_list = [x[0] for x in parsed]
    rows = []

    for i in range(len(eps_list) - 1):
        eps_a = eps_list[i]
        eps_b = eps_list[i + 1]

        ra = rank_by_eps[eps_a]
        rb = rank_by_eps[eps_b]

        # common queries only
        qids = sorted(set(ra.keys()).intersection(set(rb.keys())))
        if not qids:
            continue

        taus = []
        overlaps10 = []
        top1_changes = []

        for qid in qids:
            rank_a = ra[qid]
            rank_b = rb[qid]

            taus.append(_kendall_tau_for_query(rank_a, rank_b, k=common_topk))
            overlaps10.append(_topk_overlap(rank_a, rank_b, k=min(10, common_topk)))

            # "위협 대응 문서가 바뀌는 비율" = Top-1이 바뀐 비율
            top1_changes.append(1.0 if rank_a[0] != rank_b[0] else 0.0)

        tau_mean = float(np.nanmean(taus)) if np.any(~np.isnan(taus)) else np.nan

        rows.append({
            "seed": seed,
            "topk_used": common_topk,
            "eps_from": eps_a,
            "eps_to": eps_b,
            "kendall_tau_mean": tau_mean,
            "top10_overlap_mean": float(np.mean(overlaps10)),
            "top1_change_rate": float(np.mean(top1_changes)),
            "n_queries": int(len(qids)),
        })

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f"Done. Saved: {out_path}")


if __name__ == "__main__":
    main()
