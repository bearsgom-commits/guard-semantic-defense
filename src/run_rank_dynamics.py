import os
import yaml
import pandas as pd

from src.analysis.ranking_dynamics import load_rankings_jsonl, pairwise_dynamics

def _eps_tag(eps) -> str:
    return str(eps).replace(".", "_")

def main():
    config = yaml.safe_load(open("configs/eps_sweep_space.yaml", "r", encoding="utf-8"))

    eps_list = list(config["guard"]["eps_list"])
    seed = int(config["eval"]["seed"])
    rankings_dir = config.get("output", {}).get("rankings_dir", "results/retrieval/rankings")

    out_dir = "results/structure"
    os.makedirs(out_dir, exist_ok=True)

    # load rankings
    ranks_by_eps = {}
    for eps in eps_list:
        path = os.path.join(rankings_dir, f"rankings_eps{_eps_tag(eps)}_seed{seed}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing rankings file: {path}")
        ranks_by_eps[float(eps)] = load_rankings_jsonl(path)

    # (1) reference-to-0 dynamics
    ref_eps = float(eps_list[0])
    ref = ranks_by_eps[ref_eps]
    rows_ref = []
    for eps in map(float, eps_list[1:]):
        tau, ov, disp = pairwise_dynamics(ref, ranks_by_eps[eps])
        rows_ref.append({
            "eps_ref": ref_eps,
            "eps": eps,
            "kendall_tau": tau,
            "overlap@10": ov,
            "avg_rank_displacement": disp,
            "seed": seed
        })
    pd.DataFrame(rows_ref).to_csv(os.path.join(out_dir, "rank_tau_ref0.csv"), index=False)

    # (2) adjacent dynamics (B)
    rows_adj = []
    for i in range(len(eps_list) - 1):
        a = float(eps_list[i])
        b = float(eps_list[i + 1])
        tau, ov, disp = pairwise_dynamics(ranks_by_eps[a], ranks_by_eps[b])
        rows_adj.append({
            "eps_i": a,
            "eps_j": b,
            "kendall_tau": tau,
            "overlap@10": ov,
            "avg_rank_displacement": disp,
            "seed": seed
        })
    pd.DataFrame(rows_adj).to_csv(os.path.join(out_dir, "rank_tau_adjacent.csv"), index=False)

    # (3) overlap-only table (optional but handy for 2-page paper)
    pd.DataFrame([{
        "eps_i": r["eps_i"],
        "eps_j": r["eps_j"],
        "overlap@10": r["overlap@10"],
        "avg_rank_displacement": r["avg_rank_displacement"],
        "seed": r["seed"]
    } for r in rows_adj]).to_csv(os.path.join(out_dir, "overlap_adjacent.csv"), index=False)

    print("Done. Saved structure metrics to:", out_dir)

if __name__ == "__main__":
    main()
