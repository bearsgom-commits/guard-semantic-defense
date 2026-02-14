import json
from itertools import combinations
from typing import Dict, List, Tuple

def load_rankings_jsonl(path: str) -> Dict[int, List[int]]:
    out = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            out[int(obj["qid"])] = list(obj["topk_doc_ids"])
    return out

def topk_overlap(a: List[int], b: List[int]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / max(1, len(sa | sb))

def avg_rank_displacement(a: List[int], b: List[int]) -> float:
    pos_a = {doc: i + 1 for i, doc in enumerate(a)}
    pos_b = {doc: i + 1 for i, doc in enumerate(b)}
    common = list(set(pos_a) & set(pos_b))
    if not common:
        return 0.0
    return sum(abs(pos_a[d] - pos_b[d]) for d in common) / len(common)

def kendall_tau_on_intersection(a: List[int], b: List[int]) -> float:
    """
    Kendall tau (no ties) computed only on intersection items.
    If intersection size < 2 => 0.0
    """
    pos_a = {doc: i for i, doc in enumerate(a)}
    pos_b = {doc: i for i, doc in enumerate(b)}
    common = [d for d in a if d in pos_b]  # preserve order of a
    m = len(common)
    if m < 2:
        return 0.0

    concordant = 0
    discordant = 0
    for x, y in combinations(common, 2):
        ax = pos_a[x] - pos_a[y]
        bx = pos_b[x] - pos_b[y]
        prod = ax * bx
        if prod > 0:
            concordant += 1
        elif prod < 0:
            discordant += 1

    denom = concordant + discordant
    if denom == 0:
        return 0.0
    return (concordant - discordant) / denom

def pairwise_dynamics(
    ranks_i: Dict[int, List[int]],
    ranks_j: Dict[int, List[int]],
) -> Tuple[float, float, float]:
    qids = sorted(set(ranks_i.keys()) & set(ranks_j.keys()))
    if not qids:
        return 0.0, 0.0, 0.0

    taus, ovs, disps = [], [], []
    for qid in qids:
        a = ranks_i[qid]
        b = ranks_j[qid]
        taus.append(kendall_tau_on_intersection(a, b))
        ovs.append(topk_overlap(a, b))
        disps.append(avg_rank_displacement(a, b))

    n = len(qids)
    return sum(taus) / n, sum(ovs) / n, sum(disps) / n
