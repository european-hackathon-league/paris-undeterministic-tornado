from __future__ import annotations

"""Production-equivalent synthetic-d2 evaluation.

The synthetic harness reports raw_mrr (argsort, no Hungarian) and hungarian_acc.
But the SUBMITTED pipeline applies a Hungarian boost (assigned target forced to
rank 1) before ranking. Real d2 MRR (~0.749 for template) corresponds to that
boosted ranking, NOT raw_mrr. This script reports the boost-MRR so method
comparisons reflect what we actually submit.

Reuses d2_methods + the synthetic eval-set builder. All methods share the cached
template score matrix within one run, so registration runs once.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from synthetic_d2_eval import build_eval_set
import d2_methods


def boosted_mrr(scores: np.ndarray) -> float:
    """MRR after Hungarian assignment boosts the assigned target to rank 1
    (exactly what d2_template_retrieval does at submission time). Diagonal=truth."""
    s = scores.astype(np.float64).copy()
    n = s.shape[0]
    row, col = linear_sum_assignment(-s)
    assigned = np.empty(n, dtype=np.int64)
    assigned[row] = col
    s[np.arange(n), assigned] = s.max() + 1e6
    rr = []
    for i in range(n):
        order = np.argsort(-s[i], kind="mergesort")
        rank = int(np.where(order == i)[0][0]) + 1
        rr.append(1.0 / rank)
    return float(np.mean(rr))


def raw_mrr(scores: np.ndarray) -> float:
    n = scores.shape[0]
    rr = []
    for i in range(n):
        order = np.argsort(-scores[i], kind="mergesort")
        rank = int(np.where(order == i)[0][0]) + 1
        rr.append(1.0 / rank)
    return float(np.mean(rr))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--n-train", type=int, default=250)
    p.add_argument("--seed", type=int, default=20260627)
    p.add_argument("--grid", type=int, default=44)
    p.add_argument("--max-rot-deg", type=float, default=12.0)
    p.add_argument("--max-shift", type=float, default=6.0)
    p.add_argument("--elastic-sigma", type=float, default=8.0)
    p.add_argument("--elastic-alpha", type=float, default=3.0)
    p.add_argument("--methods", nargs="+", default=["template", "template_sinkhorn"])
    return p.parse_args()


def main():
    a = parse_args()
    data = build_eval_set(
        a.data_root, a.n_eval, a.n_train, a.seed, a.grid,
        dict(max_rot_deg=a.max_rot_deg, max_shift=a.max_shift,
             elastic_sigma=a.elastic_sigma, elastic_alpha=a.elastic_alpha),
    )
    print(f"\nprod-eval n_eval={a.n_eval} grid={a.grid} "
          f"rot={a.max_rot_deg} shift={a.max_shift} "
          f"elastic(sig={a.elastic_sigma},a={a.elastic_alpha})\n")
    print(f"  {'method':22s} {'raw_mrr':>9s} {'boost_mrr':>10s}")
    for m in a.methods:
        S = d2_methods.METHODS[m](data)
        print(f"  {m:22s} {raw_mrr(S):9.4f} {boosted_mrr(S):10.4f}", flush=True)


if __name__ == "__main__":
    main()
