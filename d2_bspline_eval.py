from __future__ import annotations

"""Standalone harness eval for deformable variants (uses deform_lib, independent of
d2_methods.METHODS which gets clobbered by parallel edits). Reports Hungarian-boost
MRR (the real submission metric)."""

import argparse
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize

from synthetic_d2_eval import build_eval_set
from d2_methods import flat_feature, _fit_pca_ridge
from deform_lib import _deformable_to_template, _deform_norm, _bspline_to_template, _bspline_norm


def boosted_mrr(S):
    S = S.astype(np.float64).copy(); n = S.shape[0]
    r, c = linear_sum_assignment(-S); a = np.empty(n, np.int64); a[r] = c
    S[np.arange(n), a] = S.max() + 1e6
    return float(np.mean([1.0 / (np.where(np.argsort(-S[i]) == i)[0][0] + 1) for i in range(n)]))


def scores(data, norm_train, norm_eval):
    tq = np.stack([flat_feature(norm_train(v, data["train_q"].mean(0))) for v in data["train_q"]])
    tt = np.stack([flat_feature(norm_train(v, data["train_t"].mean(0))) for v in data["train_t"]])
    print("  train normalized", flush=True)
    eq = np.stack([flat_feature(norm_eval(v, data["train_q"].mean(0))) for v in data["eval_q"]])
    et = np.stack([flat_feature(norm_eval(v, data["train_t"].mean(0))) for v in data["eval_t"]])
    print("  eval normalized", flush=True)
    qp, tp, rg = _fit_pca_ridge(tq, tt, 128, 100.0)
    return (normalize(rg.predict(qp.transform(eq))) @ normalize(tp.transform(et)).T).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--n-eval", type=int, default=50)
    ap.add_argument("--n-train", type=int, default=100)
    ap.add_argument("--grid", type=int, default=44)
    a = ap.parse_args()
    data = build_eval_set(a.data_root, a.n_eval, a.n_train, 20260627, a.grid,
                          dict(max_rot_deg=12, max_shift=6, elastic_sigma=8, elastic_alpha=3))
    print(f"\nn_eval={a.n_eval} n_train={a.n_train} grid={a.grid}\n")

    variants = {
        "demons_s3":  (lambda v, t: _deformable_to_template(v, t, 3.0), lambda v, t: _deform_norm(v, t, 3.0)),
        "bspline6":   (lambda v, t: _bspline_to_template(v, t, 6), lambda v, t: _bspline_norm(v, t, 6)),
        "bspline8":   (lambda v, t: _bspline_to_template(v, t, 8), lambda v, t: _bspline_norm(v, t, 8)),
    }
    for name, (nt, ne) in variants.items():
        S = scores(data, nt, ne)
        print(f"  {name:14s} boost_mrr={boosted_mrr(S):.4f}", flush=True)


if __name__ == "__main__":
    main()
