from __future__ import annotations

"""Generate the dataset2 partial using deformable (elastic) template normalization.

Pipeline per volume: rigid-align to same-modality d1 mean template, then a
regularized symmetric Demons deformable refinement (smooth=3.0) to remove the
residual elastic warp. Features feed the d1-trained PCA/Ridge cross-modal map;
Hungarian assignment as usual. Validated on the synthetic-d2 prod-eval:
template rigid boost_mrr 0.614 -> deformable(smooth3) 0.672.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize

from synthetic_d2_eval import _load_grid, read_pairs
from d2_methods import flat_feature, _fit_pca_ridge, _register_to_template
from deform_lib import _deformable_to_template


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def deform_norm(vol, tpl, smooth):
    return _deformable_to_template(_register_to_template(vol, tpl), tpl, smooth=smooth)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--grid", type=int, default=44)
    ap.add_argument("--smooth", type=float, default=3.0)
    ap.add_argument("--components", type=int, default=128)
    ap.add_argument("--alpha", type=float, default=100.0)
    ap.add_argument("--cache", type=Path, default=Path(".deformcache"))
    ap.add_argument("--out", type=Path, default=Path("submissions/_part_d2_deform_g44.csv"))
    a = ap.parse_args()
    a.cache.mkdir(exist_ok=True)
    root = a.data_root

    pairs = read_pairs(root)
    tq = np.stack([_load_grid(root, p["query_image"], a.grid) for p in pairs])
    tt = np.stack([_load_grid(root, p["target_image"], a.grid) for p in pairs])
    t1 = tq.mean(0); t2 = tt.mean(0)
    print("deforming train pairs...", flush=True)
    qf = np.stack([flat_feature(_deformable_to_template(v, t1, a.smooth)) for v in tq])
    tf = np.stack([flat_feature(_deformable_to_template(v, t2, a.smooth)) for v in tt])
    q_pca, t_pca, ridge = _fit_pca_ridge(qf, tf, a.components, a.alpha)
    print("fit PCA/Ridge on deformed train", flush=True)

    def feat(image_path, image_id, tpl, tag):
        cf = a.cache / f"{image_id}_{tag}.npy"
        if cf.exists():
            return np.load(cf)
        v = _load_grid(root, image_path, a.grid)
        f = flat_feature(deform_norm(v, tpl, a.smooth))
        np.save(cf, f)
        return f

    rows = []
    for split in ("val", "test"):
        qrows = read_csv(root / "dataset2" / f"{split}_queries.csv")
        grows = read_csv(root / "dataset2" / f"{split}_gallery.csv")
        Q = []
        for i, r in enumerate(qrows):
            f = feat(r["query_image"], r["query_id"], t1, f"q{a.grid}s{a.smooth}")
            Q.append(normalize(ridge.predict(q_pca.transform(f[None, :])))[0])
            if (i + 1) % 20 == 0: print(f"  d2/{split} q {i+1}/{len(qrows)}", flush=True)
        T = []
        for i, r in enumerate(grows):
            f = feat(r["target_image"], r["target_id"], t2, f"t{a.grid}s{a.smooth}")
            T.append(normalize(t_pca.transform(f[None, :]))[0])
            if (i + 1) % 20 == 0: print(f"  d2/{split} t {i+1}/{len(grows)}", flush=True)
        S = (np.stack(Q) @ np.stack(T).T).astype(np.float64)
        ri, ci = linear_sum_assignment(-S)
        assigned = np.empty(S.shape[0], dtype=np.int64); assigned[ri] = ci
        S[np.arange(S.shape[0]), assigned] = S.max() + 1e6
        tids = np.asarray([r["target_id"] for r in grows])
        for qi, r in enumerate(qrows):
            order = np.argsort(-S[qi], kind="mergesort")
            rows.append({"query_id": r["query_id"], "target_id_ranking": " ".join(tids[order].tolist())})
        print(f"scored d2/{split}: {S.shape}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"]); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
