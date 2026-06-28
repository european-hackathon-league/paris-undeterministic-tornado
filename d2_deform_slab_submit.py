from __future__ import annotations

"""dataset2 partial via FUSION of deformable-template scores + rigid per-slab scores.

Both are base-similarity improvements (not re-ranking), so they transfer to real d2
(unlike Sinkhorn). prod-eval: deform 0.672 -> deform+slab 0.710 (Hungarian-boost MRR).
Model (templates, PCA/Ridge maps, per-slab maps) is fit ONCE on d1 train; both splits
reuse it. d2 deform features reuse .deformcache; slab uses rigid normalization.
"""

import argparse, csv
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize

from synthetic_d2_eval import _load_grid, read_pairs
from d2_methods import flat_feature, _fit_pca_ridge, _register_to_template
from deform_lib import _deformable_to_template, _slab_feats, _zscore_rows


def read_csv(p):
    with open(p, newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=Path, default=Path("data"))
    ap.add_argument("--grid", type=int, default=44)
    ap.add_argument("--smooth", type=float, default=3.0)
    ap.add_argument("--slab-k", type=int, default=10)
    ap.add_argument("--slab-s", type=int, default=16)
    ap.add_argument("--cache", type=Path, default=Path(".deformcache"))
    ap.add_argument("--out", type=Path, default=Path("submissions/_part_d2_deformslab_g44.csv"))
    a = ap.parse_args()
    a.cache.mkdir(exist_ok=True)
    root = a.data_root; G = a.grid; SM = a.smooth

    pairs = read_pairs(root)
    tq = np.stack([_load_grid(root, p["query_image"], G) for p in pairs])
    tt = np.stack([_load_grid(root, p["target_image"], G) for p in pairs])
    t1 = tq.mean(0); t2 = tt.mean(0)

    # --- deform model ---
    print("deforming train...", flush=True)
    dqf = np.stack([flat_feature(_deformable_to_template(v, t1, SM)) for v in tq])
    dtf = np.stack([flat_feature(_deformable_to_template(v, t2, SM)) for v in tt])
    dq_pca, dt_pca, dridge = _fit_pca_ridge(dqf, dtf, 128, 100.0)

    # --- slab model (rigid-normalized, per-slab maps) ---
    print("slab train...", flush=True)
    TQs = np.stack([_slab_feats(v, a.slab_k, a.slab_s) for v in tq])  # (N,K,F) train clean
    TTs = np.stack([_slab_feats(v, a.slab_k, a.slab_s) for v in tt])
    slab_maps = [_fit_pca_ridge(TQs[:, k], TTs[:, k], 32, 100.0) for k in range(a.slab_k)]

    def deform_feat(path, iid, tpl, tag):
        cf = a.cache / f"{iid}_{tag}.npy"
        if cf.exists(): return np.load(cf)
        f = flat_feature(_deformable_to_template(_register_to_template(_load_grid(root, path, G), tpl), tpl, SM))
        np.save(cf, f); return f

    def slab_codes(vol_paths, ids, tpl, side):  # rigid-normalized slab codes
        out = []
        for p, iid in zip(vol_paths, ids):
            v = _register_to_template(_load_grid(root, p, G), tpl)
            out.append(_slab_feats(v, a.slab_k, a.slab_s))
        F = np.stack(out)  # (n,K,Fdim)
        codes = []
        for k in range(a.slab_k):
            qp, tp, rg = slab_maps[k]
            if side == "q":
                codes.append(normalize(rg.predict(qp.transform(F[:, k]))))
            else:
                codes.append(normalize(tp.transform(F[:, k])))
        return np.stack(codes, axis=1)  # (n,K,c)

    rows = []
    for split in ("val", "test"):
        qrows = read_csv(root / "dataset2" / f"{split}_queries.csv")
        grows = read_csv(root / "dataset2" / f"{split}_gallery.csv")
        # deform scores
        Q = np.stack([normalize(dridge.predict(dq_pca.transform(
            deform_feat(r["query_image"], r["query_id"], t1, f"q{G}s{SM}")[None, :])))[0] for r in qrows])
        T = np.stack([normalize(dt_pca.transform(
            deform_feat(r["target_image"], r["target_id"], t2, f"t{G}s{SM}")[None, :]))[0] for r in grows])
        S_def = (Q @ T.T).astype(np.float64)
        print(f"  deform scores {split} done", flush=True)
        # slab scores
        Qs = slab_codes([r["query_image"] for r in qrows], [r["query_id"] for r in qrows], t1, "q")
        Ts = slab_codes([r["target_image"] for r in grows], [r["target_id"] for r in grows], t2, "t")
        S_slab = np.einsum("ikc,jkc->ij", Qs, Ts).astype(np.float64)
        print(f"  slab scores {split} done", flush=True)
        # fuse
        S = _zscore_rows(S_def) + _zscore_rows(S_slab)
        ri, ci = linear_sum_assignment(-S)
        assigned = np.empty(S.shape[0], dtype=np.int64); assigned[ri] = ci
        S[np.arange(S.shape[0]), assigned] = S.max() + 1e6
        tids = np.asarray([r["target_id"] for r in grows])
        for qi, r in enumerate(qrows):
            order = np.argsort(-S[qi], kind="mergesort")
            rows.append({"query_id": r["query_id"], "target_id_ranking": " ".join(tids[order].tolist())})
        print(f"scored d2/{split}", flush=True)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"]); w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} rows to {a.out}")


if __name__ == "__main__":
    main()
