from __future__ import annotations

"""Cross-validated d1 error analysis.

d1 train has labels and is the SAME distribution as d1 val/test (registered
cross-modal T1ce->T2). So k-fold CV on the 350 train pairs is a faithful proxy
for the real d1 val/test MRR (~0.964).

For every held-out query we record, using the PRODUCTION pipeline
(template-normalized feature -> PCA/Ridge map -> cosine -> Hungarian):
  - true target rank BEFORE Hungarian (1 = already correct)
  - top1 - top2 score gap (confidence)
  - whether query<->true target are MUTUAL nearest neighbours
  - whether Hungarian assigns the correct target
  - a left/right FLIP probe: does mirroring the query raise the true-pair cosine?

The point: see whether the 4% tail is rank-2/3 (rerankable) or far (broken),
and whether a systematic preprocessing bug (e.g. L/R flip) is in play.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize

from synthetic_d2_eval import _load_grid, read_pairs
from d2_methods import _register_to_template, flat_feature, _fit_pca_ridge


def reg_feature(data_root, image_path, image_id, grid, template, cache_dir, flip=False):
    tag = f"g{grid}" + ("_flip" if flip else "")
    cache = cache_dir / f"{image_id}_{tag}.npy"
    if cache.exists():
        return np.load(cache)
    vol = _load_grid(data_root, image_path, grid)
    if flip:
        vol = vol[::-1].copy()  # mirror along first spatial axis (L/R)
    feat = flat_feature(_register_to_template(vol, template))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, feat)
    return feat


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=Path, default=Path("data"))
    p.add_argument("--cache-dir", type=Path, default=Path(".d2cache"))
    p.add_argument("--grid", type=int, default=44)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--components", type=int, default=128)
    p.add_argument("--alpha", type=float, default=100.0)
    p.add_argument("--flip-probe", action="store_true")
    args = p.parse_args()

    pairs = read_pairs(args.data_root)
    n = len(pairs)
    print(f"d1 train pairs: {n}", flush=True)

    # templates from ALL train (matches production build_model)
    tq_vol = np.stack([_load_grid(args.data_root, p["query_image"], args.grid) for p in pairs])
    tt_vol = np.stack([_load_grid(args.data_root, p["target_image"], args.grid) for p in pairs])
    t1_tpl = tq_vol.mean(axis=0)
    t2_tpl = tt_vol.mean(axis=0)
    print("templates built", flush=True)

    qf = np.zeros((n, args.grid ** 3), dtype=np.float32)
    tf = np.zeros((n, args.grid ** 3), dtype=np.float32)
    qf_flip = None
    if args.flip_probe:
        qf_flip = np.zeros((n, args.grid ** 3), dtype=np.float32)
    for i, pr in enumerate(pairs):
        qf[i] = reg_feature(args.data_root, pr["query_image"], pr["query_id"], args.grid, t1_tpl, args.cache_dir)
        tf[i] = reg_feature(args.data_root, pr["target_image"], pr["target_id"], args.grid, t2_tpl, args.cache_dir)
        if args.flip_probe:
            qf_flip[i] = reg_feature(args.data_root, pr["query_image"], pr["query_id"], args.grid, t1_tpl, args.cache_dir, flip=True)
        if (i + 1) % 25 == 0:
            print(f"  features {i+1}/{n}", flush=True)

    rng = np.random.default_rng(20260628)
    fold = rng.integers(0, args.folds, size=n)

    total_correct = 0
    errors = []  # dicts
    flip_helps = 0
    flip_tested = 0
    rank_hist = {}

    for f in range(args.folds):
        te = np.where(fold == f)[0]
        tr = np.where(fold != f)[0]
        if len(te) < 3:
            continue
        q_pca, t_pca, ridge = _fit_pca_ridge(qf[tr], tf[tr], args.components, args.alpha)
        Q = normalize(ridge.predict(q_pca.transform(qf[te])))
        T = normalize(t_pca.transform(tf[te]))
        S = (Q @ T.T).astype(np.float64)  # rows=held-out queries, cols=held-out targets; diag=truth
        m = len(te)

        # Hungarian assignment accuracy (the real metric driver)
        row, col = linear_sum_assignment(-S)
        assigned = np.empty(m, dtype=int)
        assigned[row] = col
        total_correct += int(np.sum(assigned == np.arange(m)))

        # per-query diagnostics (pre-Hungarian ranking)
        col_argmax = S.argmax(axis=0)  # for mutual-NN: best query per target
        for li in range(m):
            order = np.argsort(-S[li], kind="mergesort")
            rank = int(np.where(order == li)[0][0]) + 1
            rank_hist[rank] = rank_hist.get(rank, 0) + 1
            s_sorted = np.sort(S[li])[::-1]
            gap = float(s_sorted[0] - s_sorted[1]) if m > 1 else float("inf")
            top1_target = int(order[0])
            mutual = (col_argmax[top1_target] == li)
            if assigned[li] != li:
                gi = te[li]
                rec = dict(
                    pair=pairs[gi]["pair_id"], rank=rank, gap=gap,
                    true_score=float(S[li, li]), top1_score=float(s_sorted[0]),
                    top1_is_true=(top1_target == li), mutual_nn=bool(mutual),
                    assigned_wrong_to=int(assigned[li]),
                )
                if args.flip_probe:
                    qf_flip_te = normalize(ridge.predict(q_pca.transform(qf_flip[te])))
                    rec["flip_true_score"] = float(qf_flip_te[li] @ T[li])
                errors.append(rec)

        if args.flip_probe:
            # global flip probe: does flipping the query raise the diagonal cosine?
            Qf = normalize(ridge.predict(q_pca.transform(qf_flip[te])))
            diag = np.einsum("ij,ij->i", Q, T)
            diag_flip = np.einsum("ij,ij->i", Qf, T)
            flip_helps += int(np.sum(diag_flip > diag))
            flip_tested += m

    acc = total_correct / n
    print("\n================ d1 CV ERROR ANALYSIS ================")
    print(f"folds={args.folds} grid={args.grid}  Hungarian assignment accuracy = {acc:.4f}  (~ d1 MRR proxy)")
    print(f"errors: {len(errors)} / {n}")
    print("\npre-Hungarian true-target rank histogram:")
    for r in sorted(rank_hist):
        if r <= 5 or rank_hist[r] > 0 and r > 5:
            pass
    cum = 0
    for r in sorted(rank_hist):
        cum += rank_hist[r]
        bar = "#" * rank_hist[r]
        if r <= 10:
            print(f"  rank {r:3d}: {rank_hist[r]:4d}  {bar}")
    far = sum(v for k, v in rank_hist.items() if k > 10)
    print(f"  rank >10 : {far}")

    if errors:
        print("\nHungarian MISASSIGNMENTS (the actual losses):")
        print(f"  {'pair':24s} {'rank':>4s} {'gap':>7s} {'true_s':>7s} {'top1_s':>7s} {'mutualNN':>8s}")
        for e in sorted(errors, key=lambda x: -x["rank"]):
            extra = f"  flip_s={e['flip_true_score']:.3f}" if "flip_true_score" in e else ""
            print(f"  {e['pair']:24s} {e['rank']:4d} {e['gap']:7.3f} {e['true_score']:7.3f} {e['top1_score']:7.3f} {str(e['mutual_nn']):>8s}{extra}")
        in_top3 = sum(1 for e in errors if e["rank"] <= 3)
        print(f"\n  of {len(errors)} errors, {in_top3} have the true target in top-3 (rerankable),"
              f" {len(errors)-in_top3} are far (broken feature/registration).")

    if args.flip_probe and flip_tested:
        print(f"\nL/R FLIP probe: flipping query raised true-pair cosine in "
              f"{flip_helps}/{flip_tested} cases ({flip_helps/flip_tested:.1%}). "
              f"(>50% would indicate a systematic flip bug.)")


if __name__ == "__main__":
    main()
