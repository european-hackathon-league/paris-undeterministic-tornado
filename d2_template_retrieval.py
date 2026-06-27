from __future__ import annotations
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "nibabel>=5.3",
#   "numpy>=2.0",
#   "scipy>=1.14",
#   "scikit-learn>=1.5",
# ]
# ///

"""Dataset2 retrieval via template normalization.

Validated on the synthetic-d2 harness: template normalization scores ~0.55 MRR
vs ~0.26 for the canonical PCA-axis baseline at realistic d2 distortion.

Pipeline:
  1. Build T1 / T2 templates = mean of (downsampled) dataset1 train queries/targets.
     dataset1 train pairs are registered, so the means are sharp and the two
     templates share one grid.
  2. Fit PCA/Ridge cross-modal map on the clean dataset1 train pairs.
  3. For each dataset2 query/target: downsample, rigidly register to its
     same-modality template (intramodal intensity NCC -> robust), flatten.
  4. Score = cosine(ridge(query_pca), target_pca); optional Hungarian assignment.

Normalized features are cached in --cache-dir keyed by (image_id, grid).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import normalize

from synthetic_d2_eval import _load_grid, read_pairs, resolve_image_path
from d2_methods import _register_to_template, flat_feature, _fit_pca_ridge


DEFAULT_DATA_ROOT = Path("ehl-paris-medical-image-retrieval")
DEFAULT_CACHE = Path(".d2cache")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        w.writeheader()
        w.writerows(rows)


def normalized_feature(
    data_root: Path, image_path: str, image_id: str, grid: int, template: np.ndarray,
    cache_dir: Path, register: bool = True
) -> np.ndarray:
    tag = f"g{grid}" if register else f"g{grid}_noreg"
    cache = cache_dir / f"{image_id}_{tag}.npy"
    if cache.exists():
        return np.load(cache)
    vol = _load_grid(data_root, image_path, grid)
    feat = flat_feature(_register_to_template(vol, template) if register else vol)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache, feat)
    return feat


def build_model(data_root: Path, grid: int, components: int, alpha: float):
    pairs = read_pairs(data_root)
    tq = np.stack([_load_grid(data_root, p["query_image"], grid) for p in pairs])
    tt = np.stack([_load_grid(data_root, p["target_image"], grid) for p in pairs])
    t1_tpl = tq.mean(axis=0)
    t2_tpl = tt.mean(axis=0)
    qf = np.stack([flat_feature(v) for v in tq])
    tf = np.stack([flat_feature(v) for v in tt])
    q_pca, t_pca, ridge = _fit_pca_ridge(qf, tf, components, alpha)
    print(f"built templates + pca/ridge from {len(pairs)} train pairs", flush=True)
    return t1_tpl, t2_tpl, q_pca, t_pca, ridge


def score_pool(
    data_root, dataset, split, grid, model, cache_dir, assignment, register=True
) -> list[dict[str, str]]:
    t1_tpl, t2_tpl, q_pca, t_pca, ridge = model
    root = data_root / dataset
    qrows = read_csv(root / f"{split}_queries.csv")
    trows = read_csv(root / f"{split}_gallery.csv")

    q = []
    for i, r in enumerate(qrows):
        feat = normalized_feature(data_root, r["query_image"], r["query_id"], grid, t1_tpl, cache_dir, register)
        q.append(normalize(ridge.predict(q_pca.transform(feat[None, :])))[0])
        if (i + 1) % 20 == 0:
            print(f"  {dataset}/{split} query {i+1}/{len(qrows)}", flush=True)
    t = []
    for i, r in enumerate(trows):
        feat = normalized_feature(data_root, r["target_image"], r["target_id"], grid, t2_tpl, cache_dir, register)
        t.append(normalize(t_pca.transform(feat[None, :]))[0])
        if (i + 1) % 20 == 0:
            print(f"  {dataset}/{split} target {i+1}/{len(trows)}", flush=True)
    q = np.stack(q)
    t = np.stack(t)
    scores = (q @ t.T).astype(np.float64)

    if assignment and scores.shape[0] == scores.shape[1]:
        row_ind, col_ind = linear_sum_assignment(-scores)
        assigned = np.empty(scores.shape[0], dtype=np.int64)
        assigned[row_ind] = col_ind
        scores[np.arange(scores.shape[0]), assigned] = scores.max() + 1e6

    target_ids = np.asarray([r["target_id"] for r in trows])
    rows = []
    for qi, r in enumerate(qrows):
        order = np.argsort(-scores[qi], kind="mergesort")
        rows.append({"query_id": r["query_id"], "target_id_ranking": " ".join(target_ids[order].tolist())})
    print(f"scored {dataset}/{split}: {len(qrows)}x{len(trows)}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    p.add_argument("--datasets", nargs="+", default=["dataset2"])
    p.add_argument("--splits", nargs="+", default=["val", "test"])
    p.add_argument("--grid", type=int, default=44)
    p.add_argument("--components", type=int, default=128)
    p.add_argument("--alpha", type=float, default=100.0)
    p.add_argument("--assignment", action="store_true")
    p.add_argument("--no-register", action="store_true",
                   help="Skip template registration (use raw downsampled grid feature). "
                        "For already-aligned sets like dataset3.")
    p.add_argument("--out", type=Path, default=Path("submissions/d2_template_g44_hungarian.csv"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = build_model(args.data_root, args.grid, args.components, args.alpha)
    rows: list[dict[str, str]] = []
    for dataset in args.datasets:
        for split in args.splits:
            rows.extend(
                score_pool(args.data_root, dataset, split, args.grid, model, args.cache_dir,
                           args.assignment, register=not args.no_register)
            )
    write_csv(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
