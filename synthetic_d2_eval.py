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

"""Synthetic dataset2 local validator.

dataset2 is dataset1's source domain with *independent* random rigid
(rotation/translation) + nonlinear (elastic) warps applied to query and target.
The registered d1 CV is therefore a useless proxy for d2. This harness recreates
the d2 distortion locally on labelled d1 pairs so a matching method can be scored
(Hungarian-recovery MRR) offline, before spending a Kaggle submission.

A method is anything exposing:
    fit(train_q_paths, train_t_paths) -> state
    embed_query(path, state) -> np.ndarray   (L2-normalized row(s))
    embed_target(path, state) -> np.ndarray
Score matrix = max cosine over rows, then optional Hungarian assignment, then MRR.
"""

import argparse
import hashlib
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform, gaussian_filter, map_coordinates
from scipy.optimize import linear_sum_assignment


DEFAULT_DATA_ROOT = Path("ehl-paris-medical-image-retrieval")


def read_pairs(data_root: Path) -> list[dict[str, str]]:
    import csv

    with (data_root / "dataset1" / "train_pairs.csv").open(newline="") as f:
        return list(csv.DictReader(f))


def resolve_image_path(data_root: Path, image_path: str) -> Path:
    path = Path(image_path)
    if not path.is_absolute():
        path = data_root / path
    if path.exists():
        return path
    if path.name.endswith(".nii.gz"):
        fallback = path.with_name(path.name[:-3])
        if fallback.exists():
            return fallback
    raise FileNotFoundError(path)


def load_normalized(path: Path) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    volume = np.asanyarray(image.dataobj).astype(np.float32)
    if volume.ndim > 3:
        volume = volume[..., 0]
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.abs(volume) > 1e-6
    values = volume[mask]
    if values.size < 256:
        values = volume.reshape(-1)
    lo, hi = np.percentile(values, [1.0, 99.0]).astype(np.float32)
    scale = float(hi - lo) if float(hi - lo) > 1e-6 else 1.0
    volume = np.clip((volume - lo) / scale, 0.0, 1.0)
    volume[~mask] = 0.0
    return volume.astype(np.float32, copy=False), mask


def downsample(volume: np.ndarray, size: int) -> np.ndarray:
    """Resample a whole volume onto a fixed size**3 grid (linear)."""
    shape = np.asarray(volume.shape, dtype=np.float64)
    grid = np.stack(
        np.meshgrid(*[np.linspace(0, s - 1, size) for s in shape], indexing="ij"),
        axis=0,
    )
    sampled = map_coordinates(volume, grid, order=1, mode="constant", cval=0.0)
    return sampled.astype(np.float32, copy=False)


def _rng_for(seed: int, key: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def deform(
    volume: np.ndarray,
    rng: np.random.Generator,
    max_rot_deg: float,
    max_shift: float,
    elastic_sigma: float,
    elastic_alpha: float,
) -> np.ndarray:
    """Apply random rigid + elastic warp, mimicking dataset2 distortion."""
    shape = np.asarray(volume.shape, dtype=np.float64)
    center = (shape - 1) / 2.0

    # rigid: random rotation about each axis + translation
    angles = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg, size=3))
    rot = np.eye(3)
    for axis, ang in enumerate(angles):
        c, s = np.cos(ang), np.sin(ang)
        r = np.eye(3)
        a, b = [i for i in range(3) if i != axis]
        r[a, a], r[a, b], r[b, a], r[b, b] = c, -s, s, c
        rot = rot @ r
    shift = rng.uniform(-max_shift, max_shift, size=3)
    offset = center - rot @ center + shift
    rigid = affine_transform(volume, rot, offset=offset, order=1, mode="constant", cval=0.0)

    if elastic_alpha <= 0:
        return rigid.astype(np.float32, copy=False)

    # elastic: smooth random displacement field
    disp = [
        gaussian_filter(rng.standard_normal(rigid.shape), elastic_sigma) * elastic_alpha
        for _ in range(3)
    ]
    coords = np.stack(np.meshgrid(*[np.arange(s) for s in rigid.shape], indexing="ij"))
    warped_coords = [coords[i] + disp[i] for i in range(3)]
    warped = map_coordinates(rigid, warped_coords, order=1, mode="constant", cval=0.0)
    return warped.astype(np.float32, copy=False)


def mrr_from_scores(scores: np.ndarray) -> tuple[float, float]:
    """Diagonal is the truth (query i matches target i). Returns (raw_mrr, hungarian_acc)."""
    n = scores.shape[0]
    ranks = []
    for i in range(n):
        order = np.argsort(-scores[i], kind="mergesort")
        rank = int(np.where(order == i)[0][0]) + 1
        ranks.append(1.0 / rank)
    raw_mrr = float(np.mean(ranks))
    row_ind, col_ind = linear_sum_assignment(-scores.astype(np.float64))
    hungarian_acc = float(np.mean(col_ind[np.argsort(row_ind)] == np.arange(n)))
    return raw_mrr, hungarian_acc


def _load_grid(data_root: Path, image_path: str, grid: int) -> np.ndarray:
    vol, _ = load_normalized(resolve_image_path(data_root, image_path))
    return downsample(vol, grid)


def build_eval_set(
    data_root: Path,
    n_eval: int,
    n_train: int,
    seed: int,
    grid: int,
    deform_kwargs: dict,
) -> dict:
    """Returns downsampled train arrays (clean) + eval arrays (independently deformed).

    Train pairs stay undeformed: they mimic the registered dataset1 a method is
    fit on. Eval pairs get the synthetic d2 distortion.
    """
    pairs = read_pairs(data_root)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    train_idx = idx[:n_train]
    eval_idx = idx[n_train : n_train + n_eval]

    train_q = np.stack([_load_grid(data_root, pairs[i]["query_image"], grid) for i in train_idx])
    train_t = np.stack([_load_grid(data_root, pairs[i]["target_image"], grid) for i in train_idx])
    print(f"loaded {len(train_idx)} clean train pairs", flush=True)

    eval_q, eval_t = [], []
    for rank, i in enumerate(eval_idx):
        qv = _load_grid(data_root, pairs[i]["query_image"], grid)
        tv = _load_grid(data_root, pairs[i]["target_image"], grid)
        eval_q.append(deform(qv, _rng_for(seed, f"q{rank}"), **deform_kwargs))
        eval_t.append(deform(tv, _rng_for(seed, f"t{rank}"), **deform_kwargs))
        if (rank + 1) % 20 == 0:
            print(f"built {rank + 1}/{len(eval_idx)} deformed eval volumes", flush=True)
    return {
        "train_q": train_q,
        "train_t": train_t,
        "eval_q": np.stack(eval_q),
        "eval_t": np.stack(eval_t),
        "grid": grid,
    }


# --- methods: each returns a score matrix (rows=eval_q, cols=eval_t) -----------
# Diagonal is the true match. Implemented in d2_methods to keep this file focused.


def evaluate_method(data: dict, method: str) -> tuple[float, float]:
    import d2_methods

    fn = d2_methods.METHODS[method]
    scores = fn(data)
    return mrr_from_scores(scores)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--n-train", type=int, default=200)
    p.add_argument("--seed", type=int, default=20260627)
    p.add_argument("--grid", type=int, default=48)
    p.add_argument("--max-rot-deg", type=float, default=20.0)
    p.add_argument("--max-shift", type=float, default=10.0)
    p.add_argument("--elastic-sigma", type=float, default=8.0)
    p.add_argument("--elastic-alpha", type=float, default=6.0)
    p.add_argument("--methods", nargs="+", default=["raw_grid"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    deform_kwargs = dict(
        max_rot_deg=args.max_rot_deg,
        max_shift=args.max_shift,
        elastic_sigma=args.elastic_sigma,
        elastic_alpha=args.elastic_alpha,
    )
    data = build_eval_set(
        args.data_root, args.n_eval, args.n_train, args.seed, args.grid, deform_kwargs
    )
    print(
        f"\nsynthetic-d2  n_eval={args.n_eval} grid={args.grid} "
        f"rot={args.max_rot_deg} shift={args.max_shift} "
        f"elastic(sig={args.elastic_sigma},a={args.elastic_alpha})\n"
    )
    for method in args.methods:
        raw, acc = evaluate_method(data, method)
        print(f"  {method:28s} raw_mrr={raw:.4f}  hungarian_acc={acc:.4f}", flush=True)


if __name__ == "__main__":
    main()
