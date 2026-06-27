from __future__ import annotations
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "nibabel>=5.3",
#   "numpy>=2.0",
#   "scipy>=1.14",
# ]
# ///

"""Lesion-focused descriptors for T1 post-contrast vs T2 retrieval."""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, sobel, zoom
from scipy.optimize import linear_sum_assignment


DEFAULT_DATA_ROOT = Path("ehl-paris-medical-image-retrieval")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query_id", "target_id_ranking"])
        writer.writeheader()
        writer.writerows(rows)


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


def bbox(mask: np.ndarray, margin: int) -> tuple[slice, slice, slice]:
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return tuple(slice(0, s) for s in mask.shape)  # type: ignore[return-value]
    out = []
    for axis, values in enumerate(coords):
        out.append(slice(max(0, int(values.min()) - margin), min(mask.shape[axis], int(values.max()) + margin + 1)))
    return tuple(out)  # type: ignore[return-value]


def resize(arr: np.ndarray, shape: tuple[int, int, int], order: int = 1) -> np.ndarray:
    factors = [target / max(1, current) for target, current in zip(shape, arr.shape)]
    out = zoom(arr, factors, order=order)
    result = np.zeros(shape, dtype=np.float32)
    common = tuple(slice(0, min(a, b)) for a, b in zip(out.shape, shape))
    result[common] = out[common].astype(np.float32, copy=False)
    return result


def load_volume(path: Path, size: int, margin: int) -> tuple[np.ndarray, np.ndarray]:
    image = nib.load(str(path))
    volume = np.asanyarray(image.dataobj).astype(np.float32)
    if volume.ndim > 3:
        volume = volume[..., 0]
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.abs(volume) > 1e-6
    if int(mask.sum()) > 128:
        crop = bbox(mask, margin)
        volume = volume[crop]
        mask = mask[crop]
    foreground = volume[mask]
    if foreground.size < 256:
        foreground = volume.reshape(-1)
    lo, hi = np.percentile(foreground, [1.0, 99.5]).astype(np.float32)
    scale = float(hi - lo) if float(hi - lo) > 1e-6 else 1.0
    volume = np.clip((volume - lo) / scale, 0.0, 1.0)
    mask = resize(mask.astype(np.float32), (size, size, size), order=0) > 0.5
    volume = resize(volume, (size, size, size), order=1)
    volume[~mask] = 0.0
    return volume.astype(np.float32), mask


def soft_threshold(values: np.ndarray, mask: np.ndarray, q_low: float, q_high: float) -> np.ndarray:
    foreground = values[mask]
    if foreground.size < 256:
        foreground = values.reshape(-1)
    lo, hi = np.percentile(foreground, [q_low, q_high]).astype(np.float32)
    out = np.clip((values - lo) / (float(hi - lo) + 1e-6), 0.0, 1.0)
    out[~mask] = 0.0
    return out.astype(np.float32)


def projections(arr: np.ndarray, size: int) -> np.ndarray:
    views = []
    for axis in range(3):
        views.append(arr.mean(axis=axis))
        views.append(arr.max(axis=axis))
    return np.concatenate([v.reshape(-1) for v in views]).astype(np.float32)


def moments(arr: np.ndarray) -> np.ndarray:
    weights = np.maximum(arr, 0.0)
    total = float(weights.sum())
    if total < 1e-6:
        return np.zeros(12, dtype=np.float32)
    coords = np.stack(np.meshgrid(*[np.linspace(-1, 1, s) for s in arr.shape], indexing="ij"), axis=0)
    mean = (coords * weights).reshape(3, -1).sum(axis=1) / total
    centered = coords - mean.reshape(3, 1, 1, 1)
    var = ((centered**2) * weights).reshape(3, -1).sum(axis=1) / total
    mass = np.asarray([total / weights.size, float(weights.max()), float(np.percentile(weights[weights > 0], 90)) if np.any(weights > 0) else 0.0])
    return np.concatenate([mean, var, mass, np.asarray(arr.shape, dtype=np.float32) / max(arr.shape)]).astype(np.float32)


def feature_vector(path: Path, size: int, margin: int) -> np.ndarray:
    volume, mask = load_volume(path, size, margin)
    smooth = gaussian_filter(volume, sigma=max(1.0, size / 12.0))
    residual = np.maximum(volume - smooth, 0.0)
    high90 = soft_threshold(volume, mask, 90.0, 99.5)
    high95 = soft_threshold(volume, mask, 95.0, 99.8)
    residual_high = soft_threshold(residual, mask, 90.0, 99.5)
    grad = np.sqrt(sum(sobel(volume, axis=axis) ** 2 for axis in range(3)))
    grad_high = soft_threshold(grad, mask, 90.0, 99.5)
    channels = [high90, high95, residual_high, grad_high]
    dense = np.concatenate([channel.reshape(-1) for channel in channels])
    proj = np.concatenate([projections(channel, size) for channel in channels])
    mom = np.concatenate([moments(channel) for channel in channels])
    vec = np.concatenate([dense, proj, mom]).astype(np.float32)
    vec -= vec.mean()
    vec /= np.linalg.norm(vec) + 1e-6
    return vec


def score_pool(data_root: Path, dataset: str, split: str, size: int, margin: int, assignment: bool) -> list[dict[str, str]]:
    root = data_root / dataset
    query_rows = read_csv(root / f"{split}_queries.csv")
    target_rows = read_csv(root / f"{split}_gallery.csv")
    query_ids = [row["query_id"] for row in query_rows]
    target_ids = [row["target_id"] for row in target_rows]
    query_features = np.stack([feature_vector(resolve_image_path(data_root, row["query_image"]), size, margin) for row in query_rows])
    target_features = np.stack([feature_vector(resolve_image_path(data_root, row["target_image"]), size, margin) for row in target_rows])
    scores = query_features @ target_features.T
    if assignment:
        row_ind, col_ind = linear_sum_assignment(-scores.astype(np.float64))
        assigned = np.empty(scores.shape[0], dtype=np.int64)
        assigned[row_ind] = col_ind
        scores[np.arange(scores.shape[0]), assigned] = float(np.max(scores)) + 1e6
    target_arr = np.asarray(target_ids)
    rows = []
    for query_index, query_id in enumerate(query_ids):
        order = np.argsort(-scores[query_index], kind="mergesort")
        rows.append({"query_id": query_id, "target_id_ranking": " ".join(target_arr[order].tolist())})
    print(f"scored {dataset}/{split}: {len(query_ids)}x{len(target_ids)}")
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=["dataset1", "dataset2", "dataset3"], default=["dataset2"])
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--size", type=int, default=40)
    parser.add_argument("--margin", type=int, default=3)
    parser.add_argument("--assignment", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("submissions/d2_lesion40_hungarian.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, str]] = []
    for dataset in args.datasets:
        for split in args.splits:
            rows.extend(score_pool(args.data_root, dataset, split, args.size, args.margin, args.assignment))
    write_csv(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
