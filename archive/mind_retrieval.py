from __future__ import annotations
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "nibabel>=5.3",
#   "numpy>=2.0",
#   "scipy>=1.14",
# ]
# ///

"""MIND-like self-similarity descriptors for multi-modal MRI retrieval."""

import argparse
import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import uniform_filter, zoom
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


def bbox_from_mask(mask: np.ndarray, margin: int) -> tuple[slice, slice, slice]:
    coords = np.where(mask)
    if len(coords[0]) == 0:
        return tuple(slice(0, s) for s in mask.shape)  # type: ignore[return-value]
    slices = []
    for axis, values in enumerate(coords):
        lo = max(0, int(values.min()) - margin)
        hi = min(mask.shape[axis], int(values.max()) + margin + 1)
        slices.append(slice(lo, hi))
    return tuple(slices)  # type: ignore[return-value]


def resize_nd(arr: np.ndarray, shape: tuple[int, int, int], order: int) -> np.ndarray:
    factors = [target / max(1, current) for target, current in zip(shape, arr.shape)]
    out = zoom(arr, factors, order=order)
    result = np.zeros(shape, dtype=np.float32)
    common = tuple(slice(0, min(a, b)) for a, b in zip(out.shape, shape))
    result[common] = out[common].astype(np.float32, copy=False)
    return result


def robust_volume(path: Path, size: int, margin: int) -> np.ndarray:
    image = nib.load(str(path))
    volume = np.asanyarray(image.dataobj).astype(np.float32)
    if volume.ndim > 3:
        volume = volume[..., 0]
    volume = np.nan_to_num(volume, nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.abs(volume) > 1e-6
    if int(mask.sum()) > 128:
        volume = volume[bbox_from_mask(mask, margin)]
        mask = np.abs(volume) > 1e-6
    foreground = volume[mask]
    if foreground.size < 256:
        foreground = volume.reshape(-1)
    lo, hi = np.percentile(foreground, [1.0, 99.0]).astype(np.float32)
    scale = float(hi - lo) if float(hi - lo) > 1e-6 else 1.0
    volume = np.clip((volume - lo) / scale, 0.0, 1.0)
    volume = resize_nd(volume, (size, size, size), order=1)
    return volume.astype(np.float32, copy=False)


def shifted(arr: np.ndarray, shift: tuple[int, int, int]) -> np.ndarray:
    out = np.zeros_like(arr)
    src = []
    dst = []
    for axis_shift, dim in zip(shift, arr.shape):
        if axis_shift >= 0:
            src.append(slice(0, dim - axis_shift))
            dst.append(slice(axis_shift, dim))
        else:
            src.append(slice(-axis_shift, dim))
            dst.append(slice(0, dim + axis_shift))
    out[tuple(dst)] = arr[tuple(src)]
    return out


def mind_descriptor(volume: np.ndarray, shifts: list[tuple[int, int, int]], patch: int) -> np.ndarray:
    diffs = []
    for shift in shifts:
        diff = (volume - shifted(volume, shift)) ** 2
        diffs.append(uniform_filter(diff, size=patch, mode="nearest"))
    desc = np.stack(diffs, axis=0)
    variance = np.mean(desc, axis=0, keepdims=True)
    variance = np.maximum(variance, np.percentile(variance, 10.0) + 1e-6)
    desc = np.exp(-desc / variance)
    desc -= desc.mean(axis=0, keepdims=True)
    norm = np.sqrt(np.sum(desc * desc, axis=0, keepdims=True)) + 1e-6
    desc = desc / norm
    return desc.astype(np.float32, copy=False)


def feature_vector(path: Path, size: int, margin: int, patch: int, shifts: list[tuple[int, int, int]]) -> np.ndarray:
    volume = robust_volume(path, size, margin)
    desc = mind_descriptor(volume, shifts, patch)
    vec = desc.reshape(desc.shape[0], -1)
    # Spatial pyramid: full descriptor plus coarse 2x2x2 block averages.
    coarse = desc.reshape(desc.shape[0], 2, size // 2, 2, size // 2, 2, size // 2).mean(axis=(2, 4, 6))
    combined = np.concatenate([vec.reshape(-1), coarse.reshape(-1)], axis=0)
    combined = combined.astype(np.float32, copy=False)
    combined /= np.linalg.norm(combined) + 1e-6
    return combined


def score_pool(
    data_root: Path,
    dataset: str,
    split: str,
    size: int,
    margin: int,
    patch: int,
    shifts: list[tuple[int, int, int]],
    assignment: bool,
) -> list[dict[str, str]]:
    root = data_root / dataset
    query_rows = read_csv(root / f"{split}_queries.csv")
    target_rows = read_csv(root / f"{split}_gallery.csv")
    query_ids = [row["query_id"] for row in query_rows]
    target_ids = [row["target_id"] for row in target_rows]
    query_features = np.stack(
        [feature_vector(resolve_image_path(data_root, row["query_image"]), size, margin, patch, shifts) for row in query_rows]
    )
    target_features = np.stack(
        [feature_vector(resolve_image_path(data_root, row["target_image"]), size, margin, patch, shifts) for row in target_rows]
    )
    scores = query_features @ target_features.T
    if assignment:
        row_ind, col_ind = linear_sum_assignment(-scores.astype(np.float64))
        assigned = np.empty(scores.shape[0], dtype=np.int64)
        assigned[row_ind] = col_ind
        scores[np.arange(scores.shape[0]), assigned] = float(np.max(scores)) + 1e6
    target_ids_arr = np.asarray(target_ids)
    rows = []
    for query_index, query_id in enumerate(query_ids):
        order = np.argsort(-scores[query_index], kind="mergesort")
        rows.append({"query_id": query_id, "target_id_ranking": " ".join(target_ids_arr[order].tolist())})
    print(f"scored {dataset}/{split}: {len(query_ids)}x{len(target_ids)}")
    return rows


def parse_shifts(name: str) -> list[tuple[int, int, int]]:
    axis = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    diag = [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]
    if name == "axis":
        return axis
    if name == "axis_diag":
        return axis + diag
    raise ValueError(name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--datasets", nargs="+", choices=["dataset1", "dataset2", "dataset3"], default=["dataset2"])
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--margin", type=int, default=3)
    parser.add_argument("--patch", type=int, default=3)
    parser.add_argument("--shifts", choices=["axis", "axis_diag"], default="axis_diag")
    parser.add_argument("--assignment", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("submissions/d2_mind32_axisdiag_hungarian.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    shifts = parse_shifts(args.shifts)
    rows: list[dict[str, str]] = []
    for dataset in args.datasets:
        for split in args.splits:
            rows.extend(score_pool(args.data_root, dataset, split, args.size, args.margin, args.patch, shifts, args.assignment))
    write_csv(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
