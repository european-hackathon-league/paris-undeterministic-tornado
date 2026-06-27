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

"""PCA-axis canonicalization plus PCA/ridge cross-modal retrieval."""

import argparse
import csv
import hashlib
from itertools import product
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates, sobel
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize


DEFAULT_DATA_ROOT = Path("ehl-paris-medical-image-retrieval")
DEFAULT_CACHE_DIR = Path(".canonical_cache")


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


def canonical_volume(path: Path, size: int) -> np.ndarray:
    volume, mask = load_normalized(path)
    coords = np.column_stack(np.where(mask))
    if len(coords) < 32:
        coords = np.column_stack(np.where(np.ones_like(volume, dtype=bool)))
    center = coords.mean(axis=0)
    cov = np.cov((coords - center).T)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    vectors = vectors[:, order]
    projections = (coords - center) @ vectors
    scales = np.percentile(np.abs(projections), 99.0, axis=0)
    scales = np.maximum(scales, 1.0)
    for axis in range(3):
        skew = float(np.mean(projections[:, axis] ** 3))
        if skew < 0:
            vectors[:, axis] *= -1
            projections[:, axis] *= -1
    grid = np.stack(np.meshgrid(*[np.linspace(-1.0, 1.0, size) for _ in range(3)], indexing="ij"), axis=-1)
    physical = center + (grid.reshape(-1, 3) * scales) @ vectors.T
    sampled = map_coordinates(
        volume,
        [physical[:, 0], physical[:, 1], physical[:, 2]],
        order=1,
        mode="constant",
        cval=0.0,
    ).reshape(size, size, size)
    return sampled.astype(np.float32, copy=False)


def feature_from_canonical(volume: np.ndarray) -> np.ndarray:
    edge = np.sqrt(sum(sobel(volume, axis=axis) ** 2 for axis in range(3)))
    high = np.clip((volume - np.percentile(volume[volume > 0], 85.0)) / (np.percentile(volume[volume > 0], 99.0) - np.percentile(volume[volume > 0], 85.0) + 1e-6), 0, 1) if np.any(volume > 0) else volume
    vec = np.concatenate([volume.reshape(-1), edge.reshape(-1), high.reshape(-1)]).astype(np.float32)
    vec -= vec.mean()
    vec /= np.linalg.norm(vec) + 1e-6
    return vec


def feature_variants(path: Path, size: int) -> np.ndarray:
    volume = canonical_volume(path, size)
    variants = []
    for flips in product([False, True], repeat=3):
        variant = volume
        for axis, enabled in enumerate(flips):
            if enabled:
                variant = np.flip(variant, axis=axis)
        variants.append(feature_from_canonical(np.ascontiguousarray(variant)))
    return np.stack(variants).astype(np.float32)


def cached_feature_variants(path: Path, size: int, cache_dir: Path, force: bool) -> np.ndarray:
    path = path.resolve()
    digest = hashlib.sha256(f"{size}:{path}".encode()).hexdigest()[:20]
    cache_path = cache_dir / f"size{size}" / f"{digest}.npz"
    if cache_path.exists() and not force:
        with np.load(cache_path) as payload:
            return payload["features"].astype(np.float32, copy=False)
    features = feature_variants(path, size)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        features=features.astype(np.float32),
        source_path=np.asarray(str(path)),
        size=np.asarray(size),
    )
    return features


def fit_model(
    data_root: Path,
    size: int,
    components: int,
    alpha: float,
    cache_dir: Path,
    force_cache: bool,
    fit_flip_augmentation: bool,
) -> tuple[PCA, PCA, Ridge]:
    pairs = read_csv(data_root / "dataset1" / "train_pairs.csv")
    q_features = []
    t_features = []
    for index, row in enumerate(pairs, start=1):
        q_variants = cached_feature_variants(resolve_image_path(data_root, row["query_image"]), size, cache_dir, force_cache)
        t_variants = cached_feature_variants(resolve_image_path(data_root, row["target_image"]), size, cache_dir, force_cache)
        if fit_flip_augmentation:
            q_features.extend(q_variants)
            t_features.extend(t_variants)
        else:
            q_features.append(q_variants[0])
            t_features.append(t_variants[0])
        if index % 50 == 0:
            print(f"loaded train canonical features {index}/{len(pairs)}", flush=True)
    q = np.stack(q_features)
    t = np.stack(t_features)
    n_components = min(components, len(pairs) - 1, q.shape[1], t.shape[1])
    q_pca = PCA(n_components=n_components, whiten=True, random_state=20260627).fit(q)
    t_pca = PCA(n_components=n_components, whiten=True, random_state=20260627).fit(t)
    qz = q_pca.transform(q)
    tz = t_pca.transform(t)
    ridge = Ridge(alpha=alpha).fit(qz, tz)
    return q_pca, t_pca, ridge


def score_pool(
    data_root: Path,
    dataset: str,
    split: str,
    size: int,
    cache_dir: Path,
    force_cache: bool,
    q_pca: PCA,
    t_pca: PCA,
    ridge: Ridge,
    assignment: bool,
) -> list[dict[str, str]]:
    root = data_root / dataset
    query_rows = read_csv(root / f"{split}_queries.csv")
    target_rows = read_csv(root / f"{split}_gallery.csv")
    query_ids = [row["query_id"] for row in query_rows]
    target_ids = [row["target_id"] for row in target_rows]

    q_pred = []
    for row in query_rows:
        variants = cached_feature_variants(resolve_image_path(data_root, row["query_image"]), size, cache_dir, force_cache)
        pred = ridge.predict(q_pca.transform(variants))
        q_pred.append(normalize(pred))
    t_vecs = []
    for row in target_rows:
        variants = cached_feature_variants(resolve_image_path(data_root, row["target_image"]), size, cache_dir, force_cache)
        t_vecs.append(normalize(t_pca.transform(variants)))

    scores = np.zeros((len(query_ids), len(target_ids)), dtype=np.float32)
    for qi, query_variants in enumerate(q_pred):
        for ti, target_variants in enumerate(t_vecs):
            scores[qi, ti] = float(np.max(query_variants @ target_variants.T))
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
    print(f"scored {dataset}/{split}: {len(query_ids)}x{len(target_ids)}", flush=True)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--datasets", nargs="+", choices=["dataset1", "dataset2", "dataset3"], default=["dataset2"])
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--components", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=100.0)
    parser.add_argument("--fit-flip-augmentation", action="store_true")
    parser.add_argument("--force-cache", action="store_true")
    parser.add_argument("--assignment", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("submissions/d2_canonical_pca24_c128_hungarian.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    q_pca, t_pca, ridge = fit_model(
        args.data_root,
        args.size,
        args.components,
        args.alpha,
        args.cache_dir,
        args.force_cache,
        args.fit_flip_augmentation,
    )
    rows: list[dict[str, str]] = []
    for dataset in args.datasets:
        for split in args.splits:
            rows.extend(
                score_pool(
                    args.data_root,
                    dataset,
                    split,
                    args.size,
                    args.cache_dir,
                    args.force_cache,
                    q_pca,
                    t_pca,
                    ridge,
                    args.assignment,
                )
            )
    write_csv(args.out, rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
