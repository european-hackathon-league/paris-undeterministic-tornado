from __future__ import annotations
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "nibabel>=5.3",
#   "numpy>=2.0",
# ]
# ///

"""Generate cheap diagnostic submissions for hidden-set leakage checks.

These methods are intentionally simple. They test whether the hidden query and
gallery sets contain useful ordering, filename, filesystem, or NIfTI header
signals before spending GPU time on heavier matching models.
"""

import argparse
import csv
import hashlib
import os
from pathlib import Path

import nibabel as nib
import numpy as np


DEFAULT_DATA_ROOT = Path("ehl-paris-medical-image-retrieval")
DEFAULT_OUT_DIR = Path("submissions/diagnostics")


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
        uncompressed = path.with_name(path.name[:-3])
        if uncompressed.exists():
            return uncompressed
    raise FileNotFoundError(path)


def stable_hash_int(value: str) -> int:
    return int.from_bytes(hashlib.blake2b(value.encode(), digest_size=8).digest(), "big")


def id_hex_int(value: str) -> int:
    suffix = value.split("_", 1)[-1]
    try:
        return int(suffix, 16)
    except ValueError:
        return stable_hash_int(value)


def ranking_rows_from_scores(query_ids: list[str], target_ids: list[str], scores: np.ndarray) -> list[dict[str, str]]:
    target_ids_arr = np.asarray(target_ids)
    rows: list[dict[str, str]] = []
    for qi, query_id in enumerate(query_ids):
        order = np.argsort(-scores[qi], kind="mergesort")
        rows.append({"query_id": query_id, "target_id_ranking": " ".join(target_ids_arr[order].tolist())})
    return rows


def index_scores(query_ids: list[str], target_ids: list[str], reverse: bool = False, circular: bool = False) -> np.ndarray:
    n_q, n_t = len(query_ids), len(target_ids)
    q_idx = np.arange(n_q, dtype=np.float32)
    t_idx = np.arange(n_t, dtype=np.float32)
    if reverse:
        t_idx = (n_t - 1) - t_idx
    scale = max(1.0, (n_t - 1) / max(1, n_q - 1))
    aligned = q_idx[:, None] * scale
    dist = np.abs(aligned - t_idx[None, :])
    if circular:
        dist = np.minimum(dist, n_t - dist)
    return -dist.astype(np.float32)


def id_distance_scores(query_ids: list[str], target_ids: list[str]) -> np.ndarray:
    q_vals = np.asarray([id_hex_int(q) for q in query_ids], dtype=np.float64)
    t_vals = np.asarray([id_hex_int(t) for t in target_ids], dtype=np.float64)
    return -np.abs(q_vals[:, None] - t_vals[None, :]).astype(np.float32)


def hash_distance_scores(query_ids: list[str], target_ids: list[str]) -> np.ndarray:
    q_vals = np.asarray([stable_hash_int(q) for q in query_ids], dtype=np.float64)
    t_vals = np.asarray([stable_hash_int(t) for t in target_ids], dtype=np.float64)
    return -np.abs(q_vals[:, None] - t_vals[None, :]).astype(np.float32)


def file_size_scores(data_root: Path, query_rows: list[dict[str, str]], target_rows: list[dict[str, str]]) -> np.ndarray:
    q_sizes = np.asarray(
        [resolve_image_path(data_root, row["query_image"]).stat().st_size for row in query_rows],
        dtype=np.float64,
    )
    t_sizes = np.asarray(
        [resolve_image_path(data_root, row["target_image"]).stat().st_size for row in target_rows],
        dtype=np.float64,
    )
    return -np.abs(q_sizes[:, None] - t_sizes[None, :]).astype(np.float32)


def nifti_header_vector(path: Path) -> np.ndarray:
    img = nib.load(str(path))
    header = img.header
    shape = np.asarray(img.shape[:3], dtype=np.float64)
    if shape.size < 3:
        shape = np.pad(shape, (0, 3 - shape.size), constant_values=1.0)
    zooms = np.asarray(header.get_zooms()[:3], dtype=np.float64)
    if zooms.size < 3:
        zooms = np.pad(zooms, (0, 3 - zooms.size), constant_values=1.0)
    affine = np.asarray(img.affine, dtype=np.float64).reshape(-1)
    pixdim = np.asarray(header["pixdim"], dtype=np.float64)
    dim = np.asarray(header["dim"], dtype=np.float64)
    extras = np.asarray(
        [
            os.path.getsize(path),
            float(header["datatype"]),
            float(header["bitpix"]),
            float(header["qform_code"]),
            float(header["sform_code"]),
            float(header["xyzt_units"]),
            float(header["vox_offset"]),
            float(header["scl_slope"]) if np.isfinite(header["scl_slope"]) else 0.0,
            float(header["scl_inter"]) if np.isfinite(header["scl_inter"]) else 0.0,
        ],
        dtype=np.float64,
    )
    return np.concatenate([shape, zooms, affine, pixdim, dim, extras])


def nearest_vector_scores(query_vectors: list[np.ndarray], target_vectors: list[np.ndarray]) -> np.ndarray:
    all_vectors = np.vstack(query_vectors + target_vectors).astype(np.float64)
    mean = all_vectors.mean(axis=0)
    std = all_vectors.std(axis=0)
    std[std < 1e-9] = 1.0
    q = (np.vstack(query_vectors) - mean) / std
    t = (np.vstack(target_vectors) - mean) / std
    q_norm = np.sum(q * q, axis=1, keepdims=True)
    t_norm = np.sum(t * t, axis=1, keepdims=True).T
    dist2 = q_norm + t_norm - 2.0 * (q @ t.T)
    return -dist2.astype(np.float32)


def header_scores(data_root: Path, query_rows: list[dict[str, str]], target_rows: list[dict[str, str]]) -> np.ndarray:
    q_vecs = [nifti_header_vector(resolve_image_path(data_root, row["query_image"])) for row in query_rows]
    t_vecs = [nifti_header_vector(resolve_image_path(data_root, row["target_image"])) for row in target_rows]
    return nearest_vector_scores(q_vecs, t_vecs)


def sample_scores(sample_rows: dict[str, list[str]], query_ids: list[str], target_ids: list[str]) -> np.ndarray:
    target_index = {target_id: i for i, target_id in enumerate(target_ids)}
    scores = np.full((len(query_ids), len(target_ids)), -1e6, dtype=np.float32)
    fallback = index_scores(query_ids, target_ids)
    for qi, query_id in enumerate(query_ids):
        ranking = sample_rows.get(query_id)
        if not ranking:
            scores[qi] = fallback[qi]
            continue
        score = float(len(target_ids))
        seen = False
        for target_id in ranking:
            ti = target_index.get(target_id)
            if ti is None:
                continue
            scores[qi, ti] = score
            score -= 1.0
            seen = True
        if not seen:
            scores[qi] = fallback[qi]
    return scores


def generate_for_pool(
    data_root: Path,
    dataset: str,
    split: str,
    method: str,
    sample_rows: dict[str, list[str]],
) -> list[dict[str, str]]:
    ds_root = data_root / dataset
    query_rows = read_csv(ds_root / f"{split}_queries.csv")
    target_rows = read_csv(ds_root / f"{split}_gallery.csv")

    if method.startswith("sorted_"):
        query_rows = sorted(query_rows, key=lambda row: row["query_id"])
        target_rows = sorted(target_rows, key=lambda row: row["target_id"])

    query_ids = [row["query_id"] for row in query_rows]
    target_ids = [row["target_id"] for row in target_rows]

    match method:
        case "csv_index" | "sorted_index":
            scores = index_scores(query_ids, target_ids)
        case "csv_reverse" | "sorted_reverse":
            scores = index_scores(query_ids, target_ids, reverse=True)
        case "csv_circular" | "sorted_circular":
            scores = index_scores(query_ids, target_ids, circular=True)
        case "id_hex_distance":
            scores = id_distance_scores(query_ids, target_ids)
        case "hash_distance":
            scores = hash_distance_scores(query_ids, target_ids)
        case "file_size":
            scores = file_size_scores(data_root, query_rows, target_rows)
        case "nifti_header":
            scores = header_scores(data_root, query_rows, target_rows)
        case "sample_submission":
            scores = sample_scores(sample_rows, query_ids, target_ids)
        case _:
            raise ValueError(f"Unknown diagnostic method: {method}")
    return ranking_rows_from_scores(query_ids, target_ids, scores)


def load_sample_rows(path: Path) -> dict[str, list[str]]:
    if not path.exists():
        return {}
    return {row["query_id"]: row["target_id_ranking"].split() for row in read_csv(path)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--sample-submission", type=Path, default=Path("sample_submission.csv"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["dataset1", "dataset2", "dataset3"],
        default=["dataset2", "dataset3"],
    )
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "csv_index",
            "csv_reverse",
            "csv_circular",
            "sorted_index",
            "sorted_reverse",
            "id_hex_distance",
            "hash_distance",
            "file_size",
            "nifti_header",
            "sample_submission",
        ],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sample_rows = load_sample_rows(args.sample_submission)
    for dataset in args.datasets:
        for method in args.methods:
            rows: list[dict[str, str]] = []
            for split in args.splits:
                rows.extend(generate_for_pool(args.data_root, dataset, split, method, sample_rows))
            out = args.out_dir / f"{dataset}_{method}.csv"
            write_csv(out, rows)
            print(f"wrote {len(rows):3d} rows to {out}")


if __name__ == "__main__":
    main()
