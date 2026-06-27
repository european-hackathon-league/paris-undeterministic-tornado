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

"""Apply one-to-one assignment reranking to classical score matrices."""

import argparse
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from classical_retrieval import (
    DEFAULT_CACHE_DIR,
    DEFAULT_DATA_ROOT,
    collect_train_images,
    ensure_features,
    fit_pca_ridge,
    load_gallery_manifest,
    load_query_manifest,
    load_train_pairs,
    all_prediction_pools,
    ranking_rows_from_scores,
    score_method,
    write_csv,
    zscore_by_query,
    PCA_FEATURES,
)


def assignment_rerank_scores(scores: np.ndarray, boost: float) -> np.ndarray:
    if scores.shape[0] != scores.shape[1]:
        raise ValueError(f"Assignment rerank expects square pools, got {scores.shape}")
    row_ind, col_ind = linear_sum_assignment(-scores.astype(np.float64))
    assigned = np.empty(scores.shape[0], dtype=np.int64)
    assigned[row_ind] = col_ind
    boosted = scores.copy()
    if boost == float("inf"):
        floor = float(np.nanmin(boosted)) - 1.0
        boosted[:] = np.maximum(boosted, floor)
        boosted[np.arange(scores.shape[0]), assigned] = float(np.nanmax(scores)) + 1e6
    else:
        boosted[np.arange(scores.shape[0]), assigned] += boost
    return boosted


def blended_scores(
    methods: list[str],
    weights: list[float],
    query_ids: list[str],
    target_ids: list[str],
    cache_dir: Path,
    pca_model: object,
) -> np.ndarray:
    if len(methods) != len(weights):
        raise ValueError("--blend-methods and --blend-weights must have the same length")
    if not methods:
        raise ValueError("At least one method is required")
    total = np.zeros((len(query_ids), len(target_ids)), dtype=np.float32)
    weight_sum = 0.0
    for method, weight in zip(methods, weights):
        if weight == 0:
            continue
        scores = score_method(method, query_ids, target_ids, cache_dir, pca_model)  # type: ignore[arg-type]
        if len(methods) > 1:
            scores = zscore_by_query(scores)
        total += float(weight) * scores.astype(np.float32, copy=False)
        weight_sum += abs(float(weight))
    if weight_sum == 0:
        raise ValueError("At least one blend weight must be non-zero")
    return total


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--method", default="pca_ridge")
    parser.add_argument("--blend-methods", nargs="+")
    parser.add_argument("--blend-weights", nargs="+", type=float)
    parser.add_argument("--pca-components", type=int, default=128)
    parser.add_argument("--pca-alpha", type=float, default=100.0)
    parser.add_argument("--boost", type=float, default=float("inf"))
    parser.add_argument("--datasets", nargs="+", choices=["dataset1", "dataset2", "dataset3"], default=["dataset1", "dataset2", "dataset3"])
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--out", type=Path, default=Path("submissions/all_pca_ridge_c128_a100_hungarian.csv"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    cache_dir = args.cache_dir.resolve()
    train_pairs = load_train_pairs(data_root)

    images = collect_train_images(train_pairs)
    manifests = []
    for pool in all_prediction_pools(data_root, tuple(args.datasets), tuple(args.splits)):
        queries = load_query_manifest(data_root, pool.query_csv)
        targets = load_gallery_manifest(data_root, pool.gallery_csv)
        images.update(queries)
        images.update(targets)
        manifests.append((pool, queries, targets))
    ensure_features(images, cache_dir, force=False)

    pca_model = None
    methods = args.blend_methods or [args.method]
    weights = args.blend_weights or [1.0 for _ in methods]
    if len(methods) != len(weights):
        raise SystemExit("--blend-methods and --blend-weights must have the same length")

    if any(method in {"pca_ridge", "fusion_default_plus_pca"} for method in methods):
        pca_model = fit_pca_ridge(
            [row["query_id"] for row in train_pairs],
            [row["target_id"] for row in train_pairs],
            cache_dir,
            PCA_FEATURES,
            n_components=args.pca_components,
            alpha=args.pca_alpha,
        )

    submission_rows: list[dict[str, str]] = []
    for pool, queries, targets in manifests:
        query_ids = sorted(queries)
        target_ids = sorted(targets)
        scores = blended_scores(methods, weights, query_ids, target_ids, cache_dir, pca_model)
        scores = assignment_rerank_scores(scores, args.boost)
        submission_rows.extend(ranking_rows_from_scores(query_ids, target_ids, scores))
        print(f"assignment-ranked {pool.dataset}/{pool.split}: {len(query_ids)}x{len(target_ids)}")

    write_csv(args.out, submission_rows, ["query_id", "target_id_ranking"])
    print(f"wrote {len(submission_rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
