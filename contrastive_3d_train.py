from __future__ import annotations
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "torch>=2.7.0",
#   "numpy>=2.0",
#   "nibabel>=5.3",
# ]
# ///

"""3D contrastive retrieval model with dataset2/dataset3 style augmentation.

This is a separate experiment from the BrainIAC adapter. It trains directly on
dataset1 query/target images and uses strong spatial and intensity augmentation
to make the embedding less brittle to nonlinear deformations and surgery-like
missing regions.
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_ROOT = Path("ehl-paris-medical-image-retrieval")
DEFAULT_RUN_DIR = Path("runs/contrastive_3d")
SEED = 20260627


@dataclass(frozen=True)
class Pool:
    dataset: str
    split: str
    query_csv: Path
    gallery_csv: Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    def default(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=default)
        f.write("\n")


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


def all_prediction_pools(data_root: Path) -> list[Pool]:
    pools: list[Pool] = []
    for dataset in ("dataset1", "dataset2", "dataset3"):
        for split in ("val", "test"):
            root = data_root / dataset
            pools.append(Pool(dataset, split, root / f"{split}_queries.csv", root / f"{split}_gallery.csv"))
    return pools


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _base_subject(query_id: str) -> str:
    """Strip an augmentation suffix (q_xxx_aug00 -> q_xxx) so all variants of one
    subject share a holdout group and cannot leak across train/holdout."""
    return re.sub(r"_aug\d+$", "", query_id)


def stable_split(
    pairs: list[dict[str, str]],
    holdout_frac: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    # group pairs by base subject, then split whole groups by hash
    groups: dict[str, list[dict[str, str]]] = {}
    for row in pairs:
        groups.setdefault(_base_subject(row["query_id"]), []).append(row)
    keyed = sorted(
        groups.items(), key=lambda kv: hashlib.sha256(f"{seed}:{kv[0]}".encode()).hexdigest()
    )
    n_groups = len(keyed)
    holdout_g = max(1, min(n_groups - 1, round(n_groups * holdout_frac)))
    holdout = [row for _, rows in keyed[:holdout_g] for row in rows]
    train = [row for _, rows in keyed[holdout_g:] for row in rows]
    return train, holdout


def foreground_bbox(volume: np.ndarray, margin: int) -> tuple[slice, slice, slice]:
    finite = np.isfinite(volume)
    nonzero = finite & (np.abs(volume) > 1e-6)
    if int(nonzero.sum()) < 128:
        nonzero = finite
    coords = np.where(nonzero)
    if len(coords[0]) == 0:
        return tuple(slice(0, s) for s in volume.shape)  # type: ignore[return-value]
    slices = []
    for axis, values in enumerate(coords):
        lo = max(0, int(values.min()) - margin)
        hi = min(volume.shape[axis], int(values.max()) + margin + 1)
        slices.append(slice(lo, hi))
    return tuple(slices)  # type: ignore[return-value]


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    volume = np.nan_to_num(volume.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.abs(volume) > 1e-6
    values = volume[mask]
    if values.size < 256:
        values = volume.reshape(-1)
    lo, hi = np.percentile(values, [1.0, 99.0]).astype(np.float32)
    if not np.isfinite(hi - lo) or float(hi - lo) < 1e-6:
        lo, hi = np.percentile(values, [0.0, 100.0]).astype(np.float32)
    scale = float(hi - lo) if float(hi - lo) > 1e-6 else 1.0
    volume = np.clip((volume - lo) / scale, 0.0, 1.0)
    volume = (volume - 0.5) * 2.0
    volume[~mask] = 0.0
    return volume.astype(np.float32, copy=False)


def load_tensor(path: Path, image_size: int, crop_margin: int) -> torch.Tensor:
    image = nib.load(str(path))
    volume = np.asanyarray(image.dataobj).astype(np.float32)
    if volume.ndim > 3:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {volume.shape} for {path}")
    volume = volume[foreground_bbox(volume, crop_margin)]
    volume = normalize_volume(volume)
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(image_size, image_size, image_size), mode="trilinear", align_corners=False)
    return tensor.squeeze(0).contiguous()


class TensorCache:
    def __init__(self, data_root: Path, image_size: int, crop_margin: int) -> None:
        self.data_root = data_root
        self.image_size = image_size
        self.crop_margin = crop_margin
        self.cache: dict[str, torch.Tensor] = {}
        self.paths: dict[str, Path] = {}

    def add(self, image_id: str, image_path: str) -> None:
        self.paths[image_id] = resolve_image_path(self.data_root, image_path)

    def get(self, image_id: str) -> torch.Tensor:
        tensor = self.cache.get(image_id)
        if tensor is None:
            tensor = load_tensor(self.paths[image_id], self.image_size, self.crop_margin)
            self.cache[image_id] = tensor
        return tensor

    def preload(self, ids: list[str]) -> None:
        for index, image_id in enumerate(ids, start=1):
            self.get(image_id)
            if index % 100 == 0:
                print(f"preloaded {index}/{len(ids)} images", flush=True)


class PairTensorDataset(Dataset):
    def __init__(self, pairs: list[dict[str, str]], cache: TensorCache) -> None:
        self.pairs = pairs
        self.cache = cache

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.pairs[index]
        return self.cache.get(row["query_id"]), self.cache.get(row["target_id"])


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        groups = min(8, out_channels)
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Encoder3D(nn.Module):
    def __init__(self, embedding_dim: int, width: int, dropout: float) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1, width, 2),
            ConvBlock(width, width * 2, 2),
            ConvBlock(width * 2, width * 4, 2),
            ConvBlock(width * 4, width * 8, 2),
            ConvBlock(width * 8, width * 8, 2),
            nn.AdaptiveAvgPool3d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.LayerNorm(width * 8),
            nn.Dropout(dropout),
            nn.Linear(width * 8, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(self.features(x)), dim=1)


class DualEncoder3D(nn.Module):
    def __init__(self, embedding_dim: int, width: int, dropout: float, shared: bool) -> None:
        super().__init__()
        self.query_encoder = Encoder3D(embedding_dim, width, dropout)
        self.target_encoder = self.query_encoder if shared else Encoder3D(embedding_dim, width, dropout)
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(x)

    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        return self.target_encoder(x)

    def forward(self, query: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        scale = self.logit_scale.clamp(math.log(1.0), math.log(100.0)).exp()
        return scale * self.encode_query(query) @ self.encode_target(target).T


def random_affine_grid(batch: int, image_size: int, device: torch.device, degrees: float, translate: float, scale_range: float) -> torch.Tensor:
    angles = (torch.rand(batch, 3, device=device) * 2.0 - 1.0) * math.radians(degrees)
    sx = torch.sin(angles[:, 0])
    cx = torch.cos(angles[:, 0])
    sy = torch.sin(angles[:, 1])
    cy = torch.cos(angles[:, 1])
    sz = torch.sin(angles[:, 2])
    cz = torch.cos(angles[:, 2])

    zeros = torch.zeros(batch, device=device)
    ones = torch.ones(batch, device=device)
    rx = torch.stack(
        [ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx],
        dim=1,
    ).reshape(batch, 3, 3)
    ry = torch.stack(
        [cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy],
        dim=1,
    ).reshape(batch, 3, 3)
    rz = torch.stack(
        [cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones],
        dim=1,
    ).reshape(batch, 3, 3)
    matrix = rz @ ry @ rx
    scale = 1.0 + (torch.rand(batch, 1, 1, device=device) * 2.0 - 1.0) * scale_range
    matrix = matrix * scale
    offset = (torch.rand(batch, 3, 1, device=device) * 2.0 - 1.0) * translate
    theta = torch.cat([matrix, offset], dim=2)
    return F.affine_grid(theta, (batch, 1, image_size, image_size, image_size), align_corners=False)


def apply_elastic(x: torch.Tensor, strength: float, low_res: int) -> torch.Tensor:
    if strength <= 0:
        return x
    batch, _, depth, height, width = x.shape
    base = F.affine_grid(
        torch.eye(3, 4, device=x.device, dtype=x.dtype).unsqueeze(0).repeat(batch, 1, 1),
        x.shape,
        align_corners=False,
    )
    disp = torch.randn(batch, 3, low_res, low_res, low_res, device=x.device, dtype=x.dtype)
    disp = F.interpolate(disp, size=(depth, height, width), mode="trilinear", align_corners=False)
    disp = disp.permute(0, 2, 3, 4, 1)
    grid = base + disp * strength
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)


def apply_cutout(x: torch.Tensor, probability: float, max_radius: float) -> torch.Tensor:
    if probability <= 0 or max_radius <= 0:
        return x
    batch, _, depth, height, width = x.shape
    z = torch.linspace(-1, 1, depth, device=x.device, dtype=x.dtype).view(1, depth, 1, 1)
    y = torch.linspace(-1, 1, height, device=x.device, dtype=x.dtype).view(1, 1, height, 1)
    xcoord = torch.linspace(-1, 1, width, device=x.device, dtype=x.dtype).view(1, 1, 1, width)
    out = x
    for index in range(batch):
        if float(torch.rand((), device=x.device)) > probability:
            continue
        center = torch.rand(3, device=x.device, dtype=x.dtype) * 1.2 - 0.6
        radius = torch.rand(3, device=x.device, dtype=x.dtype) * max_radius + 0.08
        mask = (
            ((z - center[0]) / radius[0]) ** 2
            + ((y - center[1]) / radius[1]) ** 2
            + ((xcoord - center[2]) / radius[2]) ** 2
        ) < 1.0
        out[index] = torch.where(mask.unsqueeze(0), torch.zeros_like(out[index]), out[index])
    return out


def augment(x: torch.Tensor, args: argparse.Namespace) -> torch.Tensor:
    batch = x.shape[0]
    grid = random_affine_grid(batch, args.image_size, x.device, args.rotate_degrees, args.translate, args.scale_range)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)
    x = apply_elastic(x, args.elastic_strength, args.elastic_low_res)
    scale = 1.0 + (torch.rand(batch, 1, 1, 1, 1, device=x.device, dtype=x.dtype) * 2.0 - 1.0) * args.intensity_scale
    bias = (torch.rand(batch, 1, 1, 1, 1, device=x.device, dtype=x.dtype) * 2.0 - 1.0) * args.intensity_bias
    noise = torch.randn_like(x) * args.noise_std
    x = x * scale + bias + noise
    x = apply_cutout(x, args.cutout_prob, args.cutout_radius)
    return torch.clamp(x, -3.0, 3.0)


def metrics_from_scores(query_ids: list[str], target_ids: list[str], truth: dict[str, str], scores: np.ndarray) -> dict[str, float]:
    target_index = {target_id: index for index, target_id in enumerate(target_ids)}
    ranks = []
    for query_index, query_id in enumerate(query_ids):
        true_index = target_index[truth[query_id]]
        true_score = scores[query_index, true_index]
        rank = 1 + int(np.sum(scores[query_index] > true_score))
        ranks.append(rank)
    ranks_np = np.asarray(ranks, dtype=np.float32)
    return {
        "mrr": float(np.mean(1.0 / ranks_np)),
        "r1": float(np.mean(ranks_np <= 1)),
        "r5": float(np.mean(ranks_np <= 5)),
        "median_rank": float(np.median(ranks_np)),
        "mean_rank": float(np.mean(ranks_np)),
    }


@torch.no_grad()
def encode_ids(
    model: DualEncoder3D,
    cache: TensorCache,
    ids: list[str],
    side: str,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(ids), batch_size):
        batch = torch.stack([cache.get(image_id) for image_id in ids[start : start + batch_size]]).to(device)
        if side == "query":
            emb = model.encode_query(batch)
        else:
            emb = model.encode_target(batch)
        outputs.append(emb.float().cpu().numpy().astype(np.float32))
    return np.concatenate(outputs, axis=0)


def evaluate_holdout(
    model: DualEncoder3D,
    cache: TensorCache,
    holdout_pairs: list[dict[str, str]],
    all_target_ids: list[str],
    device: torch.device,
    batch_size: int,
) -> tuple[dict[str, float], dict[str, float]]:
    query_ids = [row["query_id"] for row in holdout_pairs]
    holdout_target_ids = [row["target_id"] for row in holdout_pairs]
    truth = {row["query_id"]: row["target_id"] for row in holdout_pairs}
    query_vectors = encode_ids(model, cache, query_ids, "query", device, batch_size)
    holdout_target_vectors = encode_ids(model, cache, holdout_target_ids, "target", device, batch_size)
    all_target_vectors = encode_ids(model, cache, all_target_ids, "target", device, batch_size)
    holdout_scores = query_vectors @ holdout_target_vectors.T
    all_scores = query_vectors @ all_target_vectors.T
    return (
        metrics_from_scores(query_ids, holdout_target_ids, truth, holdout_scores),
        metrics_from_scores(query_ids, all_target_ids, truth, all_scores),
    )


def ranking_rows_from_scores(query_ids: list[str], target_ids: list[str], scores: np.ndarray) -> list[dict[str, str]]:
    target_ids_arr = np.asarray(target_ids)
    rows: list[dict[str, str]] = []
    for query_index, query_id in enumerate(query_ids):
        order = np.argsort(-scores[query_index], kind="mergesort")
        rows.append({"query_id": query_id, "target_id_ranking": " ".join(target_ids_arr[order].tolist())})
    return rows


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    data_root = args.data_root.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else args.device)
    print(f"device={device}", flush=True)
    print(f"run_dir={run_dir}", flush=True)

    pairs_csv = args.train_pair_csv if args.train_pair_csv else data_root / "dataset1" / "train_pairs.csv"
    pairs = read_csv(Path(pairs_csv))
    print(f"train_pair_csv={pairs_csv} pairs={len(pairs)}", flush=True)
    train_pairs, holdout_pairs = stable_split(pairs, args.holdout_frac, args.split_seed)
    print(f"train={len(train_pairs)} holdout={len(holdout_pairs)} (subject-aware split)", flush=True)
    cache = TensorCache(data_root, args.image_size, args.crop_margin)
    for row in pairs:
        cache.add(row["query_id"], row["query_image"])
        cache.add(row["target_id"], row["target_image"])
    all_train_ids = [row["query_id"] for row in pairs] + [row["target_id"] for row in pairs]
    cache.preload(all_train_ids)
    all_target_ids = [row["target_id"] for row in pairs]

    train_dataset = PairTensorDataset(train_pairs, cache)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=len(train_dataset) >= args.batch_size)
    model = DualEncoder3D(args.embedding_dim, args.width, args.dropout, args.shared_encoder).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and args.amp))

    best_key = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        for query_cpu, target_cpu in loader:
            query = augment(query_cpu.to(device, non_blocking=True), args)
            target = augment(target_cpu.to(device, non_blocking=True), args)
            labels = torch.arange(query.shape[0], device=device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and args.amp), dtype=torch.bfloat16):
                logits = model(query, target)
                loss = (loss_fn(logits, labels) + loss_fn(logits.T, labels)) * 0.5
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            total_loss += float(loss.detach().cpu()) * query.shape[0]
            total_seen += query.shape[0]

        holdout_metrics, all_metrics = evaluate_holdout(model, cache, holdout_pairs, all_target_ids, device, args.embed_batch_size)
        row = {
            "epoch": float(epoch),
            "loss": total_loss / max(1, total_seen),
            "temperature": float(model.logit_scale.detach().clamp(math.log(1.0), math.log(100.0)).exp().cpu()),
            "holdout_mrr": holdout_metrics["mrr"],
            "holdout_r1": holdout_metrics["r1"],
            "all_gallery_mrr": all_metrics["mrr"],
            "all_gallery_r1": all_metrics["r1"],
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['loss']:.5f} holdout_mrr={row['holdout_mrr']:.4f} "
            f"all_gallery_mrr={row['all_gallery_mrr']:.4f}",
            flush=True,
        )
        key = 1000.0 * row["all_gallery_mrr"] + row["holdout_mrr"]
        if key > best_key:
            best_key = key
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            torch.save({"model": best_state, "args": vars(args), "history": history}, run_dir / "best.pt")

    if best_state is None:
        raise RuntimeError("No best state was saved")
    model.load_state_dict(best_state)
    write_csv(run_dir / "history.csv", [{k: f"{v:.8f}" for k, v in row.items()} for row in history], list(history[0].keys()))
    write_json(run_dir / "config.json", vars(args))

    if args.skip_predict:
        print("skip_predict=true; not writing submission", flush=True)
        return

    submission_rows: list[dict[str, str]] = []
    for pool in all_prediction_pools(data_root):
        query_rows = read_csv(pool.query_csv)
        target_rows = read_csv(pool.gallery_csv)
        for row in query_rows:
            cache.add(row["query_id"], row["query_image"])
        for row in target_rows:
            cache.add(row["target_id"], row["target_image"])
        query_ids = [row["query_id"] for row in query_rows]
        target_ids = [row["target_id"] for row in target_rows]
        query_vectors = encode_ids(model, cache, query_ids, "query", device, args.embed_batch_size)
        target_vectors = encode_ids(model, cache, target_ids, "target", device, args.embed_batch_size)
        scores = query_vectors @ target_vectors.T
        submission_rows.extend(ranking_rows_from_scores(query_ids, target_ids, scores))
        np.savez_compressed(
            run_dir / f"{pool.dataset}_{pool.split}_vectors.npz",
            query_ids=np.asarray(query_ids),
            target_ids=np.asarray(target_ids),
            query_vectors=query_vectors.astype(np.float32),
            target_vectors=target_vectors.astype(np.float32),
        )
        print(f"predicted {pool.dataset}/{pool.split}: {len(query_ids)}x{len(target_ids)}", flush=True)
    write_csv(run_dir / "submission.csv", submission_rows, ["query_id", "target_id_ranking"])
    print(f"wrote submission {run_dir / 'submission.csv'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--train-pair-csv", type=Path, default=None,
                        help="Override training pair CSV (e.g. augmented manifest).")
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--split-seed", type=int, default=SEED)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--crop-margin", type=int, default=4)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--shared-encoder", action="store_true")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--rotate-degrees", type=float, default=20.0)
    parser.add_argument("--translate", type=float, default=0.16)
    parser.add_argument("--scale-range", type=float, default=0.16)
    parser.add_argument("--elastic-strength", type=float, default=0.055)
    parser.add_argument("--elastic-low-res", type=int, default=5)
    parser.add_argument("--intensity-scale", type=float, default=0.4)
    parser.add_argument("--intensity-bias", type=float, default=0.25)
    parser.add_argument("--noise-std", type=float, default=0.04)
    parser.add_argument("--cutout-prob", type=float, default=0.35)
    parser.add_argument("--cutout-radius", type=float, default=0.24)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
