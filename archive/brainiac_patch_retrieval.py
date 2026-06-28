from __future__ import annotations
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "monai==1.3.2",
#   "torch>=2.7.0",
#   "numpy>=2.0",
#   "nibabel>=5.3",
#   "scipy>=1.14",
# ]
# ///

"""BrainIAC ViT patch-token matching for deformation-tolerant retrieval."""

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import ViT
from scipy.optimize import linear_sum_assignment
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_ROOT = Path("ehl-paris-medical-image-retrieval")
DEFAULT_CHECKPOINT = Path("BrainIAC.ckpt")
DEFAULT_OUT_DIR = Path("brainiac_patch_submissions")
IMAGE_SIZE = (96, 96, 96)
PATCH_GRID = (6, 6, 6)


@dataclass(frozen=True)
class Pool:
    dataset: str
    split: str
    query_csv: Path
    gallery_csv: Path


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


def all_prediction_pools(data_root: Path, datasets: tuple[str, ...], splits: tuple[str, ...]) -> list[Pool]:
    pools: list[Pool] = []
    for dataset in datasets:
        for split in splits:
            root = data_root / dataset
            pools.append(Pool(dataset, split, root / f"{split}_queries.csv", root / f"{split}_gallery.csv"))
    return pools


def normalize_nonzero(volume: np.ndarray) -> np.ndarray:
    volume = np.nan_to_num(volume.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.isfinite(volume) & (np.abs(volume) > 1e-6)
    foreground = volume[mask]
    if foreground.size < 256:
        foreground = volume[np.isfinite(volume)]
    if foreground.size == 0:
        return volume
    mean = float(foreground.mean())
    std = float(foreground.std())
    if std < 1e-6:
        volume[mask] = volume[mask] - mean
    else:
        volume[mask] = (volume[mask] - mean) / std
    volume[~mask] = 0.0
    return volume


def load_and_preprocess_volume(path: Path) -> torch.Tensor:
    image = nib.load(str(path))
    volume = np.asanyarray(image.dataobj).astype(np.float32)
    if volume.ndim > 3:
        volume = volume[..., 0]
    if volume.ndim != 3:
        raise ValueError(f"Expected 3D image, got {volume.shape} for {path}")
    volume = normalize_nonzero(volume)
    tensor = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=IMAGE_SIZE, mode="trilinear", align_corners=False)
    return tensor.squeeze(0).contiguous()


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], data_root: Path, id_key: str, image_key: str) -> None:
        self.rows = rows
        self.data_root = data_root
        self.id_key = id_key
        self.image_key = image_key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        return {
            "id": row[self.id_key],
            "image": load_and_preprocess_volume(resolve_image_path(self.data_root, row[self.image_key])),
        }


class BrainIACPatchNet(nn.Module):
    def __init__(self, checkpoint_path: Path) -> None:
        super().__init__()
        self.backbone = ViT(
            in_channels=1,
            img_size=IMAGE_SIZE,
            patch_size=(16, 16, 16),
            hidden_size=768,
            mlp_dim=3072,
            num_layers=12,
            num_heads=12,
            save_attn=False,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        backbone_state_dict = {key[9:]: value for key, value in state_dict.items() if key.startswith("backbone.")}
        self.backbone.load_state_dict(backbone_state_dict, strict=True)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        token_embeddings = self.backbone(x)[0]
        if token_embeddings.shape[1] == np.prod(PATCH_GRID):
            patches_raw = token_embeddings
            cls_raw = token_embeddings.mean(dim=1)
        else:
            cls_raw = token_embeddings[:, 0]
            patches_raw = token_embeddings[:, 1:]
        cls = F.normalize(cls_raw, dim=-1)
        patches = F.normalize(patches_raw, dim=-1)
        return cls, patches


@torch.no_grad()
def embed_manifest(
    model: BrainIACPatchNet,
    rows: list[dict[str, str]],
    data_root: Path,
    id_key: str,
    image_key: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    dataset = ManifestDataset(rows, data_root, id_key, image_key)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type != "cpu")
    ids: list[str] = []
    cls_vectors: list[np.ndarray] = []
    patch_vectors: list[np.ndarray] = []
    model.eval()
    for batch in loader:
        ids.extend(batch["id"])
        images = batch["image"].to(device)
        cls, patches = model(images)
        cls_vectors.append(cls.cpu().numpy().astype(np.float32))
        patch_vectors.append(patches.cpu().numpy().astype(np.float16))
    return ids, np.concatenate(cls_vectors, axis=0), np.concatenate(patch_vectors, axis=0)


def patch_positions(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.stack(
        torch.meshgrid(
            torch.linspace(-1.0, 1.0, PATCH_GRID[0], device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, PATCH_GRID[1], device=device, dtype=dtype),
            torch.linspace(-1.0, 1.0, PATCH_GRID[2], device=device, dtype=dtype),
            indexing="ij",
        ),
        dim=-1,
    )
    return coords.reshape(-1, 3)


@torch.no_grad()
def patch_match_scores(
    query_patches: np.ndarray,
    target_patches: np.ndarray,
    query_cls: np.ndarray,
    target_cls: np.ndarray,
    device: torch.device,
    topk: int,
    position_weight: float,
    cls_weight: float,
    target_batch_size: int,
) -> np.ndarray:
    q_patches = torch.from_numpy(query_patches.astype(np.float32)).to(device)
    t_patches = torch.from_numpy(target_patches.astype(np.float32)).to(device)
    q_cls = torch.from_numpy(query_cls.astype(np.float32)).to(device)
    t_cls = torch.from_numpy(target_cls.astype(np.float32)).to(device)
    positions = patch_positions(device, q_patches.dtype)
    penalty = torch.cdist(positions, positions, p=2) * float(position_weight)
    scores = torch.empty((q_patches.shape[0], t_patches.shape[0]), device=device, dtype=torch.float32)
    k = min(topk, q_patches.shape[1])
    for qi in range(q_patches.shape[0]):
        q = q_patches[qi]
        for start in range(0, t_patches.shape[0], target_batch_size):
            targets = t_patches[start : start + target_batch_size]
            sim = torch.einsum("pd,bkd->bpk", q, targets)
            if position_weight > 0:
                sim = sim - penalty.unsqueeze(0)
            q_to_t = sim.topk(k, dim=2).values.mean(dim=(1, 2))
            t_to_q = sim.topk(k, dim=1).values.mean(dim=(1, 2))
            score = 0.5 * (q_to_t + t_to_q)
            if cls_weight:
                score = score + float(cls_weight) * (q_cls[qi].unsqueeze(0) @ t_cls[start : start + target_batch_size].T).squeeze(0)
            scores[qi, start : start + target_batch_size] = score.float()
    return scores.cpu().numpy().astype(np.float32)


def apply_assignment(scores: np.ndarray, enabled: bool) -> np.ndarray:
    if not enabled:
        return scores
    row_ind, col_ind = linear_sum_assignment(-scores.astype(np.float64))
    assigned = np.empty(scores.shape[0], dtype=np.int64)
    assigned[row_ind] = col_ind
    out = scores.copy()
    out[np.arange(scores.shape[0]), assigned] = float(np.nanmax(scores)) + 1e6
    return out


def ranking_rows(query_ids: list[str], target_ids: list[str], scores: np.ndarray) -> list[dict[str, str]]:
    target_ids_arr = np.asarray(target_ids)
    rows = []
    for query_index, query_id in enumerate(query_ids):
        order = np.argsort(-scores[query_index], kind="mergesort")
        rows.append({"query_id": query_id, "target_id_ranking": " ".join(target_ids_arr[order].tolist())})
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--datasets", nargs="+", choices=["dataset1", "dataset2", "dataset3"], default=["dataset2"])
    parser.add_argument("--splits", nargs="+", choices=["val", "test"], default=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--target-batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--position-weights", nargs="+", type=float, default=[0.0, 0.04])
    parser.add_argument("--cls-weight", type=float, default=0.05)
    parser.add_argument("--assignment", action="store_true")
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_root = args.data_root.resolve()
    checkpoint = args.checkpoint.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({"device": str(device), "checkpoint": str(checkpoint), "out_dir": str(out_dir)}, indent=2), flush=True)
    model = BrainIACPatchNet(checkpoint).to(device)

    rows_by_weight = {weight: [] for weight in args.position_weights}
    for pool in all_prediction_pools(data_root, tuple(args.datasets), tuple(args.splits)):
        query_rows = read_csv(pool.query_csv)
        target_rows = read_csv(pool.gallery_csv)
        query_ids, query_cls, query_patches = embed_manifest(model, query_rows, data_root, "query_id", "query_image", args.batch_size, args.num_workers, device)
        target_ids, target_cls, target_patches = embed_manifest(model, target_rows, data_root, "target_id", "target_image", args.batch_size, args.num_workers, device)
        print(
            f"embedded {pool.dataset}/{pool.split}: q={query_patches.shape} t={target_patches.shape}",
            flush=True,
        )
        for position_weight in args.position_weights:
            scores = patch_match_scores(
                query_patches,
                target_patches,
                query_cls,
                target_cls,
                device,
                args.topk,
                position_weight,
                args.cls_weight,
                args.target_batch_size,
            )
            scores = apply_assignment(scores, args.assignment)
            rows_by_weight[position_weight].extend(ranking_rows(query_ids, target_ids, scores))
            np.savez_compressed(
                out_dir / f"{pool.dataset}_{pool.split}_top{args.topk}_pos{position_weight:g}_scores.npz",
                query_ids=np.asarray(query_ids),
                target_ids=np.asarray(target_ids),
                scores=scores.astype(np.float32),
            )
            print(f"scored {pool.dataset}/{pool.split} pos={position_weight:g}", flush=True)

    suffix = "_assign" if args.assignment else ""
    for position_weight, rows in rows_by_weight.items():
        out = out_dir / f"{'_'.join(args.datasets)}_patch_top{args.topk}_pos{position_weight:g}_cls{args.cls_weight:g}{suffix}.csv"
        write_csv(out, rows)
        print(f"wrote {len(rows)} rows to {out}", flush=True)


if __name__ == "__main__":
    main()
