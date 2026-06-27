from __future__ import annotations
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "monai==1.3.2",
#   "torch>=2.7.0",
#   "numpy>=2.0",
#   "nibabel>=5.3",
#   "pandas>=2.0",
# ]
# ///

import argparse
import csv
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import ViT
from torch import nn
from torch.utils.data import DataLoader, Dataset


DEFAULT_DATA_ROOT = Path("ehl-paris-medical-image-retrieval")
DEFAULT_CHECKPOINT = Path("checkpoints/BrainIAC.ckpt")
DEFAULT_RUNS_ROOT = Path("runs")
IMAGE_SIZE = (96, 96, 96)
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


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: object) -> None:
    def json_default(value: object) -> object:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=json_default)
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
    raise FileNotFoundError(f"Image path not found: {path}")


def all_prediction_pools(
    data_root: Path,
    datasets: tuple[str, ...] = ("dataset1", "dataset2", "dataset3"),
    splits: tuple[str, ...] = ("val", "test"),
) -> list[Pool]:
    pools: list[Pool] = []
    for dataset in datasets:
        for split in splits:
            ds = data_root / dataset
            pools.append(Pool(dataset=dataset, split=split, query_csv=ds / f"{split}_queries.csv", gallery_csv=ds / f"{split}_gallery.csv"))
    return pools


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(device: str) -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pair_key(row: dict[str, str]) -> str:
    return row.get("pair_id") or f"{row['query_id']}|{row['target_id']}"


def stable_split(
    pairs: list[dict[str, str]],
    holdout_frac: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    scored = []
    for row in pairs:
        digest = hashlib.sha256(f"{seed}:{pair_key(row)}".encode()).hexdigest()
        scored.append((digest, row))
    scored.sort(key=lambda x: x[0])
    holdout_n = max(1, min(len(scored) - 1, round(len(scored) * holdout_frac)))
    holdout = [row for _, row in scored[:holdout_n]]
    train = [row for _, row in scored[holdout_n:]]
    return train, holdout


def normalize_nonzero(volume: np.ndarray) -> np.ndarray:
    volume = np.nan_to_num(volume.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.isfinite(volume) & (np.abs(volume) > 1e-6)
    foreground = volume[mask]
    if foreground.size == 0:
        return volume
    mean = float(foreground.mean())
    std = float(foreground.std())
    out = volume.copy()
    if std < 1e-6:
        out[mask] = out[mask] - mean
    else:
        out[mask] = (out[mask] - mean) / std
    out[~mask] = 0.0
    return out


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


class ViTBackboneNet(nn.Module):
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
            save_attn=True,
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint.get("state_dict", checkpoint)
        backbone_state_dict = {key[9:]: value for key, value in state_dict.items() if key.startswith("backbone.")}
        self.backbone.load_state_dict(backbone_state_dict, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        token_embeddings = features[0]
        pooled = token_embeddings.mean(dim=1)
        return F.normalize(pooled, dim=1)


class ManifestImageDataset(Dataset):
    def __init__(self, rows: list[dict[str, str]], data_root: Path, id_key: str, image_key: str) -> None:
        self.rows = rows
        self.data_root = data_root
        self.id_key = id_key
        self.image_key = image_key

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.rows[index]
        image_id = row[self.id_key]
        image = load_and_preprocess_volume(resolve_image_path(self.data_root, row[self.image_key]))
        return {"id": image_id, "image": image}


def embed_manifest(
    model: ViTBackboneNet,
    rows: list[dict[str, str]],
    data_root: Path,
    id_key: str,
    image_key: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> tuple[list[str], np.ndarray]:
    dataset = ManifestImageDataset(rows, data_root, id_key, image_key)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type != "cpu")
    ids: list[str] = []
    outputs: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            ids.extend(batch["id"])
            embeddings = model(batch["image"].to(device)).detach().cpu().numpy().astype(np.float32)
            outputs.append(embeddings)
    return ids, np.concatenate(outputs, axis=0)


class AdapterModel(nn.Module):
    def __init__(self, input_dim: int = 768, hidden_dim: int = 768, output_dim: int = 256, dropout: float = 0.1) -> None:
        super().__init__()
        self.query_adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.target_adapter = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )
        self.logit_scale = nn.Parameter(torch.tensor(math.log(10.0), dtype=torch.float32))

    def encode_query(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.query_adapter(x), dim=1)

    def encode_target(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.target_adapter(x), dim=1)

    def scale(self) -> torch.Tensor:
        return self.logit_scale.clamp(math.log(1.0), math.log(100.0)).exp()

    def forward(self, query: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.scale() * self.encode_query(query) @ self.encode_target(target).T


def symmetric_contrastive_loss(logits: torch.Tensor, label_smoothing: float) -> torch.Tensor:
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (
        F.cross_entropy(logits, labels, label_smoothing=label_smoothing)
        + F.cross_entropy(logits.T, labels, label_smoothing=label_smoothing)
    ) / 2.0


def metrics_from_scores(
    query_ids: list[str],
    target_ids: list[str],
    truth: dict[str, str],
    score_matrix: np.ndarray,
) -> dict[str, float]:
    target_ids_sorted = list(target_ids)
    target_index = {target_id: index for index, target_id in enumerate(target_ids_sorted)}
    ranks = []
    for query_index, query_id in enumerate(query_ids):
        true_index = target_index[truth[query_id]]
        true_score = score_matrix[query_index, true_index]
        rank = 1 + int(np.sum(score_matrix[query_index] > true_score))
        ranks.append(rank)
    ranks_np = np.asarray(ranks, dtype=np.float32)
    return {
        "mrr": float(np.mean(1.0 / ranks_np)),
        "recall_at_1": float(np.mean(ranks_np <= 1)),
        "recall_at_5": float(np.mean(ranks_np <= 5)),
        "median_rank": float(np.median(ranks_np)),
        "mean_rank": float(np.mean(ranks_np)),
        "pairs": float(len(query_ids)),
    }


def ranking_rows_from_scores(query_ids: list[str], target_ids: list[str], score_matrix: np.ndarray) -> list[dict[str, str]]:
    rows = []
    target_ids_arr = np.asarray(target_ids)
    for query_index, query_id in enumerate(query_ids):
        order = np.argsort(-score_matrix[query_index], kind="mergesort")
        rows.append({"query_id": query_id, "target_id_ranking": " ".join(target_ids_arr[order].tolist())})
    return rows


@torch.no_grad()
def adapted_vectors(model: AdapterModel, query_vectors: np.ndarray, target_vectors: np.ndarray, device: torch.device, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    query_outputs: list[np.ndarray] = []
    target_outputs: list[np.ndarray] = []
    for start in range(0, len(query_vectors), batch_size):
        batch = torch.from_numpy(query_vectors[start : start + batch_size]).to(device)
        query_outputs.append(model.encode_query(batch).cpu().numpy().astype(np.float32))
    for start in range(0, len(target_vectors), batch_size):
        batch = torch.from_numpy(target_vectors[start : start + batch_size]).to(device)
        target_outputs.append(model.encode_target(batch).cpu().numpy().astype(np.float32))
    return np.concatenate(query_outputs, axis=0), np.concatenate(target_outputs, axis=0)


def evaluate_pair_metrics(
    model: AdapterModel | None,
    query_ids: list[str],
    query_vectors: np.ndarray,
    target_ids: list[str],
    target_vectors: np.ndarray,
    truth: dict[str, str],
    device: torch.device,
    batch_size: int,
) -> dict[str, float]:
    if model is None:
        score_matrix = query_vectors @ target_vectors.T
    else:
        adapted_query, adapted_target = adapted_vectors(model, query_vectors, target_vectors, device, batch_size)
        score_matrix = adapted_query @ adapted_target.T
    return metrics_from_scores(query_ids, target_ids, truth, score_matrix.astype(np.float32))


def save_vectors(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def train_adapter(
    model: AdapterModel,
    train_query_vectors: np.ndarray,
    train_target_vectors: np.ndarray,
    holdout_query_ids: list[str],
    holdout_query_vectors: np.ndarray,
    holdout_target_ids: list[str],
    holdout_target_vectors: np.ndarray,
    holdout_truth: dict[str, str],
    all_target_ids: list[str],
    all_target_vectors: np.ndarray,
    device: torch.device,
    epochs: int,
    early_stop_patience: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    label_smoothing: float,
    early_stop_monitor: str,
) -> tuple[
    AdapterModel,
    list[dict[str, float]],
    dict[str, float],
    dict[str, float],
    dict[str, float | int | bool | str],
]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_query_tensor = torch.from_numpy(train_query_vectors.astype(np.float32)).to(device)
    train_target_tensor = torch.from_numpy(train_target_vectors.astype(np.float32)).to(device)

    best_state = None
    best_metrics = None
    best_all_gallery_metrics = None
    best_key = -math.inf
    best_metric_epoch = 0
    best_loss = math.inf
    best_loss_epoch = 0
    epochs_since_best_loss = 0
    best_monitor_value = -math.inf if early_stop_monitor != "loss" else math.inf
    best_monitor_epoch = 0
    epochs_since_best_monitor = 0
    epochs_since_best_metric = 0
    stopped_early = False
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(train_query_tensor, train_target_tensor)
        loss = symmetric_contrastive_loss(logits, label_smoothing=label_smoothing)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        holdout_metrics = evaluate_pair_metrics(
            model,
            holdout_query_ids,
            holdout_query_vectors,
            holdout_target_ids,
            holdout_target_vectors,
            holdout_truth,
            device,
            batch_size,
        )
        all_gallery_metrics = evaluate_pair_metrics(
            model,
            holdout_query_ids,
            holdout_query_vectors,
            all_target_ids,
            all_target_vectors,
            holdout_truth,
            device,
            batch_size,
        )
        key = 1000.0 * all_gallery_metrics["mrr"] + holdout_metrics["mrr"]
        row = {
            "epoch": float(epoch),
            "loss": float(loss.detach().cpu()),
            "temperature": float(model.scale().detach().cpu()),
            "holdout_mrr": holdout_metrics["mrr"],
            "holdout_r1": holdout_metrics["recall_at_1"],
            "all_gallery_mrr": all_gallery_metrics["mrr"],
            "all_gallery_r1": all_gallery_metrics["recall_at_1"],
            "selection_key": key,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} loss={row['loss']:.5f} "
            f"holdout_mrr={row['holdout_mrr']:.4f} all_gallery_mrr={row['all_gallery_mrr']:.4f}",
            flush=True,
        )

        if row["loss"] < best_loss:
            best_loss = row["loss"]
            best_loss_epoch = epoch
            epochs_since_best_loss = 0
        else:
            epochs_since_best_loss += 1

        monitor_value = row[early_stop_monitor]
        if early_stop_monitor == "loss":
            improved_monitor = monitor_value < best_monitor_value
        else:
            improved_monitor = monitor_value > best_monitor_value
        if improved_monitor:
            best_monitor_value = monitor_value
            best_monitor_epoch = epoch
            epochs_since_best_monitor = 0
        else:
            epochs_since_best_monitor += 1

        if key > best_key:
            best_key = key
            best_metric_epoch = epoch
            epochs_since_best_metric = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_metrics = holdout_metrics
            best_all_gallery_metrics = all_gallery_metrics
        else:
            epochs_since_best_metric += 1

        if early_stop_patience > 0 and epochs_since_best_monitor >= early_stop_patience:
            stopped_early = True
            print(
                f"early stopping at epoch {epoch:03d}: "
                f"best_{early_stop_monitor}={best_monitor_value:.5f} at epoch {best_monitor_epoch:03d}, "
                f"no {early_stop_monitor} improvement for {epochs_since_best_monitor} epochs",
                flush=True,
            )
            break

    if best_state is None or best_metrics is None or best_all_gallery_metrics is None:
        raise RuntimeError("Training finished without a best checkpoint")

    model.load_state_dict(best_state)
    training_summary: dict[str, float | int | bool | str] = {
        "epochs_requested": epochs,
        "epochs_completed": len(history),
        "stopped_early": stopped_early,
        "early_stop_patience": early_stop_patience,
        "early_stop_monitor": early_stop_monitor,
        "best_monitor_value": best_monitor_value,
        "best_monitor_epoch": best_monitor_epoch,
        "best_metric_epoch": best_metric_epoch,
        "best_selection_key": best_key,
        "best_holdout_mrr": best_metrics["mrr"],
        "best_all_gallery_mrr": best_all_gallery_metrics["mrr"],
        "best_loss": best_loss,
        "best_loss_epoch": best_loss_epoch,
        "epochs_since_best_loss": epochs_since_best_loss,
        "epochs_since_best_monitor": epochs_since_best_monitor,
        "epochs_since_best_metric": epochs_since_best_metric,
        "label_smoothing": label_smoothing,
        "train_pairs": len(train_query_vectors),
        "train_loss_mode": "full_batch_symmetric_contrastive",
    }
    return model, history, best_metrics, best_all_gallery_metrics, training_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUNS_ROOT / "brainiac_adapter")
    parser.add_argument("--holdout-frac", type=float, default=0.20)
    parser.add_argument("--split-seed", type=int, default=SEED)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--embed-batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--early-stop-patience", type=int, default=15)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument(
        "--early-stop-monitor",
        choices=("loss", "holdout_mrr", "all_gallery_mrr", "selection_key"),
        default="loss",
    )
    parser.add_argument("--hidden-dim", type=int, default=768)
    parser.add_argument("--output-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    script_path = Path(__file__).resolve()

    data_root = args.data_root.resolve()
    checkpoint = args.checkpoint.resolve()
    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    weights_path = run_dir / "adapter.pt"
    device = get_device(args.device)

    print(f"training_script={script_path}", flush=True)
    print(f"run_dir={run_dir}", flush=True)
    print(f"weights_path={weights_path}", flush=True)

    train_pairs = read_csv(data_root / "dataset1" / "train_pairs.csv")
    train_split, holdout_split = stable_split(train_pairs, args.holdout_frac, args.split_seed)
    split_payload = {
        "seed": args.seed,
        "split_seed": args.split_seed,
        "holdout_frac": args.holdout_frac,
        "train_pairs": len(train_split),
        "holdout_pairs": len(holdout_split),
        "train_pair_ids": [pair_key(row) for row in train_split],
        "holdout_pair_ids": [pair_key(row) for row in holdout_split],
    }
    write_json(run_dir / "split.json", split_payload)

    backbone = ViTBackboneNet(checkpoint).to(device)

    all_train_queries = [{"query_id": row["query_id"], "query_image": row["query_image"]} for row in train_pairs]
    all_train_targets = [{"target_id": row["target_id"], "target_image": row["target_image"]} for row in train_pairs]
    train_query_ids, train_query_vectors = embed_manifest(backbone, all_train_queries, data_root, "query_id", "query_image", args.embed_batch_size, args.num_workers, device)
    train_target_ids, train_target_vectors = embed_manifest(backbone, all_train_targets, data_root, "target_id", "target_image", args.embed_batch_size, args.num_workers, device)
    query_map = {image_id: vector for image_id, vector in zip(train_query_ids, train_query_vectors)}
    target_map = {image_id: vector for image_id, vector in zip(train_target_ids, train_target_vectors)}

    train_query_matrix = np.stack([query_map[row["query_id"]] for row in train_split]).astype(np.float32)
    train_target_matrix = np.stack([target_map[row["target_id"]] for row in train_split]).astype(np.float32)
    holdout_query_ids = [row["query_id"] for row in holdout_split]
    holdout_target_ids = [row["target_id"] for row in holdout_split]
    holdout_query_matrix = np.stack([query_map[query_id] for query_id in holdout_query_ids]).astype(np.float32)
    holdout_target_matrix = np.stack([target_map[target_id] for target_id in holdout_target_ids]).astype(np.float32)
    holdout_truth = {row["query_id"]: row["target_id"] for row in holdout_split}
    all_gallery_target_ids = list(train_target_ids)
    all_gallery_target_matrix = np.stack([target_map[target_id] for target_id in all_gallery_target_ids]).astype(np.float32)

    raw_holdout_metrics = evaluate_pair_metrics(
        None,
        holdout_query_ids,
        holdout_query_matrix,
        holdout_target_ids,
        holdout_target_matrix,
        holdout_truth,
        device,
        args.batch_size,
    )
    raw_all_gallery_metrics = evaluate_pair_metrics(
        None,
        holdout_query_ids,
        holdout_query_matrix,
        all_gallery_target_ids,
        all_gallery_target_matrix,
        holdout_truth,
        device,
        args.batch_size,
    )

    adapter = AdapterModel(input_dim=train_query_matrix.shape[1], hidden_dim=args.hidden_dim, output_dim=args.output_dim, dropout=args.dropout)
    adapter, history, best_holdout_metrics, best_all_gallery_metrics, training_summary = train_adapter(
        adapter,
        train_query_matrix,
        train_target_matrix,
        holdout_query_ids,
        holdout_query_matrix,
        holdout_target_ids,
        holdout_target_matrix,
        holdout_truth,
        all_gallery_target_ids,
        all_gallery_target_matrix,
        device,
        args.epochs,
        args.early_stop_patience,
        args.batch_size,
        args.learning_rate,
        args.weight_decay,
        args.label_smoothing,
        args.early_stop_monitor,
    )

    torch.save(
        {
            "adapter_state_dict": adapter.state_dict(),
            "args": vars(args),
            "raw_holdout_metrics": raw_holdout_metrics,
            "raw_all_gallery_metrics": raw_all_gallery_metrics,
            "best_holdout_metrics": best_holdout_metrics,
            "best_all_gallery_metrics": best_all_gallery_metrics,
            "training_summary": training_summary,
        },
        weights_path,
    )
    write_json(
        run_dir / "metrics.json",
        {
            "raw_holdout_metrics": raw_holdout_metrics,
            "raw_all_gallery_metrics": raw_all_gallery_metrics,
            "best_holdout_metrics": best_holdout_metrics,
            "best_all_gallery_metrics": best_all_gallery_metrics,
            "config": vars(args),
            "training_summary": training_summary,
            "history": history,
        },
    )
    if history:
        write_csv(run_dir / "history.csv", history, list(history[0].keys()))

    submission_rows: list[dict[str, str]] = []
    for pool in all_prediction_pools(data_root):
        query_rows = read_csv(pool.query_csv)
        target_rows = read_csv(pool.gallery_csv)
        query_ids, query_vectors = embed_manifest(backbone, query_rows, data_root, "query_id", "query_image", args.embed_batch_size, args.num_workers, device)
        target_ids, target_vectors = embed_manifest(backbone, target_rows, data_root, "target_id", "target_image", args.embed_batch_size, args.num_workers, device)
        adapted_query, adapted_target = adapted_vectors(adapter, query_vectors, target_vectors, device, args.batch_size)
        score_matrix = adapted_query @ adapted_target.T
        submission_rows.extend(ranking_rows_from_scores(query_ids, target_ids, score_matrix.astype(np.float32)))
        save_vectors(
            run_dir / "vectors" / f"{pool.dataset}_{pool.split}_vectors.npz",
            {
                "dataset": np.asarray(pool.dataset),
                "split": np.asarray(pool.split),
                "query_ids": np.asarray(query_ids),
                "target_ids": np.asarray(target_ids),
                "query_vectors": adapted_query.astype(np.float32),
                "target_vectors": adapted_target.astype(np.float32),
                "raw_query_vectors": query_vectors.astype(np.float32),
                "raw_target_vectors": target_vectors.astype(np.float32),
            },
        )
        print(f"predicted {pool.dataset}/{pool.split}: {len(query_ids)}x{len(target_ids)}", flush=True)

    write_csv(run_dir / "submission.csv", submission_rows, ["query_id", "target_id_ranking"])
    print(
        json.dumps(
            {
                "training_script": str(script_path),
                "run_dir": str(run_dir),
                "weights_path": str(weights_path),
                "raw_holdout_metrics": raw_holdout_metrics,
                "raw_all_gallery_metrics": raw_all_gallery_metrics,
                "best_holdout_metrics": best_holdout_metrics,
                "best_all_gallery_metrics": best_all_gallery_metrics,
                "training_summary": training_summary,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
