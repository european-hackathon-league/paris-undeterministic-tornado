from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

DEFAULT_CACHE_ROOT = Path("artifacts/.matplotlib")
os.environ.setdefault("MPLCONFIGDIR", str(DEFAULT_CACHE_ROOT / "config"))
os.environ.setdefault("XDG_CACHE_HOME", str(DEFAULT_CACHE_ROOT / "cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOG_PATTERN = re.compile(
    r"^epoch\s+(?P<epoch>\d+)\s+"
    r"loss=(?P<loss>\d+\.\d+)\s+"
    r"holdout_mrr=(?P<holdout_mrr>\d+\.\d+)\s+"
    r"all_gallery_mrr=(?P<all_gallery_mrr>\d+\.\d+)$"
)


def parse_log(log_path: Path) -> tuple[list[int], list[float], list[float], list[float]]:
    epochs: list[int] = []
    losses: list[float] = []
    holdout_mrrs: list[float] = []
    all_gallery_mrrs: list[float] = []

    for line_number, raw_line in enumerate(log_path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        match = LOG_PATTERN.match(line)
        if match is None:
            raise ValueError(f"Unrecognized log format at line {line_number}: {line}")

        epochs.append(int(match.group("epoch")))
        losses.append(float(match.group("loss")))
        holdout_mrrs.append(float(match.group("holdout_mrr")))
        all_gallery_mrrs.append(float(match.group("all_gallery_mrr")))

    if not epochs:
        raise ValueError(f"No training rows found in {log_path}")

    return epochs, losses, holdout_mrrs, all_gallery_mrrs


def best_epoch(epochs: list[int], values: list[float]) -> tuple[int, float]:
    best_index = max(range(len(values)), key=values.__getitem__)
    return epochs[best_index], values[best_index]


def plot_metrics(
    epochs: list[int],
    losses: list[float],
    holdout_mrrs: list[float],
    all_gallery_mrrs: list[float],
    output_path: Path,
) -> None:
    figure, (loss_axis, mrr_axis) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    loss_axis.plot(epochs, losses, color="#C05A00", linewidth=2)
    loss_axis.set_title("Training Metrics from lol.log")
    loss_axis.set_ylabel("Loss")
    loss_axis.grid(alpha=0.3)

    mrr_axis.plot(epochs, holdout_mrrs, label="Holdout MRR", color="#006D77", linewidth=2)
    mrr_axis.plot(
        epochs,
        all_gallery_mrrs,
        label="All Gallery MRR",
        color="#3A86FF",
        linewidth=2,
    )
    mrr_axis.set_xlabel("Epoch")
    mrr_axis.set_ylabel("MRR")
    mrr_axis.grid(alpha=0.3)
    mrr_axis.legend()

    holdout_epoch, holdout_value = best_epoch(epochs, holdout_mrrs)
    gallery_epoch, gallery_value = best_epoch(epochs, all_gallery_mrrs)
    mrr_axis.scatter([holdout_epoch, gallery_epoch], [holdout_value, gallery_value], color="#D90429")
    mrr_axis.annotate(
        f"Best holdout: {holdout_value:.4f} @ {holdout_epoch}",
        xy=(holdout_epoch, holdout_value),
        xytext=(10, 12),
        textcoords="offset points",
    )
    mrr_axis.annotate(
        f"Best gallery: {gallery_value:.4f} @ {gallery_epoch}",
        xy=(gallery_epoch, gallery_value),
        xytext=(10, -18),
        textcoords="offset points",
    )

    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plot training metrics from lol.log")
    parser.add_argument(
        "log_path",
        nargs="?",
        default="lol.log",
        help="Path to the training log file",
    )
    parser.add_argument(
        "-o",
        "--output",
        default="artifacts/lol_training_metrics.png",
        help="Where to write the output PNG",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    log_path = Path(args.log_path)
    output_path = Path(args.output)

    epochs, losses, holdout_mrrs, all_gallery_mrrs = parse_log(log_path)
    plot_metrics(epochs, losses, holdout_mrrs, all_gallery_mrrs, output_path)

    holdout_epoch, holdout_value = best_epoch(epochs, holdout_mrrs)
    gallery_epoch, gallery_value = best_epoch(epochs, all_gallery_mrrs)
    print(f"Saved plot to {output_path}")
    print(f"Best holdout MRR: {holdout_value:.4f} at epoch {holdout_epoch}")
    print(f"Best all-gallery MRR: {gallery_value:.4f} at epoch {gallery_epoch}")


if __name__ == "__main__":
    main()
