import argparse
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def read_activation_csv(
    csv_path: Path,
) -> Tuple[List[int], Dict[str, Dict[int, List[Tuple[float, float]]]]]:
    steps_set = set()
    metrics: Dict[str, Dict[int, List[Tuple[float, float]]]] = {
        "embedding": defaultdict(list),
        "attention": defaultdict(list),
        "mlp": defaultdict(list),
        "output": defaultdict(list),
    }
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            step = int(row["step"])
            width = float(row["model_width"])
            steps_set.add(step)
            metrics["embedding"][step].append((width, float(row["embedding"])))
            metrics["attention"][step].append((width, float(row["attention"])))
            metrics["mlp"][step].append((width, float(row["mlp"])))
            metrics["output"][step].append((width, float(row["output"])))
    steps = sorted(steps_set)
    # Sort each list by width to ensure correct curve drawing
    for metric in metrics.values():
        for s in metric:
            metric[s].sort(key=lambda t: t[0])
    return steps, metrics


def plot_mup(
    steps: List[int],
    metrics: Dict[str, Dict[int, List[Tuple[float, float]]]],
    out_path: Path,
    max_steps: int | None = None,
) -> None:
    used_steps = steps if max_steps is None else [s for s in steps if s <= max_steps]
    cmap = plt.get_cmap("coolwarm")
    norm = plt.Normalize(min(used_steps), max(used_steps))

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), constrained_layout=True)
    panels = [
        ("Word Embedding", "embedding", False),
        ("Attention Output", "attention", True),
        ("FFN Output", "mlp", True),
        ("Output Logits", "output", True),
    ]

    for ax, (title, key, log_y) in zip(axes, panels):
        for step in used_steps:
            if step not in metrics[key]:
                continue
            widths, values = zip(*metrics[key][step])
            ax.plot(
                widths,
                values,
                color=cmap(norm(step)),
                marker="o",
                linewidth=2,
                markersize=3,
                alpha=0.9,
                label=f"{step}",
            )
        ax.set_title(title)
        ax.set_xlabel("Width")
        ax.set_xticks(
            sorted({w for step in used_steps for (w, _) in metrics[key].get(step, [])})
        )
        ax.grid(True, which="both", linestyle="--", linewidth=0.5, alpha=0.5)
        if log_y:
            ax.set_yscale("log")
        ax.set_ylabel("mean(|activation|)")

    # Put a shared legend on the first panel
    handles, labels = axes[0].get_legend_handles_labels()
    legend = axes[0].legend(
        handles,
        [f"Step {l}" for l in labels],
        title="Step",
        loc="upper left",
        fontsize=8,
        frameon=False,
    )
    legend.set_title("Step")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved plot to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot MUP activation curves from CSV")
    parser.add_argument(
        "--csv", type=Path, required=True, help="Path to mup_activations.csv"
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Output image path (png)"
    )
    parser.add_argument(
        "--max_steps", type=int, default=None, help="Limit to first N steps"
    )
    args = parser.parse_args()

    out_path = args.out
    if out_path is None:
        out_path = args.csv.parent / "mup_activations_plot.png"

    steps, metrics = read_activation_csv(args.csv)
    if len(steps) == 0:
        raise RuntimeError("No data found in CSV.")
    plot_mup(steps, metrics, out_path, args.max_steps)


if __name__ == "__main__":
    main()
