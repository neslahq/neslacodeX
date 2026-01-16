import argparse
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline


def format_params(num_params: int) -> str:
    """Format number of parameters as human-readable string (e.g., 64M, 1B)."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.0f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.0f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.0f}K"
    return str(num_params)


def read_hp_transfer_csv(
    csv_path: Path,
) -> Dict[Tuple[int, int], List[Tuple[float, float]]]:
    """
    Read HP transfer results CSV.
    Returns dict mapping (model_dim, num_params) -> list of (lr, val_loss) tuples.
    """
    data: Dict[Tuple[int, int], List[Tuple[float, float]]] = defaultdict(list)
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            lr = float(row["lr"])
            model_dim = int(row["model_dim"])
            num_params = int(row["num_params"])
            val_loss = float(row["val_loss"])
            data[(model_dim, num_params)].append((lr, val_loss))
    # Sort each list by learning rate
    for key in data:
        data[key].sort(key=lambda t: t[0])
    return data


def plot_hp_transfer(
    data: Dict[Tuple[int, int], List[Tuple[float, float]]],
    out_path: Path,
    title: str = "Adam",
) -> None:
    """
    Plot validation loss vs learning rate for different model sizes.
    Shows optimal LR with red dots for each model.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Sort model configs by size for consistent legend ordering
    sorted_configs = sorted(data.keys(), key=lambda x: x[1])

    # Color palette - cool colors from teal to dark blue
    colors = plt.cm.winter(np.linspace(0, 1, len(sorted_configs)))

    for (model_dim, num_params), color in zip(sorted_configs, colors):
        points = data[(model_dim, num_params)]
        if len(points) < 2:
            continue

        lrs, losses = zip(*points)
        lrs = np.array(lrs)
        losses = np.array(losses)

        print(lrs)
        print(losses)

        # Plot smooth curve using spline interpolation
        if len(lrs) >= 4:
            # Sort by lr for interpolation
            sort_idx = np.argsort(lrs)
            lrs_sorted = lrs[sort_idx]
            losses_sorted = losses[sort_idx]

            # Create smooth curve in log space
            log_lrs = np.log10(lrs_sorted)
            try:
                spline = UnivariateSpline(log_lrs, losses_sorted, s=0.1)
                log_lr_smooth = np.linspace(log_lrs.min(), log_lrs.max(), 100)
                lr_smooth = 10 ** log_lr_smooth
                loss_smooth = spline(log_lr_smooth)
                ax.plot(
                    lr_smooth,
                    loss_smooth,
                    color=color,
                    linewidth=2,
                    label=f"{model_dim} ({format_params(num_params)})",
                )
            except Exception:
                # Fallback to simple line plot
                print(f"Warning: Failed to create smooth curve for {model_dim} ({format_params(num_params)})")
                ax.plot(
                    lrs_sorted,
                    losses_sorted,
                    color=color,
                    linewidth=2,
                    marker="o",
                    markersize=4,
                    label=f"{model_dim} ({format_params(num_params)})",
                )
        else:
            print(f"Warning: Not enough data to create smooth curve for {model_dim} ({format_params(num_params)})")
            ax.plot(
                lrs,
                losses,
                color=color,
                linewidth=2,
                marker="o",
                markersize=4,
                label=f"{model_dim} ({format_params(num_params)})",
            )

        # Mark optimal learning rate with red dot
        min_idx = np.argmin(losses)
        ax.scatter(
            [lrs[min_idx]],
            [losses[min_idx]],
            color="red",
            s=50,
            zorder=5,
            edgecolors="darkred",
            linewidths=0.5,
        )

    ax.set_xscale("log", base=2)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Validation Loss")
    ax.set_title(title)
    ax.legend(title="Width (Model Size)", loc="upper left", fontsize=8, frameon=False)
    ax.grid(True, which="both", linestyle="-", linewidth=0.3, alpha=0.5)

    # Set x-axis ticks as powers of 2
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"$2^{{{int(np.log2(x))}}}$"))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved HP transfer plot to {out_path}")


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
    parser = argparse.ArgumentParser(description="Plot MUP activation curves or HP transfer results")
    subparsers = parser.add_subparsers(dest="command", help="Plot type")

    # Activation plotting subcommand
    activation_parser = subparsers.add_parser("activation", help="Plot activation curves")
    activation_parser.add_argument(
        "--csv", type=Path, required=True, help="Path to mup_activations.csv"
    )
    activation_parser.add_argument(
        "--tag", type=str, default="", help="plot tag"
    )
    activation_parser.add_argument(
        "--out", type=Path, default=None, help="Output image path (png)"
    )
    activation_parser.add_argument(
        "--max_steps", type=int, default=None, help="Limit to first N steps"
    )

    # HP transfer plotting subcommand
    hp_parser = subparsers.add_parser("hp_transfer", help="Plot HP transfer results (LR vs loss)")
    hp_parser.add_argument(
        "--csv", type=Path, required=True, help="Path to hp_transfer results.csv"
    )
    hp_parser.add_argument(
        "--out", type=Path, default=None, help="Output image path (png)"
    )
    hp_parser.add_argument(
        "--title", type=str, default="Adam", help="Plot title (e.g., optimizer name)"
    )

    args = parser.parse_args()

    if args.command == "activation":
        out_path = args.out
        if out_path is None:
            out_path = args.csv.parent / f"{args.csv.stem}.png"
        steps, metrics = read_activation_csv(args.csv)
        if len(steps) == 0:
            raise RuntimeError("No data found in CSV.")
        plot_mup(steps, metrics, out_path, args.max_steps)

    elif args.command == "hp_transfer":
        out_path = args.out
        if out_path is None:
            out_path = args.csv.parent / f"{args.csv.stem}_hp_transfer.png"
        data = read_hp_transfer_csv(args.csv)
        if len(data) == 0:
            raise RuntimeError("No data found in CSV.")
        plot_hp_transfer(data, out_path, title=args.title)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
    