import argparse
import csv
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


def format_params(num_params: int) -> str:
    """Format number of parameters as human-readable string (e.g., 64M, 1B)."""
    if num_params >= 1e9:
        return f"{num_params / 1e9:.0f}B"
    elif num_params >= 1e6:
        return f"{num_params / 1e6:.0f}M"
    elif num_params >= 1e3:
        return f"{num_params / 1e3:.0f}K"
    return str(num_params)


def format_flops(flops: float) -> str:
    """Format FLOPs budget in scientific notation without trailing .0."""
    return f"{flops:.0e}"


def read_hp_transfer_csv(
    csv_path: Path,
) -> Dict[Tuple[int, int, int | None, float | None], List[Tuple[float, float]]]:
    """
    Read HP transfer results CSV.
    Returns dict mapping (model_dim, num_params) -> list of (lr, val_loss) tuples.
    """
    data: Dict[Tuple[int, int, int | None, float | None], List[Tuple[float, float]]] = (
        defaultdict(list)
    )
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_loss = row.get("val_loss", "")
            try:
                val_loss = float(raw_loss)
            except (TypeError, ValueError):
                continue
            lr = float(row["lr"])
            model_dim = int(row["model_dim"])
            num_params = int(row["num_params"])
            depth = int(row["depth"]) if row.get("depth") else None
            flops_budget = float(row["flops_budget"]) if row.get("flops_budget") else None
            data[(model_dim, num_params, depth, flops_budget)].append((lr, val_loss))
    # Sort each list by learning rate
    for key in data:
        data[key].sort(key=lambda t: t[0])
    return data


def plot_hp_transfer(
    data: Dict[Tuple[int, int, int | None, float | None], List[Tuple[float, float]]],
    out_path: Path,
    title: str = "Adam",
) -> None:
    """
    Plot validation loss vs learning rate for different model sizes.
    Shows optimal LR with red dots for each model.
    """
    fig, ax = plt.subplots(figsize=(6, 5))

    # Sort model configs by size for consistent legend ordering
    sorted_configs = sorted(data.keys(), key=lambda x: x[0])

    # Color palette - discrete, high-contrast colors
    cmap = plt.get_cmap("tab10") if len(sorted_configs) <= 10 else plt.get_cmap("tab20")
    colors = [cmap(i % cmap.N) for i in range(len(sorted_configs))]

    num_params_values = {num_params for (_, num_params, _, _) in sorted_configs}
    flops_values = {flops for (_, _, _, flops) in sorted_configs if flops is not None}
    use_flops_label = len(flops_values) > 0 and len(num_params_values) <= 1

    for (model_dim, num_params, _depth, flops_budget), color in zip(
        sorted_configs, colors
    ):
        points = data[(model_dim, num_params, _depth, flops_budget)]
        if len(points) < 2:
            continue

        lrs, losses = zip(*points)
        lrs = np.array(lrs)
        losses = np.array(losses)

        size_label = (
            format_flops(flops_budget)
            if use_flops_label and flops_budget is not None
            else format_params(num_params)
        )
        label = f"{model_dim} ({size_label})"

        # Always compute the true best LR from raw points
        min_idx = int(np.argmin(losses))
        best_lr = float(lrs[min_idx])
        best_loss = float(losses[min_idx])

        # Plot raw points with a line to preserve true optima
        sort_idx = np.argsort(lrs)
        lrs_sorted = lrs[sort_idx]
        losses_sorted = losses[sort_idx]
        ax.plot(
            lrs_sorted,
            losses_sorted,
            color=color,
            linewidth=2,
            marker="o",
            markersize=4,
            label=label,
        )

        # Mark optimal learning rate with red dot (raw min)
        ax.scatter(
            [best_lr],
            [best_loss],
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
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(lambda x, _: f"$2^{{{int(np.log2(x))}}}$")
    )

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


def read_lr_scaling_csv(
    csv_path: Path,
) -> List[Tuple[float, float]]:
    """
    Read HP sweep results and return (flops_budget, best_lr) points.
    Uses the lowest validation loss per flops budget.
    """
    per_flops: Dict[float, List[Tuple[float, float]]] = defaultdict(list)
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_loss = row.get("val_loss", "")
            try:
                val_loss = float(raw_loss)
            except (TypeError, ValueError):
                continue
            try:
                flops_budget = float(row["flops_budget"])
                lr = float(row["lr"])
            except (TypeError, ValueError, KeyError):
                continue
            per_flops[flops_budget].append((lr, val_loss))

    points: List[Tuple[float, float]] = []
    for flops, entries in per_flops.items():
        if not entries:
            continue
        lrs = np.array([lr for lr, _ in entries], dtype=float)
        losses = np.array([loss for _, loss in entries], dtype=float)
        best_idx = int(np.argmin(losses))
        points.append((float(flops), float(lrs[best_idx])))

    points.sort(key=lambda x: x[0])
    return points


def plot_lr_scaling(
    points: List[Tuple[float, float]],
    out_path: Path,
    title: str = "LR scaling (MoE)",
    label: str = "MoE",
) -> None:
    """
    Plot LR scaling law: best_lr vs flops, with power-law fit.
    """
    if len(points) < 2:
        raise RuntimeError("Need at least two points to fit a scaling law.")

    flops = np.array([p[0] for p in points], dtype=float)
    lrs = np.array([p[1] for p in points], dtype=float)

    log_f = np.log10(flops)
    log_lr = np.log10(lrs)
    slope, intercept = np.polyfit(log_f, log_lr, 1)
    coeff = 10 ** intercept

    # Fit line for plotting
    f_fit = np.logspace(log_f.min(), log_f.max(), 200)
    lr_fit = coeff * (f_fit ** slope)

    # Simple uncertainty band from residuals in log space
    log_lr_fit = intercept + slope * log_f
    resid = log_lr - log_lr_fit
    if len(resid) > 1:
        sigma = np.std(resid, ddof=1)
    else:
        sigma = 0.0
    lr_upper = (10 ** (np.log10(lr_fit) + sigma))
    lr_lower = (10 ** (np.log10(lr_fit) - sigma))

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(flops, lrs, color="dimgray", s=35, zorder=3, label="data")
    ax.plot(
        f_fit,
        lr_fit,
        color="tab:blue",
        linestyle="--",
        linewidth=2.5,
        label=(
            f"{label}: $\\eta = {coeff:.4g} \\cdot C^{{{slope:.4f}}}$"
        ),
    )
    if sigma > 0:
        ax.fill_between(
            f_fit,
            lr_lower,
            lr_upper,
            color="tab:blue",
            alpha=0.2,
            linewidth=0,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Non-Embedding Training FLOPs")
    ax.set_ylabel("Learning Rate")
    ax.set_title(title)
    ax.grid(True, which="both", linestyle="-", linewidth=0.4, alpha=0.5)
    ax.legend(loc="upper right", frameon=False)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    print(f"Saved LR scaling plot to {out_path}")


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

    # LR scaling law plotting subcommand
    lr_parser = subparsers.add_parser(
        "lr_scaling", help="Plot LR scaling law (best LR vs FLOPs)"
    )
    lr_parser.add_argument(
        "--csv", type=Path, required=True, help="Path to hp sweep results.csv"
    )
    lr_parser.add_argument(
        "--out", type=Path, default=None, help="Output image path (png)"
    )
    lr_parser.add_argument(
        "--title", type=str, default="LR scaling (MoE)", help="Plot title"
    )
    lr_parser.add_argument(
        "--label", type=str, default="MoE", help="Legend label"
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

    elif args.command == "lr_scaling":
        out_path = args.out
        if out_path is None:
            out_path = args.csv.parent / f"{args.csv.stem}_lr_scaling.png"
        points = read_lr_scaling_csv(args.csv)
        if len(points) == 0:
            raise RuntimeError("No data found in CSV.")
        plot_lr_scaling(points, out_path, title=args.title, label=args.label)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
    