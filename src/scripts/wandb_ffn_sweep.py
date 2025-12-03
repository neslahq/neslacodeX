import argparse
import os
from pathlib import Path
from typing import Sequence

import wandb


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "src" / "codex" / "train_configs" / "debug_model.toml"
DEFAULT_RUN_SCRIPT = REPO_ROOT / "run_width_sweep.sh"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch a W&B sweep to tune Codex ffn_scale."
    )
    parser.add_argument(
        "--config-file",
        type=Path,
        default=DEFAULT_CONFIG,
        help="Absolute path to the base training TOML config.",
    )
    parser.add_argument(
        "--run-script",
        type=Path,
        default=DEFAULT_RUN_SCRIPT,
        help="Absolute path to the launcher script (defaults to torchtitan/run_train.sh).",
    )
    parser.add_argument(
        "--project",
        type=str,
        default=os.getenv("WANDB_PROJECT", "codex"),
        help="Target W&B project.",
    )
    parser.add_argument(
        "--entity",
        type=str,
        default=os.getenv("WANDB_TEAM", "nesla-lab"),
        help="Optional W&B entity (team).",
    )
    parser.add_argument(
        "--sweep-name",
        type=str,
        default="codex_ffn_scale",
        help="Friendly name for the sweep.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=("grid", "random", "bayes"),
        default="random",
        help="W&B sweep search strategy.",
    )
    parser.add_argument(
        "--metric-name",
        type=str,
        default="loss_metrics/global_avg_loss",
        help="Metric name to optimize (must match trainer logging).",
    )
    parser.add_argument(
        "--metric-goal",
        choices=("minimize", "maximize"),
        default="minimize",
        help="Whether the sweep should minimize or maximize the metric.",
    )
    parser.add_argument(
        "--ffn-scale-values",
        type=float,
        nargs="+",
        help="Discrete ffn_scale values for grid sweeps.",
    )
    parser.add_argument(
        "--ffn-scale-min",
        type=float,
        default=0.0,
        help="Lower bound for ffn_scale (used when sampling).",
    )
    parser.add_argument(
        "--ffn-scale-max",
        type=float,
        default=1.0,
        help="Upper bound for ffn_scale (used when sampling).",
    )
    parser.add_argument(
        "--ffn-scale-distribution",
        type=str,
        choices=("uniform", "log_uniform", "q_uniform"),
        default="uniform",
        help="Distribution to sample ffn_scale from when using min/max.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Optional limit for how many runs the local agent should launch.",
    )
    parser.add_argument(
        "--run-agent",
        action="store_true",
        help="Run wandb.agent locally after creating the sweep.",
    )
    parser.add_argument(
        "--ngpu",
        type=int,
        default=1,
        help="Number of GPUs to allocate per run (sets NGPU for run_train.sh).",
    )
    parser.add_argument(
        "--train-args",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded to run_train.sh (pass after '--').",
    )
    return parser.parse_args()


def _build_parameter_spec(args: argparse.Namespace) -> dict:
    if args.ffn_scale_values:
        return {"values": [float(v) for v in args.ffn_scale_values]}
    if args.ffn_scale_min is not None and args.ffn_scale_max is not None:
        if args.ffn_scale_min >= args.ffn_scale_max:
            raise ValueError("ffn_scale_min must be < ffn_scale_max.")
        return {
            "min": float(args.ffn_scale_min),
            "max": float(args.ffn_scale_max),
            "distribution": args.ffn_scale_distribution,
        }
    raise ValueError(
        "Provide either --ffn-scale-values or both --ffn-scale-min/--ffn-scale-max."
    )


def _build_command(run_script: Path, extra_args: Sequence[str] | None) -> list[str]:
    command = ["${env}", "bash", str(run_script)]
    if extra_args:
        command.extend(extra_args)
    return command


def main() -> None:
    args = parse_args()

    config_path = args.config_file.resolve()
    run_script = args.run_script.resolve()

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not run_script.exists():
        raise FileNotFoundError(f"Launcher script not found: {run_script}")

    os.environ["CONFIG_FILE"] = str(config_path)
    os.environ["NGPU"] = str(args.ngpu)
    if args.project:
        os.environ.setdefault("WANDB_PROJECT", args.project)
    if args.entity:
        os.environ.setdefault("WANDB_TEAM", args.entity)

    parameter_spec = _build_parameter_spec(args)
    command = _build_command(run_script, args.train_args)

    sweep_config = {
        "name": args.sweep_name,
        "method": args.method,
        "metric": {"name": args.metric_name, "goal": args.metric_goal},
        "parameters": {"ffn_scale": parameter_spec},
        "command": command,
    }

    sweep_id = wandb.sweep(sweep=sweep_config, project=args.project)
    print(f"Created sweep {sweep_id} for project {args.project}.")

    if args.run_agent:
        print("Starting local wandb agent...")
        wandb.agent(
            sweep_id=sweep_id,
            count=args.count,
        )
    else:
        print(
            "Launch the agent with:\n"
            f"WANDB_PROJECT={args.project} "
            f"WANDB_TEAM={args.entity or ''} "
            f"CONFIG_FILE={config_path} "
            f"NGPU={args.ngpu} "
            f"wandb agent {args.entity + '/' if args.entity else ''}{args.project}/{sweep_id}"
        )


if __name__ == "__main__":
    main()
