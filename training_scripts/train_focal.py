"""Run the FOCAL-UMS protocol reported in the paper.

Examples:
    python training_scripts/train_focal.py --orders eti --seeds 0
    python training_scripts/train_focal.py --orders eti cdm trc --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# The three ten-domain orders from the paper (Section 4.1).
TASK_ORDERS = {
    "eti": (
        "seq-eurosat-rgb,seq-tinyimg,seq-imagenet-r,seq-cub200,"
        "seq-cifar100,seq-mnist,seq-resisc45,seq-chestx,"
        "seq-cifar10,seq-cropdisease"
    ),
    "cdm": (
        "seq-cifar10,seq-cropdisease,seq-mnist,seq-resisc45,"
        "seq-eurosat-rgb,seq-tinyimg,seq-cifar100,seq-chestx,"
        "seq-imagenet-r,seq-cub200"
    ),
    "trc": (
        "seq-tinyimg,seq-resisc45,seq-cub200,seq-chestx,"
        "seq-imagenet-r,seq-eurosat-rgb,seq-mnist,seq-cifar10,"
        "seq-cropdisease,seq-cifar100"
    ),
}

# Paper configuration (Section 4.1, Implementation details).
BASE_PARAMS = {
    "ncls_per_task": 0,
    # The inherited Mammoth/LEAR registry exposes FOCAL-UMS as model "LEAR".
    "model": "LEAR",
    "backbone": "lear",
    "optimizer": "adam",
    "lr": 1e-3,
    "lr_scheduler": "cosine",
    "batch_size": 64,
    "n_epochs": 10,
    "num_workers": 16,
    "transform_type": "weak",
    "n_frq": 5898,
    "n_tasks": 10,
    "modulate_q": 1,
    "modulate_proj": 1,
    "ce_thresh": 1e-4,
    "ums_basis_rank": 768,
    "ums_svd_rank": 75,
    "ums_tau": 1e-3,
    "ums_rot_steps": 15,
    "ums_rot_lr": 1e-4,
    "ums_head_lr": 1e-3,
    "ums_rot_lambda_anchor": 2.0,
    "ums_rot_lambda_support_mse": 1.0,
    "ums_rot_lambda_support_bce": 0.1,
    "ums_rot_lambda_gaussian": 0.01,
    "ums_rot_lambda_diag": 0.1,
    "apply_to": "6,7,8,9,10,11",
    "ums_in_loop": 1,
    "ums_refresh_every": 2,
    "task_retrieval": "mahalanobis",
}


def build_command(order: str, seed: int) -> list[str]:
    datasets = TASK_ORDERS[order]
    params = {
        "seed": seed,
        "list_datasets": datasets,
        "dataset": datasets.split(",")[0],
        **BASE_PARAMS,
    }

    command = [sys.executable, "-u", "main_domain.py"]
    for key, value in params.items():
        command.extend((f"--{key}", str(value)))
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FOCAL-UMS on the paper's MDIL task orders."
    )
    parser.add_argument(
        "--orders",
        nargs="+",
        choices=TASK_ORDERS,
        default=["eti"],
        help="Task orders to run (default: eti).",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[0],
        help="Random seeds to run (default: 0). The paper uses 0 1 2.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without starting training.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for order in args.orders:
        for seed in args.seeds:
            command = build_command(order, seed)
            print(f"[FOCAL-UMS] order={order.upper()} seed={seed}")
            print(subprocess.list2cmdline(command))
            if not args.dry_run:
                subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
