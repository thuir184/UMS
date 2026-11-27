"""Run the UMS leave-one-loss-out study (B0--B5) reproducibly.

The runner keeps the training configuration fixed and changes exactly one UMS
objective weight per variant. Runs are executed sequentially so one GPU can be
used safely. Every run receives a separate log, checkpoint root, and manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


DATASET_ORDERS = [
    "seq-cifar10,seq-cropdisease,seq-mnist,seq-resisc45,seq-eurosat-rgb,seq-tinyimg,seq-cifar100,seq-chestx,seq-imagenet-r,seq-cub200",
    "seq-eurosat-rgb,seq-tinyimg,seq-imagenet-r,seq-cub200,seq-cifar100,seq-mnist,seq-resisc45,seq-chestx,seq-cifar10,seq-cropdisease",
    "seq-tinyimg,seq-resisc45,seq-cub200,seq-chestx,seq-imagenet-r,seq-eurosat-rgb,seq-mnist,seq-cifar10,seq-cropdisease,seq-cifar100",
]
ORDER_NAMES = ["CDM", "ETI", "TRC"]

BASE_WEIGHTS = {
    "ums_rot_lambda_support_mse": 1.0,   # L_val
    # B0 uses equal weights by design. B1--B5 then remove exactly one
    # objective while retaining the same scale for every remaining term.
    "ums_rot_lambda_support_bce": 1.0,   # L_mask
    "ums_rot_lambda_gaussian": 1.0,      # L_mom
    "ums_rot_lambda_diag": 1.0,          # L_off
    "ums_rot_lambda_anchor": 1.0,        # L_drift
}

VARIANTS = {
    "B0": {"name": "full", "zero": None},
    "B1": {"name": "without_value", "zero": "ums_rot_lambda_support_mse"},
    "B2": {"name": "without_mask", "zero": "ums_rot_lambda_support_bce"},
    "B3": {"name": "without_moment", "zero": "ums_rot_lambda_gaussian"},
    "B4": {"name": "without_off_diagonal", "zero": "ums_rot_lambda_diag"},
    "B5": {"name": "without_drift", "zero": "ums_rot_lambda_anchor"},
}

BASE_CONFIG = {
    "ncls_per_task": 0,
    "model": "LEAR",
    "backbone": "lear",
    "optimizer": "adam",
    "lr": 1e-3,
    "lr_scheduler": "cosine",
    "batch_size": 64,
    "n_epochs": 10,
    # Avoid serializing the full UMS/model state after every task.  The final
    # evaluated task is checkpointed once, which lowers CPU-memory/I/O peaks.
    "checkpoint_final_only": True,
    # Match the main FOCAL training configuration. The TinyImageNet loader
    # memory fix is handled in datasets/seq_tinyimagenet.py.
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
    "apply_to": "6,7,8,9,10,11",
    "ums_in_loop": 1,
    "ums_refresh_every": 2,
    "task_retrieval": "mahalanobis",
}


def parse_csv_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def parse_variants(value: str) -> list[str]:
    requested = [item.strip().upper() for item in value.split(",") if item.strip()]
    unknown = [item for item in requested if item not in VARIANTS]
    if unknown:
        raise argparse.ArgumentTypeError(f"Unknown variants: {unknown}; choose from {list(VARIANTS)}")
    return requested


def file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_command(config: dict[str, object]) -> list[str]:
    command = [sys.executable, "-u", "main_domain.py"]
    for key, value in config.items():
        command.extend([f"--{key}", str(value)])
    return command


def run_one(command: list[str], log_path: Path) -> int:
    with log_path.open("w", encoding="utf-8", buffering=1) as log_file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description="UMS B0--B5 leave-one-loss-out runner")
    parser.add_argument("--variants", type=parse_variants, default=parse_variants("B0,B1,B2,B3,B4,B5"))
    parser.add_argument("--order-ids", type=parse_csv_ints, default=parse_csv_ints("1"))
    parser.add_argument("--seeds", type=parse_csv_ints, default=parse_csv_ints("0"))
    parser.add_argument("--smoke", action="store_true", help="Use 1 epoch and stop after 2 tasks")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write manifests without training")
    parser.add_argument("--rerun-completed", action="store_true")
    parser.add_argument("--output-root", default="training_results/ums_loss_ablation_v2")
    parser.add_argument("--checkpoint-root", default="checkpoints/ums_loss_ablation_v2")
    args = parser.parse_args()

    for order_id in args.order_ids:
        if order_id < 0 or order_id >= len(DATASET_ORDERS):
            parser.error(f"order id must be 0, 1, or 2; got {order_id}")

    root = Path.cwd()
    output_root = root / args.output_root
    checkpoint_root = root / args.checkpoint_root
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoint_root.mkdir(parents=True, exist_ok=True)

    source_hashes = {
        str(path): file_sha256(root / path)
        for path in (
            Path("backbone/bilora.py"),
            Path("models/LEAR.py"),
            Path("utils/training_domain.py"),
            Path("utils/task_checkpointing.py"),
            Path("training_scripts/train_ums_loss_ablation.py"),
        )
    }

    for order_id in args.order_ids:
        dataset_order = DATASET_ORDERS[order_id]
        order_name = ORDER_NAMES[order_id]
        for seed in args.seeds:
            for variant_id in args.variants:
                variant = VARIANTS[variant_id]
                run_name = f"{variant_id}_{variant['name']}_{order_name}_seed{seed}"
                if args.smoke:
                    run_name += "_smoke"
                run_dir = output_root / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                completed_path = run_dir / "completed.json"
                if completed_path.exists() and not args.rerun_completed:
                    print(f"[SKIP] {run_name}: completion marker exists")
                    continue

                weights = deepcopy(BASE_WEIGHTS)
                if variant["zero"] is not None:
                    weights[variant["zero"]] = 0.0

                config = deepcopy(BASE_CONFIG)
                config.update(weights)
                config.update({
                    "seed": seed,
                    "list_datasets": dataset_order,
                    "dataset": dataset_order.split(",")[0],
                    "checkpoint_dir": str(checkpoint_root / run_name),
                    "wandb_name": run_name,
                })
                if args.smoke:
                    config.update({"n_epochs": 1, "stop_after": 2, "num_workers": 4})

                assert config["task_retrieval"] == "mahalanobis"
                assert all(not key.startswith("quantum_") for key in config)
                command = build_command(config)
                command_text = shlex.join(command)
                manifest = {
                    "experiment_version": "ums_loss_ablation_v2_snapshot_fix",
                    "variant_id": variant_id,
                    "variant_name": variant["name"],
                    "removed_objective": variant["zero"],
                    "mask_gate_policy": "retained for the B2 loss-only ablation",
                    "order_id": order_id,
                    "order_name": order_name,
                    "seed": seed,
                    "smoke": bool(args.smoke),
                    "evaluation_protocol": "grouped_domain_existing",
                    "started_at_utc": datetime.now(timezone.utc).isoformat(),
                    "config": config,
                    "command": command_text,
                    "source_sha256": source_hashes,
                }
                (run_dir / "manifest.json").write_text(
                    json.dumps(manifest, indent=2), encoding="utf-8"
                )

                print(f"\n{'=' * 90}\n[START] {run_name}\n{command_text}\n{'=' * 90}")
                if args.dry_run:
                    print(f"[DRY RUN] {run_name}: command validated; training was not started")
                    continue
                return_code = run_one(command, run_dir / "train.log")
                if return_code != 0:
                    failure = {"return_code": return_code, "failed_at_utc": datetime.now(timezone.utc).isoformat()}
                    (run_dir / "failed.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
                    raise SystemExit(f"{run_name} failed with exit code {return_code}")

                completion = {
                    "return_code": 0,
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                completed_path.write_text(json.dumps(completion, indent=2), encoding="utf-8")
                print(f"[DONE] {run_name}")


if __name__ == "__main__":
    main()
