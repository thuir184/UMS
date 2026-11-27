"""Re-fit and evaluate only the quantum task router from a finished checkpoint.

This is a diagnostic shortcut: the frozen ViT feature extractor and all learned
FOCAL/UMS experts are restored from the final checkpoint.  Each task's training
set is streamed once to rebuild quantum sufficient statistics, then the existing
experts are evaluated with the recalibrated router.  No FOCAL/UMS parameter is
trained or modified.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from argparse import Namespace
from typing import Any, Dict

import torch
from tqdm import tqdm


# Running ``python training_scripts/recalibrate_quantum_router.py`` sets
# sys.path[0] to training_scripts rather than the repository root.  Add the
# root explicitly so project packages such as backbone, datasets and models
# can be imported regardless of the caller's PYTHONPATH.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def _checkpoint_file(path: str) -> str:
    if os.path.isdir(path):
        path = os.path.join(path, "model", "model_checkpoint.pt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return path


def _torch_load(path: str, map_location: str = "cpu") -> Dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:  # PyTorch < 2.0
        return torch.load(path, map_location=map_location)


def _build_model_and_datasets(checkpoint_file: str, balance_tasks: bool):
    from backbone import get_backbone
    from datasets import get_all_datasets
    from main_domain import extend_args, lecun_fix
    from models import get_model
    from utils.conf import base_path, get_device, set_random_seed
    from utils.task_checkpointing import load_task_checkpoint

    saved = _torch_load(checkpoint_file)
    saved_args = saved.get("args", {})
    if not saved_args:
        raise RuntimeError("Checkpoint does not contain the original argument dictionary")

    args = Namespace(**saved_args)
    args.task_retrieval = "quantum_rls"
    args.quantum_balance_tasks = bool(balance_tasks)
    args.disable_task_checkpoints = True
    args.inference_only = True
    args.nowand = True
    args.non_verbose = False
    args.transform_type = "weak"

    lecun_fix()
    set_random_seed(getattr(args, "seed", 0))
    device = get_device()
    args.device = device
    base_path(getattr(args, "base_path", None))

    datasets = get_all_datasets(args)
    if not datasets:
        raise RuntimeError("No datasets were reconstructed from checkpoint args.list_datasets")
    extend_args(args, datasets[0])

    backbone = get_backbone(args)
    loss = datasets[0].get_loss()
    model = get_model(args, backbone, loss, datasets[0].get_transform(), dataset=datasets[0])
    model.to(device)
    model.net.to(device)
    load_task_checkpoint(checkpoint_file, model=model, optimizer=None, device=device)
    model.task_retrieval = "quantum_rls"
    model.net.eval()
    return model, datasets, device


@torch.no_grad()
def recalibrate(model, datasets, device, max_batches: int = 0):
    router = model.net.quantum_task_router
    router.reset_statistics()
    test_loaders = []

    for task_id, dataset in enumerate(datasets):
        train_loader, test_loader = dataset.get_all_data_loaders()
        test_loaders.append(test_loader)
        limit = len(train_loader) if max_batches <= 0 else min(len(train_loader), max_batches)
        progress = tqdm(
            enumerate(train_loader),
            total=limit,
            desc=f"Recalibrate quantum router task {task_id + 1}",
            ncols=150,
        )
        for batch_id, data in progress:
            if batch_id >= limit:
                break
            inputs = data[0].to(device)
            processed = model.net.vitProcess(inputs)
            if processed.size(1) == 1:
                processed = processed.expand(-1, 3, -1, -1)
            frozen_features = model.net.extract_feature(processed).detach()
            router.update_batch(frozen_features, task_id)
        progress.close()
        metrics = router.finalize_task(task_id)
        print(
            f"[Quantum-RLS recalibration] task={task_id + 1}, "
            f"samples={int(metrics['samples'])}, cond(A)={metrics['gram_condition']:.3e}"
        )

    return test_loaders


def evaluate_router_mode(model, datasets, test_loaders, balance_tasks: bool) -> Dict[str, Any]:
    """Solve one weighting mode from the same stored statistics and evaluate it."""
    from utils.training_domain import evaluate

    router = model.net.quantum_task_router
    router.balance_tasks = bool(balance_tasks)
    router.finalize_task(len(datasets) - 1)
    accuracies, _ = evaluate(model, datasets, test_loaders)
    choices = list(getattr(model, "last_task_retrieval_choices", []))
    retrieval_correct = sum(
        int(choice == task_id + 1) for task_id, choice in enumerate(choices)
    )
    retrieval_accuracy = (
        100.0 * retrieval_correct / len(choices) if choices else 0.0
    )
    return {
        "balance_tasks": bool(balance_tasks),
        "router": router.get_analysis_metrics(),
        "expert_choices": choices,
        "task_retrieval_accuracy": retrieval_accuracy,
        "class_il_accuracies": [float(value) for value in accuracies],
        "mean_class_il": float(statistics.mean(accuracies)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-fit only the quantum RLS router from a completed FOCAL-UMS checkpoint"
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to task_N or task_N/model/model_checkpoint.pt",
    )
    parser.add_argument(
        "--balance-tasks",
        type=int,
        choices=[0, 1],
        default=1,
        help="1: equal total weight per task; 0: ordinary sample-weighted ridge",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Debug only: cap train batches per task; 0 uses every batch",
    )
    parser.add_argument(
        "--output-prefix",
        default="training_results/quantum_balanced_recalibration",
        help="Output path without extension (.json and .pt are added)",
    )
    parser.add_argument(
        "--skip-evaluation",
        action="store_true",
        help="Build the router but skip test-set evaluation (useful with --max-batches for a smoke test)",
    )
    parser.add_argument(
        "--compare-unbalanced",
        action="store_true",
        help="Evaluate ordinary and task-balanced ridge from the same collected statistics",
    )
    cli = parser.parse_args()

    checkpoint_file = _checkpoint_file(cli.checkpoint)
    model, datasets, device = _build_model_and_datasets(
        checkpoint_file, balance_tasks=bool(cli.balance_tasks)
    )
    test_loaders = recalibrate(model, datasets, device, max_batches=cli.max_batches)

    if cli.skip_evaluation:
        comparisons = {}
        primary = {
            "balance_tasks": bool(cli.balance_tasks),
            "router": model.net.quantum_task_router.get_analysis_metrics(),
            "expert_choices": [],
            "task_retrieval_accuracy": None,
            "class_il_accuracies": [],
            "mean_class_il": None,
        }
    else:
        modes = [False, True] if cli.compare_unbalanced else [bool(cli.balance_tasks)]
        comparisons = {}
        for mode in modes:
            label = "balanced" if mode else "unbalanced"
            print(f"\n[Evaluate quantum router: {label}]")
            comparisons[label] = evaluate_router_mode(
                model, datasets, test_loaders, balance_tasks=mode
            )
        primary_label = "balanced" if cli.balance_tasks else "unbalanced"
        primary = comparisons[primary_label]
        # Leave the serialized router in the requested primary mode.
        model.net.quantum_task_router.balance_tasks = bool(cli.balance_tasks)
        model.net.quantum_task_router.finalize_task(len(datasets) - 1)
    result = {
        "source_checkpoint": checkpoint_file,
        "balance_tasks": bool(cli.balance_tasks),
        "debug_max_batches": int(cli.max_batches),
        "router": model.net.quantum_task_router.get_analysis_metrics(),
        "expert_choices": primary["expert_choices"],
        "task_retrieval_accuracy": primary["task_retrieval_accuracy"],
        "class_il_accuracies": primary["class_il_accuracies"],
        "mean_class_il": primary["mean_class_il"],
        "comparisons": comparisons,
    }

    output_dir = os.path.dirname(cli.output_prefix)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(cli.output_prefix + ".json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    torch.save(
        {
            "router_state_dict": model.net.quantum_task_router.state_dict(),
            "result": result,
        },
        cli.output_prefix + ".pt",
    )

    print("\n[Quantum-RLS recalibration result]")
    if cli.skip_evaluation:
        print("evaluation: skipped")
    else:
        for label, comparison in comparisons.items():
            print(f"{label} expert choices:", comparison["expert_choices"])
            print(
                f"{label} task retrieval accuracy: "
                f"{comparison['task_retrieval_accuracy']:.2f}%"
            )
            print(
                f"{label} Class-IL per task:",
                [round(value, 2) for value in comparison["class_il_accuracies"]],
            )
            print(f"{label} Mean Class-IL: {comparison['mean_class_il']:.2f}%")
    print("Saved:", cli.output_prefix + ".json")
    print("Saved:", cli.output_prefix + ".pt")


if __name__ == "__main__":
    main()
