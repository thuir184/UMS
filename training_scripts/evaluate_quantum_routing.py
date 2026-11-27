"""Compare Quantum-RLS routing rules on a completed checkpoint.

The script is evaluation-only: it restores the model and router exactly as
saved, streams each test set to collect per-sample router scores, and compares
three expert-selection rules:

* ``mean_score``: the existing dataset-wide mean-score rule;
* ``majority_vote``: one vote from each sample's top-scoring expert;
* ``oracle``: the known task expert, used only as an upper-bound diagnostic.

No model, FOCAL/UMS, or router parameter is updated.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from typing import Any, Dict, Iterable, List, Sequence

import torch
from tqdm import tqdm


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from training_scripts.recalibrate_quantum_router import (  # noqa: E402
    _build_model_and_datasets,
    _checkpoint_file,
    _torch_load,
)


def aggregate_score_batches(
    score_batches: Iterable[torch.Tensor], n_tasks: int
) -> Dict[str, Any]:
    """Aggregate per-sample task scores for mean and majority routing."""
    score_sum = torch.zeros(n_tasks, dtype=torch.float64)
    vote_counts = torch.zeros(n_tasks, dtype=torch.long)
    sample_margin_sum = 0.0
    samples = 0

    for scores in score_batches:
        scores = torch.as_tensor(scores).detach().to(device="cpu", dtype=torch.float64)
        if scores.ndim != 2 or scores.shape[1] != n_tasks:
            raise ValueError(
                f"Expected score batch [B, {n_tasks}], got {tuple(scores.shape)}"
            )
        if scores.shape[0] == 0:
            continue
        score_sum += scores.sum(dim=0)
        winners = scores.argmax(dim=1)
        vote_counts += torch.bincount(winners, minlength=n_tasks)
        if n_tasks > 1:
            top_two = scores.topk(k=2, dim=1).values
            sample_margin_sum += float((top_two[:, 0] - top_two[:, 1]).sum().item())
        samples += int(scores.shape[0])

    if samples == 0:
        raise RuntimeError("Cannot aggregate routing scores from an empty loader")

    mean_scores = score_sum / float(samples)
    vote_shares = vote_counts.to(dtype=torch.float64) / float(samples)
    mean_choice = int(mean_scores.argmax().item()) + 1
    majority_choice = int(vote_counts.argmax().item()) + 1

    if n_tasks > 1:
        mean_top_two = mean_scores.topk(k=2).values
        vote_top_two = vote_shares.topk(k=2).values
        mean_margin = float((mean_top_two[0] - mean_top_two[1]).item())
        vote_margin = float((vote_top_two[0] - vote_top_two[1]).item())
    else:
        mean_margin = math.inf
        vote_margin = 1.0

    return {
        "samples": samples,
        "mean_scores": mean_scores.tolist(),
        "mean_choice": mean_choice,
        "mean_margin": mean_margin,
        "vote_counts": vote_counts.tolist(),
        "vote_shares": vote_shares.tolist(),
        "majority_choice": majority_choice,
        "majority_share": float(vote_shares[majority_choice - 1].item()),
        "vote_margin": vote_margin,
        "mean_sample_margin": sample_margin_sum / float(samples),
    }


def _dataset_names(model, count: int) -> List[str]:
    raw_names = getattr(model.args, "list_datasets", "")
    if isinstance(raw_names, str):
        names = [name.strip() for name in raw_names.split(",") if name.strip()]
    else:
        names = [str(name) for name in raw_names]
    if len(names) != count:
        names = [f"task_{task_id + 1}" for task_id in range(count)]
    return names


@torch.no_grad()
def collect_routing_diagnostics(
    model,
    test_loaders: Sequence,
    dataset_names: Sequence[str],
    max_batches: int = 0,
) -> List[Dict[str, Any]]:
    """Collect raw Quantum-RLS routing diagnostics for every test set."""
    router = model.net.quantum_task_router
    n_tasks = len(test_loaders)
    seen = int(router.tasks_seen.item())
    if seen != n_tasks:
        raise RuntimeError(
            f"Checkpoint router has seen {seen} tasks, but {n_tasks} test loaders were built"
        )

    diagnostics: List[Dict[str, Any]] = []
    for task_id, test_loader in enumerate(test_loaders):
        limit = len(test_loader) if max_batches <= 0 else min(len(test_loader), max_batches)

        def score_batches():
            progress = tqdm(
                enumerate(test_loader),
                total=limit,
                desc=f"Route task {task_id + 1}",
                ncols=150,
            )
            try:
                for batch_id, data in progress:
                    if batch_id >= limit:
                        break
                    inputs = data[0].to(model.device)
                    processed = model.net.vitProcess(inputs)
                    if processed.size(1) == 1:
                        processed = processed.expand(-1, 3, -1, -1)
                    frozen_features = model.net.extract_feature(processed).detach()
                    yield router.scores(frozen_features)
            finally:
                progress.close()

        result = aggregate_score_batches(score_batches(), n_tasks=n_tasks)
        result.update(
            {
                "task_id": task_id + 1,
                "dataset": dataset_names[task_id],
                "oracle_choice": task_id + 1,
            }
        )
        diagnostics.append(result)
        print(
            f"[Routing] task={task_id + 1} dataset={dataset_names[task_id]} "
            f"mean={result['mean_choice']} (margin={result['mean_margin']:.6f}) "
            f"vote={result['majority_choice']} "
            f"(share={100.0 * result['majority_share']:.2f}%, "
            f"margin={100.0 * result['vote_margin']:.2f}%) "
            f"oracle={task_id + 1}"
        )
    return diagnostics


@torch.no_grad()
def evaluate_choice_rules(
    model,
    datasets: Sequence,
    test_loaders: Sequence,
    choices_by_rule: Dict[str, Sequence[int]],
    max_batches: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Evaluate all routing rules while sharing inference for equal choices."""
    n_tasks = len(test_loaders)
    if len(datasets) != n_tasks:
        raise ValueError("datasets and test_loaders must have equal lengths")
    for label, choices in choices_by_rule.items():
        if len(choices) != n_tasks:
            raise ValueError(f"Routing rule {label!r} has {len(choices)} choices, expected {n_tasks}")

    correct_by_rule = {
        label: [0 for _ in range(n_tasks)] for label in choices_by_rule
    }
    total_by_task = [0 for _ in range(n_tasks)]
    was_training = model.net.training
    model.net.eval()

    try:
        for task_id, (dataset, test_loader) in enumerate(zip(datasets, test_loaders)):
            choices_for_task = {
                label: int(choices[task_id]) for label, choices in choices_by_rule.items()
            }
            unique_choices = sorted(set(choices_for_task.values()))
            for choice in unique_choices:
                if choice < 1 or choice > len(model.net.classifierArr):
                    raise IndexError(
                        f"Expert choice {choice} is invalid for "
                        f"{len(model.net.classifierArr)} experts"
                    )

            limit = len(test_loader) if max_batches <= 0 else min(len(test_loader), max_batches)
            n_classes = dataset.N_CLASSES_PER_TASK * dataset.N_TASKS
            progress = tqdm(
                enumerate(test_loader),
                total=limit,
                desc=f"Evaluate routing rules task {task_id + 1}",
                ncols=150,
            )
            for batch_id, data in progress:
                if batch_id >= limit:
                    break
                inputs = data[0].to(model.device)
                labels = data[1].to(model.device)
                correct_for_choice = {}
                for choice in unique_choices:
                    outputs = model.myPrediction(inputs, choice - 1)
                    predictions = outputs[:, :n_classes].argmax(dim=1)
                    correct_for_choice[choice] = int((predictions == labels).sum().item())
                for label, choice in choices_for_task.items():
                    correct_by_rule[label][task_id] += correct_for_choice[choice]
                total_by_task[task_id] += int(labels.shape[0])
            progress.close()
            if total_by_task[task_id] == 0:
                raise RuntimeError(f"Empty evaluation loader for task {task_id + 1}")

            for label, choice in choices_for_task.items():
                accuracy = (
                    100.0 * correct_by_rule[label][task_id] / total_by_task[task_id]
                )
                print(
                    f"[{label}] task={task_id + 1}, expert={choice}, "
                    f"accuracy={accuracy:.2f}%"
                )
    finally:
        model.net.train(was_training)

    comparisons: Dict[str, Dict[str, Any]] = {}
    for label, choices in choices_by_rule.items():
        accuracies = [
            100.0 * correct / total
            for correct, total in zip(correct_by_rule[label], total_by_task)
        ]
        retrieval_correct = sum(
            int(choice == task_id + 1) for task_id, choice in enumerate(choices)
        )
        comparisons[label] = {
            "expert_choices": [int(choice) for choice in choices],
            "task_retrieval_accuracy": 100.0 * retrieval_correct / len(choices),
            "class_il_accuracies": accuracies,
            "mean_class_il": float(statistics.mean(accuracies)),
            "debug_max_batches": int(max_batches),
        }
    return comparisons


def _print_summary(
    dataset_names: Sequence[str],
    diagnostics: Sequence[Dict[str, Any]],
    comparisons: Dict[str, Dict[str, Any]],
) -> None:
    print("\n[Quantum-RLS routing comparison]")
    for task_id, name in enumerate(dataset_names):
        pieces = [f"{task_id + 1:2}. {name}"]
        for label in ("mean_score", "majority_vote", "oracle"):
            result = comparisons[label]
            pieces.append(
                f"{label}={result['class_il_accuracies'][task_id]:.2f}%"
                f"(e{result['expert_choices'][task_id]})"
            )
        diagnostic = diagnostics[task_id]
        pieces.append(f"vote_share={100.0 * diagnostic['majority_share']:.2f}%")
        print(" | ".join(pieces))

    for label in ("mean_score", "majority_vote", "oracle"):
        result = comparisons[label]
        print(
            f"{label}: choices={result['expert_choices']}, "
            f"retrieval={result['task_retrieval_accuracy']:.2f}%, "
            f"Mean Class-IL={result['mean_class_il']:.2f}%"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare mean-score, majority-vote, and oracle Quantum-RLS routing"
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
        default=None,
        help="Override checkpoint task balancing; omitted preserves its saved setting",
    )
    parser.add_argument(
        "--max-routing-batches",
        type=int,
        default=0,
        help="Debug only: cap routing batches per task; 0 uses all batches",
    )
    parser.add_argument(
        "--max-eval-batches",
        type=int,
        default=0,
        help="Debug only: cap accuracy batches per task; 0 uses all batches",
    )
    parser.add_argument(
        "--routing-only",
        action="store_true",
        help="Collect routing diagnostics without running classifier evaluation",
    )
    parser.add_argument(
        "--output",
        default="training_results/quantum_routing_comparison.json",
        help="JSON output path",
    )
    cli = parser.parse_args()

    checkpoint_file = _checkpoint_file(cli.checkpoint)
    saved = _torch_load(checkpoint_file)
    saved_args = saved.get("args", {})
    saved_balance = bool(saved_args.get("quantum_balance_tasks", True))
    balance_tasks = saved_balance if cli.balance_tasks is None else bool(cli.balance_tasks)

    model, datasets, _ = _build_model_and_datasets(
        checkpoint_file, balance_tasks=balance_tasks
    )
    if getattr(model, "task_retrieval", "") != "quantum_rls":
        raise RuntimeError("The restored model is not configured for Quantum-RLS routing")

    test_loaders = []
    for dataset in datasets:
        _, test_loader = dataset.get_all_data_loaders()
        test_loaders.append(test_loader)
    dataset_names = _dataset_names(model, len(datasets))

    diagnostics = collect_routing_diagnostics(
        model,
        test_loaders,
        dataset_names,
        max_batches=cli.max_routing_batches,
    )
    choices_by_rule = {
        "mean_score": [item["mean_choice"] for item in diagnostics],
        "majority_vote": [item["majority_choice"] for item in diagnostics],
        "oracle": [item["oracle_choice"] for item in diagnostics],
    }

    comparisons: Dict[str, Dict[str, Any]] = {}
    if not cli.routing_only:
        comparisons = evaluate_choice_rules(
            model,
            datasets,
            test_loaders,
            choices_by_rule,
            max_batches=cli.max_eval_batches,
        )

    result = {
        "source_checkpoint": checkpoint_file,
        "balance_tasks": balance_tasks,
        "dataset_names": dataset_names,
        "routing_diagnostics": diagnostics,
        "choices_by_rule": choices_by_rule,
        "comparisons": comparisons,
        "debug_max_routing_batches": int(cli.max_routing_batches),
        "debug_max_eval_batches": int(cli.max_eval_batches),
    }
    output_dir = os.path.dirname(cli.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(cli.output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    if comparisons:
        _print_summary(dataset_names, diagnostics, comparisons)
    else:
        print("\n[Routing-only choices]")
        for label, choices in choices_by_rule.items():
            correct = sum(choice == i + 1 for i, choice in enumerate(choices))
            print(
                f"{label}: {choices} "
                f"({correct}/{len(choices)} = {100.0 * correct / len(choices):.2f}%)"
            )
    print("Saved:", cli.output)


if __name__ == "__main__":
    main()
