#!/usr/bin/env python3
"""Standalone Task Retrieval Evaluation Script for FOCAL-UMS.

Evaluates task retrieval mechanisms on saved checkpoints without re-running training:
  1) UMS Bilinear Coordinate Resonance (Option 1)
  2) Gaussian Mahalanobis Distance (Baseline)

Supports:
  - Full checkpoints (containing model_checkpoint.pt)
  - UMS-only checkpoints (e.g. ums_only/run_name/task_9/ums/*.pt)
  - Automatic dataset loading from local data/ directory
  - Per-sample and task-level routing accuracy
  - 10x10 cross-task routing confusion matrix
  - Latency benchmarking (ms/batch)

Usage Examples:
  # 1. Evaluate from ums_only run folder on real datasets:
  python eval_task_retrieval.py --checkpoint ums_only/LEAR_seq-cifar10_None_0_10_20261004-163043_cf788a4f

  # 2. Evaluate specific task folder (e.g. task_9):
  python eval_task_retrieval.py --checkpoint ums_only/LEAR_seq-cifar10_None_0_10_20261004-163043_cf788a4f/task_9

  # 3. Fast smoke test with 50 samples per dataset on GPU 1:
  python eval_task_retrieval.py --checkpoint ums_only/LEAR_seq-cifar10_None_0_10_20261004-163043_cf788a4f --max_samples 50 --device cuda:1

  # 4. Evaluate a full checkpoint with downstream classification:
  python eval_task_retrieval.py --checkpoint checkpoints/LEAR_seq-cifar10/task_9
"""

import os
import sys
import re
import json
import time
import argparse
from argparse import Namespace
from typing import Dict, List, Tuple, Any, Optional

import torch
torch.set_num_threads(4)
import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy('file_system')
except Exception:
    pass
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# Add root directory to python path
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from utils.conf import base_path
from utils.task_checkpointing import load_task_checkpoint, serialize_for_json
from backbone import get_backbone
from models import get_model
from datasets import get_all_datasets, get_dataset_class


# Default fallback arguments for model reconstruction
DEFAULT_ARGS = {
    'model': 'LEAR',
    'backbone': 'lear',
    'dataset': 'seq-cifar10',
    'list_datasets': 'seq-cifar10,seq-cropdisease,seq-mnist,seq-resisc45,seq-eurosat-rgb,seq-tinyimg,seq-cifar100,seq-chestx,seq-imagenet-r,seq-cub200',
    'apply_to': '6,7,8,9,10,11',
    'lr': 0.001,
    'optimizer': 'adam',
    'optim_wd': 0.0,
    'optim_mom': 0.9,
    'optim_nesterov': 0,
    'batch_size': 64,
    'ncls_per_task': 0,
    'permute_classes': False,
    'validation': False,
    'validation_mode': 'current',
    'joint': 0,
    'fitting_mode': 'epochs',
    'n_epochs': 10,
    'label_perc': 1.0,
    'label_perc_by_class': 1.0,
    'noise_rate': 0.0,
    'enable_other_metrics': 0,
    'eval_future': 0,
    'non_verbose': True,
    'pretrained': True,
    'pretrain_type': 'in21k',
    'num_classes': 10,
    'transform_type': 'weak',
    'base_path': './data/',
    'ums_basis_rank': 768,
    'ums_svd_rank': 75,
    'ums_tau': 0.001,
    'ums_rot_steps': 15,
    'ums_rot_lr': 0.0001,
    'ums_head_lr': 0.001,
    'ums_rot_lambda_anchor': 2.0,
    'ums_rot_lambda_support_mse': 1.0,
    'ums_rot_lambda_support_bce': 0.1,
    'ums_rot_lambda_gaussian': 0.01,
    'n_frq': 5898,
    'n_tasks': 10,
    'modulate_q': 1,
    'modulate_proj': 1,
    'ums_in_loop': 1,
    'ums_refresh_every': 2,
    'task_retrieval': 'resonance',
    'num_workers': 0,
    'device': 'cuda:1' if (torch.cuda.is_available() and torch.cuda.device_count() > 1) else ('cuda:0' if torch.cuda.is_available() else 'cpu'),
}


def parse_cli_args():
    parser = argparse.ArgumentParser(
        description="FOCAL-UMS Standalone Task Retrieval Evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--checkpoint', '--ckpt', '--path', type=str, required=True,
                        help="Path to task directory (e.g. ums_only/run_name/task_9), model_checkpoint.pt, or run directory.")
    parser.add_argument('--task', type=int, default=None,
                        help="Task ID to evaluate if pointing to a run directory with multiple tasks. Defaults to latest task.")
    parser.add_argument('--methods', type=str, default='both', choices=['both', 'resonance', 'mahalanobis'],
                        help="Retrieval method(s) to evaluate: 'resonance', 'mahalanobis', or 'both'.")
    parser.add_argument('--device', type=str, default=None,
                        help="Execution device ('cuda', 'cuda:0', 'cuda:1', 'cpu').")
    parser.add_argument('--batch_size', type=int, default=None,
                        help="Batch size for test evaluation. Defaults to batch_size from checkpoint/config.")
    parser.add_argument('--num_workers', type=int, default=0,
                        help="Number of workers for DataLoader (default 0 to avoid /dev/shm exhaustion).")
    parser.add_argument('--max_samples', '--max_samples_per_task', type=int, default=None,
                        help="Limit number of test samples per task (for fast testing). None = all samples.")
    parser.add_argument('--list_datasets', type=str, default=None,
                        help="Override dataset list (comma-separated). Defaults to list_datasets in checkpoint.")
    parser.add_argument('--data_dir', type=str, default='./data/',
                        help="Base data directory where datasets are stored.")
    parser.add_argument('--output_json', type=str, default='task_retrieval_results.json',
                        help="Path to save detailed evaluation metrics in JSON format.")
    parser.add_argument('--eval_classification', action='store_true', default=None,
                        help="Also evaluate downstream classification accuracy (requires model_checkpoint.pt).")
    parser.add_argument('--no_eval_classification', action='store_false', dest='eval_classification',
                        help="Skip downstream classification accuracy evaluation.")
    parser.add_argument('--synthetic', action='store_true', default=False,
                        help="Use synthetic random images [B, 3, 224, 224] for testing when datasets are not locally downloaded.")
    return parser.parse_args()


def resolve_checkpoint_info(target_path: str, task_override: Optional[int] = None) -> Tuple[Optional[str], str, str, int, Dict[str, Any]]:
    """Resolves arbitrary user path to:

    (model_ckpt_path_or_none, task_dir, ums_dir, task_id, saved_args)
    """
    if not os.path.exists(target_path):
        raise FileNotFoundError(f"Specified path does not exist: {target_path}")

    model_ckpt_path = None
    task_dir = None
    ums_dir = None
    task_id = 0
    saved_args = {}

    # Case 1: Direct file
    if os.path.isfile(target_path):
        filename = os.path.basename(target_path)
        if filename == 'model_checkpoint.pt':
            model_ckpt_path = target_path
            task_dir = os.path.dirname(os.path.dirname(target_path))
            ums_dir = os.path.join(task_dir, 'ums')
        elif filename.endswith('_ums.pt'):
            ums_dir = os.path.dirname(target_path)
            task_dir = os.path.dirname(ums_dir)
            cand_pt = os.path.join(task_dir, "model", "model_checkpoint.pt")
            if os.path.isfile(cand_pt):
                model_ckpt_path = cand_pt
        else:
            task_dir = os.path.dirname(target_path)
            ums_dir = os.path.join(task_dir, 'ums')

    # Case 2: Directory
    else:
        # If user pointed directly to an 'ums' folder
        if os.path.basename(target_path.rstrip('/')) == 'ums':
            ums_dir = target_path.rstrip('/')
            task_dir = os.path.dirname(ums_dir)
        else:
            # Check if target_path contains task_* subdirectories (Run directory)
            task_subdirs = []
            for item in os.listdir(target_path):
                sub = os.path.join(target_path, item)
                if os.path.isdir(sub) and re.match(r"^task_\d+$", item):
                    t_idx = int(item.split('_')[1])
                    task_subdirs.append((t_idx, sub))

            if task_subdirs:
                task_subdirs.sort(key=lambda x: x[0])
                if task_override is not None:
                    matched = [sub for (idx, sub) in task_subdirs if idx == task_override]
                    if not matched:
                        available_tasks = [idx for (idx, _) in task_subdirs]
                        raise ValueError(f"Task {task_override} not found in {target_path}. Available tasks: {available_tasks}")
                    task_dir = matched[0]
                else:
                    task_dir = task_subdirs[-1][1]  # Highest/latest task
                ums_dir = os.path.join(task_dir, "ums")
            else:
                # Target path itself is task_dir
                task_dir = target_path.rstrip('/')
                ums_dir = os.path.join(task_dir, "ums")

    # Check for model_checkpoint.pt
    cand_model_pt = os.path.join(task_dir, "model", "model_checkpoint.pt")
    if os.path.isfile(cand_model_pt):
        model_ckpt_path = cand_model_pt

    # Retrieve task_id
    match = re.search(r"task_(\d+)", task_dir)
    if match:
        task_id = int(match.group(1))

    # Retrieve saved_args from model_checkpoint.pt or run_metadata.json
    if model_ckpt_path and os.path.isfile(model_ckpt_path):
        try:
            ckpt_obj = torch.load(model_ckpt_path, map_location='cpu')
            saved_args = ckpt_obj.get('args', {})
            task_id = ckpt_obj.get('task_id', task_id)
        except Exception as e:
            print(f"[WARNING] Could not read metadata from {model_ckpt_path}: {e}")

    # Fallback to run_metadata.json if present
    if not saved_args:
        run_dir = os.path.dirname(task_dir)
        for cand_meta in [os.path.join(run_dir, "run_metadata.json"), os.path.join(task_dir, "run_metadata.json")]:
            if os.path.isfile(cand_meta):
                try:
                    with open(cand_meta, 'r', encoding='utf-8') as f:
                        meta = json.load(f)
                        saved_args = meta.get('args', {})
                        print(f"[INFO] Loaded run arguments from: {cand_meta}")
                        break
                except Exception:
                    pass

    # Fallback to task_summary.json
    task_summary_path = os.path.join(task_dir, "task_summary.json")
    if os.path.isfile(task_summary_path):
        try:
            with open(task_summary_path, 'r', encoding='utf-8') as f:
                tsum = json.load(f)
                if 'task_id' in tsum:
                    task_id = int(tsum['task_id'])
        except Exception:
            pass

    return model_ckpt_path, task_dir, ums_dir, task_id, saved_args


def compute_sample_resonance_distances(model, x: torch.Tensor) -> torch.Tensor:
    """Computes per-sample cosine distance [B, T] using UMS Bilinear Coordinate Resonance."""
    x = x.to(model.device)
    processX = model.net.vitProcess(x)
    if processX.size(1) == 1:
        processX = processX.expand(-1, 3, -1, -1)

    with torch.no_grad():
        class_token = model.net.extract_feature(processX).detach()  # [B, D]
        n_tasks = len(model.net.fcArr)
        if n_tasks == 0:
            return torch.zeros((x.size(0), 0), device=x.device)

        experts = model.get_active_focal_experts()
        total_sample_dists = torch.zeros((x.size(0), n_tasks), device=class_token.device, dtype=torch.float32)
        valid_experts = 0

        for expert in experts:
            ums = getattr(expert, 'ums', None)
            if ums is None:
                continue
            r_cur = ums.r_cur
            n_folded = len(ums.z_store)
            if r_cur == 0 or n_folded == 0:
                continue

            tasks_eval = min(n_tasks, n_folded)
            if tasks_eval == 0:
                continue

            R = ums.rotation().detach()
            P = ums.P[:, :r_cur].detach()
            Q = ums.Q[:, :r_cur].detach()

            A = torch.matmul(R, P).to(dtype=class_token.dtype)
            B_mat = torch.matmul(R, Q).to(dtype=class_token.dtype)

            a = torch.matmul(class_token, A)
            b = torch.matmul(class_token, B_mat)
            phi = a * b
            phi_norm = F.normalize(phi, p=2, dim=-1)  # [B, r_cur]

            z_list = [ums.z_store[t][:r_cur].detach().to(device=class_token.device, dtype=class_token.dtype)
                      for t in range(tasks_eval)]
            Z = torch.stack(z_list, dim=0)            # [tasks_eval, r_cur]
            Z_norm = F.normalize(Z, p=2, dim=-1)      # [tasks_eval, r_cur]

            # Cosine similarity for each individual sample: [B, tasks_eval]
            sim = torch.matmul(phi_norm, Z_norm.t())
            dist = 1.0 - sim

            total_sample_dists[:, :tasks_eval] += dist
            valid_experts += 1

        if valid_experts > 0:
            return total_sample_dists / float(valid_experts)
        else:
            return torch.zeros((x.size(0), n_tasks), device=x.device)


def compute_sample_mahalanobis_distances(model, x: torch.Tensor) -> Optional[torch.Tensor]:
    """Computes per-sample Mahalanobis distance [B, T]."""
    if not hasattr(model.net, 'distributions') or len(model.net.distributions) == 0:
        return None

    x = x.to(model.device)
    processX = model.net.vitProcess(x)
    if processX.size(1) == 1:
        processX = processX.expand(-1, 3, -1, -1)

    with torch.no_grad():
        class_token = model.net.extract_feature(processX).detach()
        n_tasks = min(len(model.net.fcArr), len(model.net.distributions))
        if n_tasks == 0:
            return None

        sample_dists = torch.zeros((x.size(0), n_tasks), device=class_token.device, dtype=torch.float32)
        for t in range(n_tasks):
            fc = model.net.fcArr[t]
            dist = model.net.distributions[t]
            fc_feature = fc(class_token)
            delta = fc_feature - dist.mean.to(fc_feature.device)
            cov = dist.covariance_matrix.to(fc_feature.device)
            inv_cov = torch.linalg.pinv(cov)
            maha = torch.sqrt(torch.sum((delta @ inv_cov) * delta, dim=1))  # [B]
            sample_dists[:, t] = maha

        return sample_dists


def run_classification_prediction(model, inputs: torch.Tensor, pred_tasks: torch.Tensor) -> torch.Tensor:
    """Evaluates classification output using routed task heads, grouped by predicted task."""
    device = inputs.device
    B = inputs.size(0)
    final_preds = torch.zeros(B, dtype=torch.long, device=device)

    unique_tasks = torch.unique(pred_tasks)
    for t_idx in unique_tasks:
        t_val = int(t_idx.item())
        mask = (pred_tasks == t_val)
        if not mask.any():
            continue
        sub_inputs = inputs[mask]
        with torch.no_grad():
            outputs = model.myPrediction(sub_inputs, t_val)
            sub_preds = torch.argmax(outputs, dim=1)
            final_preds[mask] = sub_preds

    return final_preds


def print_table(headers: List[str], rows: List[List[Any]], title: str = ""):
    """Helper to render a cleanly formatted ASCII table."""
    if title:
        print(f"\n{'=' * 85}\n  {title}\n{'=' * 85}")
    col_widths = [len(h) for h in headers]
    for row in rows:
        for idx, val in enumerate(row):
            col_widths[idx] = max(col_widths[idx], len(str(val)))

    header_line = " | ".join(f"{h:<{col_widths[i]}}" for i, h in enumerate(headers))
    sep_line = "-+-".join("-" * col_widths[i] for i in range(len(headers)))
    print(header_line)
    print(sep_line)
    for row in rows:
        row_line = " | ".join(f"{str(v):<{col_widths[i]}}" for i, v in enumerate(row))
        print(row_line)
    print("-" * len(sep_line) + "\n")


def print_confusion_matrix(matrix: np.ndarray, title: str):
    """Renders an N x N confusion matrix."""
    n_tasks = matrix.shape[0]
    print(f"\n--- {title} (Rows: True Task, Cols: Predicted Task) ---")
    headers = ["Task"] + [f"P_{j}" for j in range(n_tasks)] + ["Total", "Acc (%)"]
    rows = []
    for i in range(n_tasks):
        total_row = int(matrix[i].sum())
        correct = int(matrix[i, i])
        acc = (correct / total_row * 100.0) if total_row > 0 else 0.0
        row_vals = [f"T_{i}"] + [str(int(matrix[i, j])) for j in range(n_tasks)] + [str(total_row), f"{acc:.2f}%"]
        rows.append(row_vals)
    print_table(headers, rows)


def main():
    cli_args = parse_cli_args()

    # 1. Resolve checkpoint paths & metadata
    model_ckpt_path, task_dir, ums_dir, task_id, saved_args_dict = resolve_checkpoint_info(
        cli_args.checkpoint, cli_args.task
    )
    is_ums_only = (model_ckpt_path is None or not os.path.isfile(model_ckpt_path))

    print(f"\n[INFO] Resolved task directory: {task_dir}")
    print(f"[INFO] Resolved UMS directory:  {ums_dir}")
    if is_ums_only:
        print("[INFO] Operating mode: UMS-ONLY (no model_checkpoint.pt found; evaluating Task Retrieval from _ums.pt matrices).")
    else:
        print(f"[INFO] Operating mode: FULL CHECKPOINT ({model_ckpt_path})")

    # 2. Merge args
    merged_args_dict = dict(DEFAULT_ARGS)
    merged_args_dict.update(saved_args_dict)

    if cli_args.device:
        merged_args_dict['device'] = cli_args.device
    if cli_args.batch_size:
        merged_args_dict['batch_size'] = cli_args.batch_size
    if cli_args.list_datasets:
        merged_args_dict['list_datasets'] = cli_args.list_datasets
    if cli_args.data_dir:
        merged_args_dict['base_path'] = cli_args.data_dir
    if cli_args.num_workers is not None:
        merged_args_dict['num_workers'] = cli_args.num_workers
    else:
        merged_args_dict['num_workers'] = 0

    args = Namespace(**merged_args_dict)
    base_path(args.base_path)

    device = torch.device(args.device)
    print(f"[INFO] Execution Device: {device}")
    print(f"[INFO] Evaluated ViT Blocks: {args.apply_to}")

    # Determine whether classification can be evaluated
    eval_classification = cli_args.eval_classification
    if eval_classification is None:
        eval_classification = (not is_ums_only)
    elif eval_classification and is_ums_only:
        print("[WARNING] --eval_classification requested but model_checkpoint.pt is absent. Disabling classification evaluation.")
        eval_classification = False

    # 3. Create datasets and loaders
    test_loaders = []
    task_names = []

    if cli_args.synthetic:
        print("[INFO] Running in SYNTHETIC data mode (dummy images [B, 3, 224, 224]).")
        n_tasks_eval = task_id + 1
        dummy_samples_per_task = cli_args.max_samples or 128
        batch_size = args.batch_size or 32

        for t in range(n_tasks_eval):
            task_names.append(f"SyntheticTask_{t}")
            x_dummy = torch.randn(dummy_samples_per_task, 3, 224, 224)
            y_dummy = torch.randint(0, 10, (dummy_samples_per_task,))
            loader = torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(x_dummy, y_dummy),
                batch_size=batch_size, shuffle=False
            )
            test_loaders.append(loader)
    else:
        print(f"[INFO] Loading datasets from: {args.base_path}")
        datasets = get_all_datasets(args)
        n_tasks_eval = min(len(datasets), task_id + 1)

        for t in range(n_tasks_eval):
            task_names.append(datasets[t].NAME)
            if hasattr(datasets[t], 'get_all_data_loaders'):
                _, t_loader = datasets[t].get_all_data_loaders()
            else:
                _, t_loader = datasets[t].get_data_loaders()
            test_loaders.append(t_loader)

    # 4. Instantiate model & backbone
    print("[INFO] Instantiating model architecture...")
    backbone = get_backbone(args)
    loss = nn.CrossEntropyLoss()
    transform = None if cli_args.synthetic else datasets[0].get_transform()
    model = get_model(args, backbone, loss, transform, dataset=datasets[0] if not cli_args.synthetic else None)

    # 5. Restore state into model
    if not is_ums_only:
        print(f"[INFO] Restoring full checkpoint from: {model_ckpt_path}")
        load_task_checkpoint(model_ckpt_path, model=model, device=device)
    else:
        print(f"[INFO] Restoring 24 UMS consolidators from: {ums_dir}")
        loaded_count = 0
        for expert in model.get_active_focal_experts():
            if hasattr(expert, 'ums') and expert.ums is not None:
                pt_file = os.path.join(ums_dir, f"{expert.ums.name}_ums.pt")
                if os.path.exists(pt_file):
                    expert.ums.load_state(pt_file, device=device)
                    loaded_count += 1
                else:
                    print(f"[WARNING] File not found: {pt_file}")
        print(f"[INFO] Successfully loaded {loaded_count} UMS consolidators.")

    model.to(device)
    model.eval()
    model.net.eval()

    # Determine number of folded tasks in UMS
    experts = model.get_active_focal_experts()
    if experts and hasattr(experts[0], 'ums') and experts[0].ums is not None:
        n_tasks_ums = len(experts[0].ums.z_store)
        r_cur = experts[0].ums.r_cur
        print(f"[INFO] UMS Consolidator Active Rank: {r_cur}/{experts[0].ums.R}, Folded Tasks: {n_tasks_ums}")
    else:
        n_tasks_ums = n_tasks_eval

    # Ensure model has enough task heads for evaluation
    if hasattr(model, 'net'):
        while len(model.net.fcArr) < max(n_tasks_eval, n_tasks_ums):
            first_fc = model.net.fcArr[0]
            model.net.fcArr.append(nn.Linear(first_fc.in_features, first_fc.out_features, device=device))
        while len(model.net.classifierArr) < max(n_tasks_eval, n_tasks_ums):
            first_cls = model.net.classifierArr[0]
            model.net.classifierArr.append(nn.Linear(first_cls.in_features, first_cls.out_features, device=device))

    n_tasks_model = len(model.net.fcArr)
    print(f"[INFO] Ready for evaluation: {n_tasks_eval} dataset domains, {n_tasks_model} candidate tasks.")

    # 6. Determine available evaluation methods
    has_mahalanobis = hasattr(model.net, 'distributions') and len(model.net.distributions) >= n_tasks_model
    methods_to_test = []

    if cli_args.methods in ['resonance', 'both']:
        methods_to_test.append('resonance')

    if cli_args.methods in ['mahalanobis', 'both']:
        if has_mahalanobis:
            methods_to_test.append('mahalanobis')
        else:
            if not is_ums_only:
                print("[WARNING] Mahalanobis distributions not found in checkpoint (likely trained with --task-retrieval resonance).")
            if cli_args.methods == 'mahalanobis':
                print("[ERROR] Cannot evaluate Mahalanobis because Gaussian distributions were not computed or stored.")
                return

    print(f"[INFO] Methods to evaluate: {methods_to_test}")

    # 7. Run Evaluation Loop
    results: Dict[str, Any] = {
        "checkpoint": cli_args.checkpoint,
        "task_id": task_id,
        "n_tasks_evaluated": n_tasks_eval,
        "eval_classification": eval_classification,
        "methods": {}
    }

    for method in methods_to_test:
        print(f"\n{'=' * 85}")
        print(f"  RUNNING TASK RETRIEVAL EVALUATION: {method.upper()}")
        print(f"{'=' * 85}")

        confusion_sample = np.zeros((n_tasks_eval, n_tasks_model), dtype=np.int64)
        task_level_correct = 0

        per_task_sample_acc = []
        per_task_oracle_acc = []
        per_task_retrieved_acc = []
        batch_latencies = []

        for t_true in range(n_tasks_eval):
            loader = test_loaders[t_true]
            t_name = task_names[t_true]

            n_samples_task = 0
            n_correct_retrieval_sample = 0
            n_correct_oracle_cls = 0
            n_correct_retrieved_cls = 0

            sum_distance_vec = torch.zeros(n_tasks_model, device=device)

            pbar = tqdm(loader, desc=f"Evaluating Task {t_true} ({t_name})", leave=False)
            for batch_idx, data in enumerate(pbar):
                inputs, labels = data[0], data[1]
                inputs = inputs.to(device)
                labels = labels.to(device)

                if cli_args.max_samples and n_samples_task >= cli_args.max_samples:
                    break

                B = inputs.size(0)
                if cli_args.max_samples and (n_samples_task + B > cli_args.max_samples):
                    take = cli_args.max_samples - n_samples_task
                    inputs = inputs[:take]
                    labels = labels[:take]
                    B = take

                # Compute per-sample distances
                t0 = time.time()
                if method == 'resonance':
                    sample_dists = compute_sample_resonance_distances(model, inputs)
                else:
                    sample_dists = compute_sample_mahalanobis_distances(model, inputs)
                t1 = time.time()
                batch_latencies.append((t1 - t0) * 1000.0)

                # Sample-level prediction: argmin across tasks
                pred_tasks = torch.argmin(sample_dists, dim=1)  # [B]

                # Update sample-level routing counts
                for p in pred_tasks:
                    p_idx = int(p.item())
                    if p_idx < n_tasks_model:
                        confusion_sample[t_true, p_idx] += 1

                n_correct_retrieval_sample += int((pred_tasks == t_true).sum().item())
                sum_distance_vec += sample_dists.sum(dim=0)
                n_samples_task += B

                # Downstream classification accuracy (if heads are loaded)
                if eval_classification:
                    with torch.no_grad():
                        # Oracle prediction
                        oracle_outputs = model.myPrediction(inputs, t_true)
                        oracle_preds = torch.argmax(oracle_outputs, dim=1)
                        n_correct_oracle_cls += int((oracle_preds == labels).sum().item())

                        # Retrieved prediction
                        retrieved_preds = run_classification_prediction(model, inputs, pred_tasks)
                        n_correct_retrieved_cls += int((retrieved_preds == labels).sum().item())

            # Task-level aggregated routing: argmin of average distance
            avg_dist_vec = sum_distance_vec / max(n_samples_task, 1)
            pred_task_level = int(torch.argmin(avg_dist_vec).item())
            if pred_task_level == t_true:
                task_level_correct += 1

            sample_retrieval_acc = (n_correct_retrieval_sample / n_samples_task * 100.0) if n_samples_task > 0 else 0.0
            per_task_sample_acc.append(sample_retrieval_acc)

            if eval_classification:
                oracle_acc = (n_correct_oracle_cls / n_samples_task * 100.0) if n_samples_task > 0 else 0.0
                retrieved_acc = (n_correct_retrieved_cls / n_samples_task * 100.0) if n_samples_task > 0 else 0.0
                per_task_oracle_acc.append(oracle_acc)
                per_task_retrieved_acc.append(retrieved_acc)

        # Store results for this method
        mean_sample_acc = float(np.mean(per_task_sample_acc))
        task_level_acc = float(task_level_correct / n_tasks_eval * 100.0)
        mean_latency = float(np.mean(batch_latencies)) if batch_latencies else 0.0

        method_dict: Dict[str, Any] = {
            "mean_sample_retrieval_acc": mean_sample_acc,
            "task_level_aggregated_acc": task_level_acc,
            "mean_latency_ms_per_batch": mean_latency,
            "per_task_sample_retrieval_acc": per_task_sample_acc,
            "confusion_matrix": confusion_sample.tolist(),
        }

        if eval_classification:
            method_dict["mean_oracle_classification_acc"] = float(np.mean(per_task_oracle_acc))
            method_dict["mean_retrieved_classification_acc"] = float(np.mean(per_task_retrieved_acc))
            method_dict["per_task_oracle_acc"] = per_task_oracle_acc
            method_dict["per_task_retrieved_acc"] = per_task_retrieved_acc

        results["methods"][method] = method_dict

        # Print Confusion Matrix
        print_confusion_matrix(confusion_sample, f"CONFUSION MATRIX - {method.upper()}")

    # 8. Summary Comparison Table
    print("\n" + "=" * 85)
    print("                      OVERALL TASK RETRIEVAL COMPARISON")
    print("=" * 85)

    summary_headers = ["Method", "Sample Routing Acc", "Task-Level Routing", "Latency (ms/batch)"]
    if eval_classification:
        summary_headers.extend(["Oracle Cls Acc", "Routed Cls Acc", "Accuracy Gap"])

    summary_rows = []
    for method in methods_to_test:
        m_data = results["methods"][method]
        row_vals = [
            method.upper(),
            f"{m_data['mean_sample_retrieval_acc']:.2f}%",
            f"{m_data['task_level_aggregated_acc']:.2f}%",
            f"{m_data['mean_latency_ms_per_batch']:.2f} ms",
        ]
        if eval_classification:
            gap = m_data["mean_oracle_classification_acc"] - m_data["mean_retrieved_classification_acc"]
            row_vals.extend([
                f"{m_data['mean_oracle_classification_acc']:.2f}%",
                f"{m_data['mean_retrieved_classification_acc']:.2f}%",
                f"{gap:.2f}%",
            ])
        summary_rows.append(row_vals)
    print_table(summary_headers, summary_rows)

    # Per-task Breakdown Table
    breakdown_headers = ["Task ID", "Domain Name"] + [f"{m.capitalize()} Routing Acc" for m in methods_to_test]
    if eval_classification:
        for m in methods_to_test:
            breakdown_headers.append(f"{m.capitalize()} Cls Acc")
        breakdown_headers.append("Oracle Cls Acc")

    breakdown_rows = []
    for t in range(n_tasks_eval):
        row = [f"Task {t}", task_names[t]]
        for m in methods_to_test:
            row.append(f"{results['methods'][m]['per_task_sample_retrieval_acc'][t]:.2f}%")
        if eval_classification:
            for m in methods_to_test:
                row.append(f"{results['methods'][m]['per_task_retrieved_acc'][t]:.2f}%")
            row.append(f"{results['methods'][methods_to_test[0]]['per_task_oracle_acc'][t]:.2f}%")
        breakdown_rows.append(row)

    print_table(breakdown_headers, breakdown_rows, title="PER-TASK PERFORMANCE BREAKDOWN")

    # 9. Save JSON results
    if cli_args.output_json:
        out_path = cli_args.output_json
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(serialize_for_json(results), f, indent=2)
        print(f"[INFO] Detailed evaluation report successfully saved to: {out_path}\n")


if __name__ == '__main__':
    main()
