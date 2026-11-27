"""Task Checkpointing & Analysis Module for FOCAL-UMS.

Saves structured diagnostic data and model checkpoints after each continual learning task:
  1) UMS Consolidator: shared bases (P, Q), rotation (W_rot, R_o), coordinates (z_store),
     basis capacity, orthonormality errors, rotation errors, block coherence, and metrics.
  2) FocalExpert: task-partitioned frequency coefficients and UMS consolidator state.
  3) Model State Checkpoint: full model state_dict, per-task bottlenecks (fcArr) & classifiers (classifierArr),
     Mahalanobis Gaussian distributions (mean & covariance), optimizer state, args, eval results.

Directory layout:
  checkpoints/
  └── {run_name}/
      ├── run_metadata.json
      ├── overall_analysis.json
      ├── task_{t}/
      │   ├── model/
      │   │   └── model_checkpoint.pt
      │   ├── ums/
      │   │   ├── ums_summary.json
      │   │   ├── {layer_name}_ums.pt
      │   │   └── {layer_name}_analysis.json
      │   ├── focal/
      │   │   └── focal_summary.json
      │   └── task_summary.json
"""

import os
import json
import copy
import datetime
from argparse import Namespace
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import numpy as np


def serialize_for_json(obj: Any) -> Any:
    """Recursively converts objects (Tensors, NumPy types, Namespaces, sets) to JSON-serializable types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, (torch.Tensor,)):
        if obj.numel() == 1:
            val = obj.item()
            return val if (not isinstance(val, float) or (not np.isnan(val) and not np.isinf(val))) else str(val)
        return serialize_for_json(obj.detach().cpu().tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        val = float(obj)
        return val if (not np.isnan(val) and not np.isinf(val)) else str(val)
    if isinstance(obj, (np.ndarray,)):
        return serialize_for_json(obj.tolist())
    if isinstance(obj, (list, tuple)):
        return [serialize_for_json(item) for item in obj]
    if isinstance(obj, (set,)):
        return [serialize_for_json(item) for item in sorted(list(obj))]
    if isinstance(obj, (dict,)):
        return {str(k): serialize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, Namespace):
        return serialize_for_json(vars(obj))
    # Fallback for any other objects
    return str(obj)


def get_or_create_checkpoint_dirs(base_dir: str, run_name: str, task_id: int) -> Dict[str, str]:
    """Creates and returns the structured directories for a given run and task."""
    run_dir = os.path.join(base_dir, run_name)
    task_dir = os.path.join(run_dir, f"task_{task_id}")
    model_dir = os.path.join(task_dir, "model")
    ums_dir = os.path.join(task_dir, "ums")
    focal_dir = os.path.join(task_dir, "focal")
    moe_dir = os.path.join(task_dir, "moe")

    for d in [run_dir, task_dir, model_dir, ums_dir, focal_dir, moe_dir]:
        os.makedirs(d, exist_ok=True)

    return {
        "run_dir": run_dir,
        "task_dir": task_dir,
        "model_dir": model_dir,
        "ums_dir": ums_dir,
        "focal_dir": focal_dir,
        "moe_dir": moe_dir,
    }


def find_bilora_moe_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Discovers all BiLORA_MoE modules in a model."""
    from backbone.bilora import BiLORA_MoE

    found = []
    # If the model has net.local_vitmodel.blocks (like LEAR)
    if hasattr(model, 'net') and hasattr(model.net, 'local_vitmodel') and hasattr(model.net.local_vitmodel, 'blocks'):
        blocks = model.net.local_vitmodel.blocks
        trainable_blocks = getattr(model, 'trainable_blocks', list(range(len(blocks))))
        for idx in trainable_blocks:
            if idx < len(blocks):
                block = blocks[idx]
                if hasattr(block, 'attn') and isinstance(block.attn, BiLORA_MoE):
                    layer_name = getattr(block.attn, 'layer_name', f"Block{idx}") or f"Block{idx}"
                    found.append((layer_name, block.attn))

    # Also search recursively in named_modules in case architecture differs
    if len(found) == 0:
        for name, mod in model.named_modules():
            if isinstance(mod, BiLORA_MoE):
                layer_name = getattr(mod, 'layer_name', name) or name
                found.append((layer_name, mod))

    return found


def find_bilora_focal_modules(model: nn.Module) -> List[Tuple[str, nn.Module]]:
    """Discovers all BiLORA_Focal modules in a model."""
    from backbone.bilora import BiLORA_Focal
    found = []
    if hasattr(model, 'net') and hasattr(model.net, 'local_vitmodel') and hasattr(model.net.local_vitmodel, 'blocks'):
        blocks = model.net.local_vitmodel.blocks
        trainable_blocks = getattr(model, 'trainable_blocks', list(range(len(blocks))))
        for idx in trainable_blocks:
            if idx < len(blocks):
                block = blocks[idx]
                if hasattr(block, 'attn') and isinstance(block.attn, BiLORA_Focal):
                    layer_name = getattr(block.attn, 'layer_name', f"Block{idx}") or f"Block{idx}"
                    found.append((layer_name, block.attn))
    if len(found) == 0:
        for name, mod in model.named_modules():
            if isinstance(mod, BiLORA_Focal):
                layer_name = getattr(mod, 'layer_name', name) or name
                found.append((layer_name, mod))
    return found


def save_ums_consolidators(
    model: nn.Module,
    ums_dir: str,
    task_id: int,
    include_readout: bool = True
) -> Dict[str, Any]:
    """Saves all UMS consolidators in the model and generates ums_summary.json.

    Returns the aggregated summary dictionary.
    """
    bilora_modules = find_bilora_focal_modules(model)
    if not bilora_modules:
        bilora_modules = find_bilora_moe_modules(model)
    module_metrics = {}
    ranks_used = []
    recon_errors = []
    raw_recon_errors = []
    pre_recon_errors = []
    support_f1_scores = []
    support_ious = []
    off_diagonal_ratios = []
    target_mean_errors = []
    target_std_errors = []
    old_task_drifts = []
    coherences = []
    rot_deviations = []
    capacity = None

    for layer_name, bilora in bilora_modules:
        modules_to_save = []
        for attr in ['focal_q', 'focal_k', 'focal_v', 'focal_proj', 'moe_k', 'moe_v']:
            mod = getattr(bilora, attr, None)
            if mod is not None and hasattr(mod, 'ums') and mod.ums is not None:
                modules_to_save.append(mod)

        for module in modules_to_save:
            if hasattr(module, 'ums') and module.ums is not None:
                if getattr(module, '__class__', None) and module.__class__.__name__ == 'FocalExpert':
                    mod_pt, mod_json = module.save_state(ums_dir, task_id=task_id, include_readout=include_readout)
                else:
                    mod_pt, mod_json = module.ums.save_state(ums_dir, task_id=task_id, include_readout=include_readout)
            elif hasattr(module, 'save_state'):
                mod_pt, mod_json = module.save_state(ums_dir, task_id=task_id)
            else:
                continue
            mod_metrics = module.ums.get_analysis_metrics(task_id=task_id)
            module_metrics[module.ums.name] = {
                "pt_file": os.path.basename(mod_pt),
                "analysis_file": os.path.basename(mod_json),
                "basis_rank_used": mod_metrics["basis_rank_used_r_cur"],
                "basis_capacity": mod_metrics["basis_capacity_R"],
                "capacity_used_pct": mod_metrics["basis_capacity_used_pct"],
                "block_coherence": mod_metrics["overall_block_coherence"],
                "rotation_deviation": mod_metrics["rotation_deviation_from_identity"],
                "fold_history": mod_metrics.get("fold_history", []),
            }
            ranks_used.append(mod_metrics["basis_rank_used_r_cur"])
            coherences.append(mod_metrics["overall_block_coherence"])
            rot_deviations.append(mod_metrics["rotation_deviation_from_identity"])
            capacity = mod_metrics["basis_capacity_R"]
            if mod_metrics.get("fold_history"):
                latest_fold = mod_metrics["fold_history"][-1]
                recon_errors.append(latest_fold.get("relative_recon_error", 0.0))
                raw_recon_errors.append(latest_fold.get("raw_relative_recon_error", 0.0))
                pre = latest_fold.get("pre_refinement", {})
                post = latest_fold.get("post_refinement", {})
                pre_recon_errors.append(pre.get("gated_relative_recon_error", 0.0))
                support_f1_scores.append(post.get("support_f1", 0.0))
                support_ious.append(post.get("support_iou", 0.0))
                off_diagonal_ratios.append(post.get("off_diagonal_energy_ratio", 0.0))
                target_mean_errors.append(post.get("target_mean_abs_error", 0.0))
                target_std_errors.append(post.get("target_std_abs_error", 0.0))
                old_task_drifts.append(latest_fold.get("avg_old_task_modulation_drift", 0.0))

    summary = {
        "task_id": int(task_id),
        "total_ums_consolidators": len(module_metrics),
        "basis_capacity_R": capacity or 0,
        "avg_basis_rank_used": float(np.mean(ranks_used)) if ranks_used else 0.0,
        "max_basis_rank_used": int(np.max(ranks_used)) if ranks_used else 0,
        "min_basis_rank_used": int(np.min(ranks_used)) if ranks_used else 0,
        "avg_capacity_used_pct": float(np.mean(ranks_used) / max(capacity or 1, 1) * 100) if ranks_used else 0.0,
        "avg_relative_recon_error": float(np.mean(recon_errors)) if recon_errors else 0.0,
        "avg_raw_relative_recon_error": float(np.mean(raw_recon_errors)) if raw_recon_errors else 0.0,
        "avg_pre_refinement_recon_error": float(np.mean(pre_recon_errors)) if pre_recon_errors else 0.0,
        "avg_support_f1": float(np.mean(support_f1_scores)) if support_f1_scores else 0.0,
        "avg_support_iou": float(np.mean(support_ious)) if support_ious else 0.0,
        "avg_off_diagonal_energy_ratio": float(np.mean(off_diagonal_ratios)) if off_diagonal_ratios else 0.0,
        "avg_target_mean_abs_error": float(np.mean(target_mean_errors)) if target_mean_errors else 0.0,
        "avg_target_std_abs_error": float(np.mean(target_std_errors)) if target_std_errors else 0.0,
        "avg_old_task_modulation_drift": float(np.mean(old_task_drifts)) if old_task_drifts else 0.0,
        "avg_block_coherence": float(np.mean(coherences)) if coherences else 0.0,
        "max_block_coherence": float(np.max(coherences)) if coherences else 0.0,
        "avg_rotation_deviation": float(np.mean(rot_deviations)) if rot_deviations else 0.0,
        "per_module_metrics": module_metrics,
    }

    summary_path = os.path.join(ums_dir, "ums_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(serialize_for_json(summary), f, indent=2)

    return summary


def save_focal_experts(
    model: nn.Module,
    focal_dir: str,
    task_id: int
) -> Dict[str, Any]:
    """Saves FocalExpert state (task-partitioned coeff) and generates focal_summary.json."""
    bilora_modules = find_bilora_focal_modules(model)
    if not bilora_modules:
        bilora_modules = find_bilora_moe_modules(model)
    module_metrics = {}
    for layer_name, bilora in bilora_modules:
        for attr in ['focal_q', 'focal_k', 'focal_v', 'focal_proj', 'moe_k', 'moe_v']:
            mod = getattr(bilora, attr, None)
            if mod is not None and hasattr(mod, 'get_analysis_metrics'):
                if hasattr(mod, 'save_state'):
                    try:
                        mod.save_state(focal_dir, task_id=task_id)
                    except Exception:
                        pass
                module_metrics[mod.name] = mod.get_analysis_metrics(task_id=task_id)
    summary = {
        'task_id': int(task_id),
        'total_focal_modules': len(module_metrics),
        'per_module_metrics': module_metrics,
    }
    summary_path = os.path.join(focal_dir, "focal_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(serialize_for_json(summary), f, indent=2)
    return summary


# Backward compatibility alias
def save_moe_and_experts(model, moe_dir, task_id):
    return save_focal_experts(model, moe_dir, task_id)


def save_model_checkpoint(
    model: nn.Module,
    model_dir: str,
    task_id: int,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    eval_results: Optional[Any] = None,
    loss_history: Optional[Any] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Saves the complete model checkpoint to model/model_checkpoint.pt.

    Includes model state_dict, optimizer, scheduler, per-task linear heads (fcArr, classifierArr),
    Mahalanobis Gaussian distribution parameters, and training metadata.
    """
    ckpt_path = os.path.join(model_dir, "model_checkpoint.pt")

    # Serialize per-task heads and distributions if present (e.g. LEAR)
    fc_heads_state = []
    classifier_heads_state = []
    distributions_data = []

    if hasattr(model, 'net'):
        if hasattr(model.net, 'fcArr'):
            for fc in model.net.fcArr:
                fc_heads_state.append(copy.deepcopy(fc.state_dict()))
        if hasattr(model.net, 'classifierArr'):
            for cls in model.net.classifierArr:
                classifier_heads_state.append(copy.deepcopy(cls.state_dict()))
        if hasattr(model.net, 'distributions'):
            for dist in model.net.distributions:
                distributions_data.append({
                    "mean": dist.mean.detach().cpu(),
                    "covariance_matrix": dist.covariance_matrix.detach().cpu(),
                })

    args_dict = {}
    if hasattr(model, 'args') and hasattr(model.args, '__dict__'):
        args_dict = serialize_for_json(vars(model.args))

    checkpoint = {
        "task_id": int(task_id),
        "timestamp": datetime.datetime.now().isoformat(),
        "model_state_dict": model.state_dict(),
        "fc_heads_state_dict": fc_heads_state,
        "classifier_heads_state_dict": classifier_heads_state,
        "distributions": distributions_data,
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "eval_results": serialize_for_json(eval_results),
        "loss_history": serialize_for_json(loss_history),
        "args": args_dict,
        "extra_metadata": serialize_for_json(extra_metadata or {}),
    }

    torch.save(checkpoint, ckpt_path)
    return ckpt_path


def update_overall_analysis(
    run_dir: str,
    task_id: int,
    task_summary: Dict[str, Any],
    model_args: Optional[Any] = None
) -> str:
    """Updates overall_analysis.json at run level with cross-task trajectory data."""
    analysis_path = os.path.join(run_dir, "overall_analysis.json")

    overall: Dict[str, Any] = {
        "run_name": os.path.basename(run_dir),
        "last_updated": datetime.datetime.now().isoformat(),
        "args": serialize_for_json(vars(model_args)) if model_args is not None and hasattr(model_args, '__dict__') else {},
        "tasks": {},
        "trajectories": {
            "tasks": [],
            "avg_basis_rank_used": [],
            "avg_capacity_used_pct": [],
            "avg_relative_recon_error": [],
            "avg_block_coherence": [],
            "avg_masked_experts": [],
            "eval_accuracies": [],
        }
    }

    if os.path.exists(analysis_path):
        try:
            with open(analysis_path, 'r', encoding='utf-8') as f:
                loaded = json.load(f)
                if isinstance(loaded, dict):
                    overall.update(loaded)
        except Exception:
            pass

    overall["last_updated"] = datetime.datetime.now().isoformat()
    overall["tasks"][f"task_{task_id}"] = task_summary

    # Recompute progression trajectories sorted by task ID
    task_keys = sorted(overall["tasks"].keys(), key=lambda k: int(k.split('_')[-1]))
    trajectories = {
        "tasks": [],
        "avg_basis_rank_used": [],
        "avg_capacity_used_pct": [],
        "avg_relative_recon_error": [],
        "avg_block_coherence": [],
        "avg_masked_experts": [],
        "eval_accuracies": [],
    }

    for k in task_keys:
        t_data = overall["tasks"][k]
        t_id = t_data.get("task_id", int(k.split('_')[-1]))
        trajectories["tasks"].append(t_id)

        ums_sum = t_data.get("ums_summary", {})
        trajectories["avg_basis_rank_used"].append(ums_sum.get("avg_basis_rank_used", 0.0))
        trajectories["avg_capacity_used_pct"].append(ums_sum.get("avg_capacity_used_pct", 0.0))
        trajectories["avg_relative_recon_error"].append(ums_sum.get("avg_relative_recon_error", 0.0))
        trajectories["avg_block_coherence"].append(ums_sum.get("avg_block_coherence", 0.0))

        focal_sum = t_data.get('focal_summary', t_data.get('moe_summary', {}))
        trajectories["avg_masked_experts"].append(focal_sum.get('total_focal_modules', 0))

        trajectories["eval_accuracies"].append(t_data.get("eval_results", None))

    overall["trajectories"] = trajectories

    with open(analysis_path, 'w', encoding='utf-8') as f:
        json.dump(serialize_for_json(overall), f, indent=2)

    return analysis_path


def save_task_checkpoint(
    model: nn.Module,
    task_id: int,
    base_dir: str = "checkpoints",
    run_name: Optional[str] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    eval_results: Optional[Any] = None,
    loss_history: Optional[Any] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    include_readout: bool = True,
    verbose: bool = True
) -> str:
    """Master saving function: saves UMS Consolidator, FocalExpert state, and Model state checkpoint.

    Args:
        model: The continual model (e.g. LEAR).
        task_id: The integer task index (e.g. 0, 1, ...).
        base_dir: Root directory for checkpoints (default: "checkpoints").
        run_name: Optional identifier for the run/experiment. If None, derived from args.
        optimizer: Optional optimizer to save state.
        scheduler: Optional scheduler to save state.
        eval_results: Optional accuracy/test results from evaluation.
        loss_history: Optional training loss records.
        extra_metadata: Optional dictionary of additional metadata.
        include_readout: Whether to compute and save reconstructed modulation matrices.
        verbose: Whether to print summary log to console.

    Returns:
        The path to the saved task directory.
    """
    # Callers in the training loop may save the same task again after
    # evaluation.  Reconstruct router metadata here so that this later save
    # does not overwrite the diagnostics collected by the model hook.
    resolved_extra_metadata = dict(extra_metadata or {})
    task_retrieval = getattr(model, 'task_retrieval', None)
    if task_retrieval is not None:
        resolved_extra_metadata.setdefault('task_retrieval', task_retrieval)
    quantum_router = getattr(getattr(model, 'net', None), 'quantum_task_router', None)
    if quantum_router is not None and hasattr(quantum_router, 'get_analysis_metrics'):
        resolved_extra_metadata['quantum_router'] = quantum_router.get_analysis_metrics()

    # 1. Resolve run name
    if run_name is None:
        if hasattr(model, 'args') and model.args is not None:
            ckpt_name = getattr(model.args, 'ckpt_name', None)
            if ckpt_name:
                run_name = str(ckpt_name)
            else:
                m_name = getattr(model.args, 'model', 'model')
                d_name = getattr(model.args, 'dataset', 'dataset')
                run_name = f"{m_name}_{d_name}"
        else:
            run_name = "focal_ums_run"

    # 2. Form directories
    dirs = get_or_create_checkpoint_dirs(base_dir, run_name, task_id)
    run_dir = dirs["run_dir"]
    task_dir = dirs["task_dir"]
    model_dir = dirs["model_dir"]
    ums_dir = dirs["ums_dir"]
    focal_dir = dirs["focal_dir"]
    moe_dir = dirs.get("moe_dir", os.path.join(task_dir, "moe"))

    # 3. Save run metadata if not exists
    meta_path = os.path.join(run_dir, "run_metadata.json")
    if not os.path.exists(meta_path):
        run_meta = {
            "run_name": run_name,
            "created_at": datetime.datetime.now().isoformat(),
            "model_class": model.__class__.__name__,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "args": serialize_for_json(vars(model.args)) if hasattr(model, 'args') and hasattr(model.args, '__dict__') else {},
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(run_meta, f, indent=2)

    # 4. Save 1) UMS Consolidator
    ums_summary = save_ums_consolidators(
        model=model,
        ums_dir=ums_dir,
        task_id=task_id,
        include_readout=include_readout
    )

    # 5. Save 2) FocalExpert state
    focal_summary = save_focal_experts(
        model=model,
        focal_dir=focal_dir,
        task_id=task_id
    )
    if moe_dir and moe_dir != focal_dir:
        save_focal_experts(
            model=model,
            focal_dir=moe_dir,
            task_id=task_id
        )
        moe_summary_path = os.path.join(moe_dir, "moe_summary.json")
        with open(moe_summary_path, 'w', encoding='utf-8') as f:
            json.dump(serialize_for_json(focal_summary), f, indent=2)

    # 6. Save 3) Model State Checkpoint
    model_ckpt_path = save_model_checkpoint(
        model=model,
        model_dir=model_dir,
        task_id=task_id,
        optimizer=optimizer,
        scheduler=scheduler,
        eval_results=eval_results,
        loss_history=loss_history,
        extra_metadata=resolved_extra_metadata
    )

    # 7. Write task_summary.json
    task_summary = {
        "task_id": int(task_id),
        "timestamp": datetime.datetime.now().isoformat(),
        "model_checkpoint_file": os.path.relpath(model_ckpt_path, task_dir),
        "eval_results": serialize_for_json(eval_results),
        "loss_history": serialize_for_json(loss_history),
        "ums_summary": ums_summary,
        "focal_summary": focal_summary,
        "moe_summary": focal_summary,
        "extra_metadata": serialize_for_json(resolved_extra_metadata),
    }
    task_summary_path = os.path.join(task_dir, "task_summary.json")
    with open(task_summary_path, 'w', encoding='utf-8') as f:
        json.dump(serialize_for_json(task_summary), f, indent=2)

    # 8. Update run-level overall_analysis.json
    update_overall_analysis(
        run_dir=run_dir,
        task_id=task_id,
        task_summary=task_summary,
        model_args=getattr(model, 'args', None)
    )

    if verbose:
        print("\n" + "=" * 80)
        print(f"[CHECKPOINT] Task {task_id} successfully saved to: {task_dir}")
        print(f"  ├── 1) Model: {os.path.basename(model_ckpt_path)}")
        print(f"  ├── 2) UMS:   {ums_summary['total_ums_consolidators']} consolidators saved (avg rank used: {ums_summary['avg_basis_rank_used']:.1f}/{ums_summary['basis_capacity_R']}, avg recon error: {ums_summary['avg_relative_recon_error']:.2%}, avg coherence: {ums_summary['avg_block_coherence']:.3e})")
        print(f"  └── 3) Focal: {focal_summary['total_focal_modules']} focal modules saved")
        print("=" * 80 + "\n")

    return task_dir


def load_task_checkpoint(
    checkpoint_path_or_task_dir: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Loads a saved task checkpoint, restoring model weights, per-task heads, distributions, and UMS states.

    Args:
        checkpoint_path_or_task_dir: Path to model_checkpoint.pt or the task directory.
        model: Target model to load weights into.
        optimizer: Optional optimizer to restore state.
        device: Device to load tensors onto.

    Returns:
        The loaded checkpoint dictionary.
    """
    if os.path.isdir(checkpoint_path_or_task_dir):
        ckpt_path = os.path.join(checkpoint_path_or_task_dir, "model", "model_checkpoint.pt")
    else:
        ckpt_path = checkpoint_path_or_task_dir

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    target_device = device or getattr(model, 'device', 'cpu')
    try:
        saved_obj = torch.load(ckpt_path, map_location=target_device, weights_only=False)
    except TypeError:  # PyTorch < 2.0
        saved_obj = torch.load(ckpt_path, map_location=target_device)

    # 1. Rebuild variable-size per-task heads before loading the complete
    # state_dict.  A freshly constructed domain model only owns one head, and
    # that head can have a different number of output classes from the first
    # serialized head.  ``strict=False`` does not ignore tensor shape
    # mismatches, so every saved head must have its checkpoint shape first.
    if hasattr(model, 'net'):
        if "fc_heads_state_dict" in saved_obj and hasattr(model.net, 'fcArr'):
            saved_fcs = saved_obj["fc_heads_state_dict"]
            rebuilt_fcs = nn.ModuleList()
            for state in saved_fcs:
                out_features, in_features = state["weight"].shape
                rebuilt_fcs.append(
                    nn.Linear(
                        in_features,
                        out_features,
                        bias="bias" in state,
                        device=target_device,
                    )
                )
            if rebuilt_fcs:
                model.net.fcArr = rebuilt_fcs

        if "classifier_heads_state_dict" in saved_obj and hasattr(model.net, 'classifierArr'):
            saved_clss = saved_obj["classifier_heads_state_dict"]
            rebuilt_classifiers = nn.ModuleList()
            for state in saved_clss:
                out_features, in_features = state["weight"].shape
                rebuilt_classifiers.append(
                    nn.Linear(
                        in_features,
                        out_features,
                        bias="bias" in state,
                        device=target_device,
                    )
                )
            if rebuilt_classifiers:
                model.net.classifierArr = rebuilt_classifiers
                model.net.classifier = model.net.classifierArr[-1]
                if hasattr(model.net, 'c_expert'):
                    model.net.c_expert = len(model.net.classifierArr) - 1

    # 2. Load all regular model parameters after the dynamic modules have the
    # correct shapes.  Missing buffers from newer code remain backward
    # compatible through strict=False.
    model.load_state_dict(saved_obj["model_state_dict"], strict=False)

    # Explicitly restore the per-task heads as an authoritative fallback for
    # older checkpoints whose main model_state_dict omitted ModuleList items.
    if hasattr(model, 'net'):
        if "fc_heads_state_dict" in saved_obj and hasattr(model.net, 'fcArr'):
            for idx, state in enumerate(saved_obj["fc_heads_state_dict"]):
                model.net.fcArr[idx].load_state_dict(state)
                model.net.fcArr[idx].to(target_device)

        if "classifier_heads_state_dict" in saved_obj and hasattr(model.net, 'classifierArr'):
            for idx, state in enumerate(saved_obj["classifier_heads_state_dict"]):
                model.net.classifierArr[idx].load_state_dict(state)
                model.net.classifierArr[idx].to(target_device)

        # 3. Restore Mahalanobis distributions
        if "distributions" in saved_obj and hasattr(model.net, 'distributions'):
            from torch.distributions import MultivariateNormal
            model.net.distributions = []
            for d in saved_obj["distributions"]:
                mu = d["mean"].to(target_device)
                cov = d["covariance_matrix"].to(target_device)
                model.net.distributions.append(MultivariateNormal(mu, cov))


    # 4. Restore UMS consolidator states (P, Q, W_rot, z_store, r_cur)
    task_dir = os.path.dirname(ckpt_path) if os.path.isfile(ckpt_path) else checkpoint_path_or_task_dir
    if os.path.basename(task_dir) == 'model':
        task_dir = os.path.dirname(task_dir)
    ums_dir = os.path.join(task_dir, 'ums') if os.path.isdir(os.path.join(task_dir, 'ums')) else None

    # Map each UMS module to its attribute path in model
    ums_path_map = {}
    for mod_path, m in model.named_modules():
        if getattr(m, '__class__', None) and m.__class__.__name__ == 'UMSConsolidator':
            ums_path_map[m] = mod_path

    bilora_modules = find_bilora_focal_modules(model)
    if not bilora_modules:
        bilora_modules = find_bilora_moe_modules(model)

    model_state = saved_obj.get("model_state_dict", {})

    for _, bilora in bilora_modules:
        for attr in ['focal_q', 'focal_k', 'focal_v', 'focal_proj', 'moe_k', 'moe_v']:
            mod = getattr(bilora, attr, None)
            if mod is not None and hasattr(mod, 'ums') and mod.ums is not None:
                ums = mod.ums
                loaded_from_pt = False
                # Option A: Load from dedicated .pt in ums/ folder
                if ums_dir is not None:
                    pt_file = os.path.join(ums_dir, f"{ums.name}_ums.pt")
                    if os.path.exists(pt_file):
                        try:
                            ums.load_state(pt_file, device=target_device)
                            loaded_from_pt = True
                        except Exception as e:
                            print(f"[WARNING] Could not load {pt_file}: {e}")

                # Option B: Fallback to reconstructing z_store from model_state_dict
                if not loaded_from_pt:
                    mod_path = ums_path_map.get(ums, "")
                    if mod_path:
                        z_keys = [k for k in model_state if k.startswith(f"{mod_path}.z_store.")]
                    else:
                        z_keys = [k for k in model_state if f".{attr}.ums.z_store." in k or f"{ums.name}.z_store." in k]

                    if z_keys:
                        z_keys.sort(key=lambda k: int(k.split('.')[-1]))
                        ums.z_store = nn.ParameterList([
                            nn.Parameter(model_state[k].to(target_device), requires_grad=False)
                            for k in z_keys
                        ])
                    ums.r_cur = int((ums.P.norm(dim=0) > 1e-8).sum().item())

    # 5. Restore optimizer
    if optimizer is not None and saved_obj.get("optimizer_state_dict") is not None:
        try:
            optimizer.load_state_dict(saved_obj["optimizer_state_dict"])
        except Exception as e:
            print(f"[WARNING] Could not restore optimizer state: {e}")

    print(f"[INFO] Successfully loaded task {saved_obj.get('task_id')} checkpoint from: {ckpt_path}")
    return saved_obj


def load_task_analysis_summary(run_dir_or_checkpoint_dir: str) -> Dict[str, Any]:
    """Loads overall_analysis.json for research analysis and visualization."""
    if os.path.isfile(run_dir_or_checkpoint_dir):
        path = run_dir_or_checkpoint_dir
    else:
        path = os.path.join(run_dir_or_checkpoint_dir, "overall_analysis.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Analysis summary file not found: {path}")

    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
