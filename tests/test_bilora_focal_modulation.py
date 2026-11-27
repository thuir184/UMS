"""Unit tests for BiLORA_Focal with Query (Q) and Output Projection (Proj) modulation via FocalExpert."""
import os
import shutil
import tempfile
import torch
import torch.nn as nn

from backbone.bilora import BiLORA_Focal, FocalExpert
from utils.task_checkpointing import (
    find_bilora_focal_modules,
    save_ums_consolidators,
    save_focal_experts,
)


def test_bilora_focal_init_all_experts():
    dim, n_tasks, n_frq = 64, 4, 16
    attn = BiLORA_Focal(
        dim=dim, num_heads=4, n_tasks=n_tasks, n_frq=n_frq,
        modulate_q=True, modulate_proj=True,
        ums_cfg=dict(basis_rank=16, svd_rank=4, rot_steps=2)
    )
    assert attn.modulate_q is True
    assert attn.modulate_proj is True
    assert isinstance(attn.focal_q, FocalExpert)
    assert isinstance(attn.focal_k, FocalExpert)
    assert isinstance(attn.focal_v, FocalExpert)
    assert isinstance(attn.focal_proj, FocalExpert)
    assert attn.focal_q.name == "Focal-Query"
    assert attn.focal_k.name == "Focal-Key"
    assert attn.focal_v.name == "Focal-Val"
    assert attn.focal_proj.name == "Focal-Proj"
    print("[OK] test_bilora_focal_init_all_experts passed")


def test_bilora_focal_selective_modulation():
    dim, n_tasks, n_frq = 64, 4, 16
    # Only K, V (modulate_q=False, modulate_proj=False)
    attn_kv = BiLORA_Focal(
        dim=dim, num_heads=4, n_tasks=n_tasks, n_frq=n_frq,
        modulate_q=False, modulate_proj=False,
        ums_cfg=dict(basis_rank=16, svd_rank=4, rot_steps=2)
    )
    assert attn_kv.focal_q is None
    assert attn_kv.focal_k is not None
    assert attn_kv.focal_v is not None
    assert attn_kv.focal_proj is None

    # Forward works without Q and Proj
    x = torch.randn(2, 8, dim)
    out = attn_kv(x)
    assert out.shape == (2, 8, dim)
    print("[OK] test_bilora_focal_selective_modulation passed")


def test_bilora_focal_forward_and_gradient_flow():
    torch.manual_seed(42)
    dim, n_tasks, n_frq = 64, 4, 16
    attn = BiLORA_Focal(
        dim=dim, num_heads=4, n_tasks=n_tasks, n_frq=n_frq,
        modulate_q=True, modulate_proj=True,
        ums_cfg=dict(basis_rank=16, svd_rank=4, rot_steps=2)
    )

    B, N = 2, 8
    x = torch.randn(B, N, dim)

    out = attn(x)
    assert out.shape == (B, N, dim)
    assert torch.isfinite(out).all()

    # Backward pass and gradient check with non-linear loss
    loss = (out ** 2).sum()
    loss.backward()

    for name, expert in [
        ('focal_q', attn.focal_q),
        ('focal_k', attn.focal_k),
        ('focal_v', attn.focal_v),
        ('focal_proj', attn.focal_proj),
    ]:
        assert expert.coeff.grad is not None, f"Gradient for {name}.coeff is None!"
        assert torch.isfinite(expert.coeff.grad).all(), f"Gradient for {name}.coeff has inf/nan!"
        # Check current task slice received gradients
        cur_task = min(len(expert.ums.z_store), expert.n_tasks - 1)
        slice_grad = expert.coeff.grad[cur_task * expert.n_frq_per_task : (cur_task + 1) * expert.n_frq_per_task]
        assert slice_grad.abs().sum().item() > 0, f"{name}.coeff for task {cur_task} received zero gradient!"

    print("[OK] test_bilora_focal_forward_and_gradient_flow passed")


def test_bilora_focal_end_task_and_forced_eval():
    torch.manual_seed(42)
    dim, n_tasks, n_frq = 64, 3, 16
    attn = BiLORA_Focal(
        dim=dim, num_heads=4, n_tasks=n_tasks, n_frq=n_frq,
        modulate_q=True, modulate_proj=True,
        ums_cfg=dict(basis_rank=16, svd_rank=4, rot_steps=2)
    )

    B, N = 2, 6
    x = torch.randn(B, N, dim)

    # Train and fold task 0
    attn(x)
    attn.end_task(0)

    for name, expert in [
        ('focal_q', attn.focal_q),
        ('focal_k', attn.focal_k),
        ('focal_v', attn.focal_v),
        ('focal_proj', attn.focal_proj),
    ]:
        assert len(expert.ums.z_store) == 1, f"{name} ums did not fold task 0!"
        assert 0 in expert.task_assignment_history, f"{name} task_assignment_history missing task 0!"

    # Forced task evaluation mode
    attn.set_force_task(0)
    out_eval = attn(x)
    assert out_eval.shape == (B, N, dim)
    assert torch.isfinite(out_eval).all()

    attn.clear_force_task()

    # Train and fold task 1
    attn(x)
    attn.end_task(1)

    for name, expert in [
        ('focal_q', attn.focal_q),
        ('focal_k', attn.focal_k),
        ('focal_v', attn.focal_v),
        ('focal_proj', attn.focal_proj),
    ]:
        assert len(expert.ums.z_store) == 2, f"{name} ums did not fold task 1!"

    # Evaluate task 0 and task 1 separately
    attn.set_force_task(0)
    out_t0 = attn(x)
    attn.set_force_task(1)
    out_t1 = attn(x)
    attn.clear_force_task()

    assert not torch.allclose(out_t0, out_t1), "Outputs for task 0 and task 1 under forced eval should differ!"
    print("[OK] test_bilora_focal_end_task_and_forced_eval passed")


def test_bilora_focal_checkpointing():
    torch.manual_seed(42)
    dim, n_tasks, n_frq = 64, 2, 16

    class DummyBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.attn = BiLORA_Focal(
                dim=dim, num_heads=4, n_tasks=n_tasks, n_frq=n_frq,
                layer_name="Block0",
                modulate_q=True, modulate_proj=True,
                ums_cfg=dict(basis_rank=16, svd_rank=4, rot_steps=2)
            )

    class DummyViT(nn.Module):
        def __init__(self):
            super().__init__()
            self.blocks = nn.ModuleList([DummyBlock()])

    class DummyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.local_vitmodel = DummyViT()

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = DummyNet()
            self.trainable_blocks = [0]

    model = DummyModel()
    attn = model.net.local_vitmodel.blocks[0].attn

    # Fold task 0
    x = torch.randn(2, 6, dim)
    attn(x)
    attn.end_task(0)

    temp_dir = tempfile.mkdtemp()
    try:
        ums_dir = os.path.join(temp_dir, "ums")
        focal_dir = os.path.join(temp_dir, "focal")
        os.makedirs(ums_dir, exist_ok=True)
        os.makedirs(focal_dir, exist_ok=True)

        ums_sum = save_ums_consolidators(model, ums_dir, task_id=0)
        focal_sum = save_focal_experts(model, focal_dir, task_id=0)

        # 4 modules (focal_q, focal_k, focal_v, focal_proj)
        assert ums_sum["total_ums_consolidators"] == 4, f"Expected 4 UMS consolidators, got {ums_sum['total_ums_consolidators']}"
        assert focal_sum["total_focal_modules"] == 4, f"Expected 4 Focal modules, got {focal_sum['total_focal_modules']}"

        expected_names = {"Block0-Query", "Block0-Key", "Block0-Val", "Block0-Proj"}
        assert set(ums_sum["per_module_metrics"].keys()) == expected_names
        assert set(focal_sum["per_module_metrics"].keys()) == expected_names

        # Check analysis metrics
        metrics = attn.get_analysis_metrics(0)
        assert 'focal_q' in metrics and 'focal_proj' in metrics
        assert 'ums_q' in metrics and 'ums_proj' in metrics
    finally:
        shutil.rmtree(temp_dir)

    print("[OK] test_bilora_focal_checkpointing passed")


if __name__ == "__main__":
    test_bilora_focal_init_all_experts()
    test_bilora_focal_selective_modulation()
    test_bilora_focal_forward_and_gradient_flow()
    test_bilora_focal_end_task_and_forced_eval()
    test_bilora_focal_checkpointing()
    print("\nALL BILORA FOCAL MODULATION TESTS PASSED!")
