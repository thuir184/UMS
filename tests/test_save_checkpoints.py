"""Unit tests for UMS Consolidator, MoE, and model checkpoint saving and analysis."""

import os
import shutil
import tempfile
import json
from argparse import Namespace
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from backbone.bilora import UMSConsolidator, MoE, BiLORA_MoE
from utils.task_checkpointing import (
    save_task_checkpoint,
    load_task_checkpoint,
    load_task_analysis_summary,
    serialize_for_json,
)


def _small_permutation(moe, dim):
    moe.full_permutation = torch.randperm(dim * dim).tolist()
    moe.create_bilora_indices()


def test_ums_save_and_analysis():
    print("\n--- Running test_ums_save_and_analysis ---")
    tmpdir = tempfile.mkdtemp()
    try:
        torch.manual_seed(42)
        dim, R, r = 32, 16, 4
        ums = UMSConsolidator(dim=dim, basis_rank=R, svd_rank=r, tau=1e-3, rot_steps=2, alpha=300., name="TestBlock-Key")

        # Fold task 0
        S_0 = torch.randn(dim, dim) * 0.1
        ums.fold(0, S_0)

        # Fold task 1
        S_1 = torch.randn(dim, dim) * 0.1
        ums.fold(1, S_1)

        # Save state
        pt_path, json_path = ums.save_state(tmpdir, task_id=1, include_readout=True)
        assert os.path.exists(pt_path), f"File {pt_path} does not exist"
        assert os.path.exists(json_path), f"File {json_path} does not exist"

        # Verify .pt
        loaded_pt = torch.load(pt_path, map_location="cpu")
        assert "name" in loaded_pt and loaded_pt["name"] == "TestBlock-Key"
        assert "tensors" in loaded_pt
        tensors = loaded_pt["tensors"]
        assert "P" in tensors and "Q" in tensors and "W_rot" in tensors and "R_o" in tensors
        assert "z_store" in tensors and tensors["z_store"].shape[0] == 2
        assert "w_readout" in tensors and tensors["w_readout"].shape == (dim, dim)

        # Verify .json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data["name"] == "TestBlock-Key"
        metrics = data["metrics"]
        assert metrics["basis_capacity_R"] == R
        assert metrics["basis_rank_used_r_cur"] > 0
        assert "basis_orthonormality_error_P" in metrics
        assert "rotation_orthogonality_error" in metrics
        assert "overall_block_coherence" in metrics
        assert "coordinates_stats" in metrics
        assert "task_0" in metrics["coordinates_stats"] and "task_1" in metrics["coordinates_stats"]
        assert "coordinates_cross_similarity" in metrics
        assert "task_0_vs_task_1" in metrics["coordinates_cross_similarity"]
        assert len(metrics["fold_history"]) == 2

        # Verify load_state
        fresh_ums = UMSConsolidator(dim=dim, basis_rank=R, svd_rank=r, name="Fresh")
        fresh_ums.load_state(pt_path)
        assert fresh_ums.r_cur == ums.r_cur
        assert torch.allclose(fresh_ums.P, ums.P)
        assert len(fresh_ums.z_store) == 2

        print("[OK] test_ums_save_and_analysis passed!")
    finally:
        shutil.rmtree(tmpdir)


def test_moe_save_and_analysis():
    print("\n--- Running test_moe_save_and_analysis ---")
    tmpdir = tempfile.mkdtemp()
    try:
        torch.manual_seed(42)
        dim, num_experts, n_frq, n_tasks = 32, 8, 16, 2
        moe = MoE(dim=dim, num_experts=num_experts, n_frq=n_frq, n_tasks=n_tasks, topk=1,
                  device="cpu", ums_cfg=dict(basis_rank=16, svd_rank=4, rot_steps=2), name="TestBlock-Val")
        _small_permutation(moe, dim)
        moe.set_current_epoch(10)

        # Record indices
        moe.flag_to_save_lora_indices = True
        for _ in range(4):
            x = torch.randn(2, dim)
            _ = moe(x)

        moe.end_task(task_id=0)

        # Save state
        pt_path, json_path = moe.save_state(tmpdir, task_id=0)
        assert os.path.exists(pt_path)
        assert os.path.exists(json_path)

        # Verify .pt
        loaded_pt = torch.load(pt_path, map_location="cpu")
        assert "tensors" in loaded_pt
        assert "router_weight" in loaded_pt["tensors"]
        assert "coeff" in loaded_pt["tensors"]
        assert "expert_mask" in loaded_pt["tensors"]

        # Verify .json
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        assert data["name"] == "TestBlock-Val"
        metrics = data["metrics"]
        assert metrics["masked_experts_count"] > 0
        assert metrics["remaining_experts_count"] < num_experts
        assert "expert_coeff_norms_summary" in metrics
        assert "task_history" in metrics

        # Verify load_state
        fresh_moe = MoE(dim=dim, num_experts=num_experts, n_frq=n_frq, n_tasks=n_tasks, topk=1, device="cpu")
        fresh_moe.load_state(pt_path)
        assert torch.allclose(fresh_moe.expert_mask, moe.expert_mask)
        assert torch.allclose(fresh_moe.coeff, moe.coeff)

        print("[OK] test_moe_save_and_analysis passed!")
    finally:
        shutil.rmtree(tmpdir)


def test_bilora_moe_save_and_analysis():
    print("\n--- Running test_bilora_moe_save_and_analysis ---")
    tmpdir = tempfile.mkdtemp()
    ums_dir = os.path.join(tmpdir, "ums")
    moe_dir = os.path.join(tmpdir, "moe")
    try:
        torch.manual_seed(42)
        dim, n_tasks = 32, 2
        attn = BiLORA_MoE(dim=dim, num_heads=2, n_frq=16, num_experts=8, topk=1,
                          n_tasks=n_tasks, layer_name="Block9",
                          ums_cfg=dict(basis_rank=16, svd_rank=4, rot_steps=2))
        _small_permutation(attn.moe_k, dim)
        _small_permutation(attn.moe_v, dim)
        attn.set_current_epoch(10)

        attn.turn_on_saving_experts()
        x = torch.randn(2, 4, dim)
        _ = attn(x)
        attn.turn_off_saving_experts(task=0)

        attn.save_state(ums_dir=ums_dir, moe_dir=moe_dir, task_id=0)

        # Check files
        assert os.path.exists(os.path.join(ums_dir, "Block9-Key_ums.pt"))
        assert os.path.exists(os.path.join(ums_dir, "Block9-Key_analysis.json"))
        assert os.path.exists(os.path.join(ums_dir, "Block9-Val_ums.pt"))
        assert os.path.exists(os.path.join(ums_dir, "Block9-Val_analysis.json"))

        assert os.path.exists(os.path.join(moe_dir, "Block9-Key_moe.pt"))
        assert os.path.exists(os.path.join(moe_dir, "Block9-Key_analysis.json"))
        assert os.path.exists(os.path.join(moe_dir, "Block9-Val_moe.pt"))
        assert os.path.exists(os.path.join(moe_dir, "Block9-Val_analysis.json"))

        metrics = attn.get_analysis_metrics(task_id=0)
        assert "moe_k" in metrics and "moe_v" in metrics
        assert "ums_k" in metrics and "ums_v" in metrics
        assert metrics["layer_name"] == "Block9"

        print("[OK] test_bilora_moe_save_and_analysis passed!")
    finally:
        shutil.rmtree(tmpdir)


class DummyBlock(nn.Module):
    def __init__(self, attn):
        super().__init__()
        self.attn = attn


class DummyViT(nn.Module):
    def __init__(self, blocks):
        super().__init__()
        self.blocks = nn.ModuleList(blocks)


class DummyNet(nn.Module):
    def __init__(self, blocks, model_dim=32, fc_dim=16, num_classes=5):
        super().__init__()
        self.local_vitmodel = DummyViT(blocks)
        self.fcArr = [nn.Linear(model_dim, fc_dim)]
        self.classifierArr = [nn.Linear(fc_dim, num_classes)]
        self.distributions = []


class DummyContinualModel(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net
        self.device = 'cpu'
        self.current_task = 0
        self.trainable_blocks = [0]
        self.args = Namespace(
            model='DummyLEAR',
            dataset='seq-cifar10',
            ckpt_name='test_experiment',
            n_epochs=1,
            checkpoint_dir='checkpoints',
            disable_task_checkpoints=False,
            save_recon_matrices=False,
        )


def test_full_task_checkpoint_pipeline():
    print("\n--- Running test_full_task_checkpoint_pipeline ---")
    tmpdir = tempfile.mkdtemp()
    try:
        torch.manual_seed(42)
        dim = 32
        attn = BiLORA_MoE(dim=dim, num_heads=2, n_frq=16, num_experts=8, topk=1,
                          n_tasks=2, layer_name="Block0",
                          ums_cfg=dict(basis_rank=16, svd_rank=4, rot_steps=2))
        _small_permutation(attn.moe_k, dim)
        _small_permutation(attn.moe_v, dim)

        block = DummyBlock(attn)
        net = DummyNet([block], model_dim=dim, fc_dim=16, num_classes=5)

        # Add Gaussian distribution for task 0
        mu = torch.zeros(16)
        cov = torch.eye(16)
        net.distributions.append(MultivariateNormal(mu, cov))

        model = DummyContinualModel(net)
        opt = torch.optim.SGD(model.parameters(), lr=0.01)

        # Simulate task 0 training
        attn.turn_on_saving_experts()
        x = torch.randn(2, 4, dim)
        _ = attn(x)
        attn.turn_off_saving_experts(task=0)

        # 1. Save Task 0 Checkpoint
        saved_task_dir = save_task_checkpoint(
            model=model,
            task_id=0,
            base_dir=tmpdir,
            run_name="test_experiment",
            optimizer=opt,
            eval_results=[0.85],
            loss_history=[0.42, 0.38]
        )

        assert os.path.exists(saved_task_dir)
        run_dir = os.path.dirname(saved_task_dir)

        # Check directory structure
        assert os.path.exists(os.path.join(run_dir, "run_metadata.json"))
        assert os.path.exists(os.path.join(run_dir, "overall_analysis.json"))
        assert os.path.exists(os.path.join(saved_task_dir, "task_summary.json"))

        model_ckpt = os.path.join(saved_task_dir, "model", "model_checkpoint.pt")
        assert os.path.exists(model_ckpt)

        ums_summary = os.path.join(saved_task_dir, "ums", "ums_summary.json")
        assert os.path.exists(ums_summary)
        assert os.path.exists(os.path.join(saved_task_dir, "ums", "Block0-Key_ums.pt"))
        assert os.path.exists(os.path.join(saved_task_dir, "ums", "Block0-Key_analysis.json"))

        moe_summary = os.path.join(saved_task_dir, "moe", "moe_summary.json")
        assert os.path.exists(moe_summary)
        assert os.path.exists(os.path.join(saved_task_dir, "moe", "Block0-Key_moe.pt"))
        assert os.path.exists(os.path.join(saved_task_dir, "moe", "Block0-Key_analysis.json"))

        # 2. Simulate Task 1
        model.current_task = 1
        # Add head for task 1
        net.fcArr.append(nn.Linear(dim, 16))
        net.classifierArr.append(nn.Linear(16, 5))
        net.distributions.append(MultivariateNormal(mu + 1.0, cov))

        attn.turn_on_saving_experts()
        _ = attn(x)
        attn.turn_off_saving_experts(task=1)

        saved_task1_dir = save_task_checkpoint(
            model=model,
            task_id=1,
            base_dir=tmpdir,
            run_name="test_experiment",
            optimizer=opt,
            eval_results=[0.82, 0.88],
            loss_history=[0.31, 0.25]
        )

        assert os.path.exists(saved_task1_dir)

        # 3. Check overall_analysis.json trajectory across task 0 and task 1
        overall = load_task_analysis_summary(run_dir)
        assert "tasks" in overall
        assert "task_0" in overall["tasks"] and "task_1" in overall["tasks"]
        traj = overall["trajectories"]
        assert len(traj["tasks"]) == 2
        assert len(traj["avg_basis_rank_used"]) == 2
        assert len(traj["avg_masked_experts"]) == 2
        print("Overall Trajectory:", traj)

        # 4. Test loading checkpoint
        fresh_attn = BiLORA_MoE(dim=dim, num_heads=2, n_frq=16, num_experts=8, topk=1,
                                n_tasks=2, layer_name="Block0",
                                ums_cfg=dict(basis_rank=16, svd_rank=4))
        _small_permutation(fresh_attn.moe_k, dim)
        _small_permutation(fresh_attn.moe_v, dim)
        fresh_net = DummyNet([DummyBlock(fresh_attn)], model_dim=dim, fc_dim=16, num_classes=5)
        fresh_model = DummyContinualModel(fresh_net)

        loaded_data = load_task_checkpoint(saved_task1_dir, fresh_model)
        assert loaded_data["task_id"] == 1
        assert len(fresh_model.net.fcArr) == 2
        assert len(fresh_model.net.classifierArr) == 2
        assert len(fresh_model.net.distributions) == 2
        assert fresh_attn.moe_k.ums.r_cur > 0

        print("[OK] test_full_task_checkpoint_pipeline passed!")
    finally:
        shutil.rmtree(tmpdir)


if __name__ == "__main__":
    test_ums_save_and_analysis()
    test_moe_save_and_analysis()
    test_bilora_moe_save_and_analysis()
    test_full_task_checkpoint_pipeline()
    print("\n=========================================")
    print("ALL CHECKPOINTING TESTS PASSED!")
    print("=========================================")
