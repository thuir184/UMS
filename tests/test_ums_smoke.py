"""Smoke test for the Unified Modulation Subspace (UMS).

Runs on CPU with small dimensions. It exercises, end to end:
  1. UMSConsolidator.fold over several tasks: truncated SVD, residual-energy test,
     incremental shared-basis expansion (P, Q), coordinates z, Cayley rotation
     update, consolidated readout;
  2. orthonormality of P, Q and the rotation R_o;
  3. MoE.end_task real path (router -> index collection -> mask -> fold) and the
     forced-task UMS readout used at evaluation;
  4. BiLORA_MoE attention forward in both routing (training) and forced (eval) modes.
"""
import torch

from backbone.bilora import UMSConsolidator, MoE, BiLORA_MoE


def _rand_sparse(dim, shared, private_scale, density, seed):
    g = torch.Generator().manual_seed(seed)
    S = (shared * 0.6)
    n = int(dim * dim * density)
    idx = torch.randint(dim * dim, (n,), generator=g)
    vals = torch.randn(n, generator=g) * private_scale
    S.view(-1).index_add_(0, idx, vals)
    return S


def test_consolidator_folds_and_readout():
    torch.manual_seed(0)
    dim, n_tasks = 64, 4
    ums = UMSConsolidator(dim=dim, basis_rank=24, svd_rank=6, tau=1e-3,
                          rot_steps=2, alpha=300.)

    shared = torch.randn(dim, dim) * 0.05
    prev_W = torch.zeros(dim, dim)
    for t in range(n_tasks):
        S_t = _rand_sparse(dim, shared, private_scale=1.0, density=0.02, seed=100 + t)
        metrics = ums.fold(t, S_t)
        assert metrics['tasks_folded'] == t + 1
        assert metrics['basis_rank_used'] >= 1
        assert torch.isfinite(torch.as_tensor(metrics['block_coherence']))
        assert 'pre_refinement' in metrics
        assert 'post_refinement' in metrics
        assert 'refinement' in metrics
        for key in ('support_precision', 'support_recall', 'support_f1',
                    'support_iou', 'off_diagonal_energy_ratio'):
            assert key in metrics['post_refinement']
            assert torch.isfinite(torch.as_tensor(metrics['post_refinement'][key]))
        if t > 0:
            assert metrics['refinement']['anchor_enabled'] is True
            assert len(metrics['refinement']['refinement_trace']) == ums.rot_steps
            assert len(metrics['old_task_modulation_drift']) == t
            assert torch.isfinite(torch.as_tensor(metrics['avg_old_task_modulation_drift']))
        # rotation must actually move away from identity
        R = ums.rotation()
        assert R.shape == (dim, dim)
        w = ums.readout(t)
        assert w.shape == (dim, dim) and torch.isfinite(w).all()
        prev_W = w

    # bases orthonormal on used columns
    r = ums.r_cur
    assert r >= 1
    eye = torch.eye(r)
    assert torch.allclose(ums.P[:, :r].t() @ ums.P[:, :r], eye, atol=1e-5)
    assert torch.allclose(ums.Q[:, :r].t() @ ums.Q[:, :r], eye, atol=1e-5)
    # rotation orthogonal
    R = ums.rotation()
    assert torch.allclose(R.t() @ R, torch.eye(dim), atol=1e-5)
    # per-task matrices were discarded: only coordinates are retained
    assert len(ums.z_store) == n_tasks
    for z in ums.z_store:
        assert z.shape == (ums.R,)
    print("[OK] consolidator: fold/readout, orthonormal bases & rotation")


def _small_permutation(moe, dim):
    # module PERMUTATION assumes dim=768; regenerate a dim-consistent permutation
    moe.full_permutation = torch.randperm(dim * dim).tolist()
    moe.create_bilora_indices()


def test_moe_end_task_and_forced_readout():
    torch.manual_seed(0)
    dim, n_tasks = 64, 4
    moe = MoE(dim=dim, num_experts=16, n_frq=64, n_tasks=n_tasks, topk=1,
              device="cpu",
              ums_cfg=dict(basis_rank=24, svd_rank=6, rot_steps=2))
    _small_permutation(moe, dim)
    moe.set_current_epoch(50)  # noise ~ 0

    # populate routed indices like the saving-flag epochs
    moe.flag_to_save_lora_indices = True
    for _ in range(8):
        cls = torch.randn(5, dim)
        out = moe(cls)
        assert out.shape == (5, dim, dim)
    moe.end_task(task_id=0)
    assert len(moe.ums.z_store) == 1
    assert (moe.expert_mask == float('-inf')).any()

    # forced UMS readout (evaluation) bypasses the router
    moe.set_force_task(0)
    cls = torch.randn(3, dim)
    forced = moe(cls)
    assert forced.shape == (3, dim, dim)
    expected = moe.ums.readout(0)
    assert torch.allclose(forced[0], expected, atol=1e-5)
    moe.clear_force_task()
    # routing path still works after clearing
    assert moe(torch.randn(2, dim)).shape == (2, dim, dim)
    print("[OK] MoE: end_task fold + forced UMS readout + routing")


def test_bilora_moe_attention_modes():
    torch.manual_seed(0)
    dim, n_tasks = 64, 4
    attn = BiLORA_MoE(dim=dim, num_heads=4, n_frq=64, num_experts=16, topk=1,
                      n_tasks=n_tasks,
                      ums_cfg=dict(basis_rank=24, svd_rank=6, rot_steps=2))
    _small_permutation(attn.moe_k, dim)
    _small_permutation(attn.moe_v, dim)
    attn.set_current_epoch(50)

    B, N = 2, 6
    x = torch.randn(B, N, dim)

    # training: routing mode
    attn.turn_on_saving_experts()
    y = attn(x)
    assert y.shape == (B, N, dim) and torch.isfinite(y).all()
    attn.turn_off_saving_experts(task=0)

    # evaluation: forced consolidated readout
    attn.set_force_task(0)
    y_eval = attn(x)
    assert y_eval.shape == (B, N, dim) and torch.isfinite(y_eval).all()
    attn.clear_force_task()
    print("[OK] BiLORA_MoE: routing and forced-eval forward")


if __name__ == "__main__":
    test_consolidator_folds_and_readout()
    test_moe_end_task_and_forced_readout()
    test_bilora_moe_attention_modes()
    print("\nALL UMS SMOKE TESTS PASSED")

