"""CPU tests for the fixed quantum feature map and recursive ridge router."""

import torch

from models.utils.quantum_task_router import FixedQuantumFeatureMap, QuantumRLSTaskRouter


def test_quantum_feature_map_is_fixed_and_finite():
    torch.manual_seed(3)
    x = torch.randn(5, 12)
    fmap_a = FixedQuantumFeatureMap(input_dim=12, n_qubits=3, n_layers=2, seed=7)
    fmap_b = FixedQuantumFeatureMap(input_dim=12, n_qubits=3, n_layers=2, seed=7)

    phi_a = fmap_a(x)
    phi_b = fmap_b(x)
    assert phi_a.shape == (5, 10)  # 3 observables/qubit + bias
    assert torch.isfinite(phi_a).all()
    assert torch.allclose(phi_a, phi_b, atol=1e-6)
    assert torch.allclose(phi_a[:, -1], torch.ones(5))
    assert (phi_a[:, :-1].abs() <= 1.0 + 1e-6).all()


def test_quantum_rls_updates_and_roundtrips_state():
    torch.manual_seed(11)
    router = QuantumRLSTaskRouter(
        input_dim=12,
        n_tasks=3,
        n_qubits=3,
        n_layers=1,
        ridge=1e-2,
        seed=5,
        balance_tasks=False,
    )

    x0 = torch.randn(7, 12) - 1.0
    phi0 = router.feature_map(x0).double()
    router.update_batch(x0, task_id=0)
    metrics0 = router.finalize_task(0)
    assert metrics0["samples"] == 7
    assert router.scores(torch.randn(2, 12)).shape == (2, 1)

    x1 = torch.randn(9, 12) + 1.0
    phi1 = router.feature_map(x1).double()
    router.update_batch(x1, task_id=1)
    metrics1 = router.finalize_task(1)
    assert metrics1["samples"] == 9

    # Recursive sufficient statistics must reproduce the exact closed-form
    # ridge solution over all samples observed so far.
    phi = torch.cat((phi0, phi1), dim=0)
    targets = torch.zeros(phi.shape[0], 3, dtype=torch.float64)
    targets[: len(phi0), 0] = 1.0
    targets[len(phi0) :, 1] = 1.0
    expected = torch.linalg.solve(
        1e-2 * torch.eye(router.feature_dim, dtype=torch.float64) + phi.t() @ phi,
        phi.t() @ targets,
    )
    assert torch.allclose(router.weights, expected, atol=1e-9)

    distances = router.distances(torch.randn(4, 12))
    assert len(distances) == 2
    assert all(isinstance(value, float) for value in distances)

    restored = QuantumRLSTaskRouter(
        input_dim=12,
        n_tasks=3,
        n_qubits=3,
        n_layers=1,
        ridge=1e-2,
        seed=5,
        balance_tasks=False,
    )
    restored.load_state_dict(router.state_dict())
    probe = torch.randn(3, 12)
    assert torch.allclose(router.scores(probe), restored.scores(probe), atol=1e-6)
    assert restored.tasks_seen.item() == 2
    assert restored.sample_counts.tolist() == [7, 9, 0]


def test_task_balanced_rls_matches_closed_form():
    torch.manual_seed(17)
    router = QuantumRLSTaskRouter(
        input_dim=12,
        n_tasks=2,
        n_qubits=3,
        n_layers=1,
        ridge=1e-2,
        seed=9,
        balance_tasks=True,
    )
    x0 = torch.randn(3, 12) - 0.5
    x1 = torch.randn(11, 12) + 0.5
    phi0 = router.feature_map(x0).double()
    phi1 = router.feature_map(x1).double()

    router.update_batch(x0, task_id=0)
    router.finalize_task(0)
    router.update_batch(x1, task_id=1)
    router.finalize_task(1)

    expected_gram = (
        1e-2 * torch.eye(router.feature_dim, dtype=torch.float64)
        + phi0.t() @ phi0 / len(phi0)
        + phi1.t() @ phi1 / len(phi1)
    )
    expected_cross = torch.stack(
        (phi0.mean(dim=0), phi1.mean(dim=0)), dim=1
    )
    expected_weights = torch.linalg.solve(expected_gram, expected_cross)
    assert torch.allclose(router.gram, expected_gram, atol=1e-9)
    assert torch.allclose(router.weights, expected_weights, atol=1e-9)

