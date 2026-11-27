"""Fixed quantum feature map with an information-form recursive ridge router.

The quantum circuit is deliberately a feature generator only.  It has no
trainable parameters, so the feature space remains fixed over the continual
stream.  In task-balanced mode, the router keeps per-task ridge sufficient
statistics and solves

    A = lambda I + sum_t (Phi_t^T Phi_t) / N_t
    B[:, t] = (Phi_t^T 1) / N_t

through A W = B after each task.  Thus every task contributes equal total
weight even when its dataset contains many more samples.  The original
sample-weighted ridge system remains available for controlled comparison.
No previous images are retained.

This implementation uses a small exact state-vector simulator written in
PyTorch.  It therefore needs no external quantum package and is intended for
8--16 qubit experiments, with 8 qubits as the inexpensive first test.
"""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedQuantumFeatureMap(nn.Module):
    """Map frozen classical features to local quantum observables.

    The fixed pipeline is:

        per-sample layer norm -> fixed orthogonal reduction -> angle encoding
        -> shallow RY/RZ + ring-CNOT circuit -> Z, X and nearest-neighbour ZZ.

    Local observables are used instead of a global fidelity kernel to keep the
    output dimension fixed and avoid constructing an O(N^2) Gram matrix.
    """

    def __init__(
        self,
        input_dim: int = 768,
        n_qubits: int = 8,
        n_layers: int = 2,
        seed: int = 0,
        shots: int = 0,
    ) -> None:
        super().__init__()
        if n_qubits < 2:
            raise ValueError("quantum_n_qubits must be at least 2")
        if n_qubits > 16:
            raise ValueError(
                "The built-in exact simulator is capped at 16 qubits; "
                "use a dedicated backend for larger circuits."
            )
        if n_layers < 1:
            raise ValueError("quantum_layers must be at least 1")
        if shots < 0:
            raise ValueError("quantum_shots must be non-negative (0 means exact expectations)")

        self.input_dim = int(input_dim)
        self.n_qubits = int(n_qubits)
        self.n_layers = int(n_layers)
        self.shots = int(shots)
        self.state_dim = 1 << self.n_qubits

        # A deterministic, fixed dimensionality reducer.  QR makes its columns
        # orthonormal, and keeping it frozen is essential for exact RLS reuse.
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        reducer_raw = torch.randn(self.input_dim, self.n_qubits, generator=generator)
        reducer, _ = torch.linalg.qr(reducer_raw, mode="reduced")
        self.register_buffer("reducer", reducer.to(dtype=torch.float32))

        basis = torch.arange(self.state_dim, dtype=torch.long)
        z_signs = []
        x_permutations = []
        cnot_permutations = []
        for qubit in range(self.n_qubits):
            bit_mask = 1 << (self.n_qubits - 1 - qubit)
            bit_is_one = (basis & bit_mask) != 0
            z_signs.append(torch.where(bit_is_one, -1.0, 1.0))
            x_permutations.append(basis ^ bit_mask)

            target = (qubit + 1) % self.n_qubits
            target_mask = 1 << (self.n_qubits - 1 - target)
            cnot_perm = basis.clone()
            cnot_perm[bit_is_one] ^= target_mask
            cnot_permutations.append(cnot_perm)

        z_signs_t = torch.stack(z_signs, dim=0)
        self.register_buffer("z_signs", z_signs_t, persistent=False)
        self.register_buffer(
            "zz_signs",
            z_signs_t * torch.roll(z_signs_t, shifts=-1, dims=0),
            persistent=False,
        )
        self.register_buffer(
            "x_permutations", torch.stack(x_permutations, dim=0), persistent=False
        )
        self.register_buffer(
            "cnot_permutations", torch.stack(cnot_permutations, dim=0), persistent=False
        )

        # Z_i, X_i, Z_i Z_{i+1}, followed by a constant bias feature.
        self.observable_dim = 3 * self.n_qubits
        self.output_dim = self.observable_dim + 1

    def _apply_ry(self, state: torch.Tensor, theta: torch.Tensor, qubit: int) -> torch.Tensor:
        batch = state.shape[0]
        left = 1 << qubit
        right = 1 << (self.n_qubits - qubit - 1)
        state_view = state.reshape(batch, left, 2, right)
        amp0, amp1 = state_view[:, :, 0, :], state_view[:, :, 1, :]
        cos = torch.cos(theta / 2).reshape(batch, 1, 1)
        sin = torch.sin(theta / 2).reshape(batch, 1, 1)
        out0 = cos * amp0 - sin * amp1
        out1 = sin * amp0 + cos * amp1
        return torch.stack((out0, out1), dim=2).reshape(batch, self.state_dim)

    def _apply_rz(self, state: torch.Tensor, theta: torch.Tensor, qubit: int) -> torch.Tensor:
        batch = state.shape[0]
        left = 1 << qubit
        right = 1 << (self.n_qubits - qubit - 1)
        state_view = state.reshape(batch, left, 2, right)
        phase0 = torch.exp((-0.5j) * theta).reshape(batch, 1, 1)
        phase1 = torch.exp((0.5j) * theta).reshape(batch, 1, 1)
        out0 = phase0 * state_view[:, :, 0, :]
        out1 = phase1 * state_view[:, :, 1, :]
        return torch.stack((out0, out1), dim=2).reshape(batch, self.state_dim)

    def _sample_expectations(self, expectations: torch.Tensor) -> torch.Tensor:
        if self.shots <= 0:
            return expectations
        probability_plus = ((expectations + 1.0) / 2.0).clamp(0.0, 1.0)
        total = torch.full_like(probability_plus, float(self.shots))
        plus_counts = torch.binomial(total, probability_plus)
        return (2.0 * plus_counts / float(self.shots)) - 1.0

    @torch.no_grad()
    def forward(self, frozen_features: torch.Tensor) -> torch.Tensor:
        if frozen_features.ndim != 2 or frozen_features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected frozen features [B, {self.input_dim}], got {tuple(frozen_features.shape)}"
            )

        normalized = F.layer_norm(frozen_features.float(), (self.input_dim,))
        reduced = normalized @ self.reducer.to(device=normalized.device)
        angles = math.pi * torch.tanh(reduced)

        # H^{\otimes n}|0> is the uniform superposition.  Starting from it avoids
        # materialising n separate constant Hadamard gate applications.
        amplitude = 1.0 / math.sqrt(float(self.state_dim))
        state = torch.full(
            (angles.shape[0], self.state_dim),
            complex(amplitude, 0.0),
            dtype=torch.complex64,
            device=angles.device,
        )

        for layer in range(self.n_layers):
            scale = float(layer + 1)
            for qubit in range(self.n_qubits):
                theta = scale * angles[:, qubit]
                state = self._apply_ry(state, theta, qubit)
                state = self._apply_rz(state, theta, qubit)
            for qubit in range(self.n_qubits):
                permutation = self.cnot_permutations[qubit].to(device=state.device)
                state = state.index_select(1, permutation)

        probabilities = state.abs().square().real
        z = probabilities @ self.z_signs.to(device=state.device, dtype=probabilities.dtype).t()
        zz = probabilities @ self.zz_signs.to(device=state.device, dtype=probabilities.dtype).t()

        x_values = []
        for qubit in range(self.n_qubits):
            permutation = self.x_permutations[qubit].to(device=state.device)
            flipped = state.index_select(1, permutation)
            x_values.append((state.conj() * flipped).sum(dim=1).real)
        x = torch.stack(x_values, dim=1)

        observables = torch.cat((z, x, zz), dim=1).clamp(-1.0, 1.0)
        observables = self._sample_expectations(observables)
        observables = F.normalize(observables, p=2, dim=1, eps=1e-8)
        bias = torch.ones(observables.shape[0], 1, device=observables.device)
        return torch.cat((observables, bias), dim=1)


class QuantumRLSTaskRouter(nn.Module):
    """Task-ID classifier using fixed quantum features and exact ridge statistics."""

    def __init__(
        self,
        input_dim: int,
        n_tasks: int,
        n_qubits: int = 8,
        n_layers: int = 2,
        ridge: float = 1e-3,
        seed: int = 0,
        shots: int = 0,
        balance_tasks: bool = True,
    ) -> None:
        super().__init__()
        if n_tasks < 1:
            raise ValueError("n_tasks must be positive")
        if ridge <= 0:
            raise ValueError("quantum_ridge must be positive")

        self.n_tasks = int(n_tasks)
        self.ridge = float(ridge)
        self.balance_tasks = bool(balance_tasks)
        self.feature_map = FixedQuantumFeatureMap(
            input_dim=input_dim,
            n_qubits=n_qubits,
            n_layers=n_layers,
            seed=seed,
            shots=shots,
        )
        feature_dim = self.feature_map.output_dim

        # Float64 statistics make the small linear solve more stable.  Fixed
        # shapes ensure ordinary state_dict checkpoints restore without custom
        # loading code, even when only a prefix of tasks has been observed.
        self.register_buffer(
            "task_grams",
            torch.zeros(self.n_tasks, feature_dim, feature_dim, dtype=torch.float64),
        )
        self.register_buffer("gram", torch.eye(feature_dim, dtype=torch.float64) * self.ridge)
        self.register_buffer("cross", torch.zeros(feature_dim, self.n_tasks, dtype=torch.float64))
        self.register_buffer("weights", torch.zeros(feature_dim, self.n_tasks, dtype=torch.float64))
        self.register_buffer("sample_counts", torch.zeros(self.n_tasks, dtype=torch.long))
        self.register_buffer("tasks_seen", torch.zeros((), dtype=torch.long))

    @property
    def feature_dim(self) -> int:
        return self.feature_map.output_dim

    @torch.no_grad()
    def update_batch(self, frozen_features: torch.Tensor, task_id: int) -> None:
        task_id = int(task_id)
        if task_id < 0 or task_id >= self.n_tasks:
            raise IndexError(f"task_id={task_id} is outside [0, {self.n_tasks - 1}]")
        phi = self.feature_map(frozen_features).to(dtype=torch.float64)
        self.task_grams[task_id].add_(phi.t() @ phi)
        # All examples in this end-of-task pass have one-hot target task_id.
        self.cross[:, task_id].add_(phi.sum(dim=0))
        self.sample_counts[task_id].add_(phi.shape[0])

    @torch.no_grad()
    def reset_statistics(self) -> None:
        """Clear learned routing statistics while preserving the fixed feature map."""
        self.task_grams.zero_()
        self.gram.copy_(
            torch.eye(self.feature_dim, dtype=torch.float64, device=self.gram.device)
            * self.ridge
        )
        self.cross.zero_()
        self.weights.zero_()
        self.sample_counts.zero_()
        self.tasks_seen.zero_()

    @torch.no_grad()
    def _build_ridge_system(self) -> torch.Tensor:
        """Rebuild A and B from per-task sufficient statistics.

        With task balancing, every task contributes its mean squared loss:

            A = lambda I + sum_t (Phi_t^T Phi_t) / N_t
            B_t = (Phi_t^T 1) / N_t

        Setting ``balance_tasks=False`` recovers ordinary sample-weighted ridge.
        """
        system = (
            torch.eye(self.feature_dim, dtype=torch.float64, device=self.gram.device)
            * self.ridge
        )
        rhs = torch.zeros_like(self.cross)
        for task_id in range(self.n_tasks):
            count = int(self.sample_counts[task_id].item())
            if count == 0:
                continue
            scale = (1.0 / float(count)) if self.balance_tasks else 1.0
            system.add_(self.task_grams[task_id], alpha=scale)
            rhs[:, task_id].copy_(self.cross[:, task_id] * scale)
        self.gram.copy_(system)
        return rhs

    @torch.no_grad()
    def finalize_task(self, task_id: int) -> Dict[str, float]:
        task_id = int(task_id)
        if self.sample_counts[task_id].item() == 0:
            raise RuntimeError(f"Cannot finalize quantum router task {task_id}: no samples accumulated")
        rhs = self._build_ridge_system()
        solution = torch.linalg.solve(self.gram, rhs)
        self.weights.copy_(solution)
        self.tasks_seen.fill_(max(int(self.tasks_seen.item()), task_id + 1))

        eigvals = torch.linalg.eigvalsh(self.gram)
        min_eig = eigvals.min().clamp_min(1e-18)
        condition = (eigvals.max() / min_eig).item()
        return {
            "task_id": float(task_id),
            "samples": float(self.sample_counts[task_id].item()),
            "tasks_seen": float(self.tasks_seen.item()),
            "feature_dim": float(self.feature_dim),
            "gram_condition": float(condition),
        }

    @torch.no_grad()
    def scores(self, frozen_features: torch.Tensor) -> torch.Tensor:
        seen = int(self.tasks_seen.item())
        if seen == 0:
            raise RuntimeError("Quantum RLS router has not been fitted to any task")
        phi = self.feature_map(frozen_features).to(dtype=torch.float64)
        return (phi @ self.weights[:, :seen]).to(dtype=torch.float32)

    @torch.no_grad()
    def distances(self, frozen_features: torch.Tensor) -> list[float]:
        # training_domain.evaluate minimizes distance, so negate the RLS score.
        mean_scores = self.scores(frozen_features).mean(dim=0)
        return (-mean_scores).cpu().tolist()

    def get_analysis_metrics(self) -> Dict[str, object]:
        eigvals = torch.linalg.eigvalsh(self.gram.detach())
        condition = (eigvals.max() / eigvals.min().clamp_min(1e-18)).item()
        return {
            "type": "quantum_rls",
            "n_qubits": self.feature_map.n_qubits,
            "n_layers": self.feature_map.n_layers,
            "shots": self.feature_map.shots,
            "feature_dim": self.feature_dim,
            "ridge": self.ridge,
            "balance_tasks": self.balance_tasks,
            "tasks_seen": int(self.tasks_seen.item()),
            "sample_counts": self.sample_counts.detach().cpu().tolist(),
            "gram_condition": float(condition),
        }
