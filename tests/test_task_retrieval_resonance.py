"""Unit tests for the selectable LEAR task-retrieval strategies."""
import torch
import torch.nn as nn
from argparse import Namespace
from backbone.LEAR import LEAR_backbone
from models.LEAR import LEAR
from utils.args import ArgumentParser, add_management_args, add_configuration_args, add_experiment_args


import torchvision.transforms as transforms


class DummyDataset:
    def __init__(self, n_classes=10):
        self.N_CLASSES = n_classes
        self.N_CLASSES_PER_TASK = n_classes
        self.N_TASKS = 2
        self.SETTING = 'domain-il'
        self.train_loader = [
            (torch.randn(4, 3, 224, 224), torch.randint(0, n_classes, (4,)))
            for _ in range(2)
        ]

    def get_normalization_transform(self):
        return transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))


def test_parser_and_init():
    parser = ArgumentParser()
    LEAR.get_parser(parser)

    # Test default
    args_default = parser.parse_args(["--apply-to", "11"])
    assert args_default.task_retrieval == "mahalanobis"

    # Test resonance with hyphen
    args_res = parser.parse_args(["--apply-to", "11", "--task-retrieval", "resonance"])
    assert args_res.task_retrieval == "resonance"

    # Test resonance with underscore
    args_res_u = parser.parse_args(["--apply-to", "11", "--task_retrieval", "resonance"])
    assert args_res_u.task_retrieval == "resonance"

    # Test mahalanobis explicit
    args_mah = parser.parse_args(["--apply-to", "11", "--task-retrieval", "mahalanobis"])
    assert args_mah.task_retrieval == "mahalanobis"

    # Test fixed quantum features + recursive ridge
    args_quantum = parser.parse_args([
        "--apply-to", "11",
        "--task-retrieval", "quantum_rls",
        "--quantum-n-qubits", "8",
    ])
    assert args_quantum.task_retrieval == "quantum_rls"
    assert args_quantum.quantum_n_qubits == 8

    print("[OK] ArgumentParser tests passed.")


def test_resonance_distance_computation():
    torch.manual_seed(42)
    parser = ArgumentParser()
    add_management_args(parser)
    add_experiment_args(parser)
    add_configuration_args(parser, Namespace(dataset='seq-cifar10'))
    LEAR.get_parser(parser)

    args = parser.parse_args([
        "--apply-to", "11",
        "--lr", "1e-3",
        "--task-retrieval", "resonance",
        "--n_frq", "64",
        "--n_tasks", "2",
        "--ums_basis_rank", "16",
        "--ums_svd_rank", "4",
        "--ums_rot_steps", "0",
        "--ums_tau", "1e-3",
    ])
    setattr(args, 'modulate_q', True)
    setattr(args, 'modulate_proj', True)
    setattr(args, 'disable_task_checkpoints', True)
    setattr(args, 'lr', 1e-3)
    setattr(args, 'optimizer', 'adam')
    setattr(args, 'optim_wd', 0.0)
    setattr(args, 'optim_mom', 0.9)
    setattr(args, 'optim_nesterov', False)

    backbone = LEAR_backbone(num_classes=10)
    loss = nn.CrossEntropyLoss()
    dataset = DummyDataset(n_classes=10)

    model = LEAR(backbone=backbone, loss=loss, args=args, transform=None, dataset=dataset)
    model.to(model.device)
    assert model.task_retrieval == "resonance"

    # Simulate end of task 0
    model._current_task = 0
    # Provide synthetic gradients or coeff values so fold has non-zero matrix
    for expert in model.get_active_focal_experts():
        expert.coeff.data.normal_()

    model.end_task(dataset)
    assert len(model.net.distributions) == 0, "Resonance mode should not populate distributions!"

    # In task 0, len(fcArr) is 1
    x_test = torch.randn(2, 3, 224, 224)
    dists = model.cal_expert_dist(x_test)
    assert len(dists) == 1
    assert isinstance(dists[0], float)
    assert 0.0 <= dists[0] <= 2.0
    print(f"[OK] Task 0 distances: {dists}")

    # Simulate task 1
    model._current_task = 1
    model.begin_task(dataset)
    assert len(model.net.fcArr) == 2

    for expert in model.get_active_focal_experts():
        expert.coeff.data.normal_()

    model.end_task(dataset)
    assert len(model.net.distributions) == 0

    # In task 1, len(fcArr) is 2
    dists_task1 = model.cal_expert_dist(x_test)
    assert len(dists_task1) == 2
    for d in dists_task1:
        assert isinstance(d, float)
        assert 0.0 <= d <= 2.0
    print(f"[OK] Task 1 distances (2 tasks): {dists_task1}")

    print("[OK] Resonance distance computation test passed.")


def test_mahalanobis_mode_compatibility():
    torch.manual_seed(42)
    parser = ArgumentParser()
    add_management_args(parser)
    add_experiment_args(parser)
    add_configuration_args(parser, Namespace(dataset='seq-cifar10'))
    LEAR.get_parser(parser)

    args = parser.parse_args([
        "--apply-to", "11",
        "--lr", "1e-3",
        "--task-retrieval", "mahalanobis",
        "--n_frq", "64",
        "--n_tasks", "2",
        "--ums_basis_rank", "16",
        "--ums_svd_rank", "4",
        "--ums_rot_steps", "0",
        "--ums_tau", "1e-3",
    ])
    setattr(args, 'modulate_q', True)
    setattr(args, 'modulate_proj', True)
    setattr(args, 'disable_task_checkpoints', True)
    setattr(args, 'lr', 1e-3)
    setattr(args, 'optimizer', 'adam')
    setattr(args, 'optim_wd', 0.0)
    setattr(args, 'optim_mom', 0.9)
    setattr(args, 'optim_nesterov', False)

    backbone = LEAR_backbone(num_classes=10)
    loss = nn.CrossEntropyLoss()
    dataset = DummyDataset(n_classes=10)

    model = LEAR(backbone=backbone, loss=loss, args=args, transform=None, dataset=dataset)
    model.to(model.device)
    assert model.task_retrieval == "mahalanobis"

    # Task 0
    model._current_task = 0
    model.end_task(dataset)
    assert len(model.net.distributions) == 1, "Mahalanobis mode MUST populate distributions!"

    x_test = torch.randn(2, 3, 224, 224)
    dists = model.cal_expert_dist(x_test)
    assert len(dists) == 1
    assert isinstance(dists[0], float)
    print(f"[OK] Mahalanobis task 0 distances: {dists}")

    print("[OK] Mahalanobis mode compatibility test passed.")


if __name__ == '__main__':
    test_parser_and_init()
    test_resonance_distance_computation()
    test_mahalanobis_mode_compatibility()
    print("\nALL TASK RETRIEVAL TESTS PASSED!")
