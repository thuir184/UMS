import math

import pytest
import torch

from training_scripts.evaluate_quantum_routing import aggregate_score_batches


def test_score_aggregation_distinguishes_mean_from_majority_vote():
    # Two high-confidence outliers move the mean to expert 2, while three of
    # five individual samples still vote for expert 1.
    batches = [
        torch.tensor([[0.6, 0.5], [0.6, 0.5], [0.6, 0.5]]),
        torch.tensor([[0.0, 2.0], [0.0, 2.0]]),
    ]

    result = aggregate_score_batches(batches, n_tasks=2)

    assert result["samples"] == 5
    assert result["mean_choice"] == 2
    assert result["majority_choice"] == 1
    assert result["vote_counts"] == [3, 2]
    assert result["majority_share"] == pytest.approx(0.6)
    assert result["vote_margin"] == pytest.approx(0.2)
    assert result["mean_sample_margin"] == pytest.approx(0.86)


def test_score_aggregation_is_batch_partition_invariant():
    scores = torch.tensor(
        [[0.9, 0.1, 0.0], [0.2, 0.7, 0.1], [0.6, 0.3, 0.1]],
        dtype=torch.float32,
    )

    together = aggregate_score_batches([scores], n_tasks=3)
    partitioned = aggregate_score_batches([scores[:1], scores[1:]], n_tasks=3)

    assert together["mean_scores"] == pytest.approx(partitioned["mean_scores"])
    assert together["vote_counts"] == partitioned["vote_counts"]
    assert together["mean_choice"] == partitioned["mean_choice"]
    assert together["majority_choice"] == partitioned["majority_choice"]


def test_score_aggregation_handles_one_task():
    result = aggregate_score_batches([torch.tensor([[0.2], [0.8]])], n_tasks=1)

    assert result["mean_choice"] == 1
    assert result["majority_choice"] == 1
    assert result["vote_counts"] == [2]
    assert result["majority_share"] == pytest.approx(1.0)
    assert math.isinf(result["mean_margin"])


def test_score_aggregation_rejects_wrong_shape_and_empty_input():
    with pytest.raises(ValueError, match="Expected score batch"):
        aggregate_score_batches([torch.ones(2, 3)], n_tasks=2)

    with pytest.raises(RuntimeError, match="empty loader"):
        aggregate_score_batches([], n_tasks=2)
