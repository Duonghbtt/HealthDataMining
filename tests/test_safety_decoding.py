from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.models.ddi_regularization import (
    compute_ddi_penalty_for_set,
    rerank_prediction_set,
    score_candidate_set,
)


def test_compute_ddi_penalty_for_set_counts_pairs() -> None:
    ddi_matrix = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    selected = torch.tensor([1.0, 1.0, 1.0, 0.0], dtype=torch.float32)

    stats = compute_ddi_penalty_for_set(selected, ddi_matrix)

    assert stats["num_selected_drugs"] == pytest.approx(3.0)
    assert stats["num_pairs"] == pytest.approx(3.0)
    assert stats["num_ddi_pairs"] == pytest.approx(2.0)
    assert stats["ddi_rate"] == pytest.approx(2.0 / 3.0)


def test_score_candidate_set_trades_off_utility_and_ddi() -> None:
    ddi_matrix = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    probs = torch.tensor([0.9, 0.8, 0.3], dtype=torch.float32)

    risky = score_candidate_set(probs, [0, 1], ddi_matrix, beta_ddi=1.0, gamma_size=0.0)
    safer = score_candidate_set(probs, [0], ddi_matrix, beta_ddi=1.0, gamma_size=0.0)

    assert risky["ddi_penalty"] > safer["ddi_penalty"]
    assert safer["total_score"] > risky["total_score"]


def test_soft_constrained_rerank_keeps_non_empty_predictions() -> None:
    ddi_matrix = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    probs = torch.tensor([[0.52, 0.51, 0.10, 0.09]], dtype=torch.float32)

    outputs = rerank_prediction_set(
        probs,
        ddi_matrix,
        decode_mode="soft_constrained_rerank",
        threshold=0.95,
        top_m=3,
        beta_ddi=0.75,
        gamma_size=0.10,
        min_drugs=1,
        target_avg_drugs=1.5,
        ensure_non_empty=True,
    )

    prediction_mask = outputs["prediction_mask"]
    assert tuple(prediction_mask.shape) == (1, 4)
    assert int(prediction_mask.sum().item()) >= 1
    assert bool(torch.isfinite(outputs["total_score"]).all().item())
