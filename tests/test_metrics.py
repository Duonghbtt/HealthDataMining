from __future__ import annotations

import pytest


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
pytest.importorskip("sklearn")

from sklearn.metrics import average_precision_score

from src.evaluation.metrics import (
    compute_core_metrics,
    compute_core_metrics_for_mask,
    compute_prauc,
    select_binary_predictions,
)


def test_compute_prauc_is_samplewise_and_skips_samples_without_positives() -> None:
    drug_probs = np.array(
        [
            [0.9, 0.1, 0.8],
            [0.2, 0.3, 0.1],
            [0.2, 0.8, 0.7],
        ],
        dtype=np.float32,
    )
    y_true = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    expected = (
        average_precision_score(y_true[0], drug_probs[0]) +
        average_precision_score(y_true[2], drug_probs[2])
    ) / 2.0

    assert compute_prauc(drug_probs, y_true) == pytest.approx(expected)


def test_compute_core_metrics_uses_corrected_prauc_definition() -> None:
    drug_probs = torch.tensor(
        [
            [0.9, 0.1, 0.8],
            [0.2, 0.3, 0.1],
            [0.2, 0.8, 0.7],
        ],
        dtype=torch.float32,
    )
    y_true = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    ddi_matrix = torch.zeros((3, 3), dtype=torch.float32)

    metrics = compute_core_metrics(
        y_true=y_true,
        y_score=drug_probs,
        threshold=0.5,
        ddi_matrix=ddi_matrix,
    )

    assert metrics["prauc"] == pytest.approx(
        compute_prauc(drug_probs.detach().cpu().numpy(), y_true.detach().cpu().numpy())
    )
    assert metrics["avg_predicted_drugs"] == pytest.approx(4.0 / 3.0)


def test_compute_core_metrics_for_mask_handles_subgroup_selection() -> None:
    drug_probs = torch.tensor(
        [
            [0.9, 0.1, 0.8],
            [0.2, 0.7, 0.1],
            [0.6, 0.6, 0.2],
        ],
        dtype=torch.float32,
    )
    y_true = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    ddi_matrix = torch.zeros((3, 3), dtype=torch.float32)
    subgroup_mask = torch.tensor([True, False, True], dtype=torch.bool)

    metrics = compute_core_metrics_for_mask(
        y_true=y_true,
        y_score=drug_probs,
        threshold=0.5,
        ddi_matrix=ddi_matrix,
        sample_mask=subgroup_mask,
    )

    assert metrics["num_samples"] == pytest.approx(2.0)
    assert metrics["avg_predicted_drugs"] == pytest.approx(2.0)


def test_select_binary_predictions_supports_topk_and_percentile() -> None:
    drug_probs = torch.tensor(
        [
            [0.9, 0.1, 0.8, 0.2],
            [0.2, 0.7, 0.1, 0.6],
        ],
        dtype=torch.float32,
    )

    topk_binary = select_binary_predictions(drug_probs, prediction_method="topk", top_k=2)
    percentile_binary = select_binary_predictions(drug_probs, prediction_method="percentile", percentile=75.0)

    assert topk_binary.shape == drug_probs.shape
    assert percentile_binary.shape == drug_probs.shape
    assert torch.equal(topk_binary.sum(dim=1), torch.tensor([2, 2]))
    assert torch.equal(percentile_binary.sum(dim=1), torch.tensor([2, 2]))
