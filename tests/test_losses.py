from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

import src.training.losses as losses_module
from src.training.losses import (
    MedicationRecommendationLoss,
    build_medication_loss_config,
    compute_medication_losses,
    extract_last_valid_targets,
    resolve_lambda_ddi_current,
)


def test_extract_last_valid_targets_uses_last_valid_visit() -> None:
    target_drugs = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [9.0, 9.0, 9.0]],
            [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    visit_mask = torch.tensor(
        [
            [1, 1, 0],
            [1, 1, 1],
        ],
        dtype=torch.bool,
    )

    target_current = extract_last_valid_targets(target_drugs, visit_mask)

    assert torch.equal(target_current[0], torch.tensor([0.0, 1.0, 0.0]))
    assert torch.equal(target_current[1], torch.tensor([0.0, 0.0, 0.0]))


def test_compute_medication_losses_returns_all_components() -> None:
    ddi_adj = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    logits = torch.tensor(
        [
            [0.5, -0.2, 0.1],
            [-0.4, 0.7, 0.3],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor(
        [
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            [[0.0, 0.0, 1.0], [0.0, 1.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    visit_mask = torch.tensor(
        [
            [1, 1],
            [1, 1],
        ],
        dtype=torch.bool,
    )

    loss_outputs = compute_medication_losses(
        drug_logits=logits,
        target_drugs=targets,
        visit_mask=visit_mask,
        ddi_matrix=ddi_adj,
        lambda_ddi=0.1,
    )
    drug_probs = torch.sigmoid(logits)
    expected_ddi_loss = (
        (drug_probs.unsqueeze(2) * drug_probs.unsqueeze(1)) * ddi_adj.unsqueeze(0)
    ).sum(dim=(1, 2)).mean()

    assert loss_outputs["target_current"].shape == (2, 3)
    assert torch.isfinite(loss_outputs["prediction_loss"])
    assert torch.isfinite(loss_outputs["pred_bce_loss"])
    assert torch.isfinite(loss_outputs["margin_loss"])
    assert torch.isfinite(loss_outputs["ddi_loss"])
    assert torch.isfinite(loss_outputs["weighted_ddi_loss"])
    assert torch.isfinite(loss_outputs["lambda_ddi_current"])
    assert torch.isfinite(loss_outputs["total_loss"])
    assert torch.allclose(loss_outputs["ddi_loss"], expected_ddi_loss)
    assert torch.allclose(
        loss_outputs["total_loss"],
        loss_outputs["prediction_loss"] + 0.1 * loss_outputs["ddi_loss"],
    )


def test_resolve_lambda_ddi_current_supports_stagewise_linear_ramp() -> None:
    loss_config = build_medication_loss_config(
        loss_config={
            "use_ddi": True,
            "lambda_ddi": 0.1,
            "stage_training": True,
            "ddi_schedule": {
                "enabled": True,
                "type": "linear",
                "start_epoch": 3,
                "end_epoch": 5,
                "start_value": 0.01,
                "end_value": 0.1,
            },
        }
    )

    assert resolve_lambda_ddi_current(current_epoch=1, loss_config=loss_config) == pytest.approx(0.0)
    assert resolve_lambda_ddi_current(current_epoch=3, loss_config=loss_config) == pytest.approx(0.01)
    assert resolve_lambda_ddi_current(current_epoch=4, loss_config=loss_config) == pytest.approx(0.055)
    assert resolve_lambda_ddi_current(current_epoch=5, loss_config=loss_config) == pytest.approx(0.1)


def test_compute_medication_losses_builds_pos_weight_from_avg_pos() -> None:
    logits = torch.tensor(
        [
            [0.4, -0.1, 0.2],
            [-0.3, 0.8, 0.1],
        ],
        dtype=torch.float32,
        requires_grad=True,
    )
    targets = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    expected_pos_weight = torch.full((3,), 2.0, dtype=torch.float32)

    loss_from_avg_pos = compute_medication_losses(
        drug_logits=logits,
        target_drugs=targets,
        avg_pos=1.0,
        reduction="mean",
    )
    loss_from_explicit_weight = compute_medication_losses(
        drug_logits=logits,
        target_drugs=targets,
        pos_weight=expected_pos_weight,
        reduction="mean",
    )

    assert torch.allclose(
        loss_from_avg_pos["prediction_loss"],
        loss_from_explicit_weight["prediction_loss"],
    )


def test_compute_medication_losses_rejects_mismatched_drug_probs() -> None:
    logits = torch.tensor([[0.2, -0.4, 0.7]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)

    with pytest.raises(ValueError, match="drug_probs must match torch.sigmoid\\(drug_logits\\)"):
        compute_medication_losses(
            drug_logits=logits,
            target_drugs=targets,
            drug_probs=torch.tensor([[0.1, 0.1, 0.1]], dtype=torch.float32),
        )


def test_compute_medication_losses_asserts_on_non_grad_logits_during_training() -> None:
    logits = torch.tensor([[0.2, -0.4, 0.7]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)

    with pytest.raises(AssertionError, match="drug_logits must be raw logits"):
        compute_medication_losses(
            drug_logits=logits,
            target_drugs=targets,
        )

    loss_outputs = compute_medication_losses(
        drug_logits=logits,
        target_drugs=targets,
        training=False,
    )
    assert torch.isfinite(loss_outputs["prediction_loss"])


def test_medication_recommendation_loss_warns_when_lambda_ddi_is_zero(capsys: pytest.CaptureFixture[str]) -> None:
    losses_module._LAMBDA_DDI_DISABLED_WARNING_SHOWN = False
    criterion = MedicationRecommendationLoss(lambda_ddi=0.0)
    logits = torch.tensor([[0.2, -0.4, 0.7]], dtype=torch.float32, requires_grad=True)
    targets = torch.tensor([[1.0, 0.0, 1.0]], dtype=torch.float32)

    criterion(
        drug_logits=logits,
        target_drugs=targets,
    )
    captured = capsys.readouterr()

    assert "WARNING: lambda_ddi=0, DDI loss is disabled. DDI rate will not be optimized." in captured.out
