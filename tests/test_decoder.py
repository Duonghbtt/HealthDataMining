from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.models.medication_decoder import MedicationDecoder
from src.training.losses import MedicationRecommendationLoss


@pytest.fixture
def fused_repr() -> torch.Tensor:
    return torch.tensor(
        [
            [0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2, 0.4],
            [-0.4, 0.6, 0.2, -0.3, 0.1, 0.7, -0.2, 0.0],
        ],
        dtype=torch.float32,
    )


def test_medication_decoder_forward_shapes_and_metadata(fused_repr: torch.Tensor) -> None:
    decoder = MedicationDecoder(
        hidden_dim=8,
        drug_vocab_size=6,
        dropout=0.0,
        top_k_metadata=3,
    )

    outputs = decoder(fused_repr, top_k=2)

    assert outputs["drug_logits"].shape == (2, 6)
    assert outputs["drug_probs"].shape == (2, 6)
    assert outputs["recommendation_metadata"]["batch_size"] == 2
    assert outputs["recommendation_metadata"]["hidden_dim"] == 8
    assert outputs["recommendation_metadata"]["drug_vocab_size"] == 6
    assert outputs["recommendation_metadata"]["decoder_mode"] == "independent"
    assert outputs["recommendation_metadata"]["label_correlation_enabled"] is False
    assert torch.allclose(outputs["drug_logits"], outputs["base_drug_logits"])
    assert torch.allclose(
        outputs["label_correlation_logits"],
        torch.zeros_like(outputs["label_correlation_logits"]),
    )
    assert float(outputs["recommendation_metadata"]["correlation_residual_norm"].item()) == 0.0
    assert float(outputs["recommendation_metadata"]["logit_shift_mean_abs"].item()) == 0.0
    assert float(outputs["recommendation_metadata"]["logit_shift_max_abs"].item()) == 0.0
    assert outputs["recommendation_metadata"]["topk_indices"].shape == (2, 2)
    assert outputs["recommendation_metadata"]["topk_scores"].shape == (2, 2)
    assert torch.isfinite(outputs["drug_logits"]).all()
    assert torch.isfinite(outputs["drug_probs"]).all()
    assert torch.all(outputs["drug_probs"] >= 0.0)
    assert torch.all(outputs["drug_probs"] <= 1.0)


def test_medication_decoder_label_correlation_residual_mode_is_opt_in(
    fused_repr: torch.Tensor,
) -> None:
    decoder = MedicationDecoder(
        hidden_dim=8,
        drug_vocab_size=6,
        dropout=0.0,
        top_k_metadata=0,
        decoder_mode="label_correlation_residual",
        correlation_dim=4,
        patient_residual_weight=0.2,
        coprescription_residual_weight=0.1,
        correlation_dropout=0.0,
    )

    outputs = decoder(fused_repr)

    assert outputs["drug_logits"].shape == (2, 6)
    assert outputs["base_drug_logits"].shape == (2, 6)
    assert outputs["patient_correlation_logits"].shape == (2, 6)
    assert outputs["coprescription_correlation_logits"].shape == (2, 6)
    assert outputs["label_correlation_logits"].shape == (2, 6)
    assert outputs["recommendation_metadata"]["decoder_mode"] == "label_correlation_residual"
    assert outputs["recommendation_metadata"]["label_correlation_enabled"] is True
    assert outputs["recommendation_metadata"]["correlation_dim"] == 4
    assert outputs["correlation_residual_norm"].ndim == 0
    assert outputs["logit_shift_mean_abs"].ndim == 0
    assert outputs["logit_shift_max_abs"].ndim == 0
    assert torch.isfinite(outputs["drug_logits"]).all()
    assert not torch.allclose(outputs["drug_logits"], outputs["base_drug_logits"])
    assert float(outputs["correlation_residual_norm"].item()) > 0.0
    assert float(outputs["logit_shift_mean_abs"].item()) > 0.0
    assert float(outputs["logit_shift_max_abs"].item()) > 0.0


def test_medication_decoder_legacy_label_correlation_flag_enables_residual_mode(
    fused_repr: torch.Tensor,
) -> None:
    decoder = MedicationDecoder(
        hidden_dim=8,
        drug_vocab_size=6,
        dropout=0.0,
        top_k_metadata=0,
        decoder_mode="independent",
        label_correlation_enabled=True,
        correlation_dim=4,
        patient_residual_weight=0.2,
        coprescription_residual_weight=0.1,
        correlation_dropout=0.0,
    )

    outputs = decoder(fused_repr)

    assert outputs["recommendation_metadata"]["decoder_mode"] == "label_correlation_residual"
    assert outputs["recommendation_metadata"]["label_correlation_enabled"] is True
    assert not torch.allclose(outputs["drug_logits"], outputs["base_drug_logits"])


def test_medication_decoder_label_correlation_residual_backward_pass() -> None:
    decoder = MedicationDecoder(
        hidden_dim=8,
        drug_vocab_size=6,
        dropout=0.0,
        top_k_metadata=0,
        decoder_mode="label_correlation_residual",
        correlation_dim=4,
        patient_residual_weight=0.2,
        coprescription_residual_weight=0.1,
        correlation_dropout=0.0,
    )
    fused_repr = torch.randn(3, 8, dtype=torch.float32, requires_grad=True)

    outputs = decoder(fused_repr)
    loss = outputs["drug_logits"].sum()
    loss.backward()

    assert fused_repr.grad is not None
    assert torch.isfinite(fused_repr.grad).all()
    assert decoder.decoder[0].weight.grad is not None
    assert decoder.drug_correlation_embedding is not None
    assert decoder.drug_correlation_embedding.weight.grad is not None
    assert decoder.patient_correlation_projection is not None
    projection = decoder.patient_correlation_projection[0]
    assert projection.weight.grad is not None


def test_medication_decoder_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="decoder_mode"):
        MedicationDecoder(hidden_dim=8, drug_vocab_size=6, decoder_mode="attention")


def test_medication_recommendation_loss_adds_sampled_pairwise_ranking() -> None:
    logits = torch.tensor(
        [
            [-1.0, 1.0, 0.5],
            [2.0, -0.5, 0.0],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    base_loss = MedicationRecommendationLoss(lambda_ddi=0.0, ranking_lambda=0.0)
    ranking_loss = MedicationRecommendationLoss(
        lambda_ddi=0.0,
        ranking_lambda=0.5,
        ranking_objective="margin",
        ranking_margin=1.0,
        ranking_num_negatives=2,
        ranking_hard_negative_fraction=1.0,
    )

    base_outputs = base_loss(drug_logits=logits, target_drugs=targets)
    ranking_outputs = ranking_loss(drug_logits=logits, target_drugs=targets)

    assert float(ranking_outputs["ranking_loss"].item()) > 0.0
    assert float(ranking_outputs["weighted_ranking_loss"].item()) == pytest.approx(
        float(ranking_outputs["ranking_loss"].item()) * 0.5
    )
    assert float(ranking_outputs["total_loss"].item()) == pytest.approx(
        float(base_outputs["total_loss"].item())
        + float(ranking_outputs["weighted_ranking_loss"].item())
    )


def test_medication_recommendation_loss_supports_asymmetric_focal_objective() -> None:
    logits = torch.tensor([[-6.0, 3.0]], dtype=torch.float32)
    targets = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    bce_loss = MedicationRecommendationLoss(lambda_ddi=0.0, objective="bce")
    asymmetric_loss = MedicationRecommendationLoss(
        lambda_ddi=0.0,
        objective="asymmetric_focal",
        asymmetric_gamma_pos=0.0,
        asymmetric_gamma_neg=4.0,
        asymmetric_clip=0.05,
    )

    bce_outputs = bce_loss(drug_logits=logits, target_drugs=targets)
    asymmetric_outputs = asymmetric_loss(drug_logits=logits, target_drugs=targets)

    assert asymmetric_outputs["objective"] == "asymmetric_focal"
    assert torch.isfinite(asymmetric_outputs["prediction_loss"])
    assert float(asymmetric_outputs["prediction_loss"].item()) < float(
        bce_outputs["prediction_loss"].item()
    )
    assert float(asymmetric_outputs["asymmetric_gamma_neg"].item()) == pytest.approx(4.0)
    assert float(asymmetric_outputs["asymmetric_clip"].item()) == pytest.approx(0.05)


def test_asymmetric_focal_keeps_positive_class_weighting() -> None:
    logits = torch.zeros((1, 2), dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    unweighted_loss = MedicationRecommendationLoss(
        lambda_ddi=0.0,
        objective="asymmetric_focal",
        asymmetric_gamma_pos=0.0,
        asymmetric_gamma_neg=0.0,
        asymmetric_clip=0.0,
    )
    weighted_loss = MedicationRecommendationLoss(
        lambda_ddi=0.0,
        objective="asymmetric_focal",
        asymmetric_gamma_pos=0.0,
        asymmetric_gamma_neg=0.0,
        asymmetric_clip=0.0,
        pos_weight=torch.tensor([3.0, 1.0], dtype=torch.float32),
    )

    unweighted_outputs = unweighted_loss(drug_logits=logits, target_drugs=targets)
    weighted_outputs = weighted_loss(drug_logits=logits, target_drugs=targets)

    assert float(weighted_outputs["prediction_loss"].item()) > float(
        unweighted_outputs["prediction_loss"].item()
    )


def test_asymmetric_focal_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError, match="asymmetric_gamma_neg"):
        MedicationRecommendationLoss(
            lambda_ddi=0.0,
            objective="asymmetric_focal",
            asymmetric_gamma_neg=-1.0,
        )

    with pytest.raises(ValueError, match="asymmetric_clip"):
        MedicationRecommendationLoss(
            lambda_ddi=0.0,
            objective="asymmetric_focal",
            asymmetric_clip=1.0,
        )
