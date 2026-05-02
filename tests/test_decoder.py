from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.models.medication_decoder import MedicationDecoder


@pytest.fixture
def fused_repr() -> torch.Tensor:
    return torch.tensor(
        [
            [0.1, -0.2, 0.3, 0.0, 0.5, -0.1, 0.2, 0.4],
            [-0.4, 0.6, 0.2, -0.3, 0.1, 0.7, -0.2, 0.0],
        ],
        dtype=torch.float32,
    )


def test_medication_decoder_forward_shapes_and_probabilities(fused_repr: torch.Tensor) -> None:
    decoder = MedicationDecoder(
        hidden_dim=8,
        drug_vocab_size=6,
        dropout=0.0,
    )
    assert hasattr(decoder, "proj")
    assert hasattr(decoder, "fc")
    assert hasattr(decoder, "residual_fc")

    outputs = decoder(context_vector=fused_repr)

    assert outputs["drug_logits"].shape == (2, 6)
    assert outputs["final_logits"].shape == (2, 6)
    assert outputs["drug_probs"].shape == (2, 6)
    assert outputs["logits_new"].shape == (2, 6)
    assert outputs["logits_copy"].shape == (2, 6)
    assert outputs["gate"].shape == (2, 6)
    assert outputs["gate_raw"].shape == (2, 1)
    assert outputs["copy_signal"].shape == (2, 6)
    assert torch.isfinite(outputs["drug_logits"]).all()
    assert torch.isfinite(outputs["drug_probs"]).all()
    assert torch.all(outputs["drug_probs"] >= 0.0)
    assert torch.all(outputs["drug_probs"] <= 1.0)
    assert torch.allclose(outputs["drug_probs"], torch.sigmoid(outputs["drug_logits"]))
    assert torch.allclose(outputs["gate"], torch.ones_like(outputs["gate"]))


def test_copy_reuse_decoder_emits_new_copy_and_gate_outputs(fused_repr: torch.Tensor) -> None:
    decoder = MedicationDecoder(
        hidden_dim=8,
        drug_vocab_size=6,
        dropout=0.0,
        decoder_mode="copy_reuse_v2",
        gate_type="scalar",
    )

    outputs = decoder(
        context_vector=fused_repr,
        current_state=fused_repr,
        history_context=torch.ones_like(fused_repr) * 0.25,
        retrieval_context=torch.ones_like(fused_repr) * -0.10,
        history_med_bag=torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        retrieval_med_bag=torch.tensor(
            [
                [0.0, 0.5, 0.0, 0.0, 0.5, 0.0],
                [0.0, 0.0, 0.0, 0.7, 0.0, 0.3],
            ],
            dtype=torch.float32,
        ),
    )

    assert outputs["drug_logits"].shape == (2, 6)
    assert outputs["logits_new"].shape == (2, 6)
    assert outputs["logits_copy"].shape == (2, 6)
    assert outputs["copy_signal"].shape == (2, 6)
    assert outputs["gate_raw"].shape == (2, 1)
    assert outputs["gate"].shape == (2, 6)
    assert outputs["copy_source_weights"].shape == (2, 3)
    assert outputs["copy_source_mask"].shape == (2, 3)
    assert torch.isfinite(outputs["drug_logits"]).all()
    assert torch.isfinite(outputs["logits_copy"]).all()
    assert torch.all(outputs["gate"] >= 0.0)
    assert torch.all(outputs["gate"] <= 1.0)
    assert torch.allclose(outputs["gate"][:, :1], outputs["gate_raw"])


def test_copy_reuse_decoder_falls_back_to_predict_new_when_copy_sources_are_empty(
    fused_repr: torch.Tensor,
) -> None:
    decoder = MedicationDecoder(
        hidden_dim=8,
        drug_vocab_size=6,
        dropout=0.0,
        decoder_mode="copy_reuse_v2",
        gate_type="scalar",
    )

    outputs = decoder(
        context_vector=fused_repr,
        current_state=fused_repr,
        history_context=torch.zeros_like(fused_repr),
        retrieval_context=torch.zeros_like(fused_repr),
        history_med_bag=torch.zeros(2, 6, dtype=torch.float32),
        retrieval_med_bag=torch.zeros(2, 6, dtype=torch.float32),
    )

    assert torch.allclose(outputs["gate"], torch.ones_like(outputs["gate"]))
    assert torch.allclose(outputs["drug_logits"], outputs["logits_new"])
