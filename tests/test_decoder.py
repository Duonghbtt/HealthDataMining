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
    assert outputs["recommendation_metadata"]["topk_indices"].shape == (2, 2)
    assert outputs["recommendation_metadata"]["topk_scores"].shape == (2, 2)
    assert torch.isfinite(outputs["drug_logits"]).all()
    assert torch.isfinite(outputs["drug_probs"]).all()
    assert torch.all(outputs["drug_probs"] >= 0.0)
    assert torch.all(outputs["drug_probs"] <= 1.0)
