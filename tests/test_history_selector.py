from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.models.history_selector import SelfHistorySelector


def test_visit_only_selector_masks_current_visit_and_renormalizes_topk() -> None:
    selector = SelfHistorySelector(
        hidden_dim=4,
        dropout=0.0,
        selection_mode="visit_only",
        top_k=1,
        attention_type="softmax_topk",
    )
    current_state = torch.tensor(
        [[0.5, 0.1, 0.2, 0.3]],
        dtype=torch.float32,
    )
    history_states = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.2, 0.8, 0.0, 0.0],
                [0.5, 0.1, 0.2, 0.3],
                [0.0, 0.0, 0.0, 0.0],
            ]
        ],
        dtype=torch.float32,
    )
    visit_mask = torch.tensor([[1, 1, 1, 0]], dtype=torch.bool)

    outputs = selector(
        current_state=current_state,
        history_states=history_states,
        visit_mask=visit_mask,
    )

    assert outputs["selected_history_context"].shape == (1, 4)
    assert outputs["visit_attention_weights"].shape == (1, 4)
    assert outputs["selected_visit_indices"].shape == (1, 1)
    assert outputs["selected_visit_mask"].shape == (1, 4)
    assert outputs["visit_attention_weights"][0, 2].item() == pytest.approx(0.0)
    assert outputs["visit_attention_weights"][0, 3].item() == pytest.approx(0.0)
    assert outputs["visit_attention_weights"].sum().item() == pytest.approx(1.0)
    assert int((outputs["visit_attention_weights"] > 0).sum().item()) == 1
    assert outputs["selected_visit_mask"].sum().item() == 1


def test_visit_attribute_selector_emits_attribute_attention_data() -> None:
    selector = SelfHistorySelector(
        hidden_dim=4,
        dropout=0.0,
        selection_mode="visit_attribute",
        top_k=2,
        attention_type="softmax_topk",
    )
    current_state = torch.tensor(
        [[0.3, 0.2, 0.1, 0.4]],
        dtype=torch.float32,
    )
    history_states = torch.tensor(
        [
            [
                [0.8, 0.1, 0.0, 0.0],
                [0.4, 0.4, 0.1, 0.0],
                [0.3, 0.2, 0.1, 0.4],
            ]
        ],
        dtype=torch.float32,
    )
    visit_mask = torch.tensor([[1, 1, 1]], dtype=torch.bool)
    modality_history_states = {
        "diagnosis": history_states + 0.1,
        "procedure": history_states + 0.2,
        "lab_vital": history_states + 0.3,
        "medication_history": history_states + 0.4,
    }

    outputs = selector(
        current_state=current_state,
        history_states=history_states,
        visit_mask=visit_mask,
        modality_history_states=modality_history_states,
    )

    assert outputs["selection_mode"] == "visit_attribute"
    assert outputs["selected_history_context"].shape == (1, 4)
    assert outputs["medication_history_context"].shape == (1, 4)
    assert set(outputs["attribute_contexts"].keys()) == {
        "diag_context",
        "proc_context",
        "lab_vital_context",
        "med_context",
    }
    assert set(outputs["attribute_attention_weights"].keys()) == {
        "diagnosis",
        "procedure",
        "lab_vital",
        "medication_history",
    }
    assert "visit_context" in outputs["attribute_fusion_weights"]
    assert "med_context" in outputs["attribute_fusion_weights"]
    assert outputs["attribute_attention_weights"]["diagnosis"].shape == (1, 3)
    assert outputs["attribute_attention_weights"]["medication_history"].sum().item() == pytest.approx(1.0)
    assert torch.isfinite(outputs["selected_history_context"]).all()


def test_selection_mode_none_returns_safe_zeros() -> None:
    selector = SelfHistorySelector(hidden_dim=4, dropout=0.0, selection_mode="none")
    current_state = torch.tensor([[0.1, 0.2, 0.3, 0.4]], dtype=torch.float32)
    history_states = torch.tensor([[[0.1, 0.2, 0.3, 0.4]]], dtype=torch.float32)
    visit_mask = torch.tensor([[1]], dtype=torch.bool)

    outputs = selector(
        current_state=current_state,
        history_states=history_states,
        visit_mask=visit_mask,
    )

    assert torch.equal(outputs["selected_history_context"], torch.zeros_like(current_state))
    assert outputs["selected_visit_indices"].shape == (1, 0)
    assert outputs["attribute_contexts"] == {}
    assert outputs["selected_visit_mask"].shape == (1, 1)
