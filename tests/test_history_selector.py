from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.models.history_selector import HistorySelector
from src.retrieval.memory_bank import MemoryBank, build_last_visit_queries
from src.retrieval.topk_retriever import retrieve_topk

from tests.helpers import load_handover_records, load_vocab_size


def _build_memory_bank() -> MemoryBank:
    return MemoryBank(
        visit_states=torch.tensor(
            [
                [1.00, 0.00],
                [0.96, 0.04],
                [0.95, 0.03],
                [0.92, 0.02],
                [0.00, 1.00],
            ],
            dtype=torch.float32,
        ),
        visit_repr=torch.tensor(
            [
                [1.00, 0.00],
                [0.96, 0.04],
                [0.95, 0.03],
                [0.92, 0.02],
                [0.00, 1.00],
            ],
            dtype=torch.float32,
        ),
        subject_ids=[101, 101, 102, 102, 104],
        hadm_ids=[201, 201, 202, 202, 204],
        stay_ids=[301, 301, 302, 302, 304],
        visit_index=[0, 1, 0, 1, 0],
        visit_time_days=[1.0, 3.0, 2.0, 4.0, 3.0],
        visit_time_text=[
            "2020-01-01 00:00:00",
            "2020-01-03 00:00:00",
            "2020-01-02 00:00:00",
            "2020-01-04 00:00:00",
            "2020-01-03 00:00:00",
        ],
        target_drugs=[(1,), (1, 2), (1, 3), (1, 2), (4,)],
        num_steps=[2, 2, 2, 2, 1],
        diag_code_sets=[(10, 20), (10, 20, 30), (10, 20, 31), (10, 20, 30), (99,)],
        proc_code_sets=[(1,), (1, 2), (1,), (1, 2), (8,)],
        lab_feature_sets=[(0, 1), (0, 1, 2), (0, 1), (0, 1, 2), (7,)],
        vital_feature_sets=[(0,), (0, 1), (0,), (0, 1), (5,)],
        split="train",
    )


def test_history_selector_handles_personal_history_and_padding() -> None:
    selector = HistorySelector(hidden_dim=2, dropout=0.0, score_bias_weight=0.25)
    current_state = torch.tensor([[0.95, 0.05], [0.20, 0.80]], dtype=torch.float32)
    state_sequence = torch.tensor(
        [
            [[1.0, 0.0], [0.9, 0.1], [0.0, 0.0]],
            [[0.2, 0.8], [0.0, 0.0], [0.0, 0.0]],
        ],
        dtype=torch.float32,
    )
    visit_mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)
    retrieval_payload = {
        "neighbor_indices": torch.tensor([[2, -1], [0, 4]], dtype=torch.long),
        "neighbor_scores": torch.tensor([[0.8, float("-inf")], [0.3, 0.1]], dtype=torch.float32),
        "matched_visit_indices": torch.tensor([[1, -1], [0, 0]], dtype=torch.long),
        "neighbor_stay_ids": torch.tensor([[302, -1], [301, 304]], dtype=torch.long),
    }

    out = selector(
        current_state=current_state,
        state_sequence=state_sequence,
        visit_mask=visit_mask,
        retrieval_payload=retrieval_payload,
        memory_bank=_build_memory_bank(),
    )

    metadata = out["evidence_metadata"]
    assert out["self_history_context"].shape == (2, 2)
    assert out["neighbor_history_context"].shape == (2, 2)
    assert metadata["self_history_available_mask"].tolist() == [True, False]
    assert metadata["self_history_top_index"][1].item() == -1
    assert metadata["neighbor_mask"][0, 1].item() is False
    assert metadata["neighbor_weights"][0, 1].item() == 0.0
    assert metadata["self_attribute_weights"].shape == (2, 3, 5)
    assert metadata["neighbor_attribute_weights"].shape == (2, 2, 5)
    assert torch.isfinite(metadata["self_attribute_weights"]).all()
    assert torch.isfinite(metadata["neighbor_attribute_weights"]).all()
    assert metadata["group_aware_selection_used"] is False
    assert torch.isfinite(out["self_history_context"]).all()
    assert torch.isfinite(out["neighbor_history_context"]).all()


def test_history_selector_without_neighbors_returns_zero_neighbor_context() -> None:
    selector = HistorySelector(hidden_dim=2, dropout=0.0)
    out = selector(
        current_state=torch.tensor([[0.5, 0.5]], dtype=torch.float32),
        state_sequence=torch.tensor([[[0.5, 0.5]]], dtype=torch.float32),
        visit_mask=torch.tensor([[1]], dtype=torch.bool),
    )
    metadata = out["evidence_metadata"]
    assert torch.equal(out["neighbor_history_context"], torch.zeros_like(out["neighbor_history_context"]))
    assert metadata["neighbor_available_mask"].tolist() == [False]
    assert metadata["neighbor_indices"].shape == (1, 0)


def test_history_selector_top_k_and_score_breakdown() -> None:
    selector = HistorySelector(
        hidden_dim=2,
        dropout=0.0,
        score_bias_weight=0.5,
        self_top_k=1,
        neighbor_top_k=1,
        use_retrieval_bias=True,
    )
    out = selector(
        current_state=torch.tensor([[0.95, 0.05]], dtype=torch.float32),
        state_sequence=torch.tensor([[[1.0, 0.0], [0.95, 0.05], [0.9, 0.1]]], dtype=torch.float32),
        visit_mask=torch.tensor([[1, 1, 1]], dtype=torch.bool),
        retrieval_payload={
            "neighbor_indices": torch.tensor([[1, 2]], dtype=torch.long),
            "neighbor_scores": torch.tensor([[0.2, 0.9]], dtype=torch.float32),
            "matched_visit_indices": torch.tensor([[0, 1]], dtype=torch.long),
            "neighbor_stay_ids": torch.tensor([[301, 302]], dtype=torch.long),
        },
        memory_bank=_build_memory_bank(),
    )
    metadata = out["evidence_metadata"]
    assert metadata["self_history_selected_count"].tolist() == [1]
    assert metadata["neighbor_selected_count"].tolist() == [1]
    assert metadata["self_history_content_scores"].shape == metadata["self_history_scores"].shape
    assert metadata["neighbor_content_scores"].shape == metadata["neighbor_scores"].shape
    assert metadata["neighbor_retrieval_bias"].shape == metadata["neighbor_scores"].shape
    assert metadata["self_attribute_scores"].shape == metadata["self_attribute_weights"].shape
    assert metadata["neighbor_attribute_scores"].shape == metadata["neighbor_attribute_weights"].shape
    assert metadata["selection_config"]["neighbor_top_k"] == 1


def test_history_selector_supports_group_aware_reweight_and_attribute_fallbacks() -> None:
    selector = HistorySelector(hidden_dim=2, dropout=0.0, group_reweight_weight=0.5)
    out = selector(
        current_state=torch.tensor([[0.8, 0.2]], dtype=torch.float32),
        state_sequence=torch.tensor([[[1.0, 0.0], [0.9, 0.1], [0.7, 0.3]]], dtype=torch.float32),
        visit_mask=torch.tensor([[1, 1, 1]], dtype=torch.bool),
        retrieval_payload={
            "neighbor_indices": torch.tensor([[0, 2]], dtype=torch.long),
            "neighbor_scores": torch.tensor([[0.2, 0.7]], dtype=torch.float32),
            "matched_visit_indices": torch.tensor([[0, 1]], dtype=torch.long),
            "neighbor_stay_ids": torch.tensor([[301, 302]], dtype=torch.long),
        },
        memory_bank=_build_memory_bank(),
        group_context=torch.tensor([[1.0, 0.0]], dtype=torch.float32),
        group_available_mask=torch.tensor([True]),
        attribute_payload={
            "self": {
                "masks": {
                    "lab": torch.tensor([[True, False, False]], dtype=torch.bool),
                    "vital": torch.tensor([[False, False, False]], dtype=torch.bool),
                }
            }
        },
    )
    metadata = out["evidence_metadata"]
    assert metadata["group_aware_selection_used"] is True
    assert metadata["self_group_reweight_scores"].shape == metadata["self_history_scores"].shape
    assert metadata["neighbor_group_reweight_scores"].shape == metadata["neighbor_scores"].shape
    assert torch.isfinite(metadata["self_group_influence"]).all()
    assert torch.isfinite(metadata["neighbor_group_influence"]).all()
    assert metadata["self_attribute_fallback_mask"].dtype == torch.bool
    assert metadata["self_attribute_fallback_mask"].shape == metadata["self_history_mask"].shape
    assert metadata["self_attribute_weights"].shape[-1] == 5
    assert torch.isfinite(metadata["self_attribute_weights"]).all()


def test_history_selector_on_real_handover_batch() -> None:
    pyarrow = pytest.importorskip("pyarrow")
    _ = pyarrow
    from src.data.dataset import collate_batch
    from src.models.patient_state_encoder import PatientStateEncoder

    records = load_handover_records(limit=4)
    batch = collate_batch(records)
    model = PatientStateEncoder(
        diagnosis_vocab_size=load_vocab_size("diagnosis"),
        procedure_vocab_size=load_vocab_size("procedure"),
        drug_vocab_size=load_vocab_size("drug"),
        num_lab_features=batch["lab_values"].shape[-1],
        num_vital_features=batch["vital_values"].shape[-1],
        code_embedding_dim=8,
        medication_embedding_dim=8,
        numeric_projection_dim=4,
        time_embedding_dim=4,
        visit_hidden_dim=16,
        hidden_dim=12,
        dropout=0.0,
    )
    encoder_outputs = model(batch)
    bank = MemoryBank.build_from_batch(records, encoder_outputs, split="train")
    query_states, query_metadata = build_last_visit_queries(records, encoder_outputs, split="train")
    retrieval_payload = retrieve_topk(
        query_states,
        query_metadata,
        bank,
        top_k=2,
        temporal_decay_alpha=0.05,
    )

    selector = HistorySelector(hidden_dim=12, dropout=0.0)
    out = selector(
        current_state=encoder_outputs["pooled_state"],
        state_sequence=encoder_outputs["state_sequence"],
        visit_mask=encoder_outputs["visit_mask"],
        retrieval_payload=retrieval_payload,
        memory_bank=bank,
    )

    assert out["self_history_context"].shape == (4, 12)
    assert out["neighbor_history_context"].shape == (4, 12)
    assert out["evidence_metadata"]["neighbor_indices"].shape[0] == 4
    assert out["evidence_metadata"]["self_attribute_weights"].shape[:2] == out["evidence_metadata"]["self_history_mask"].shape
    assert out["evidence_metadata"]["neighbor_attribute_weights"].shape[:2] == out["evidence_metadata"]["neighbor_mask"].shape
    assert torch.isfinite(out["self_history_context"]).all()
    assert torch.isfinite(out["neighbor_history_context"]).all()
