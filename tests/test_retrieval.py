from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.retrieval.dynamic_graph import build_edge_artifact
from src.retrieval.memory_bank import MemoryBank, build_last_visit_queries
from src.retrieval.topk_retriever import (
    _retrieve_patient_neighbors_reference,
    retrieve_patient_neighbors,
    retrieve_personal_history,
    validate_retrieval_payload,
)


def _build_memory_bank() -> MemoryBank:
    return MemoryBank(
        visit_states=torch.tensor(
            [
                [1.00, 0.00],  # stay 301 visit 0
                [0.96, 0.04],  # stay 301 visit 1
                [0.95, 0.03],  # stay 302 visit 0
                [0.92, 0.02],  # stay 302 visit 1
                [0.00, 1.00],  # stay 304 visit 0
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


def _query_metadata() -> dict[str, object]:
    return {
        "stay_ids": [301, 399],
        "subject_ids": [101, 199],
        "hadm_ids": [201, 299],
        "visit_indices": [1, 0],
        "visit_time_days": [3.0, 4.0],
        "diag_code_sets": [(10, 20, 30), (10, 20, 30)],
        "proc_code_sets": [(1, 2), (1, 2)],
        "lab_feature_sets": [(0, 1, 2), (0, 1, 2)],
        "vital_feature_sets": [(0, 1), (0, 1)],
        "split": ["train", "train"],
    }


def test_retrieve_personal_history_only_looks_backward() -> None:
    bank = _build_memory_bank()
    out = retrieve_personal_history(
        torch.tensor([[0.96, 0.04]], dtype=torch.float32),
        {
            "stay_ids": [301],
            "visit_indices": [1],
            "visit_time_days": [3.0],
            "diag_code_sets": [(10, 20, 30)],
            "proc_code_sets": [(1, 2)],
            "lab_feature_sets": [(0, 1, 2)],
            "vital_feature_sets": [(0, 1)],
        },
        bank,
        top_k=3,
        temporal_decay_alpha=0.1,
    )
    assert out["indices"].shape == (1, 1)
    assert out["indices"][0, 0].item() == 0


def test_retrieve_patient_neighbors_sorts_scores_and_groups_by_stay() -> None:
    bank = _build_memory_bank()
    payload = retrieve_patient_neighbors(
        torch.tensor([[0.96, 0.04]], dtype=torch.float32),
        {key: [value[0]] for key, value in _query_metadata().items()},
        bank,
        top_k=3,
        temporal_decay_alpha=0.2,
    )
    validate_retrieval_payload(payload)
    assert payload["neighbor_stay_ids"].shape == (1, 2)
    assert payload["neighbor_stay_ids"][0, 0].item() == 302
    assert payload["neighbor_scores"][0, 0] >= payload["neighbor_scores"][0, 1]
    assert payload["matched_visit_indices"][0, 0].item() == 1


def test_cross_patient_retrieval_excludes_same_stay() -> None:
    bank = _build_memory_bank()
    payload = retrieve_patient_neighbors(
        torch.tensor([[0.96, 0.04]], dtype=torch.float32),
        {key: [value[0]] for key, value in _query_metadata().items()},
        bank,
        top_k=3,
        temporal_decay_alpha=0.2,
    )
    assert 301 not in payload["neighbor_stay_ids"][0].tolist()


def test_batch_query_supports_more_than_one_patient() -> None:
    bank = _build_memory_bank()
    payload = retrieve_patient_neighbors(
        torch.tensor([[0.96, 0.04], [0.95, 0.03]], dtype=torch.float32),
        _query_metadata(),
        bank,
        top_k=2,
        temporal_decay_alpha=0.2,
    )
    assert payload["neighbor_indices"].shape[0] == 2
    assert payload["query_visit_indices"].shape == (2,)


def test_temporal_similarity_changes_ranking() -> None:
    bank = _build_memory_bank()
    payload = retrieve_patient_neighbors(
        torch.tensor([[0.96, 0.04]], dtype=torch.float32),
        {
            "stay_ids": [399],
            "subject_ids": [199],
            "hadm_ids": [299],
            "visit_indices": [0],
            "visit_time_days": [4.0],
            "diag_code_sets": [(10, 20, 30)],
            "proc_code_sets": [(1, 2)],
            "lab_feature_sets": [(0, 1, 2)],
            "vital_feature_sets": [(0, 1)],
            "split": ["train"],
        },
        bank,
        top_k=2,
        temporal_decay_alpha=0.5,
    )
    assert payload["neighbor_stay_ids"][0, 0].item() == 302
    assert payload["neighbor_time_gaps_days"][0, 0] <= payload["neighbor_time_gaps_days"][0, 1]


def test_retrieval_rejects_cross_split_by_default() -> None:
    bank = _build_memory_bank()
    with pytest.raises(ValueError, match="Cross-split retrieval is disabled"):
        retrieve_patient_neighbors(
            torch.tensor([[0.96, 0.04]], dtype=torch.float32),
            {
                "stay_ids": [399],
                "subject_ids": [199],
                "hadm_ids": [299],
                "visit_indices": [0],
                "visit_time_days": [4.0],
                "diag_code_sets": [(10, 20, 30)],
                "proc_code_sets": [(1, 2)],
                "lab_feature_sets": [(0, 1, 2)],
                "vital_feature_sets": [(0, 1)],
                "split": ["test"],
            },
            bank,
            top_k=2,
            temporal_decay_alpha=0.2,
        )


def test_retrieval_rejects_mixed_query_splits() -> None:
    bank = _build_memory_bank()
    with pytest.raises(ValueError, match="Mixed query splits are not supported"):
        retrieve_patient_neighbors(
            torch.tensor([[0.96, 0.04], [0.95, 0.03]], dtype=torch.float32),
            {
                "stay_ids": [399, 400],
                "subject_ids": [199, 200],
                "hadm_ids": [299, 300],
                "visit_indices": [0, 0],
                "visit_time_days": [4.0, 4.0],
                "diag_code_sets": [(10, 20, 30), (10, 20, 30)],
                "proc_code_sets": [(1, 2), (1, 2)],
                "lab_feature_sets": [(0, 1, 2), (0, 1, 2)],
                "vital_feature_sets": [(0, 1), (0, 1)],
                "split": ["train", "val"],
            },
            bank,
            top_k=2,
            temporal_decay_alpha=0.2,
        )


def test_retrieval_train_bank_only_allows_cross_split_eval_queries() -> None:
    bank = _build_memory_bank()
    payload = retrieve_patient_neighbors(
        torch.tensor([[0.96, 0.04]], dtype=torch.float32),
        {
            "stay_ids": [399],
            "subject_ids": [199],
            "hadm_ids": [299],
            "visit_indices": [0],
            "visit_time_days": [4.0],
            "diag_code_sets": [(10, 20, 30)],
            "proc_code_sets": [(1, 2)],
            "lab_feature_sets": [(0, 1, 2)],
            "vital_feature_sets": [(0, 1)],
            "split": ["val"],
        },
        bank,
        top_k=2,
        temporal_decay_alpha=0.2,
        cross_split_policy="train_bank_only",
    )
    validate_retrieval_payload(payload)
    assert payload["query_split"] == "val"
    assert payload["bank_split"] == "train"
    assert payload["cross_split_policy"] == "train_bank_only"


def test_retrieval_scoring_mode_changes_neighbor_scores() -> None:
    bank = _build_memory_bank()
    query = torch.tensor([[0.96, 0.04]], dtype=torch.float32)
    metadata = {
        "stay_ids": [399],
        "subject_ids": [199],
        "hadm_ids": [299],
        "visit_indices": [0],
        "visit_time_days": [4.0],
        "diag_code_sets": [(10, 20, 30)],
        "proc_code_sets": [(1, 2)],
        "lab_feature_sets": [(0, 1, 2)],
        "vital_feature_sets": [(0, 1)],
        "split": ["train"],
    }
    static_payload = retrieve_patient_neighbors(
        query,
        metadata,
        bank,
        top_k=2,
        temporal_decay_alpha=0.5,
        scoring_mode="static_cosine",
    )
    temporal_payload = retrieve_patient_neighbors(
        query,
        metadata,
        bank,
        top_k=2,
        temporal_decay_alpha=0.5,
        scoring_mode="temporal_cosine",
    )
    assert static_payload["retrieval_scoring_mode"] == "static_cosine"
    assert temporal_payload["retrieval_scoring_mode"] == "temporal_cosine"
    assert temporal_payload["neighbor_scores"][0, 0].item() != pytest.approx(
        static_payload["neighbor_scores"][0, 0].item()
    )


@pytest.mark.parametrize("scoring_mode", ["static_cosine", "temporal_cosine", "temporal_relevance"])
def test_optimized_bruteforce_matches_reference(scoring_mode: str) -> None:
    bank = _build_memory_bank()
    query = torch.tensor([[0.96, 0.04], [0.95, 0.03]], dtype=torch.float32)
    metadata = {
        "stay_ids": [399, 400],
        "subject_ids": [199, 200],
        "hadm_ids": [299, 300],
        "visit_indices": [0, 0],
        "visit_time_days": [4.0, 4.0],
        "diag_code_sets": [(10, 20, 30), (10, 20, 30)],
        "proc_code_sets": [(1, 2), (1, 2)],
        "lab_feature_sets": [(0, 1, 2), (0, 1, 2)],
        "vital_feature_sets": [(0, 1), (0, 1)],
        "split": ["val", "val"],
    }
    optimized = retrieve_patient_neighbors(
        query,
        metadata,
        bank,
        top_k=2,
        temporal_decay_alpha=0.2,
        backend="bruteforce",
        use_faiss_if_available=False,
        cross_split_policy="train_bank_only",
        scoring_mode=scoring_mode,
    )
    reference = _retrieve_patient_neighbors_reference(
        query,
        metadata,
        bank,
        top_k=2,
        temporal_decay_alpha=0.2,
        backend="bruteforce",
        use_faiss_if_available=False,
        cross_split_policy="train_bank_only",
        scoring_mode=scoring_mode,
    )

    assert optimized["query_split"] == reference["query_split"]
    assert optimized["bank_split"] == reference["bank_split"]
    assert optimized["cross_split_policy"] == reference["cross_split_policy"]
    for field in (
        "neighbor_indices",
        "neighbor_scores",
        "neighbor_static_scores",
        "neighbor_time_gaps_days",
        "neighbor_subject_ids",
        "neighbor_hadm_ids",
        "neighbor_stay_ids",
        "matched_visit_indices",
        "aux_personal_history_indices",
        "aux_personal_history_scores",
    ):
        assert torch.equal(optimized[field], reference[field]), field


def test_build_last_visit_queries_uses_record_split_when_not_overridden() -> None:
    records = [
        {
            "subject_id": 1,
            "hadm_id": 11,
            "stay_id": 111,
            "split": "test",
            "intime": "2020-01-01 00:00:00",
            "steps": [
                {
                    "delta_hours": 0.0,
                    "diagnosis_ids": [2],
                    "procedure_ids": [3],
                    "lab_mask": [1, 0],
                    "vital_mask": [1, 1],
                }
            ],
        }
    ]
    encoder_outputs = {
        "state_sequence": torch.tensor([[[1.0, 0.0]]], dtype=torch.float32),
        "visit_mask": torch.tensor([[True]], dtype=torch.bool),
    }
    query_states, metadata = build_last_visit_queries(records, encoder_outputs)
    assert query_states.shape == (1, 2)
    assert metadata["split"] == ["test"]


def test_faiss_backend_falls_back_when_package_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    bank = _build_memory_bank()
    monkeypatch.setattr("src.retrieval.topk_retriever.FaissIndex.is_available", lambda: False)
    payload = retrieve_patient_neighbors(
        torch.tensor([[0.96, 0.04]], dtype=torch.float32),
        {key: [value[0]] for key, value in _query_metadata().items()},
        bank,
        top_k=2,
        temporal_decay_alpha=0.2,
        backend="faiss",
        use_faiss_if_available=True,
    )
    assert payload["backend"] == "bruteforce"


def test_dynamic_graph_stays_edge_artifact_only() -> None:
    bank = _build_memory_bank()
    payload = retrieve_patient_neighbors(
        torch.tensor([[0.96, 0.04]], dtype=torch.float32),
        {key: [value[0]] for key, value in _query_metadata().items()},
        bank,
        top_k=2,
        temporal_decay_alpha=0.2,
    )
    graph_payload = build_edge_artifact(payload)
    assert graph_payload["split"] == "train"
    assert len(graph_payload["edges"]) == 2
    assert set(graph_payload["edges"][0]) == {
        "src_stay_id",
        "dst_stay_id",
        "score",
        "time_gap_days",
        "rank",
        "split",
        "matched_visit_index",
    }
