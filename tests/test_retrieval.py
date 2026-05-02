from __future__ import annotations

import pytest


torch = pytest.importorskip("torch")

from src.data.dataset import collate_batch
from src.retrieval.faiss_index import VisitFaissIndex
from src.retrieval.memory_bank import VisitMemoryBank
from src.retrieval.topk_retriever import TopKVisitRetriever


def test_visit_memory_bank_builds_visit_level_records() -> None:
    bank = VisitMemoryBank(split_name="train")
    bank.add_batch(
        patient_ids=[101, 202],
        visit_embeddings=torch.tensor(
            [
                [[1.0, 0.0, 0.0, 0.0], [0.8, 0.2, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0, 0.0], [0.0, 0.8, 0.2, 0.0], [0.0, 0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
        ),
        medication_evidence=torch.tensor(
            [
                [[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]],
                [[0, 0, 1, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]],
            ],
            dtype=torch.float32,
        ),
        visit_mask=torch.tensor([[1, 1, 0], [1, 1, 0]], dtype=torch.bool),
        time_delta_hours=torch.tensor([[0.0, 24.0, 0.0], [0.0, 12.0, 0.0]], dtype=torch.float32),
    )

    assert bank.num_visits == 4
    assert bank.export_embeddings().shape == (4, 4)
    assert bank.export_medication_evidence().shape == (4, 5)
    assert bank.export_metadata()["visit_indices"].tolist() == [0, 1, 0, 1]


def test_future_leakage_filtering_blocks_same_patient_future_and_exact_match() -> None:
    bank = VisitMemoryBank(split_name="train")
    for visit_index in range(3):
        bank.add(
            patient_id=11,
            visit_index=visit_index,
            visit_time=float(visit_index),
            visit_embedding=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
            medication_evidence=torch.tensor([float(visit_index), 0.0, 0.0], dtype=torch.float32),
        )
    bank.add(
        patient_id=22,
        visit_index=0,
        visit_time=0.0,
        visit_embedding=torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32),
        medication_evidence=torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
    )

    candidates = bank.get_candidate_pool(
        patient_id=11,
        visit_index=1,
        visit_time=1.0,
        allow_same_patient=True,
        exclude_future=True,
        exclude_exact_match=True,
    )

    assert set(candidates["visit_indices"].tolist()) == {0, 0}
    assert set(candidates["patient_ids"].tolist()) == {11, 22}
    assert 1 not in candidates["visit_indices"].tolist()
    assert 2 not in candidates["visit_indices"].tolist()


def test_topk_retriever_returns_weighted_medication_context() -> None:
    bank = VisitMemoryBank(split_name="train")
    bank.add(
        patient_id=31,
        visit_index=0,
        visit_time=0.0,
        visit_embedding=torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
        medication_evidence=torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    )
    bank.add(
        patient_id=32,
        visit_index=0,
        visit_time=0.0,
        visit_embedding=torch.tensor([0.8, 0.2, 0.0, 0.0], dtype=torch.float32),
        medication_evidence=torch.tensor([0.0, 1.0, 0.0, 0.0, 0.0], dtype=torch.float32),
    )
    bank.add(
        patient_id=33,
        visit_index=0,
        visit_time=0.0,
        visit_embedding=torch.tensor([0.0, 1.0, 0.0, 0.0], dtype=torch.float32),
        medication_evidence=torch.tensor([0.0, 0.0, 1.0, 0.0, 0.0], dtype=torch.float32),
    )

    retriever = TopKVisitRetriever(
        hidden_dim=4,
        drug_vocab_size=5,
        top_k=2,
        backend="bruteforce",
        allow_same_patient=False,
        exclude_future=True,
        exclude_exact_match=True,
        similarity_mode="cosine_decay",
        temporal_decay_alpha=0.1,
    )
    retriever.set_memory_bank(bank)
    outputs = retriever.retrieve(
        current_state=torch.tensor([[0.9, 0.1, 0.0, 0.0]], dtype=torch.float32),
        current_patient_ids=[999],
        current_visit_indices=[1],
        current_visit_times=torch.tensor([1.0], dtype=torch.float32),
    )

    assert outputs["aggregated_retrieval_context"].shape == (1, 4)
    assert outputs["retrieval_medication_context"].shape == (1, 5)
    assert outputs["retrieved_indices"].shape == (1, 2)
    assert outputs["retrieved_scores"].shape == (1, 2)
    assert outputs["retrieval_weights"].shape == (1, 2)
    assert outputs["retrieved_medication_evidence"].shape == (1, 2, 5)
    assert outputs["retrieval_weights"][0].sum().item() == pytest.approx(1.0)
    assert int(outputs["valid_candidate_counts"][0].item()) == 3


def test_faiss_index_falls_back_without_optional_dependency() -> None:
    index = VisitFaissIndex(backend="faiss", use_faiss_if_available=False)
    index.build_index(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
    outputs = index.search(torch.tensor([[1.0, 0.0]], dtype=torch.float32), top_k=1)

    assert outputs["indices"].shape == (1, 1)
    assert outputs["scores"].shape == (1, 1)
    assert int(outputs["indices"][0, 0].item()) == 0


def test_collate_batch_exposes_absolute_visit_times_when_intime_exists() -> None:
    batch = collate_batch(
        [
            {
                "subject_id": 1,
                "hadm_id": 10,
                "stay_id": 100,
                "patient_id": 1,
                "num_steps": 2,
                "intime": "2020-01-01 00:00:00",
                "steps": [
                    {"delta_hours": 0.0, "target_drugs": [0]},
                    {"delta_hours": 24.0, "target_drugs": [1]},
                ],
            }
        ]
    )

    assert "visit_time_absolute_hours" in batch
    assert "visit_time_absolute_mask" in batch
    assert bool(batch["visit_time_absolute_mask"][0, 0].item())
    assert bool(batch["visit_time_absolute_mask"][0, 1].item())
    assert batch["visit_time_absolute_hours"][0, 1].item() > batch["visit_time_absolute_hours"][0, 0].item()
    assert bool(batch["has_absolute_visit_time"][0].item())


def test_absolute_time_filter_blocks_cross_patient_future_candidates() -> None:
    bank = VisitMemoryBank(split_name="train")
    bank.add(
        patient_id=11,
        visit_index=0,
        visit_time=10.0,
        has_absolute_time=True,
        visit_embedding=torch.tensor([1.0, 0.0], dtype=torch.float32),
        medication_evidence=torch.tensor([1.0, 0.0], dtype=torch.float32),
    )
    bank.add(
        patient_id=22,
        visit_index=0,
        visit_time=12.0,
        has_absolute_time=True,
        visit_embedding=torch.tensor([0.0, 1.0], dtype=torch.float32),
        medication_evidence=torch.tensor([0.0, 1.0], dtype=torch.float32),
    )

    candidates = bank.get_candidate_pool(
        patient_id=99,
        visit_index=0,
        visit_time=11.0,
        query_has_absolute_time=True,
        allow_same_patient=False,
        exclude_future=True,
        exclude_exact_match=True,
        exclude_future_all_patients_if_absolute_time=True,
        require_absolute_time_for_cross_patient_temporal_filter=False,
    )

    assert candidates["patient_ids"].tolist() == [11]
    assert candidates["visit_times"].tolist() == [10.0]


def test_retriever_policy_reports_when_absolute_time_is_missing() -> None:
    bank = VisitMemoryBank(split_name="train")
    bank.add(
        patient_id=1,
        visit_index=0,
        visit_time=0.0,
        has_absolute_time=False,
        visit_embedding=torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        medication_evidence=torch.tensor([1.0, 0.0], dtype=torch.float32),
    )
    retriever = TopKVisitRetriever(hidden_dim=3, drug_vocab_size=2, top_k=1)
    retriever.set_memory_bank(bank)
    policy = retriever.describe_leakage_policy()

    assert not bool(policy["has_absolute_time"])
    assert bool(policy["same_patient_future_blocked"])
    assert not bool(policy["cross_patient_absolute_temporal_filter"])
