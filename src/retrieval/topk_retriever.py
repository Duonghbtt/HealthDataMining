from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

from src.models.temporal_similarity import cosine_similarity_matrix, temporal_decay_weights
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.memory_bank import MemoryBank


RETRIEVAL_PAYLOAD_FIELDS = (
    "query_stay_ids",
    "query_split",
    "neighbor_indices",
    "neighbor_scores",
    "neighbor_static_scores",
    "neighbor_time_gaps_days",
    "neighbor_subject_ids",
    "neighbor_hadm_ids",
    "neighbor_stay_ids",
    "backend",
    "bank_split",
    "retrieval_scoring_mode",
    "cross_split_policy",
)

_SUPPORTED_RETRIEVAL_SCORING_MODES = frozenset(
    {
        "static_cosine",
        "temporal_cosine",
        "temporal_relevance",
        "legacy",
    }
)
_SUPPORTED_CROSS_SPLIT_POLICIES = frozenset({"same_split", "train_bank_only", "allow_all"})


def _metadata_list(metadata: Mapping[str, Any], key: str, length: int) -> list[Any]:
    values = metadata.get(key)
    if values is None:
        return [None] * length
    if isinstance(values, torch.Tensor):
        return values.flatten().cpu().tolist()
    values = list(values)
    if len(values) != length:
        raise ValueError(f"Query metadata `{key}` must have length {length}, got {len(values)}")
    return values


def _jaccard(left: Sequence[int], right: Sequence[int]) -> float:
    left_set = set(int(item) for item in left)
    right_set = set(int(item) for item in right)
    if not left_set and not right_set:
        return 0.0
    union = left_set | right_set
    if not union:
        return 0.0
    return len(left_set & right_set) / float(len(union))


def _relevance_gate(query_meta: Mapping[str, Any], candidate_meta: Mapping[str, Any]) -> torch.Tensor:
    diag_scores = []
    proc_scores = []
    lab_scores = []
    vital_scores = []
    for row_index in range(len(query_meta["diag_code_sets"])):
        diag_scores.append(_jaccard(query_meta["diag_code_sets"][row_index], candidate_meta["diag_code_sets"][row_index]))
        proc_scores.append(_jaccard(query_meta["proc_code_sets"][row_index], candidate_meta["proc_code_sets"][row_index]))
        lab_scores.append(_jaccard(query_meta["lab_feature_sets"][row_index], candidate_meta["lab_feature_sets"][row_index]))
        vital_scores.append(_jaccard(query_meta["vital_feature_sets"][row_index], candidate_meta["vital_feature_sets"][row_index]))
    diag_tensor = torch.tensor(diag_scores, dtype=torch.float32)
    proc_tensor = torch.tensor(proc_scores, dtype=torch.float32)
    lab_tensor = torch.tensor(lab_scores, dtype=torch.float32)
    vital_tensor = torch.tensor(vital_scores, dtype=torch.float32)
    return 1.0 + 0.45 * diag_tensor + 0.25 * proc_tensor + 0.15 * lab_tensor + 0.15 * vital_tensor


def validate_retrieval_payload(payload: Mapping[str, Any]) -> None:
    missing = [field for field in RETRIEVAL_PAYLOAD_FIELDS if field not in payload]
    if missing:
        raise ValueError(f"Retrieval payload is missing required fields: {missing}")
    query_count = len(payload["query_stay_ids"])
    expected_width: int | None = None
    for field in (
        "neighbor_indices",
        "neighbor_scores",
        "neighbor_static_scores",
        "neighbor_time_gaps_days",
        "neighbor_subject_ids",
        "neighbor_hadm_ids",
        "neighbor_stay_ids",
    ):
        value = payload[field]
        if not isinstance(value, torch.Tensor):
            raise ValueError(f"Retrieval payload field `{field}` must be a torch.Tensor")
        if value.ndim != 2 or value.shape[0] != query_count:
            raise ValueError(f"Retrieval payload field `{field}` must have shape (B, K)")
        if expected_width is None:
            expected_width = int(value.shape[1])
        elif int(value.shape[1]) != expected_width:
            raise ValueError("All neighbor_* tensors in retrieval payload must share the same width")
    for optional_field in (
        "query_visit_indices",
        "matched_visit_indices",
        "aux_personal_history_indices",
        "aux_personal_history_scores",
        "aux_transversal_visit_indices",
    ):
        if optional_field in payload and not isinstance(payload[optional_field], torch.Tensor):
            raise ValueError(f"Optional retrieval payload field `{optional_field}` must be a torch.Tensor")


def _resolve_retrieval_scoring_mode(scoring_mode: str) -> str:
    resolved = str(scoring_mode).strip().lower()
    if resolved == "legacy":
        resolved = "temporal_relevance"
    if resolved not in _SUPPORTED_RETRIEVAL_SCORING_MODES:
        raise ValueError(
            f"Unsupported retrieval scoring_mode `{scoring_mode}`. "
            f"Expected one of {sorted(_SUPPORTED_RETRIEVAL_SCORING_MODES)}."
        )
    return resolved


def _resolve_cross_split_policy(
    *,
    allow_cross_split: bool | None = None,
    cross_split_policy: str | None = None,
) -> str:
    if cross_split_policy is None:
        return "allow_all" if bool(allow_cross_split) else "same_split"
    resolved = str(cross_split_policy).strip().lower()
    if resolved not in _SUPPORTED_CROSS_SPLIT_POLICIES:
        raise ValueError(
            f"Unsupported cross_split_policy `{cross_split_policy}`. "
            f"Expected one of {sorted(_SUPPORTED_CROSS_SPLIT_POLICIES)}."
        )
    return resolved


def _validate_split(query_split: Sequence[Any], bank_split: str, *, cross_split_policy: str) -> str:
    non_null_query_splits = {str(split) for split in query_split if split is not None}
    if len(non_null_query_splits) > 1:
        raise ValueError(
            "Mixed query splits are not supported because retrieval payload exposes a single query_split: "
            f"{sorted(non_null_query_splits)}"
        )
    resolved_query_split = bank_split if not non_null_query_splits else sorted(non_null_query_splits)[0]
    if cross_split_policy == "same_split" and non_null_query_splits and non_null_query_splits != {bank_split}:
        raise ValueError(
            f"Cross-split retrieval is disabled. Query splits={sorted(non_null_query_splits)} bank split={bank_split}."
        )
    if cross_split_policy == "train_bank_only" and str(bank_split) != "train":
        raise ValueError(
            "cross_split_policy=train_bank_only requires the retrieval bank to be built from the train split. "
            f"Received bank split={bank_split!r}."
        )
    return resolved_query_split


def _combine_neighbor_scores(
    *,
    static_scores: torch.Tensor,
    temporal_weights: torch.Tensor,
    relevance_gate: torch.Tensor,
    scoring_mode: str,
) -> torch.Tensor:
    resolved_mode = _resolve_retrieval_scoring_mode(scoring_mode)
    if resolved_mode == "static_cosine":
        return static_scores
    if resolved_mode == "temporal_cosine":
        return static_scores * temporal_weights
    return static_scores * temporal_weights * relevance_gate


def _candidate_indices_for_neighbors(
    query_states: torch.Tensor,
    memory_bank: MemoryBank,
    *,
    top_k: int,
    backend: str,
    use_faiss_if_available: bool,
) -> tuple[str, list[torch.Tensor] | None]:
    if backend == "faiss" and use_faiss_if_available and FaissIndex.is_available():
        search_k = min(len(memory_bank), max(top_k * 8, top_k))
        faiss_index = FaissIndex.build(memory_bank.visit_states)
        _, candidate_indices = faiss_index.search(query_states, search_k)
        return "faiss", [candidate_indices[row_index].unique() for row_index in range(candidate_indices.shape[0])]
    return "bruteforce", None


def _build_candidate_meta(memory_bank: MemoryBank, candidate_idx: list[int]) -> dict[str, Any]:
    return {
        "diag_code_sets": [memory_bank.diag_code_sets[index] for index in candidate_idx],
        "proc_code_sets": [memory_bank.proc_code_sets[index] for index in candidate_idx],
        "lab_feature_sets": [memory_bank.lab_feature_sets[index] for index in candidate_idx],
        "vital_feature_sets": [memory_bank.vital_feature_sets[index] for index in candidate_idx],
    }


def _pad_tensor_rows(rows: list[torch.Tensor], *, fill_value: float | int, dtype: torch.dtype) -> torch.Tensor:
    width = max((int(row.shape[0]) for row in rows), default=0)
    output = torch.full((len(rows), width), fill_value, dtype=dtype)
    for row_index, row in enumerate(rows):
        if row.numel():
            output[row_index, : row.shape[0]] = row.to(dtype=dtype)
    return output


def _stay_max_scores(
    scores: torch.Tensor,
    stay_group_ids: torch.Tensor,
    *,
    num_stays: int,
) -> torch.Tensor:
    output = torch.full((int(num_stays),), float("-inf"), dtype=scores.dtype)
    if hasattr(output, "scatter_reduce_"):
        output.scatter_reduce_(0, stay_group_ids, scores, reduce="amax", include_self=True)
        return output

    for row_index, stay_position in enumerate(stay_group_ids.tolist()):
        value = float(scores[row_index].item())
        if value > float(output[stay_position].item()):
            output[stay_position] = value
    return output


def _relevance_upper_bound(base_scores: torch.Tensor) -> torch.Tensor:
    return torch.where(base_scores >= 0.0, base_scores * 2.0, base_scores)


def _best_visit_entry(
    *,
    row_indices: torch.Tensor,
    final_scores: torch.Tensor,
    static_scores: torch.Tensor,
    query_time_day: float,
    memory_bank: MemoryBank,
) -> dict[str, Any]:
    if row_indices.numel() <= 0:
        raise ValueError("row_indices must contain at least one candidate")
    best_offset = int(torch.argmax(final_scores).item())
    global_index = int(row_indices[best_offset].item())
    return {
        "global_index": global_index,
        "score": float(final_scores[best_offset].item()),
        "static_score": float(static_scores[best_offset].item()),
        "time_gap": float(abs(float(query_time_day) - float(memory_bank.visit_time_days[global_index].item()))),
        "subject_id": int(memory_bank.subject_ids[global_index].item()),
        "hadm_id": int(memory_bank.hadm_ids[global_index].item()),
        "visit_index": int(memory_bank.visit_index[global_index].item()),
        "stay_id": int(memory_bank.stay_ids[global_index].item()),
    }


def _sort_ranked_entries(entries: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    if not entries:
        return []
    ranked = sorted(
        entries,
        key=lambda item: (-float(item["score"]), int(item["global_index"])),
    )
    return ranked[: int(top_k)]


def _relevance_gate_for_rows(
    *,
    query_row_meta: Mapping[str, Sequence[int]],
    candidate_indices: torch.Tensor,
    memory_bank: MemoryBank,
) -> torch.Tensor:
    diag_scores = []
    proc_scores = []
    lab_scores = []
    vital_scores = []
    for global_index in candidate_indices.tolist():
        candidate_index = int(global_index)
        diag_scores.append(
            _jaccard(query_row_meta["diag_code_sets"], memory_bank.diag_code_sets[candidate_index])
        )
        proc_scores.append(
            _jaccard(query_row_meta["proc_code_sets"], memory_bank.proc_code_sets[candidate_index])
        )
        lab_scores.append(
            _jaccard(query_row_meta["lab_feature_sets"], memory_bank.lab_feature_sets[candidate_index])
        )
        vital_scores.append(
            _jaccard(query_row_meta["vital_feature_sets"], memory_bank.vital_feature_sets[candidate_index])
        )
    diag_tensor = torch.tensor(diag_scores, dtype=torch.float32)
    proc_tensor = torch.tensor(proc_scores, dtype=torch.float32)
    lab_tensor = torch.tensor(lab_scores, dtype=torch.float32)
    vital_tensor = torch.tensor(vital_scores, dtype=torch.float32)
    return 1.0 + 0.45 * diag_tensor + 0.25 * proc_tensor + 0.15 * lab_tensor + 0.15 * vital_tensor


def _empty_neighbor_payload_rows() -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32)


def _retrieve_patient_neighbors_optimized_bruteforce(
    query_visit_states: torch.Tensor,
    query_metadata: Mapping[str, Any],
    memory_bank: MemoryBank,
    *,
    top_k: int,
    temporal_decay_alpha: float,
    allow_cross_split: bool | None = None,
    cross_split_policy: str | None = None,
    scoring_mode: str,
) -> dict[str, Any]:
    query_count = int(query_visit_states.shape[0])
    query_stay_ids = _metadata_list(query_metadata, "stay_ids", query_count)
    query_subject_ids = _metadata_list(query_metadata, "subject_ids", query_count)
    query_hadm_ids = _metadata_list(query_metadata, "hadm_ids", query_count)
    query_visit_indices = _metadata_list(query_metadata, "visit_indices", query_count)
    query_visit_times = _metadata_list(query_metadata, "visit_time_days", query_count)
    query_split = _metadata_list(query_metadata, "split", query_count)
    resolved_cross_split_policy = _resolve_cross_split_policy(
        allow_cross_split=allow_cross_split,
        cross_split_policy=cross_split_policy,
    )
    resolved_query_split = _validate_split(
        query_split,
        memory_bank.split,
        cross_split_policy=resolved_cross_split_policy,
    )
    resolved_scoring_mode = _resolve_retrieval_scoring_mode(scoring_mode)
    query_meta = {
        "diag_code_sets": _metadata_list(query_metadata, "diag_code_sets", query_count),
        "proc_code_sets": _metadata_list(query_metadata, "proc_code_sets", query_count),
        "lab_feature_sets": _metadata_list(query_metadata, "lab_feature_sets", query_count),
        "vital_feature_sets": _metadata_list(query_metadata, "vital_feature_sets", query_count),
    }

    query_tensor = torch.nn.functional.normalize(
        torch.as_tensor(query_visit_states, dtype=torch.float32).cpu(),
        p=2,
        dim=-1,
        eps=1.0e-12,
    )
    static_scores_matrix = query_tensor @ memory_bank.normalized_visit_states.T
    if resolved_scoring_mode == "static_cosine":
        base_scores_matrix = static_scores_matrix
    else:
        base_scores_matrix = static_scores_matrix * temporal_decay_weights(
            torch.tensor(query_visit_times, dtype=torch.float32),
            memory_bank.visit_time_days,
            alpha=temporal_decay_alpha,
        )

    all_visit_indices = torch.arange(len(memory_bank), dtype=torch.long)
    query_stay_tensor = torch.tensor(query_stay_ids, dtype=torch.long)
    cross_patient_mask = memory_bank.stay_ids.unsqueeze(0) != query_stay_tensor.unsqueeze(1)
    aux_transversal_visit_rows = [
        all_visit_indices[cross_patient_mask[row_index]].clone()
        for row_index in range(query_count)
    ]

    stay_group_ids = memory_bank.stay_group_ids
    rows_by_stay = memory_bank.rows_by_stay
    num_stays = len(rows_by_stay)

    neighbor_indices_rows: list[torch.Tensor] = []
    neighbor_scores_rows: list[torch.Tensor] = []
    neighbor_static_rows: list[torch.Tensor] = []
    neighbor_time_gap_rows: list[torch.Tensor] = []
    neighbor_subject_rows: list[torch.Tensor] = []
    neighbor_hadm_rows: list[torch.Tensor] = []
    neighbor_stay_rows: list[torch.Tensor] = []
    matched_visit_rows: list[torch.Tensor] = []

    personal_history = retrieve_personal_history(
        query_visit_states,
        query_metadata,
        memory_bank,
        top_k=top_k,
        temporal_decay_alpha=temporal_decay_alpha,
        scoring_mode=resolved_scoring_mode,
    )

    for row_index in range(query_count):
        valid_visit_mask = cross_patient_mask[row_index]
        if not bool(valid_visit_mask.any().item()):
            empty_long, empty_float = _empty_neighbor_payload_rows()
            neighbor_indices_rows.append(empty_long)
            neighbor_scores_rows.append(empty_float)
            neighbor_static_rows.append(empty_float)
            neighbor_time_gap_rows.append(empty_float)
            neighbor_subject_rows.append(empty_long)
            neighbor_hadm_rows.append(empty_long)
            neighbor_stay_rows.append(empty_long)
            matched_visit_rows.append(empty_long)
            continue

        masked_base_scores = base_scores_matrix[row_index].masked_fill(~valid_visit_mask, float("-inf"))
        masked_static_scores = static_scores_matrix[row_index].masked_fill(~valid_visit_mask, float("-inf"))

        ranked_entries: list[dict[str, Any]] = []
        if resolved_scoring_mode in {"static_cosine", "temporal_cosine"}:
            stay_scores = _stay_max_scores(
                masked_base_scores,
                stay_group_ids,
                num_stays=num_stays,
            )
            finite_mask = torch.isfinite(stay_scores)
            valid_stay_positions = torch.nonzero(finite_mask, as_tuple=False).flatten()
            keep = min(int(top_k), int(valid_stay_positions.shape[0]))
            if keep > 0:
                top_stay_positions = valid_stay_positions[
                    torch.topk(stay_scores[valid_stay_positions], k=keep, dim=0).indices
                ]
                for stay_position in top_stay_positions.tolist():
                    row_indices = rows_by_stay[int(stay_position)]
                    row_indices = row_indices[valid_visit_mask[row_indices]]
                    if row_indices.numel() <= 0:
                        continue
                    ranked_entries.append(
                        _best_visit_entry(
                            row_indices=row_indices,
                            final_scores=masked_base_scores[row_indices],
                            static_scores=masked_static_scores[row_indices],
                            query_time_day=float(query_visit_times[row_index]),
                            memory_bank=memory_bank,
                        )
                    )
        else:
            stay_upper = _stay_max_scores(
                _relevance_upper_bound(masked_base_scores),
                stay_group_ids,
                num_stays=num_stays,
            )
            finite_mask = torch.isfinite(stay_upper)
            ordered_stay_positions = torch.nonzero(finite_mask, as_tuple=False).flatten()
            if ordered_stay_positions.numel() > 0:
                ordered_stay_positions = ordered_stay_positions[
                    torch.argsort(stay_upper[ordered_stay_positions], descending=True)
                ]

            current_threshold = float("-inf")
            query_row_meta = {
                key: query_meta[key][row_index]
                for key in query_meta
            }
            for stay_position in ordered_stay_positions.tolist():
                if len(ranked_entries) >= int(top_k) and float(stay_upper[stay_position].item()) <= current_threshold:
                    break
                row_indices = rows_by_stay[int(stay_position)]
                row_indices = row_indices[valid_visit_mask[row_indices]]
                if row_indices.numel() <= 0:
                    continue
                exact_gate = _relevance_gate_for_rows(
                    query_row_meta=query_row_meta,
                    candidate_indices=row_indices,
                    memory_bank=memory_bank,
                )
                final_scores = masked_base_scores[row_indices] * exact_gate
                ranked_entries.append(
                    _best_visit_entry(
                        row_indices=row_indices,
                        final_scores=final_scores,
                        static_scores=masked_static_scores[row_indices],
                        query_time_day=float(query_visit_times[row_index]),
                        memory_bank=memory_bank,
                    )
                )
                ranked_entries = _sort_ranked_entries(ranked_entries, top_k=int(top_k))
                if len(ranked_entries) >= int(top_k):
                    current_threshold = float(ranked_entries[-1]["score"])

        ranked = _sort_ranked_entries(ranked_entries, top_k=int(top_k))
        neighbor_indices_rows.append(torch.tensor([item["global_index"] for item in ranked], dtype=torch.long))
        neighbor_scores_rows.append(torch.tensor([item["score"] for item in ranked], dtype=torch.float32))
        neighbor_static_rows.append(torch.tensor([item["static_score"] for item in ranked], dtype=torch.float32))
        neighbor_time_gap_rows.append(torch.tensor([item["time_gap"] for item in ranked], dtype=torch.float32))
        neighbor_subject_rows.append(torch.tensor([item["subject_id"] for item in ranked], dtype=torch.long))
        neighbor_hadm_rows.append(torch.tensor([item["hadm_id"] for item in ranked], dtype=torch.long))
        neighbor_stay_rows.append(torch.tensor([item["stay_id"] for item in ranked], dtype=torch.long))
        matched_visit_rows.append(torch.tensor([item["visit_index"] for item in ranked], dtype=torch.long))

    payload = {
        "query_stay_ids": [int(value) for value in query_stay_ids],
        "query_split": resolved_query_split,
        "query_visit_indices": torch.tensor(query_visit_indices, dtype=torch.long),
        "query_diag_code_sets": list(query_meta["diag_code_sets"]),
        "query_proc_code_sets": list(query_meta["proc_code_sets"]),
        "query_lab_feature_sets": list(query_meta["lab_feature_sets"]),
        "query_vital_feature_sets": list(query_meta["vital_feature_sets"]),
        "neighbor_indices": _pad_tensor_rows(neighbor_indices_rows, fill_value=-1, dtype=torch.long),
        "neighbor_scores": _pad_tensor_rows(neighbor_scores_rows, fill_value=float("-inf"), dtype=torch.float32),
        "neighbor_static_scores": _pad_tensor_rows(neighbor_static_rows, fill_value=float("-inf"), dtype=torch.float32),
        "neighbor_time_gaps_days": _pad_tensor_rows(neighbor_time_gap_rows, fill_value=float("inf"), dtype=torch.float32),
        "neighbor_subject_ids": _pad_tensor_rows(neighbor_subject_rows, fill_value=-1, dtype=torch.long),
        "neighbor_hadm_ids": _pad_tensor_rows(neighbor_hadm_rows, fill_value=-1, dtype=torch.long),
        "neighbor_stay_ids": _pad_tensor_rows(neighbor_stay_rows, fill_value=-1, dtype=torch.long),
        "matched_visit_indices": _pad_tensor_rows(matched_visit_rows, fill_value=-1, dtype=torch.long),
        "aux_personal_history_indices": personal_history["indices"],
        "aux_personal_history_scores": personal_history["scores"],
        "aux_transversal_visit_indices": _pad_tensor_rows(aux_transversal_visit_rows, fill_value=-1, dtype=torch.long),
        "backend": "bruteforce",
        "bank_split": memory_bank.split,
        "retrieval_scoring_mode": resolved_scoring_mode,
        "cross_split_policy": resolved_cross_split_policy,
        "query_subject_ids": torch.tensor(query_subject_ids, dtype=torch.long),
        "query_hadm_ids": torch.tensor(query_hadm_ids, dtype=torch.long),
    }
    validate_retrieval_payload(payload)
    return payload


def retrieve_personal_history(
    query_visit_states: torch.Tensor,
    query_metadata: Mapping[str, Any],
    memory_bank: MemoryBank,
    *,
    top_k: int,
    temporal_decay_alpha: float,
    scoring_mode: str = "temporal_relevance",
) -> dict[str, torch.Tensor]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if temporal_decay_alpha <= 0:
        raise ValueError("temporal_decay_alpha must be > 0")
    query_visit_states = torch.as_tensor(query_visit_states, dtype=torch.float32).cpu()
    query_count = int(query_visit_states.shape[0])
    query_stay_ids = _metadata_list(query_metadata, "stay_ids", query_count)
    query_visit_indices = _metadata_list(query_metadata, "visit_indices", query_count)
    query_visit_time_days = _metadata_list(query_metadata, "visit_time_days", query_count)
    query_meta = {
        "diag_code_sets": _metadata_list(query_metadata, "diag_code_sets", query_count),
        "proc_code_sets": _metadata_list(query_metadata, "proc_code_sets", query_count),
        "lab_feature_sets": _metadata_list(query_metadata, "lab_feature_sets", query_count),
        "vital_feature_sets": _metadata_list(query_metadata, "vital_feature_sets", query_count),
    }

    all_indices: list[torch.Tensor] = []
    all_scores: list[torch.Tensor] = []
    for row_index in range(query_count):
        mask = (memory_bank.stay_ids == int(query_stay_ids[row_index])) & (
            memory_bank.visit_index < int(query_visit_indices[row_index])
        )
        candidate_indices = torch.nonzero(mask, as_tuple=False).flatten()
        if candidate_indices.numel() == 0:
            all_indices.append(torch.empty(0, dtype=torch.long))
            all_scores.append(torch.empty(0, dtype=torch.float32))
            continue
        static_scores = cosine_similarity_matrix(
            query_visit_states[row_index : row_index + 1],
            memory_bank.visit_states[candidate_indices],
        )[0]
        temporal_weights = temporal_decay_weights(
            [query_visit_time_days[row_index]],
            memory_bank.visit_time_days[candidate_indices],
            alpha=temporal_decay_alpha,
        )[0]
        candidate_meta = _build_candidate_meta(memory_bank, candidate_indices.tolist())
        row_query_meta = {key: [value[row_index]] * int(candidate_indices.shape[0]) for key, value in query_meta.items()}
        relevance_gate = _relevance_gate(row_query_meta, candidate_meta)
        final_scores = _combine_neighbor_scores(
            static_scores=static_scores,
            temporal_weights=temporal_weights,
            relevance_gate=relevance_gate,
            scoring_mode=scoring_mode,
        )
        keep = min(top_k, int(final_scores.shape[0]))
        row_scores, row_order = torch.topk(final_scores, k=keep, dim=0)
        all_indices.append(candidate_indices[row_order])
        all_scores.append(row_scores)

    return {
        "indices": _pad_tensor_rows(all_indices, fill_value=-1, dtype=torch.long),
        "scores": _pad_tensor_rows(all_scores, fill_value=float("-inf"), dtype=torch.float32),
    }


def _retrieve_patient_neighbors_reference(
    query_visit_states: torch.Tensor,
    query_metadata: Mapping[str, Any],
    memory_bank: MemoryBank,
    *,
    top_k: int,
    temporal_decay_alpha: float,
    backend: str = "bruteforce",
    use_faiss_if_available: bool = True,
    allow_cross_split: bool | None = None,
    cross_split_policy: str | None = None,
    scoring_mode: str = "temporal_relevance",
) -> dict[str, Any]:
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    if temporal_decay_alpha <= 0:
        raise ValueError("temporal_decay_alpha must be > 0")
    query_visit_states = torch.as_tensor(query_visit_states, dtype=torch.float32).cpu()
    if query_visit_states.ndim != 2:
        raise ValueError("query_visit_states must have shape (B, D)")

    query_count = int(query_visit_states.shape[0])
    query_stay_ids = _metadata_list(query_metadata, "stay_ids", query_count)
    query_subject_ids = _metadata_list(query_metadata, "subject_ids", query_count)
    query_hadm_ids = _metadata_list(query_metadata, "hadm_ids", query_count)
    query_visit_indices = _metadata_list(query_metadata, "visit_indices", query_count)
    query_visit_times = _metadata_list(query_metadata, "visit_time_days", query_count)
    query_split = _metadata_list(query_metadata, "split", query_count)
    resolved_cross_split_policy = _resolve_cross_split_policy(
        allow_cross_split=allow_cross_split,
        cross_split_policy=cross_split_policy,
    )
    resolved_scoring_mode = _resolve_retrieval_scoring_mode(scoring_mode)
    resolved_query_split = _validate_split(
        query_split,
        memory_bank.split,
        cross_split_policy=resolved_cross_split_policy,
    )
    query_meta = {
        "diag_code_sets": _metadata_list(query_metadata, "diag_code_sets", query_count),
        "proc_code_sets": _metadata_list(query_metadata, "proc_code_sets", query_count),
        "lab_feature_sets": _metadata_list(query_metadata, "lab_feature_sets", query_count),
        "vital_feature_sets": _metadata_list(query_metadata, "vital_feature_sets", query_count),
    }

    resolved_backend, faiss_candidates = _candidate_indices_for_neighbors(
        query_visit_states,
        memory_bank,
        top_k=top_k,
        backend=backend,
        use_faiss_if_available=use_faiss_if_available,
    )

    neighbor_indices_rows: list[torch.Tensor] = []
    neighbor_scores_rows: list[torch.Tensor] = []
    neighbor_static_rows: list[torch.Tensor] = []
    neighbor_time_gap_rows: list[torch.Tensor] = []
    neighbor_subject_rows: list[torch.Tensor] = []
    neighbor_hadm_rows: list[torch.Tensor] = []
    neighbor_stay_rows: list[torch.Tensor] = []
    matched_visit_rows: list[torch.Tensor] = []
    aux_transversal_visit_rows: list[torch.Tensor] = []

    personal_history = retrieve_personal_history(
        query_visit_states,
        query_metadata,
        memory_bank,
        top_k=top_k,
        temporal_decay_alpha=temporal_decay_alpha,
        scoring_mode=resolved_scoring_mode,
    )

    for row_index in range(query_count):
        if faiss_candidates is None:
            candidate_indices = torch.arange(len(memory_bank), dtype=torch.long)
        else:
            candidate_indices = faiss_candidates[row_index]
        cross_patient_mask = memory_bank.stay_ids[candidate_indices] != int(query_stay_ids[row_index])
        candidate_indices = candidate_indices[cross_patient_mask]
        aux_transversal_visit_rows.append(candidate_indices.clone())
        if candidate_indices.numel() == 0:
            empty_long = torch.empty(0, dtype=torch.long)
            empty_float = torch.empty(0, dtype=torch.float32)
            neighbor_indices_rows.append(empty_long)
            neighbor_scores_rows.append(empty_float)
            neighbor_static_rows.append(empty_float)
            neighbor_time_gap_rows.append(empty_float)
            neighbor_subject_rows.append(empty_long)
            neighbor_hadm_rows.append(empty_long)
            neighbor_stay_rows.append(empty_long)
            matched_visit_rows.append(empty_long)
            continue

        candidate_states = memory_bank.visit_states[candidate_indices]
        static_scores = cosine_similarity_matrix(query_visit_states[row_index : row_index + 1], candidate_states)[0]
        temporal_weights = temporal_decay_weights(
            [query_visit_times[row_index]],
            memory_bank.visit_time_days[candidate_indices],
            alpha=temporal_decay_alpha,
        )[0]
        candidate_meta = _build_candidate_meta(memory_bank, candidate_indices.tolist())
        row_query_meta = {key: [value[row_index]] * int(candidate_indices.shape[0]) for key, value in query_meta.items()}
        relevance_gate = _relevance_gate(row_query_meta, candidate_meta)
        final_scores = _combine_neighbor_scores(
            static_scores=static_scores,
            temporal_weights=temporal_weights,
            relevance_gate=relevance_gate,
            scoring_mode=resolved_scoring_mode,
        )
        time_gaps = torch.abs(torch.tensor(query_visit_times[row_index], dtype=torch.float32) - memory_bank.visit_time_days[candidate_indices])

        best_by_stay: dict[int, dict[str, Any]] = {}
        for local_index, global_index in enumerate(candidate_indices.tolist()):
            stay_id = int(memory_bank.stay_ids[global_index].item())
            score = float(final_scores[local_index].item())
            previous = best_by_stay.get(stay_id)
            if previous is None or score > previous["score"]:
                best_by_stay[stay_id] = {
                    "global_index": global_index,
                    "score": score,
                    "static_score": float(static_scores[local_index].item()),
                    "time_gap": float(time_gaps[local_index].item()),
                    "subject_id": int(memory_bank.subject_ids[global_index].item()),
                    "hadm_id": int(memory_bank.hadm_ids[global_index].item()),
                    "visit_index": int(memory_bank.visit_index[global_index].item()),
                }

        ranked = sorted(best_by_stay.values(), key=lambda item: item["score"], reverse=True)[:top_k]
        neighbor_indices_rows.append(torch.tensor([item["global_index"] for item in ranked], dtype=torch.long))
        neighbor_scores_rows.append(torch.tensor([item["score"] for item in ranked], dtype=torch.float32))
        neighbor_static_rows.append(torch.tensor([item["static_score"] for item in ranked], dtype=torch.float32))
        neighbor_time_gap_rows.append(torch.tensor([item["time_gap"] for item in ranked], dtype=torch.float32))
        neighbor_subject_rows.append(torch.tensor([item["subject_id"] for item in ranked], dtype=torch.long))
        neighbor_hadm_rows.append(torch.tensor([item["hadm_id"] for item in ranked], dtype=torch.long))
        neighbor_stay_rows.append(torch.tensor([int(memory_bank.stay_ids[item["global_index"]].item()) for item in ranked], dtype=torch.long))
        matched_visit_rows.append(torch.tensor([item["visit_index"] for item in ranked], dtype=torch.long))

    payload = {
        "query_stay_ids": [int(value) for value in query_stay_ids],
        "query_split": resolved_query_split,
        "query_visit_indices": torch.tensor(query_visit_indices, dtype=torch.long),
        "query_diag_code_sets": list(query_meta["diag_code_sets"]),
        "query_proc_code_sets": list(query_meta["proc_code_sets"]),
        "query_lab_feature_sets": list(query_meta["lab_feature_sets"]),
        "query_vital_feature_sets": list(query_meta["vital_feature_sets"]),
        "neighbor_indices": _pad_tensor_rows(neighbor_indices_rows, fill_value=-1, dtype=torch.long),
        "neighbor_scores": _pad_tensor_rows(neighbor_scores_rows, fill_value=float("-inf"), dtype=torch.float32),
        "neighbor_static_scores": _pad_tensor_rows(neighbor_static_rows, fill_value=float("-inf"), dtype=torch.float32),
        "neighbor_time_gaps_days": _pad_tensor_rows(neighbor_time_gap_rows, fill_value=float("inf"), dtype=torch.float32),
        "neighbor_subject_ids": _pad_tensor_rows(neighbor_subject_rows, fill_value=-1, dtype=torch.long),
        "neighbor_hadm_ids": _pad_tensor_rows(neighbor_hadm_rows, fill_value=-1, dtype=torch.long),
        "neighbor_stay_ids": _pad_tensor_rows(neighbor_stay_rows, fill_value=-1, dtype=torch.long),
        "matched_visit_indices": _pad_tensor_rows(matched_visit_rows, fill_value=-1, dtype=torch.long),
        "aux_personal_history_indices": personal_history["indices"],
        "aux_personal_history_scores": personal_history["scores"],
        "aux_transversal_visit_indices": _pad_tensor_rows(aux_transversal_visit_rows, fill_value=-1, dtype=torch.long),
        "backend": resolved_backend,
        "bank_split": memory_bank.split,
        "retrieval_scoring_mode": resolved_scoring_mode,
        "cross_split_policy": resolved_cross_split_policy,
        "query_subject_ids": torch.tensor(query_subject_ids, dtype=torch.long),
        "query_hadm_ids": torch.tensor(query_hadm_ids, dtype=torch.long),
    }
    validate_retrieval_payload(payload)
    return payload


def retrieve_patient_neighbors(
    query_visit_states: torch.Tensor,
    query_metadata: Mapping[str, Any],
    memory_bank: MemoryBank,
    *,
    top_k: int,
    temporal_decay_alpha: float,
    backend: str = "bruteforce",
    use_faiss_if_available: bool = True,
    allow_cross_split: bool | None = None,
    cross_split_policy: str | None = None,
    scoring_mode: str = "temporal_relevance",
) -> dict[str, Any]:
    resolved_backend = str(backend).strip().lower()
    if resolved_backend == "bruteforce":
        return _retrieve_patient_neighbors_optimized_bruteforce(
            query_visit_states,
            query_metadata,
            memory_bank,
            top_k=top_k,
            temporal_decay_alpha=temporal_decay_alpha,
            allow_cross_split=allow_cross_split,
            cross_split_policy=cross_split_policy,
            scoring_mode=scoring_mode,
        )
    return _retrieve_patient_neighbors_reference(
        query_visit_states,
        query_metadata,
        memory_bank,
        top_k=top_k,
        temporal_decay_alpha=temporal_decay_alpha,
        backend=backend,
        use_faiss_if_available=use_faiss_if_available,
        allow_cross_split=allow_cross_split,
        cross_split_policy=cross_split_policy,
        scoring_mode=scoring_mode,
    )


def retrieve_topk(
    query_embeddings: torch.Tensor,
    query_metadata: Mapping[str, Any],
    memory_bank: MemoryBank,
    *,
    top_k: int,
    temporal_decay_alpha: float,
    exclude_same_stay: bool = True,
    backend: str = "bruteforce",
    use_faiss_if_available: bool = True,
    allow_cross_split: bool | None = None,
    cross_split_policy: str | None = None,
    scoring_mode: str = "temporal_relevance",
) -> dict[str, Any]:
    if not exclude_same_stay:
        raise ValueError("Visit-centric retrieval only supports exclude_same_stay=True for cross-patient neighbors")
    return retrieve_patient_neighbors(
        query_embeddings,
        query_metadata,
        memory_bank,
        top_k=top_k,
        temporal_decay_alpha=temporal_decay_alpha,
        backend=backend,
        use_faiss_if_available=use_faiss_if_available,
        allow_cross_split=allow_cross_split,
        cross_split_policy=cross_split_policy,
        scoring_mode=scoring_mode,
    )
