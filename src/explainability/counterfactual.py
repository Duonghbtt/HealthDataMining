from __future__ import annotations

import copy
from typing import Any, Mapping, Sequence

import torch

from src.models.fusion import BRANCH_ORDER
from src.models.full_model import RetrievalEvidenceFusionModel


_DEFAULT_INTERVENTIONS = (
    "mask_last_visit_diagnosis",
    "mask_last_visit_procedure",
    "mask_last_visit_lab",
    "mask_last_visit_vital",
    "mask_last_visit_med_history",
    "drop_self_evidence",
    "drop_neighbor_evidence",
    "drop_group_evidence",
)

_INTERVENTION_CONFIGS: dict[str, dict[str, str]] = {
    "mask_last_visit_diagnosis": {"kind": "feature_mask", "description": "ẩn nhóm chẩn đoán ở visit cuối"},
    "mask_last_visit_procedure": {"kind": "feature_mask", "description": "ẩn nhóm thủ thuật ở visit cuối"},
    "mask_last_visit_lab": {"kind": "feature_mask", "description": "ẩn nhóm xét nghiệm ở visit cuối"},
    "mask_last_visit_vital": {"kind": "feature_mask", "description": "ẩn nhóm dấu hiệu sinh tồn ở visit cuối"},
    "mask_last_visit_med_history": {"kind": "feature_mask", "description": "ẩn tiền sử thuốc ở visit cuối"},
    "drop_self_evidence": {"kind": "branch_drop", "description": "bỏ evidence từ lịch sử của chính bệnh nhân"},
    "drop_neighbor_evidence": {"kind": "branch_drop", "description": "bỏ evidence từ các ca tương tự"},
    "drop_group_evidence": {"kind": "branch_drop", "description": "bỏ evidence từ nhóm bệnh nhân tương đồng"},
}


def _ensure_model(model: RetrievalEvidenceFusionModel) -> RetrievalEvidenceFusionModel:
    if not isinstance(model, RetrievalEvidenceFusionModel):
        raise TypeError("run_counterfactual_analysis expects RetrievalEvidenceFusionModel.")
    missing = [
        name
        for name in ("encoder", "history_selector", "fusion_module", "medication_decoder")
        if getattr(model, name, None) is None
    ]
    if missing:
        raise ValueError(f"Counterfactual analysis requires full model modules, missing: {missing}")
    return model


def _clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.clone() if isinstance(value, torch.Tensor) else copy.deepcopy(value)
        for key, value in batch.items()
    }


def _require_single_sample(batch: Mapping[str, Any]) -> int:
    visit_mask = batch.get("visit_mask")
    if not isinstance(visit_mask, torch.Tensor):
        raise TypeError("batch must contain `visit_mask` as a torch.Tensor.")
    if visit_mask.ndim != 2:
        raise ValueError(f"visit_mask must have shape (B, T), got {tuple(visit_mask.shape)}")
    if int(visit_mask.shape[0]) != 1:
        raise ValueError(f"Counterfactual analysis supports batch size 1 only, got {int(visit_mask.shape[0])}.")
    valid_count = int(visit_mask[0].to(dtype=torch.long).sum().item())
    if valid_count <= 0:
        raise ValueError("Counterfactual analysis requires at least one valid visit.")
    return valid_count - 1


def _resolve_mode(model: RetrievalEvidenceFusionModel, mode: str | None) -> str:
    resolved = model.mode if mode is None else mode
    if resolved not in {"core", "extended"}:
        raise ValueError(f"mode must be `core` or `extended`, got {resolved!r}")
    return resolved


def _resolve_top_k(top_k: int, vocab_size: int) -> int:
    if int(top_k) <= 0:
        raise ValueError(f"top_k must be positive, got {top_k!r}")
    return min(int(top_k), int(vocab_size))


def _resolve_ddi_upper(ddi_matrix: torch.Tensor | None) -> torch.Tensor | None:
    if ddi_matrix is None:
        return None
    matrix = torch.as_tensor(ddi_matrix, dtype=torch.float32)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"ddi_matrix must have square shape (D, D), got {tuple(matrix.shape)}")
    binary = torch.maximum((matrix > 0).to(dtype=torch.bool), (matrix > 0).transpose(0, 1))
    return torch.triu(binary, diagonal=1).cpu()


def _available_ddi_upper(model: RetrievalEvidenceFusionModel, ddi_matrix: torch.Tensor | None) -> torch.Tensor | None:
    if ddi_matrix is not None:
        return _resolve_ddi_upper(ddi_matrix)
    ddi_upper = getattr(getattr(model, "ddi_regularizer", None), "ddi_upper", None)
    return None if ddi_upper is None else torch.as_tensor(ddi_upper, dtype=torch.bool).detach().cpu()


def _to_cpu_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype) if dtype is not None else torch.as_tensor(value)
    return tensor.detach().cpu()


def _predicted_indices(drug_probs: torch.Tensor, threshold: float) -> list[int]:
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold!r}")
    if drug_probs.ndim != 2 or int(drug_probs.shape[0]) != 1:
        raise ValueError(f"drug_probs must have shape (1, D), got {tuple(drug_probs.shape)}")
    return sorted(int(i) for i in torch.nonzero(drug_probs[0] >= float(threshold), as_tuple=False).flatten().tolist())


def _top_recommendations(drug_probs: torch.Tensor, top_k: int) -> list[dict[str, float | int]]:
    k = _resolve_top_k(top_k, int(drug_probs.shape[1]))
    scores, indices = torch.topk(drug_probs[0], k=k, dim=-1)
    return [{"drug_index": int(index), "score": float(score)} for score, index in zip(scores.tolist(), indices.tolist())]


def _interacting_pairs(predicted_drugs: Sequence[int], ddi_upper: torch.Tensor | None) -> list[list[int]]:
    if ddi_upper is None or len(predicted_drugs) < 2:
        return []
    idx = torch.as_tensor(list(predicted_drugs), dtype=torch.long)
    subset = ddi_upper.index_select(0, idx).index_select(1, idx)
    pairs: list[list[int]] = []
    for row_index in range(subset.shape[0]):
        for col_index in range(row_index + 1, subset.shape[1]):
            if bool(subset[row_index, col_index].item()):
                pairs.append([int(predicted_drugs[row_index]), int(predicted_drugs[col_index])])
    return pairs


def _safety_summary(
    *,
    drug_probs: torch.Tensor,
    threshold: float,
    ddi_penalty: float | None,
    ddi_upper: torch.Tensor | None,
) -> dict[str, Any]:
    predicted_drugs = _predicted_indices(drug_probs, threshold)
    interacting_pairs = _interacting_pairs(predicted_drugs, ddi_upper)
    return {
        "predicted_drug_count": len(predicted_drugs),
        "predicted_drugs": list(predicted_drugs),
        "ddi_penalty": ddi_penalty,
        "has_thresholded_ddi": bool(interacting_pairs),
        "thresholded_interacting_pair_count": len(interacting_pairs),
        "thresholded_total_pair_count": len(predicted_drugs) * (len(predicted_drugs) - 1) // 2,
        "thresholded_interacting_pairs": interacting_pairs,
    }


def _fusion_weights_dict(fusion_outputs: Mapping[str, Any]) -> dict[str, float]:
    weights = _to_cpu_tensor(fusion_outputs["fusion_weights"], dtype=torch.float32)
    branch_order = fusion_outputs.get("branch_order", BRANCH_ORDER)
    return {str(name): float(weights[0, index].item()) for index, name in enumerate(branch_order)}


def _top_history_item(*, weights: Any, mask: Any, indices: Any, stay_ids: Any | None = None, matched_visit_indices: Any | None = None) -> dict[str, Any] | None:
    weight_tensor = _to_cpu_tensor(weights, dtype=torch.float32)
    mask_tensor = _to_cpu_tensor(mask, dtype=torch.bool)
    if weight_tensor.ndim != 2 or mask_tensor.ndim != 2 or weight_tensor.numel() == 0 or not bool(mask_tensor[0].any().item()):
        return None
    pos = int(weight_tensor[0].masked_fill(~mask_tensor[0], float("-inf")).argmax(dim=-1).item())
    payload: dict[str, Any] = {
        "weight": float(weight_tensor[0, pos].item()),
        "index": int(_to_cpu_tensor(indices, dtype=torch.long)[0, pos].item()),
    }
    if stay_ids is not None:
        payload["stay_id"] = int(_to_cpu_tensor(stay_ids, dtype=torch.long)[0, pos].item())
    if matched_visit_indices is not None:
        payload["matched_visit_index"] = int(_to_cpu_tensor(matched_visit_indices, dtype=torch.long)[0, pos].item())
    return payload


def _top_attribute_groups(evidence_metadata: Mapping[str, Any], max_items: int = 2) -> dict[str, list[dict[str, Any]]]:
    attribute_order = list(evidence_metadata.get("attribute_order", []))
    if not attribute_order:
        return {"self": [], "neighbor": []}
    result: dict[str, list[dict[str, Any]]] = {}
    for prefix in ("self", "neighbor"):
        weights_value = evidence_metadata.get(f"{prefix}_attribute_weights")
        mask_value = evidence_metadata.get(f"{prefix}_attribute_mask")
        if weights_value is None or mask_value is None:
            result[prefix] = []
            continue
        weights = _to_cpu_tensor(weights_value, dtype=torch.float32)
        mask = _to_cpu_tensor(mask_value, dtype=torch.bool)
        if weights.ndim != 3 or mask.ndim != 3 or weights.numel() == 0 or not bool(mask.any().item()):
            result[prefix] = []
            continue
        rows = []
        for attribute_index, attribute_name in enumerate(attribute_order):
            valid = weights[0, :, attribute_index][mask[0, :, attribute_index]]
            if valid.numel() > 0:
                rows.append({"attribute": str(attribute_name), "mean_weight": float(valid.mean().item())})
        rows.sort(key=lambda item: item["mean_weight"], reverse=True)
        result[prefix] = rows[:max_items]
    return result


def _evidence_summary(raw_outputs: Mapping[str, Any]) -> dict[str, Any]:
    evidence_metadata = raw_outputs["evidence_metadata"]
    fusion_outputs = raw_outputs["fusion_outputs"]
    branch_mask = _to_cpu_tensor(fusion_outputs["branch_mask"], dtype=torch.bool)
    branch_order = fusion_outputs.get("branch_order", BRANCH_ORDER)
    return {
        "dominant_branch": str(fusion_outputs["dominant_branch_name"][0]),
        "fusion_weights": _fusion_weights_dict(fusion_outputs),
        "branch_available": {str(name): bool(branch_mask[0, index].item()) for index, name in enumerate(branch_order)},
        "top_self_history": _top_history_item(
            weights=evidence_metadata.get("self_history_weights", torch.zeros(1, 0)),
            mask=evidence_metadata.get("self_history_mask", torch.zeros(1, 0, dtype=torch.bool)),
            indices=evidence_metadata.get("self_history_indices", torch.zeros(1, 0, dtype=torch.long)),
        ),
        "top_neighbor_history": _top_history_item(
            weights=evidence_metadata.get("neighbor_weights", torch.zeros(1, 0)),
            mask=evidence_metadata.get("neighbor_mask", torch.zeros(1, 0, dtype=torch.bool)),
            indices=evidence_metadata.get("neighbor_indices", torch.zeros(1, 0, dtype=torch.long)),
            stay_ids=evidence_metadata.get("neighbor_stay_ids"),
            matched_visit_indices=evidence_metadata.get("neighbor_matched_visit_indices"),
        ),
        "top_attribute_groups": _top_attribute_groups(evidence_metadata),
    }


def _validate_interventions(candidate_interventions: Sequence[str] | None) -> list[str]:
    names = list(_DEFAULT_INTERVENTIONS if candidate_interventions is None else candidate_interventions)
    unknown = [name for name in names if name not in _INTERVENTION_CONFIGS]
    if unknown:
        raise ValueError(f"Unsupported counterfactual interventions: {unknown}")
    return names


def _feature_available(batch: Mapping[str, Any], intervention_name: str, last_visit_index: int) -> bool:
    feature_map = {
        "mask_last_visit_diagnosis": "diag_mask",
        "mask_last_visit_procedure": "proc_mask",
        "mask_last_visit_lab": "lab_mask",
        "mask_last_visit_vital": "vital_mask",
        "mask_last_visit_med_history": "med_history_mask",
    }
    mask_tensor = batch.get(feature_map[intervention_name])
    return bool(isinstance(mask_tensor, torch.Tensor) and mask_tensor.ndim == 3 and int(mask_tensor.shape[0]) == 1 and int(mask_tensor.shape[2]) > 0 and mask_tensor[0, last_visit_index].any().item())


def _apply_feature_intervention(batch: Mapping[str, Any], intervention_name: str, *, last_visit_index: int) -> dict[str, Any]:
    perturbed = _clone_batch(batch)
    if intervention_name == "mask_last_visit_diagnosis":
        perturbed["diag_codes"][0, last_visit_index] = 0
        perturbed["diag_mask"][0, last_visit_index] = False
    elif intervention_name == "mask_last_visit_procedure":
        perturbed["proc_codes"][0, last_visit_index] = 0
        perturbed["proc_mask"][0, last_visit_index] = False
    elif intervention_name == "mask_last_visit_lab":
        perturbed["lab_values"][0, last_visit_index] = 0.0
        perturbed["lab_mask"][0, last_visit_index] = False
    elif intervention_name == "mask_last_visit_vital":
        perturbed["vital_values"][0, last_visit_index] = 0.0
        perturbed["vital_mask"][0, last_visit_index] = False
    elif intervention_name == "mask_last_visit_med_history":
        perturbed["med_history"][0, last_visit_index] = 0
        perturbed["med_history_mask"][0, last_visit_index] = False
    else:
        raise ValueError(f"Unsupported feature intervention: {intervention_name}")
    return perturbed


def _branch_drop_target(intervention_name: str) -> str:
    return {
        "drop_self_evidence": "self",
        "drop_neighbor_evidence": "neighbor",
        "drop_group_evidence": "group",
    }[intervention_name]


def _run_variant(
    model: RetrievalEvidenceFusionModel,
    batch: Mapping[str, Any],
    *,
    threshold: float,
    top_k: int,
    mode: str | None,
    retrieval_payload: Mapping[str, Any] | None,
    memory_bank: Any,
    records: Sequence[Mapping[str, Any]] | None,
    query_metadata: Mapping[str, Any] | None,
    query_states: torch.Tensor | None,
    ddi_upper: torch.Tensor | None,
    drop_branches: Sequence[str] | None = None,
) -> dict[str, Any]:
    encoder_outputs = model.encoder(dict(batch))
    current_state = encoder_outputs["pooled_state"]
    resolved_mode = _resolve_mode(model, mode)
    resolved_retrieval_payload, retrieval_mode = model._build_retrieval_payload_if_possible(
        encoder_outputs,
        mode=resolved_mode,
        retrieval_payload=retrieval_payload,
        memory_bank=memory_bank,
        records=records,
        query_metadata=query_metadata,
        query_states=query_states,
    )
    retrieval_available = resolved_retrieval_payload is not None
    retrieval_used = retrieval_available and memory_bank is not None
    group_outputs = model._build_group_outputs(
        current_state=current_state,
        retrieval_payload=resolved_retrieval_payload if retrieval_used else None,
        memory_bank=memory_bank if retrieval_used else None,
    )
    selection_outputs = model.history_selector(
        current_state=current_state,
        state_sequence=encoder_outputs["state_sequence"],
        visit_mask=encoder_outputs["visit_mask"],
        retrieval_payload=resolved_retrieval_payload,
        memory_bank=memory_bank,
        group_context=group_outputs.get("group_context"),
        group_available_mask=group_outputs.get("group_available_mask"),
        attribute_payload=None,
    )
    evidence_metadata = dict(selection_outputs["evidence_metadata"])
    evidence_metadata["group_metadata"] = group_outputs.get("group_metadata", [])
    branch_masks = {
        "current": torch.ones(current_state.shape[0], dtype=torch.bool, device=current_state.device),
        "self": evidence_metadata["self_history_available_mask"],
        "neighbor": evidence_metadata["neighbor_available_mask"],
        "group": evidence_metadata["group_available_mask"],
    }
    self_context = selection_outputs["self_history_context"]
    neighbor_context = selection_outputs["neighbor_history_context"]
    group_context = selection_outputs.get("group_context")
    for branch_name in set(drop_branches or []):
        if branch_name == "self":
            self_context = torch.zeros_like(current_state)
            branch_masks["self"] = torch.zeros_like(branch_masks["self"], dtype=torch.bool)
        elif branch_name == "neighbor":
            neighbor_context = torch.zeros_like(current_state)
            branch_masks["neighbor"] = torch.zeros_like(branch_masks["neighbor"], dtype=torch.bool)
        elif branch_name == "group":
            group_context = torch.zeros_like(current_state)
            branch_masks["group"] = torch.zeros_like(branch_masks["group"], dtype=torch.bool)
    fusion_outputs = model.fusion_module(
        current_state=current_state,
        self_history_context=self_context,
        neighbor_history_context=neighbor_context,
        group_context=group_context,
        branch_masks=branch_masks,
    )
    decoder_outputs = model.medication_decoder(fusion_outputs["fused_repr"], top_k=top_k)
    drug_probs = decoder_outputs["drug_probs"].detach().cpu()
    ddi_penalty = None
    if model.ddi_regularizer is not None:
        ddi_penalty = float(model.ddi_regularizer.compute_penalty_per_sample(decoder_outputs["drug_probs"]).mean().detach().cpu().item())
    return {
        "fusion_outputs": fusion_outputs,
        "decoder_outputs": decoder_outputs,
        "evidence_metadata": evidence_metadata,
        "retrieval_mode": retrieval_mode,
        "retrieval_available": retrieval_available,
        "retrieval_used": retrieval_used,
        "safety_summary": _safety_summary(drug_probs=drug_probs, threshold=threshold, ddi_penalty=ddi_penalty, ddi_upper=ddi_upper),
    }


def _mean_abs_topk_delta(baseline_probs: torch.Tensor, variant_probs: torch.Tensor, *, top_k: int) -> float:
    before_top = {item["drug_index"] for item in _top_recommendations(baseline_probs, top_k)}
    after_top = {item["drug_index"] for item in _top_recommendations(variant_probs, top_k)}
    compare_indices = sorted(before_top | after_top)
    if not compare_indices:
        return 0.0
    index_tensor = torch.as_tensor(compare_indices, dtype=torch.long)
    return float(torch.abs(baseline_probs[0].index_select(0, index_tensor) - variant_probs[0].index_select(0, index_tensor)).mean().item())


def _ddi_delta(before_summary: Mapping[str, Any], after_summary: Mapping[str, Any]) -> float:
    before_penalty = before_summary.get("ddi_penalty")
    after_penalty = after_summary.get("ddi_penalty")
    if before_penalty is not None and after_penalty is not None:
        return float(after_penalty) - float(before_penalty)
    return float(after_summary.get("thresholded_interacting_pair_count", 0)) - float(before_summary.get("thresholded_interacting_pair_count", 0))


def _build_intervention_record(
    *,
    name: str,
    description: str,
    kind: str,
    baseline: Mapping[str, Any],
    baseline_raw: Mapping[str, Any],
    variant_raw: Mapping[str, Any],
    top_k: int,
    threshold: float,
) -> dict[str, Any]:
    baseline_probs = baseline_raw["decoder_outputs"]["drug_probs"].detach().cpu()
    variant_probs = variant_raw["decoder_outputs"]["drug_probs"].detach().cpu()
    predicted_before = list(baseline["predicted_drugs"])
    predicted_after = list(variant_raw["safety_summary"]["predicted_drugs"])
    entered_drugs = sorted(set(predicted_after) - set(predicted_before))
    removed_drugs = sorted(set(predicted_before) - set(predicted_after))
    changed_drug_count = len(entered_drugs) + len(removed_drugs)
    ddi_change = _ddi_delta(baseline["safety_summary"], variant_raw["safety_summary"])
    mean_delta = _mean_abs_topk_delta(baseline_probs, variant_probs, top_k=top_k)
    return {
        "name": name,
        "description": description,
        "kind": kind,
        "available": True,
        "topk_before": baseline["top_recommendations"],
        "topk_after": _top_recommendations(variant_probs, top_k),
        "predicted_drugs_before": predicted_before,
        "predicted_drugs_after": predicted_after,
        "entered_drugs": entered_drugs,
        "removed_drugs": removed_drugs,
        "changed_drug_count": changed_drug_count,
        "mean_abs_topk_prob_delta": mean_delta,
        "ddi_penalty_before": baseline["safety_summary"].get("ddi_penalty"),
        "ddi_penalty_after": variant_raw["safety_summary"].get("ddi_penalty"),
        "ddi_delta": ddi_change,
        "fusion_weights_before": baseline["fusion_weights"],
        "fusion_weights_after": _fusion_weights_dict(variant_raw["fusion_outputs"]),
        "dominant_branch_before": baseline["dominant_branch"],
        "dominant_branch_after": str(variant_raw["fusion_outputs"]["dominant_branch_name"][0]),
        "impact_score": float(changed_drug_count) + mean_delta + abs(ddi_change),
        "threshold": float(threshold),
    }


def run_counterfactual_analysis(
    model: RetrievalEvidenceFusionModel,
    batch: Mapping[str, Any],
    *,
    threshold: float = 0.5,
    top_k: int = 5,
    mode: str = "core",
    retrieval_payload: Mapping[str, Any] | None = None,
    memory_bank: Any = None,
    records: Sequence[Mapping[str, Any]] | None = None,
    query_metadata: Mapping[str, Any] | None = None,
    query_states: torch.Tensor | None = None,
    ddi_matrix: torch.Tensor | None = None,
    candidate_interventions: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Run a lightweight counterfactual analysis for a single patient batch."""

    model = _ensure_model(model)
    intervention_names = _validate_interventions(candidate_interventions)
    last_visit_index = _require_single_sample(batch)
    ddi_upper = _available_ddi_upper(model, ddi_matrix)

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            baseline_raw = _run_variant(
                model,
                batch,
                threshold=threshold,
                top_k=top_k,
                mode=mode,
                retrieval_payload=retrieval_payload,
                memory_bank=memory_bank,
                records=records,
                query_metadata=query_metadata,
                query_states=query_states,
                ddi_upper=ddi_upper,
            )
            baseline_probs = baseline_raw["decoder_outputs"]["drug_probs"].detach().cpu()
            baseline_top = _top_recommendations(baseline_probs, top_k)
            baseline_evidence = _evidence_summary(baseline_raw)
            baseline = {
                "top_recommendations": baseline_top,
                "predicted_drugs": list(baseline_raw["safety_summary"]["predicted_drugs"]),
                "fusion_weights": _fusion_weights_dict(baseline_raw["fusion_outputs"]),
                "dominant_branch": baseline_evidence["dominant_branch"],
                "evidence_summary": baseline_evidence,
                "safety_summary": baseline_raw["safety_summary"],
                "recommendation_metadata": {
                    "topk_indices": [int(item["drug_index"]) for item in baseline_top],
                    "topk_scores": [float(item["score"]) for item in baseline_top],
                },
                "retrieval_mode": str(baseline_raw["retrieval_mode"]),
                "retrieval_available": bool(baseline_raw["retrieval_available"]),
                "retrieval_used": bool(baseline_raw["retrieval_used"]),
            }

            interventions: list[dict[str, Any]] = []
            for intervention_name in intervention_names:
                config = _INTERVENTION_CONFIGS[intervention_name]
                kind = config["kind"]
                description = config["description"]
                if kind == "feature_mask":
                    if not _feature_available(batch, intervention_name, last_visit_index):
                        interventions.append({"name": intervention_name, "description": description, "kind": kind, "available": False, "reason": "feature_group_not_present_on_last_visit"})
                        continue
                    variant_batch = _apply_feature_intervention(batch, intervention_name, last_visit_index=last_visit_index)
                    variant_raw = _run_variant(
                        model,
                        variant_batch,
                        threshold=threshold,
                        top_k=top_k,
                        mode=mode,
                        retrieval_payload=retrieval_payload,
                        memory_bank=memory_bank,
                        records=records,
                        query_metadata=query_metadata,
                        query_states=query_states,
                        ddi_upper=ddi_upper,
                    )
                else:
                    dropped_branch = _branch_drop_target(intervention_name)
                    if not bool(baseline_raw["fusion_outputs"]["branch_mask"][0, BRANCH_ORDER.index(dropped_branch)].item()):
                        interventions.append({"name": intervention_name, "description": description, "kind": kind, "available": False, "reason": "evidence_branch_not_available"})
                        continue
                    variant_raw = _run_variant(
                        model,
                        batch,
                        threshold=threshold,
                        top_k=top_k,
                        mode=mode,
                        retrieval_payload=retrieval_payload,
                        memory_bank=memory_bank,
                        records=records,
                        query_metadata=query_metadata,
                        query_states=query_states,
                        ddi_upper=ddi_upper,
                        drop_branches=[dropped_branch],
                    )
                interventions.append(
                    _build_intervention_record(
                        name=intervention_name,
                        description=description,
                        kind=kind,
                        baseline=baseline,
                        baseline_raw=baseline_raw,
                        variant_raw=variant_raw,
                        top_k=top_k,
                        threshold=threshold,
                    )
                )

            available = [item for item in interventions if item.get("available")]
            return {
                "baseline": baseline,
                "interventions": interventions,
                "best_counterfactual": max(available, key=lambda item: float(item["impact_score"])) if available else None,
                "metadata": {
                    "threshold": float(threshold),
                    "top_k": int(top_k),
                    "mode": str(mode),
                    "candidate_interventions": intervention_names,
                    "last_visit_index": int(last_visit_index),
                    "retrieval_used": bool(baseline_raw["retrieval_used"]),
                },
            }
    finally:
        model.train(was_training)


__all__ = ["run_counterfactual_analysis"]
