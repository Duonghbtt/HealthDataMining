from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from src.models.fusion import BRANCH_ORDER
from src.utils.io import ensure_dir, resolve_path, write_csv_gz, write_json


def _to_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    return torch.as_tensor(value, dtype=dtype) if dtype is not None else torch.as_tensor(value)


def _attribute_dict(
    attribute_order: list[str],
    weights: torch.Tensor,
    scores: torch.Tensor,
    row_index: int,
    position: int,
) -> dict[str, dict[str, float]]:
    return {
        attribute_name: {
            "weight": float(weights[row_index, position, attr_index].item()),
            "score": float(scores[row_index, position, attr_index].item()),
        }
        for attr_index, attribute_name in enumerate(attribute_order)
    }


def build_attention_payload(
    selection_outputs: Mapping[str, Any],
    fusion_outputs: Mapping[str, Any],
) -> dict[str, Any]:
    metadata = selection_outputs["evidence_metadata"]
    fusion_weights = _to_tensor(fusion_outputs["fusion_weights"], dtype=torch.float32).cpu()
    payload = {"queries": []}
    batch_size = int(fusion_weights.shape[0])
    self_weights = _to_tensor(metadata["self_history_weights"], dtype=torch.float32).cpu()
    self_indices = _to_tensor(metadata["self_history_indices"], dtype=torch.long).cpu()
    self_mask = _to_tensor(metadata["self_history_mask"], dtype=torch.bool).cpu()
    neighbor_weights = _to_tensor(metadata["neighbor_weights"], dtype=torch.float32).cpu()
    neighbor_indices = _to_tensor(metadata["neighbor_indices"], dtype=torch.long).cpu()
    neighbor_mask = _to_tensor(metadata["neighbor_mask"], dtype=torch.bool).cpu()
    neighbor_stay_ids = _to_tensor(metadata["neighbor_stay_ids"], dtype=torch.long).cpu()
    matched_visit_indices = _to_tensor(metadata["neighbor_matched_visit_indices"], dtype=torch.long).cpu()
    self_selected_mask = _to_tensor(metadata.get("self_history_selected_mask", self_mask), dtype=torch.bool).cpu()
    neighbor_selected_mask = _to_tensor(metadata.get("neighbor_selected_mask", neighbor_mask), dtype=torch.bool).cpu()
    self_content_scores = _to_tensor(metadata.get("self_history_content_scores", metadata["self_history_scores"]), dtype=torch.float32).cpu()
    neighbor_content_scores = _to_tensor(metadata.get("neighbor_content_scores", metadata["neighbor_scores"]), dtype=torch.float32).cpu()
    neighbor_retrieval_scores = _to_tensor(metadata.get("neighbor_retrieval_scores", torch.zeros_like(neighbor_weights)), dtype=torch.float32).cpu()
    neighbor_retrieval_bias = _to_tensor(metadata.get("neighbor_retrieval_bias", torch.zeros_like(neighbor_weights)), dtype=torch.float32).cpu()
    attribute_order = list(metadata.get("attribute_order", []))
    self_attribute_weights = _to_tensor(
        metadata.get(
            "self_attribute_weights",
            torch.zeros(self_weights.shape[0], self_weights.shape[1], len(attribute_order), dtype=torch.float32),
        ),
        dtype=torch.float32,
    ).cpu()
    self_attribute_scores = _to_tensor(
        metadata.get(
            "self_attribute_scores",
            torch.zeros_like(self_attribute_weights),
        ),
        dtype=torch.float32,
    ).cpu()
    neighbor_attribute_weights = _to_tensor(
        metadata.get(
            "neighbor_attribute_weights",
            torch.zeros(neighbor_weights.shape[0], neighbor_weights.shape[1], len(attribute_order), dtype=torch.float32),
        ),
        dtype=torch.float32,
    ).cpu()
    neighbor_attribute_scores = _to_tensor(
        metadata.get(
            "neighbor_attribute_scores",
            torch.zeros_like(neighbor_attribute_weights),
        ),
        dtype=torch.float32,
    ).cpu()
    self_group_influence = _to_tensor(
        metadata.get("self_group_influence", torch.zeros_like(self_weights)),
        dtype=torch.float32,
    ).cpu()
    self_group_reweight_scores = _to_tensor(
        metadata.get("self_group_reweight_scores", torch.zeros_like(self_weights)),
        dtype=torch.float32,
    ).cpu()
    neighbor_group_influence = _to_tensor(
        metadata.get("neighbor_group_influence", torch.zeros_like(neighbor_weights)),
        dtype=torch.float32,
    ).cpu()
    neighbor_group_reweight_scores = _to_tensor(
        metadata.get("neighbor_group_reweight_scores", torch.zeros_like(neighbor_weights)),
        dtype=torch.float32,
    ).cpu()
    group_aware_selection_mask = _to_tensor(
        metadata.get("group_aware_selection_mask", torch.zeros(batch_size, dtype=torch.bool)),
        dtype=torch.bool,
    ).cpu()
    branch_entropy = _to_tensor(fusion_outputs.get("branch_entropy", torch.zeros(batch_size)), dtype=torch.float32).cpu()
    dominant_branch_name = fusion_outputs.get("dominant_branch_name", ["current"] * batch_size)
    selection_config = dict(metadata.get("selection_config", {}))

    for row_index in range(batch_size):
        self_rows = []
        for position in range(self_weights.shape[1]):
            if not bool(self_mask[row_index, position].item()):
                continue
            self_rows.append(
                {
                    "visit_index": int(self_indices[row_index, position].item()),
                    "weight": float(self_weights[row_index, position].item()),
                    "content_score": float(self_content_scores[row_index, position].item()),
                    "attribute_weights": _attribute_dict(
                        attribute_order,
                        self_attribute_weights,
                        self_attribute_scores,
                        row_index,
                        position,
                    ),
                    "dominant_attribute": attribute_order[
                        int(self_attribute_weights[row_index, position].argmax(dim=-1).item())
                    ]
                    if attribute_order
                    else "",
                    "group_reweight_score": float(self_group_reweight_scores[row_index, position].item()),
                    "group_influence": float(self_group_influence[row_index, position].item()),
                    "selected": bool(self_selected_mask[row_index, position].item()),
                }
            )
        neighbor_rows = []
        for position in range(neighbor_weights.shape[1]):
            if not bool(neighbor_mask[row_index, position].item()):
                continue
            neighbor_rows.append(
                {
                    "bank_index": int(neighbor_indices[row_index, position].item()),
                    "stay_id": int(neighbor_stay_ids[row_index, position].item()),
                    "matched_visit_index": int(matched_visit_indices[row_index, position].item()),
                    "weight": float(neighbor_weights[row_index, position].item()),
                    "content_score": float(neighbor_content_scores[row_index, position].item()),
                    "retrieval_score": float(neighbor_retrieval_scores[row_index, position].item()),
                    "retrieval_bias": float(neighbor_retrieval_bias[row_index, position].item()),
                    "attribute_weights": _attribute_dict(
                        attribute_order,
                        neighbor_attribute_weights,
                        neighbor_attribute_scores,
                        row_index,
                        position,
                    ),
                    "dominant_attribute": attribute_order[
                        int(neighbor_attribute_weights[row_index, position].argmax(dim=-1).item())
                    ]
                    if attribute_order
                    else "",
                    "group_reweight_score": float(neighbor_group_reweight_scores[row_index, position].item()),
                    "group_influence": float(neighbor_group_influence[row_index, position].item()),
                    "selected": bool(neighbor_selected_mask[row_index, position].item()),
                }
            )
        payload["queries"].append(
            {
                "query_index": row_index,
                "fusion_weights": {
                    branch_name: float(fusion_weights[row_index, branch_index].item())
                    for branch_index, branch_name in enumerate(BRANCH_ORDER)
                },
                "branch_entropy": float(branch_entropy[row_index].item()),
                "dominant_branch": str(dominant_branch_name[row_index]),
                "group_aware_selection": bool(group_aware_selection_mask[row_index].item()),
                "selection_config": selection_config,
                "self_history": self_rows,
                "neighbor_history": neighbor_rows,
            }
        )
    return payload


def build_selection_summary(selection_outputs: Mapping[str, Any]) -> dict[str, Any]:
    metadata = selection_outputs["evidence_metadata"]
    self_available = _to_tensor(metadata["self_history_available_mask"], dtype=torch.bool).cpu()
    neighbor_available = _to_tensor(metadata["neighbor_available_mask"], dtype=torch.bool).cpu()
    self_selected_count = _to_tensor(metadata.get("self_history_selected_count", torch.zeros_like(self_available, dtype=torch.long)), dtype=torch.long).cpu()
    neighbor_selected_count = _to_tensor(metadata.get("neighbor_selected_count", torch.zeros_like(neighbor_available, dtype=torch.long)), dtype=torch.long).cpu()
    attribute_order = list(metadata.get("attribute_order", []))
    self_attribute_weights = _to_tensor(
        metadata.get(
            "self_attribute_weights",
            torch.zeros(self_available.shape[0], 0, len(attribute_order), dtype=torch.float32),
        ),
        dtype=torch.float32,
    ).cpu()
    neighbor_attribute_weights = _to_tensor(
        metadata.get(
            "neighbor_attribute_weights",
            torch.zeros(neighbor_available.shape[0], 0, len(attribute_order), dtype=torch.float32),
        ),
        dtype=torch.float32,
    ).cpu()
    self_attribute_mask = _to_tensor(
        metadata.get("self_attribute_mask", torch.zeros_like(self_attribute_weights, dtype=torch.bool)),
        dtype=torch.bool,
    ).cpu()
    neighbor_attribute_mask = _to_tensor(
        metadata.get("neighbor_attribute_mask", torch.zeros_like(neighbor_attribute_weights, dtype=torch.bool)),
        dtype=torch.bool,
    ).cpu()
    group_aware_selection_mask = _to_tensor(
        metadata.get("group_aware_selection_mask", torch.zeros(self_available.shape[0], dtype=torch.bool)),
        dtype=torch.bool,
    ).cpu()
    mean_attribute_weights: dict[str, float] = {}
    mean_attribute_weight_sources = 0
    dominant_attribute_counts = {attribute_name: 0 for attribute_name in attribute_order}
    for branch_weights, branch_mask in (
        (self_attribute_weights, self_attribute_mask),
        (neighbor_attribute_weights, neighbor_attribute_mask),
    ):
        if branch_weights.numel() == 0:
            continue
        valid_candidates = branch_mask.any(dim=-1)
        if not bool(valid_candidates.any().item()):
            continue
        valid_weights = branch_weights[valid_candidates]
        mean_attribute_weight_sources += 1
        for attr_index, attribute_name in enumerate(attribute_order):
            current_value = mean_attribute_weights.get(attribute_name, 0.0)
            mean_attribute_weights[attribute_name] = current_value + float(valid_weights[:, attr_index].mean().item())
        for attr_index in valid_weights.argmax(dim=-1).tolist():
            dominant_attribute_counts[attribute_order[int(attr_index)]] += 1
    if mean_attribute_weights and mean_attribute_weight_sources > 0:
        mean_attribute_weights = {
            attribute_name: value / float(mean_attribute_weight_sources)
            for attribute_name, value in mean_attribute_weights.items()
        }
    return {
        "selection_config": dict(metadata.get("selection_config", {})),
        "self_available_rate": float(self_available.to(dtype=torch.float32).mean().item()) if self_available.numel() else 0.0,
        "neighbor_available_rate": float(neighbor_available.to(dtype=torch.float32).mean().item()) if neighbor_available.numel() else 0.0,
        "self_selected_mean": float(self_selected_count.to(dtype=torch.float32).mean().item()) if self_selected_count.numel() else 0.0,
        "neighbor_selected_mean": float(neighbor_selected_count.to(dtype=torch.float32).mean().item()) if neighbor_selected_count.numel() else 0.0,
        "self_selected_max": int(self_selected_count.max().item()) if self_selected_count.numel() else 0,
        "neighbor_selected_max": int(neighbor_selected_count.max().item()) if neighbor_selected_count.numel() else 0,
        "mean_attribute_weights": mean_attribute_weights,
        "dominant_attribute_counts": dominant_attribute_counts,
        "group_aware_selection_rate": float(group_aware_selection_mask.to(dtype=torch.float32).mean().item())
        if group_aware_selection_mask.numel()
        else 0.0,
    }


def build_branch_summary(fusion_outputs: Mapping[str, Any]) -> dict[str, Any]:
    fusion_weights = _to_tensor(fusion_outputs["fusion_weights"], dtype=torch.float32).cpu()
    branch_entropy = _to_tensor(
        fusion_outputs.get("branch_entropy", torch.zeros(fusion_weights.shape[0])),
        dtype=torch.float32,
    ).cpu()
    normalized_branch_entropy = _to_tensor(
        fusion_outputs.get("normalized_branch_entropy", torch.zeros(fusion_weights.shape[0])),
        dtype=torch.float32,
    ).cpu()
    dominant_branch_name = fusion_outputs.get("dominant_branch_name", ["current"] * fusion_weights.shape[0])
    branch_mask = _to_tensor(fusion_outputs.get("branch_mask", torch.ones_like(fusion_weights, dtype=torch.bool)), dtype=torch.bool).cpu()
    branch_contribution_norms = _to_tensor(
        fusion_outputs.get("branch_contribution_norms", torch.zeros_like(fusion_weights)),
        dtype=torch.float32,
    ).cpu()
    branch_collapse_flag = _to_tensor(
        fusion_outputs.get("branch_collapse_flag", torch.zeros(fusion_weights.shape[0], dtype=torch.bool)),
        dtype=torch.bool,
    ).cpu()
    branch_collapse_score = _to_tensor(
        fusion_outputs.get("branch_collapse_score", torch.zeros(fusion_weights.shape[0])),
        dtype=torch.float32,
    ).cpu()
    branch_balance_score = _to_tensor(
        fusion_outputs.get("branch_balance_score", torch.zeros(fusion_weights.shape[0])),
        dtype=torch.float32,
    ).cpu()
    fusion_entropy_loss = _to_tensor(
        fusion_outputs.get("fusion_entropy_loss", torch.zeros(fusion_weights.shape[0])),
        dtype=torch.float32,
    ).cpu()
    fusion_balance_loss = _to_tensor(
        fusion_outputs.get("fusion_balance_loss", torch.zeros(fusion_weights.shape[0])),
        dtype=torch.float32,
    ).cpu()
    mean_weights = {
        branch_name: float(fusion_weights[:, branch_index].mean().item())
        for branch_index, branch_name in enumerate(BRANCH_ORDER)
    }
    std_weights = {
        branch_name: float(fusion_weights[:, branch_index].std(unbiased=False).item())
        for branch_index, branch_name in enumerate(BRANCH_ORDER)
    }
    availability_rate = {
        branch_name: float(branch_mask[:, branch_index].to(dtype=torch.float32).mean().item())
        for branch_index, branch_name in enumerate(BRANCH_ORDER)
    }
    mean_contribution_norms = {
        branch_name: float(branch_contribution_norms[:, branch_index].mean().item())
        for branch_index, branch_name in enumerate(BRANCH_ORDER)
    }
    dominant_counts = {branch_name: 0 for branch_name in BRANCH_ORDER}
    for name in dominant_branch_name:
        dominant_counts[str(name)] = dominant_counts.get(str(name), 0) + 1
    return {
        "num_queries": int(fusion_weights.shape[0]),
        "fusion_strategy": fusion_outputs.get("fusion_strategy", "gated"),
        "mean_weights": mean_weights,
        "std_weights": std_weights,
        "availability_rate": availability_rate,
        "mean_contribution_norms": mean_contribution_norms,
        "branch_entropy_mean": float(branch_entropy.mean().item()),
        "branch_entropy_std": float(branch_entropy.std(unbiased=False).item()) if branch_entropy.numel() else 0.0,
        "normalized_branch_entropy_mean": float(normalized_branch_entropy.mean().item()) if normalized_branch_entropy.numel() else 0.0,
        "branch_collapse_rate": float(branch_collapse_flag.to(dtype=torch.float32).mean().item()) if branch_collapse_flag.numel() else 0.0,
        "branch_collapse_score_mean": float(branch_collapse_score.mean().item()) if branch_collapse_score.numel() else 0.0,
        "branch_balance_score_mean": float(branch_balance_score.mean().item()) if branch_balance_score.numel() else 0.0,
        "fusion_entropy_loss_mean": float(fusion_entropy_loss.mean().item()) if fusion_entropy_loss.numel() else 0.0,
        "fusion_balance_loss_mean": float(fusion_balance_loss.mean().item()) if fusion_balance_loss.numel() else 0.0,
        "dominant_branch_counts": dominant_counts,
    }


def build_faithfulness_payload(fusion_outputs: Mapping[str, Any]) -> dict[str, Any]:
    fusion_weights = _to_tensor(fusion_outputs["fusion_weights"], dtype=torch.float32).cpu()
    branch_mask = _to_tensor(fusion_outputs["branch_mask"], dtype=torch.bool).cpu()
    strategy = str(fusion_outputs.get("fusion_strategy", "gated"))
    branch_contexts = torch.stack(
        [_to_tensor(fusion_outputs["branch_contexts"][branch_name], dtype=torch.float32).cpu() for branch_name in BRANCH_ORDER],
        dim=1,
    )
    fused_repr = _to_tensor(fusion_outputs["fused_repr"], dtype=torch.float32).cpu()
    branch_logits = _to_tensor(
        fusion_outputs.get("branch_logits", torch.zeros_like(fusion_weights)),
        dtype=torch.float32,
    ).cpu()

    rows = []
    for row_index in range(fusion_weights.shape[0]):
        for branch_index, branch_name in enumerate(BRANCH_ORDER):
            if not bool(branch_mask[row_index, branch_index].item()):
                continue
            if strategy == "gated":
                removal_mask = branch_mask[row_index].clone()
                removal_mask[branch_index] = False
                if removal_mask.any():
                    removed_logits = branch_logits[row_index].masked_fill(~removal_mask, float("-inf"))
                    removed_weights = torch.softmax(removed_logits, dim=0)
                    removed_weights = torch.where(removal_mask, removed_weights, torch.zeros_like(removed_weights))
                    removed_weights = removed_weights / removed_weights.sum().clamp(min=1.0)
                    removed_repr = (branch_contexts[row_index] * removed_weights.unsqueeze(-1)).sum(dim=0)
                else:
                    removed_repr = torch.zeros_like(fused_repr[row_index])
            else:
                removed_repr = fused_repr[row_index] - (
                    fusion_weights[row_index, branch_index] * branch_contexts[row_index, branch_index]
                )
            rows.append(
                {
                    "query_index": row_index,
                    "branch": branch_name,
                    "faithfulness_shift_norm": float((fused_repr[row_index] - removed_repr).norm().item()),
                    "faithfulness_relative_shift": float(
                        (fused_repr[row_index] - removed_repr).norm().item() / fused_repr[row_index].norm().clamp(min=1.0e-12).item()
                    ),
                    "branch_weight": float(fusion_weights[row_index, branch_index].item()),
                }
            )
    return {"rows": rows}


def build_attention_rows(
    selection_outputs: Mapping[str, Any],
    fusion_outputs: Mapping[str, Any],
) -> list[dict[str, Any]]:
    payload = build_attention_payload(selection_outputs, fusion_outputs)
    rows = []
    for query in payload["queries"]:
        base_row = {"query_index": query["query_index"], **query["fusion_weights"]}
        if not query["self_history"] and not query["neighbor_history"]:
            rows.append(base_row)
            continue
        for row in query["self_history"]:
            rows.append(
                {
                    **base_row,
                    "evidence_type": "self",
                    "candidate_index": row["visit_index"],
                    "candidate_weight": row["weight"],
                    "candidate_score": row["content_score"],
                    "dominant_attribute": row.get("dominant_attribute", ""),
                    "attribute_weights_json": json.dumps(row.get("attribute_weights", {}), sort_keys=True),
                    "group_reweight_score": row.get("group_reweight_score", 0.0),
                    "group_influence": row.get("group_influence", 0.0),
                    "candidate_selected": row["selected"],
                    "matched_visit_index": row["visit_index"],
                    "stay_id": "",
                }
            )
        for row in query["neighbor_history"]:
            rows.append(
                {
                    **base_row,
                    "evidence_type": "neighbor",
                    "candidate_index": row["bank_index"],
                    "candidate_weight": row["weight"],
                    "candidate_score": row["content_score"],
                    "candidate_selected": row["selected"],
                    "dominant_attribute": row.get("dominant_attribute", ""),
                    "attribute_weights_json": json.dumps(row.get("attribute_weights", {}), sort_keys=True),
                    "group_reweight_score": row.get("group_reweight_score", 0.0),
                    "group_influence": row.get("group_influence", 0.0),
                    "retrieval_score": row["retrieval_score"],
                    "retrieval_bias": row["retrieval_bias"],
                    "matched_visit_index": row["matched_visit_index"],
                    "stay_id": row["stay_id"],
                }
            )
    return rows


def artifact_dir(project_root: str | Path) -> Path:
    return ensure_dir(resolve_path(project_root, "outputs/figures"))


def save_attention_artifacts(
    project_root: str | Path,
    *,
    name: str,
    selection_outputs: Mapping[str, Any],
    fusion_outputs: Mapping[str, Any],
) -> dict[str, Path]:
    output_dir = artifact_dir(project_root)
    json_path = write_json(
        output_dir / f"{name}_attention.json",
        {
            "queries": build_attention_payload(selection_outputs, fusion_outputs)["queries"],
            "selection_summary": build_selection_summary(selection_outputs),
            "branch_summary": build_branch_summary(fusion_outputs),
            "faithfulness": build_faithfulness_payload(fusion_outputs),
        },
    )
    csv_path = write_csv_gz(
        output_dir / f"{name}_attention.csv.gz",
        build_attention_rows(selection_outputs, fusion_outputs),
        fieldnames=[
            "query_index",
            *BRANCH_ORDER,
            "evidence_type",
            "candidate_index",
            "candidate_weight",
            "candidate_score",
            "dominant_attribute",
            "attribute_weights_json",
            "group_reweight_score",
            "group_influence",
            "candidate_selected",
            "retrieval_score",
            "retrieval_bias",
            "matched_visit_index",
            "stay_id",
        ],
    )
    return {"json": json_path, "csv": csv_path}
