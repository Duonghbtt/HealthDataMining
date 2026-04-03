from __future__ import annotations

from typing import Any, Mapping


_BRANCH_LABELS = {
    "current": "trạng thái hiện tại",
    "self": "lịch sử của chính bệnh nhân",
    "neighbor": "các ca tương tự",
    "group": "nhóm bệnh nhân tương đồng",
}


def _drug_name(drug_index: int, drug_idx_to_token: Mapping[Any, Any] | None) -> str:
    if drug_idx_to_token is None:
        return f"drug_{drug_index}"
    for candidate in (drug_index, str(drug_index)):
        if candidate in drug_idx_to_token:
            return str(drug_idx_to_token[candidate])
    return f"drug_{drug_index}"


def _format_top_recommendations(
    recommendations: list[dict[str, Any]],
    *,
    drug_idx_to_token: Mapping[Any, Any] | None,
    max_drugs: int,
) -> str:
    if not recommendations:
        return "chưa có thuốc gợi ý nổi bật"
    rows = []
    for item in recommendations[: max(1, int(max_drugs))]:
        drug_index = int(item["drug_index"])
        score = float(item.get("score", 0.0))
        rows.append(f"{_drug_name(drug_index, drug_idx_to_token)} ({score:.2f})")
    return ", ".join(rows)


def _format_drug_set(
    drug_indices: list[int],
    *,
    drug_idx_to_token: Mapping[Any, Any] | None,
    max_drugs: int = 4,
) -> str:
    if not drug_indices:
        return "không có thay đổi thuốc"
    return ", ".join(
        _drug_name(int(drug_index), drug_idx_to_token)
        for drug_index in drug_indices[: max(1, int(max_drugs))]
    )


def _evidence_text(baseline: Mapping[str, Any]) -> str:
    evidence_summary = baseline.get("evidence_summary", {})
    dominant_branch = str(baseline.get("dominant_branch", "current"))
    branch_label = _BRANCH_LABELS.get(dominant_branch, dominant_branch)

    details: list[str] = []
    top_self = evidence_summary.get("top_self_history")
    if isinstance(top_self, Mapping):
        details.append(f"visit lịch sử quan trọng nhất là #{int(top_self['index'])}")

    top_neighbor = evidence_summary.get("top_neighbor_history")
    if isinstance(top_neighbor, Mapping):
        stay_id = top_neighbor.get("stay_id")
        matched_visit_index = top_neighbor.get("matched_visit_index")
        if stay_id is not None and matched_visit_index is not None:
            details.append(
                f"ca tương tự nổi bật là stay {int(stay_id)} ở visit ghép #{int(matched_visit_index)}"
            )
        elif stay_id is not None:
            details.append(f"ca tương tự nổi bật là stay {int(stay_id)}")

    top_attribute_groups = evidence_summary.get("top_attribute_groups", {})
    attributes: list[str] = []
    for branch_name in ("self", "neighbor"):
        rows = top_attribute_groups.get(branch_name, [])
        if rows:
            attributes.extend(str(item["attribute"]) for item in rows[:1])
    if attributes:
        details.append(f"nhóm tín hiệu nổi bật gồm {', '.join(attributes[:2])}")

    if not details:
        return f"Nhánh chi phối hiện tại là {branch_label}, nhưng evidence chi tiết chưa đủ mạnh để tóm tắt thêm."
    return f"Nhánh chi phối hiện tại là {branch_label}; " + "; ".join(details[:2]) + "."


def _safety_text(baseline: Mapping[str, Any]) -> str:
    safety_summary = baseline.get("safety_summary", {})
    predicted_count = int(safety_summary.get("predicted_drug_count", 0))
    ddi_penalty = safety_summary.get("ddi_penalty")
    if bool(safety_summary.get("has_thresholded_ddi", False)):
        pair_count = int(safety_summary.get("thresholded_interacting_pair_count", 0))
        return (
            f"Có cảnh báo an toàn: phát hiện {pair_count} cặp thuốc có nguy cơ DDI "
            f"trong {predicted_count} thuốc được gợi ý."
        )
    if ddi_penalty is not None:
        return (
            f"Chưa thấy cặp DDI nổi bật ở ngưỡng hiện tại; penalty DDI mềm của mô hình là "
            f"{float(ddi_penalty):.4f}."
        )
    return "Chưa thấy cảnh báo an toàn nổi bật từ phần DDI ở cấu hình hiện tại."


def _counterfactual_text(
    best_counterfactual: Mapping[str, Any] | None,
    *,
    drug_idx_to_token: Mapping[Any, Any] | None,
) -> str:
    if not isinstance(best_counterfactual, Mapping):
        return "Chưa tìm được counterfactual đủ rõ để minh họa thay đổi khuyến nghị."

    description = str(best_counterfactual.get("description", "thay đổi evidence"))
    entered = _format_drug_set(
        [int(item) for item in best_counterfactual.get("entered_drugs", [])],
        drug_idx_to_token=drug_idx_to_token,
    )
    removed = _format_drug_set(
        [int(item) for item in best_counterfactual.get("removed_drugs", [])],
        drug_idx_to_token=drug_idx_to_token,
    )
    ddi_delta = float(best_counterfactual.get("ddi_delta", 0.0))

    change_parts: list[str] = []
    if best_counterfactual.get("entered_drugs"):
        change_parts.append(f"mô hình thêm {entered}")
    if best_counterfactual.get("removed_drugs"):
        change_parts.append(f"mô hình bỏ {removed}")
    if not change_parts:
        change_parts.append("danh sách thuốc thay đổi nhẹ về xác suất nhưng chưa đổi mạnh theo threshold")

    if ddi_delta > 0:
        safety_part = "nguy cơ DDI tăng"
    elif ddi_delta < 0:
        safety_part = "nguy cơ DDI giảm"
    else:
        safety_part = "nguy cơ DDI gần như không đổi"

    return f"Nếu {description}, thì {'; '.join(change_parts)} và {safety_part}."


def build_nl_explanation(
    counterfactual_payload: Mapping[str, Any],
    *,
    drug_idx_to_token: Mapping[Any, Any] | None = None,
    max_drugs: int = 5,
) -> dict[str, Any]:
    """Build short Vietnamese explanation text from a counterfactual payload."""

    baseline = counterfactual_payload.get("baseline", {})
    best_counterfactual = counterfactual_payload.get("best_counterfactual")

    recommendation_text = (
        "Thuốc gợi ý nổi bật: "
        + _format_top_recommendations(
            list(baseline.get("top_recommendations", [])),
            drug_idx_to_token=drug_idx_to_token,
            max_drugs=max_drugs,
        )
        + "."
    )
    evidence_text = _evidence_text(baseline)
    safety_text = _safety_text(baseline)
    counterfactual_text = _counterfactual_text(
        best_counterfactual,
        drug_idx_to_token=drug_idx_to_token,
    )
    summary_text = " ".join(
        part
        for part in (
            recommendation_text,
            evidence_text,
            safety_text,
            counterfactual_text,
        )
        if part
    )

    return {
        "recommendation_text": recommendation_text,
        "evidence_text": evidence_text,
        "safety_text": safety_text,
        "counterfactual_text": counterfactual_text,
        "summary_text": summary_text,
    }


__all__ = ["build_nl_explanation"]
