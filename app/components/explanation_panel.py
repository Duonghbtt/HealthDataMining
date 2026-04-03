from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - UI dependency
    import streamlit as st
except ImportError:  # pragma: no cover - UI dependency
    st = None

from app.components.recommendation_panel import compute_patient_safety_summary, format_drug_name


def _require_streamlit() -> None:
    if st is None:  # pragma: no cover - UI dependency
        raise RuntimeError("streamlit is required for the app components")


def _to_cpu_tensor(value: Any, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=dtype) if dtype is not None else torch.as_tensor(value)
    return tensor.detach().cpu()


def _top_self_history(outputs: Mapping[str, Any]) -> dict[str, Any] | None:
    evidence = outputs.get("evidence_metadata", {})
    weights = evidence.get("self_history_weights")
    mask = evidence.get("self_history_mask")
    indices = evidence.get("self_history_indices")
    if weights is None or mask is None or indices is None:
        return None
    weight_tensor = _to_cpu_tensor(weights, dtype=torch.float32)
    mask_tensor = _to_cpu_tensor(mask, dtype=torch.bool)
    index_tensor = _to_cpu_tensor(indices, dtype=torch.long)
    if weight_tensor.ndim != 2 or mask_tensor.ndim != 2 or index_tensor.ndim != 2:
        return None
    if weight_tensor.shape[0] == 0 or not bool(mask_tensor[0].any().item()):
        return None
    top_pos = int(weight_tensor[0].masked_fill(~mask_tensor[0], float("-inf")).argmax(dim=-1).item())
    return {
        "index": int(index_tensor[0, top_pos].item()),
        "weight": float(weight_tensor[0, top_pos].item()),
    }


def _top_neighbor_history(outputs: Mapping[str, Any]) -> dict[str, Any] | None:
    evidence = outputs.get("evidence_metadata", {})
    weights = evidence.get("neighbor_weights")
    mask = evidence.get("neighbor_mask")
    stay_ids = evidence.get("neighbor_stay_ids")
    visit_indices = evidence.get("neighbor_matched_visit_indices")
    if weights is None or mask is None or stay_ids is None:
        return None
    weight_tensor = _to_cpu_tensor(weights, dtype=torch.float32)
    mask_tensor = _to_cpu_tensor(mask, dtype=torch.bool)
    stay_tensor = _to_cpu_tensor(stay_ids, dtype=torch.long)
    visit_tensor = _to_cpu_tensor(visit_indices, dtype=torch.long) if visit_indices is not None else None
    if weight_tensor.ndim != 2 or mask_tensor.ndim != 2 or stay_tensor.ndim != 2:
        return None
    if weight_tensor.shape[0] == 0 or not bool(mask_tensor[0].any().item()):
        return None
    top_pos = int(weight_tensor[0].masked_fill(~mask_tensor[0], float("-inf")).argmax(dim=-1).item())
    payload = {
        "stay_id": int(stay_tensor[0, top_pos].item()),
        "weight": float(weight_tensor[0, top_pos].item()),
    }
    if visit_tensor is not None and visit_tensor.ndim == 2:
        payload["matched_visit_index"] = int(visit_tensor[0, top_pos].item())
    return payload


def render_evidence_summary(outputs: Mapping[str, Any]) -> None:
    _require_streamlit()

    st.subheader("Evidence Summary")
    dominant_branch = outputs.get("dominant_branch_name", ["current"])
    dominant = str(dominant_branch[0]) if isinstance(dominant_branch, list) and dominant_branch else str(dominant_branch)
    st.write(f"Dominant branch: `{dominant}`")

    fusion_weights = outputs.get("fusion_weights")
    branch_order = outputs.get("branch_order", ["current", "self", "neighbor", "group"])
    if isinstance(fusion_weights, torch.Tensor) and fusion_weights.ndim == 2 and fusion_weights.shape[0] > 0:
        rows = [
            {"branch": str(branch_name), "weight": float(fusion_weights[0, index].item())}
            for index, branch_name in enumerate(branch_order)
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    top_self = _top_self_history(outputs)
    top_neighbor = _top_neighbor_history(outputs)
    if top_self is not None:
        st.caption(
            f"Top self-history visit: #{int(top_self['index'])} "
            f"(weight={float(top_self['weight']):.3f})"
        )
    if top_neighbor is not None:
        text = f"Top neighbor stay: {int(top_neighbor['stay_id'])}"
        if "matched_visit_index" in top_neighbor:
            text += f" | matched visit #{int(top_neighbor['matched_visit_index'])}"
        text += f" | weight={float(top_neighbor['weight']):.3f}"
        st.caption(text)


def render_safety_summary(
    runtime: Mapping[str, Any],
    outputs: Mapping[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    _require_streamlit()
    summary = compute_patient_safety_summary(runtime, outputs, threshold=threshold)

    st.subheader("Safety / DDI")
    cols = st.columns(4)
    cols[0].metric("Threshold", f"{float(threshold):.2f}")
    cols[1].metric("Predicted Drugs", int(summary["predicted_drug_count"]))
    cols[2].metric("DDI Warning", "Yes" if summary["has_ddi"] else "No")
    cols[3].metric(
        "DDI Penalty",
        "-" if summary["ddi_penalty"] is None else f"{float(summary['ddi_penalty']):.4f}",
    )

    if summary["predicted_names"]:
        st.write("Predicted set")
        st.write(", ".join(summary["predicted_names"]))
    else:
        st.info("Khong co thuoc nao vuot threshold hien tai.")

    if summary["interacting_pairs"]:
        st.warning("Phat hien cap thuoc co nguy co DDI theo threshold hien tai.")
        st.dataframe(summary["interacting_pairs"], use_container_width=True, hide_index=True)
    else:
        st.success("Chua thay cap DDI noi bat theo threshold hien tai.")
    return summary


def render_counterfactual_panel(
    runtime: Mapping[str, Any],
    payload: Mapping[str, Any],
    nl_explanation: Mapping[str, Any] | None,
) -> None:
    _require_streamlit()

    baseline = payload.get("baseline", {})
    st.subheader("Counterfactual")
    top_recommendations = list(baseline.get("top_recommendations", []))
    if top_recommendations:
        st.write("Baseline top recommendations")
        rows = [
            {
                "drug_index": int(item["drug_index"]),
                "drug_name": format_drug_name(runtime, int(item["drug_index"])),
                "score": float(item["score"]),
            }
            for item in top_recommendations
        ]
        st.dataframe(rows, use_container_width=True, hide_index=True)

    best = payload.get("best_counterfactual")
    if best is None:
        st.info("Chua tim duoc counterfactual kha dung.")
    else:
        best_rows = [
            {
                "name": str(best.get("name", "")),
                "description": str(best.get("description", "")),
                "impact_score": float(best.get("impact_score", 0.0)),
                "changed_drug_count": int(best.get("changed_drug_count", 0)),
                "ddi_delta": float(best.get("ddi_delta", 0.0)),
            }
        ]
        st.write("Best counterfactual")
        st.dataframe(best_rows, use_container_width=True, hide_index=True)

    interventions = list(payload.get("interventions", []))
    if interventions:
        rows = []
        for item in interventions:
            rows.append(
                {
                    "name": str(item.get("name", "")),
                    "available": bool(item.get("available", False)),
                    "impact_score": float(item.get("impact_score", 0.0)) if item.get("available", False) else None,
                    "changed_drugs": int(item.get("changed_drug_count", 0)) if item.get("available", False) else None,
                    "reason": item.get("reason", ""),
                }
            )
        st.write("Interventions")
        st.dataframe(rows, use_container_width=True, hide_index=True)

    if nl_explanation:
        st.subheader("Natural Language Explanation")
        st.write(nl_explanation.get("summary_text", ""))
        with st.expander("Recommendation text", expanded=False):
            st.write(nl_explanation.get("recommendation_text", ""))
        with st.expander("Evidence text", expanded=False):
            st.write(nl_explanation.get("evidence_text", ""))
        with st.expander("Safety text", expanded=False):
            st.write(nl_explanation.get("safety_text", ""))
        with st.expander("Counterfactual text", expanded=False):
            st.write(nl_explanation.get("counterfactual_text", ""))


__all__ = [
    "render_counterfactual_panel",
    "render_evidence_summary",
    "render_safety_summary",
]
