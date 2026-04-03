from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - UI dependency
    import streamlit as st
except ImportError:  # pragma: no cover - UI dependency
    st = None

from app.components.explanation_panel import render_counterfactual_panel
from app.components.recommendation_panel import (
    _move_batch_to_device,
    ensure_memory_bank,
    get_app_runtime,
    initialize_app_state,
)
from src.explainability import build_nl_explanation, run_counterfactual_analysis


def _page_link(page: str, label: str) -> None:
    if hasattr(st, "page_link"):
        st.page_link(page, label=label)
    else:  # pragma: no cover - UI dependency
        st.write(f"{label}: `{page}`")


def main() -> None:
    if st is None:  # pragma: no cover - UI dependency
        raise RuntimeError("streamlit is required to run this page")

    initialize_app_state()
    runtime = get_app_runtime()

    st.title("Counterfactual Explanation")
    inference_outputs = st.session_state.get("inference_outputs")
    patient_batch = st.session_state.get("patient_batch")
    patient_record = st.session_state.get("patient_record")

    if inference_outputs is None or patient_batch is None or patient_record is None:
        st.info("Can co patient input va recommendation truoc khi chay counterfactual.")
        _page_link("app/pages/2_recommendation.py", "Go to Recommendation")
        return

    threshold = st.slider(
        "Threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(st.session_state.get("ui_threshold") or runtime.get("default_threshold", 0.5)),
        step=0.05,
    )
    top_k = st.slider(
        "Top-K",
        min_value=1,
        max_value=20,
        value=int(st.session_state.get("ui_top_k") or runtime.get("default_top_k", 10)),
        step=1,
    )

    if st.button("Run Counterfactual", use_container_width=True):
        if not runtime.get("model_available", False):
            st.error("Model khong san sang.")
        else:
            try:
                memory_bank = None
                if str(inference_outputs.get("effective_mode", "core")) == "extended":
                    bank_split = inference_outputs.get("memory_bank_split") or st.session_state.get("patient_source_split") or "train"
                    memory_bank, bank_error = ensure_memory_bank(runtime, str(bank_split))
                    if memory_bank is None and bank_error is not None:
                        st.warning(f"Khong the tai memory bank cho counterfactual, fallback core: {bank_error}")

                batch_on_device = _move_batch_to_device(patient_batch, runtime["device"])
                payload = run_counterfactual_analysis(
                    runtime["model"],
                    batch_on_device,
                    threshold=threshold,
                    top_k=top_k,
                    mode=str(inference_outputs.get("effective_mode", "core")),
                    memory_bank=memory_bank,
                    records=[patient_record],
                    ddi_matrix=runtime.get("ddi_matrix"),
                )
                nl_payload = build_nl_explanation(
                    payload,
                    drug_idx_to_token=runtime.get("drug_idx_to_token"),
                    max_drugs=top_k,
                )
                st.session_state["counterfactual_payload"] = payload
                st.session_state["nl_explanation"] = nl_payload
                st.session_state["ui_threshold"] = threshold
                st.session_state["ui_top_k"] = top_k
                st.success("Da tao counterfactual explanation.")
            except Exception as exc:
                st.error(f"Counterfactual failed: {type(exc).__name__}: {exc}")

    if st.button("Refresh NL Explanation", use_container_width=True):
        payload = st.session_state.get("counterfactual_payload")
        if payload is None:
            st.info("Chua co counterfactual payload de lam moi.")
        else:
            nl_payload = build_nl_explanation(
                payload,
                drug_idx_to_token=runtime.get("drug_idx_to_token"),
                max_drugs=top_k,
            )
            st.session_state["nl_explanation"] = nl_payload
            st.success("Da lam moi natural language explanation.")

    counterfactual_payload = st.session_state.get("counterfactual_payload")
    nl_explanation = st.session_state.get("nl_explanation")
    if counterfactual_payload is None:
        st.info("Bam `Run Counterfactual` de tao explanation cho ca hien tai.")
        return

    render_counterfactual_panel(runtime, counterfactual_payload, nl_explanation)


if __name__ == "__main__":  # pragma: no cover - UI dependency
    main()
