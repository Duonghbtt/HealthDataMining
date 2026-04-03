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

from app.components.explanation_panel import render_evidence_summary
from app.components.patient_form import render_patient_form
from app.components.recommendation_panel import (
    get_app_runtime,
    initialize_app_state,
    render_recommendation_panel,
    render_runtime_status,
    run_patient_inference,
)


def main() -> None:
    if st is None:  # pragma: no cover - UI dependency
        raise RuntimeError("streamlit is required to run this page")

    initialize_app_state()
    runtime = get_app_runtime()

    st.title("Medication Recommendation")
    render_runtime_status(runtime, compact=True)

    patient_context = render_patient_form(runtime)

    settings_col, action_col = st.columns([2, 1])
    with settings_col:
        use_retrieval = st.checkbox(
            "Use Retrieval If Available",
            value=True,
            disabled=not runtime.get("model_available", False),
        )
        top_k = st.slider(
            "Top-K recommendations",
            min_value=1,
            max_value=20,
            value=int(st.session_state.get("ui_top_k") or runtime.get("default_top_k", 10)),
            step=1,
        )
        threshold = st.slider(
            "Prediction threshold",
            min_value=0.05,
            max_value=0.95,
            value=float(st.session_state.get("ui_threshold") or runtime.get("default_threshold", 0.5)),
            step=0.05,
        )
    with action_col:
        disabled = not runtime.get("model_available", False) or patient_context.get("batch") is None
        if st.button("Run Recommendation", use_container_width=True, disabled=disabled):
            try:
                outputs = run_patient_inference(
                    runtime,
                    batch=patient_context["batch"],
                    records=[patient_context["record"]] if patient_context.get("record") is not None else None,
                    use_retrieval=bool(use_retrieval),
                    source_split=patient_context.get("source_split"),
                    top_k=top_k,
                )
                st.session_state["inference_outputs"] = outputs
                st.session_state["retrieval_payload"] = outputs.get("retrieval_payload")
                st.session_state["ui_threshold"] = threshold
                st.session_state["ui_top_k"] = top_k
                st.session_state["counterfactual_payload"] = None
                st.session_state["nl_explanation"] = None
                st.session_state["safety_summary"] = None
                if outputs.get("retrieval_payload") is not None:
                    st.success("Da tao retrieval payload cho page Similar Cases.")
                else:
                    st.info("Khong co retrieval payload tu lan inference nay.")
            except Exception as exc:
                st.error(f"Inference failed: {type(exc).__name__}: {exc}")

    inference_outputs = st.session_state.get("inference_outputs")
    if inference_outputs is None:
        st.info("Hay load/build patient input roi bam `Run Recommendation`.")
        return

    render_recommendation_panel(runtime, inference_outputs, top_k=top_k, threshold=threshold)
    render_evidence_summary(inference_outputs)


if __name__ == "__main__":  # pragma: no cover - UI dependency
    main()
