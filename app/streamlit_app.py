from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - UI dependency
    import streamlit as st
except ImportError:  # pragma: no cover - UI dependency
    st = None

from app.components.recommendation_panel import (
    clear_app_runtime_cache,
    clear_app_session,
    get_app_runtime,
    initialize_app_state,
    render_runtime_status,
)


def _rerun() -> None:
    if hasattr(st, "rerun"):
        st.rerun()
    else:  # pragma: no cover - old streamlit fallback
        st.experimental_rerun()


def _page_link(page: str, label: str) -> None:
    if hasattr(st, "page_link"):
        st.page_link(page, label=label)
    else:  # pragma: no cover - UI dependency
        st.write(f"- {label}: `{page}`")


def main() -> None:
    if st is None:  # pragma: no cover - UI dependency
        raise RuntimeError("streamlit is required to run this app")

    st.set_page_config(page_title="ClinRec Demo", layout="wide")
    initialize_app_state()

    st.title("ClinRec Demo")
    st.caption(
        "App demo cho recommendation, safety/DDI va counterfactual explanation. "
        "Page Similar Cases cua Nguoi 2 se doc `retrieval_payload` tu session state khi co san."
    )

    left, right = st.columns([1, 1])
    with left:
        if st.button("Load / Refresh Artifacts", use_container_width=True):
            clear_app_runtime_cache()
            initialize_app_state()
            _ = get_app_runtime(force_refresh=True)
            _rerun()
    with right:
        if st.button("Clear Session", use_container_width=True):
            clear_app_session()
            _rerun()

    runtime = get_app_runtime()
    render_runtime_status(runtime, compact=False)

    st.subheader("Workflow")
    _page_link("app/pages/2_recommendation.py", "Page 2: Recommendation")
    _page_link("app/pages/3_safety_ddi.py", "Page 3: Safety / DDI")
    _page_link("app/pages/4_counterfactual.py", "Page 4: Counterfactual")
    _page_link("app/pages/1_similar_cases.py", "Page 1: Similar Cases")

    st.subheader("Session State")
    session_rows = [
        {"key": "patient_record", "available": st.session_state.get("patient_record") is not None},
        {"key": "patient_batch", "available": st.session_state.get("patient_batch") is not None},
        {"key": "inference_outputs", "available": st.session_state.get("inference_outputs") is not None},
        {"key": "retrieval_payload", "available": st.session_state.get("retrieval_payload") is not None},
        {"key": "counterfactual_payload", "available": st.session_state.get("counterfactual_payload") is not None},
        {"key": "nl_explanation", "available": st.session_state.get("nl_explanation") is not None},
    ]
    st.dataframe(session_rows, use_container_width=True, hide_index=True)

    if not runtime.get("model_available", False):
        st.warning(
            "Chua co checkpoint hop le trong `outputs/checkpoints`. "
            "App van mo duoc, nhung cac page inference se bi disable cho toi khi co checkpoint."
        )


if __name__ == "__main__":  # pragma: no cover - UI dependency
    main()
