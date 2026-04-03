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

from app.components.explanation_panel import render_safety_summary
from app.components.recommendation_panel import get_app_runtime, initialize_app_state


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

    st.title("Safety / DDI Analysis")
    inference_outputs = st.session_state.get("inference_outputs")
    if inference_outputs is None:
        st.info("Chua co ket qua inference. Hay chay recommendation truoc.")
        _page_link("app/pages/2_recommendation.py", "Go to Recommendation")
        return

    threshold = st.slider(
        "Threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(st.session_state.get("ui_threshold") or runtime.get("default_threshold", 0.5)),
        step=0.05,
    )

    analyze = st.button("Analyze Safety", use_container_width=True)
    recompute = st.button("Recompute With Current Threshold", use_container_width=True)

    if analyze or recompute or st.session_state.get("safety_summary") is None:
        try:
            summary = render_safety_summary(runtime, inference_outputs, threshold=threshold)
            st.session_state["safety_summary"] = summary
            st.session_state["ui_threshold"] = threshold
        except Exception as exc:
            st.error(f"Safety analysis failed: {type(exc).__name__}: {exc}")
            return
    else:
        render_safety_summary(runtime, inference_outputs, threshold=threshold)


if __name__ == "__main__":  # pragma: no cover - UI dependency
    main()
