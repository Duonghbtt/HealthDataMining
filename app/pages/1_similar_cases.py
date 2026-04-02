from __future__ import annotations

from typing import Any

import torch

from app.components.similarity_panel import render_similarity_panel


try:  # pragma: no cover - UI dependency
    import streamlit as st
except ImportError:  # pragma: no cover - UI dependency
    st = None


def _demo_payload() -> dict[str, Any]:
    return {
        "query_stay_ids": [900001],
        "query_split": "train",
        "neighbor_indices": torch.tensor([[0, 1, 2]], dtype=torch.long),
        "neighbor_scores": torch.tensor([[0.93, 0.81, 0.77]], dtype=torch.float32),
        "neighbor_static_scores": torch.tensor([[0.95, 0.88, 0.84]], dtype=torch.float32),
        "neighbor_time_gaps_days": torch.tensor([[1.0, 3.5, 6.0]], dtype=torch.float32),
        "neighbor_subject_ids": torch.tensor([[1201, 1202, 1203]], dtype=torch.long),
        "neighbor_hadm_ids": torch.tensor([[2201, 2202, 2203]], dtype=torch.long),
        "neighbor_stay_ids": torch.tensor([[3201, 3202, 3203]], dtype=torch.long),
        "backend": "bruteforce",
        "bank_split": "train",
    }


def main() -> None:
    if st is None:  # pragma: no cover - UI dependency
        raise RuntimeError("streamlit is required to run this page")
    st.set_page_config(page_title="Similar Cases", layout="wide")
    st.title("Similar Cases")
    st.caption(
        "Trang này khóa schema retrieval payload ngay từ pha 1 và chỉ đọc payload thô. "
        "Không phụ thuộc `similar_case_report` ở giai đoạn hiện tại."
    )
    payload = st.session_state.get("retrieval_payload") or _demo_payload()
    render_similarity_panel(payload, query_index=0)


if __name__ == "__main__":  # pragma: no cover - UI dependency
    main()
