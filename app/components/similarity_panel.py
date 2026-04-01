from __future__ import annotations

from typing import Any, Mapping

from src.retrieval.topk_retriever import validate_retrieval_payload


try:  # pragma: no cover - UI dependency
    import streamlit as st
except ImportError:  # pragma: no cover - UI dependency
    st = None


def payload_rows(payload: Mapping[str, Any], *, query_index: int = 0) -> list[dict[str, Any]]:
    validate_retrieval_payload(payload)
    rows: list[dict[str, Any]] = []
    for rank, (
        stay_id,
        subject_id,
        hadm_id,
        score,
        static_score,
        time_gap_days,
    ) in enumerate(
        zip(
            payload["neighbor_stay_ids"][query_index].tolist(),
            payload["neighbor_subject_ids"][query_index].tolist(),
            payload["neighbor_hadm_ids"][query_index].tolist(),
            payload["neighbor_scores"][query_index].tolist(),
            payload["neighbor_static_scores"][query_index].tolist(),
            payload["neighbor_time_gaps_days"][query_index].tolist(),
        ),
        start=1,
    ):
        if int(stay_id) < 0:
            continue
        rows.append(
            {
                "rank": rank,
                "stay_id": int(stay_id),
                "subject_id": int(subject_id),
                "hadm_id": int(hadm_id),
                "score": float(score),
                "static_score": float(static_score),
                "time_gap_days": float(time_gap_days),
            }
        )
    return rows


def render_similarity_panel(payload: Mapping[str, Any], *, query_index: int = 0) -> None:
    if st is None:  # pragma: no cover - UI dependency
        raise RuntimeError("streamlit is required to render the similarity panel")
    rows = payload_rows(payload, query_index=query_index)
    st.caption(
        "Pha 1: trang này đọc trực tiếp retrieval payload. "
        "similar_case_report sẽ được nối ở pha sau."
    )
    st.write(
        f"Query stay_id: `{payload['query_stay_ids'][query_index]}` | "
        f"split: `{payload['query_split']}` | backend: `{payload['backend']}`"
    )
    if not rows:
        st.warning("Không có similar case nào trong retrieval payload.")
        return
    st.dataframe(rows, use_container_width=True, hide_index=True)
