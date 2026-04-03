from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - UI dependency
    import streamlit as st
except ImportError:  # pragma: no cover - UI dependency
    st = None

from app.components.recommendation_panel import (
    collate_batch,
    format_vocab_name,
    get_dataset,
    initialize_app_state,
    load_demo_record,
)


def _require_streamlit() -> None:
    if st is None:  # pragma: no cover - UI dependency
        raise RuntimeError("streamlit is required for the app components")


def _parse_int_list(text: str) -> list[int]:
    values: list[int] = []
    for part in text.split(","):
        token = part.strip()
        if token:
            values.append(int(token))
    return values


def _parse_float_list(text: str) -> list[float]:
    values: list[float] = []
    for part in text.split(","):
        token = part.strip()
        if token:
            values.append(float(token))
    return values


def _sanitize_indices(values: list[int], upper_bound: int) -> tuple[list[int], list[int]]:
    valid: list[int] = []
    invalid: list[int] = []
    for value in values:
        if 0 <= int(value) < int(upper_bound):
            valid.append(int(value))
        else:
            invalid.append(int(value))
    return valid, invalid


def _build_dense_numeric(values: list[float], size: int) -> tuple[list[float], list[bool]]:
    dense = [0.0] * int(size)
    mask = [False] * int(size)
    for index, value in enumerate(values[: int(size)]):
        dense[index] = float(value)
        mask[index] = True
    return dense, mask


def _feature_preview_rows(metadata: Mapping[str, Any], *, limit: int = 5) -> list[dict[str, Any]]:
    rows = []
    sorted_items = sorted(
        metadata.items(),
        key=lambda item: int(item[1].get("index", 0)),
    )
    for token, payload in sorted_items[: int(limit)]:
        rows.append(
            {
                "index": int(payload.get("index", 0)),
                "token": str(token),
                "label": str(payload.get("label", token)),
            }
        )
    return rows


def _clear_downstream_state() -> None:
    for key in (
        "inference_outputs",
        "retrieval_payload",
        "counterfactual_payload",
        "nl_explanation",
        "safety_summary",
    ):
        st.session_state[key] = None


def _record_summary(record: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict[str, Any]:
    steps = list(record.get("steps", []))
    last_step = steps[-1] if steps else {}
    return {
        "subject_id": int(record.get("subject_id", -1)),
        "hadm_id": int(record.get("hadm_id", -1)),
        "stay_id": int(record.get("stay_id", -1)),
        "num_steps": int(record.get("num_steps", len(steps))),
        "last_visit_diagnosis_count": len(last_step.get("diagnosis_ids", [])),
        "last_visit_procedure_count": len(last_step.get("procedure_ids", [])),
        "last_visit_med_history_count": len(last_step.get("med_history_ids", [])),
        "diagnosis_preview": ", ".join(
            format_vocab_name(runtime.get("diagnosis_idx_to_token"), index)
            for index in last_step.get("diagnosis_ids", [])[:3]
        ),
        "procedure_preview": ", ".join(
            format_vocab_name(runtime.get("procedure_idx_to_token"), index)
            for index in last_step.get("procedure_ids", [])[:3]
        ),
    }


def render_patient_form(runtime: Mapping[str, Any]) -> dict[str, Any]:
    _require_streamlit()
    initialize_app_state()

    st.subheader("Patient Input")
    input_mode = st.radio(
        "Input source",
        options=("Demo sample", "Manual quick entry"),
        horizontal=True,
        key="patient_input_mode",
    )

    if input_mode == "Demo sample":
        split = st.selectbox("Demo split", options=("val", "test"), index=0, key="demo_split")
        try:
            dataset = get_dataset(runtime, split)
            dataset_size = len(dataset)
            max_index = max(dataset_size - 1, 0)
            sample_index = st.number_input(
                "Sample index",
                min_value=0,
                max_value=max_index,
                value=min(int(st.session_state.get("demo_sample_index", 0) or 0), max_index),
                step=1,
                key="demo_sample_index",
            )
            if st.button("Load Demo Sample", use_container_width=True):
                record = load_demo_record(runtime, split, int(sample_index))
                batch = collate_batch([record])
                st.session_state["patient_record"] = record
                st.session_state["patient_batch"] = batch
                st.session_state["patient_source"] = "demo"
                st.session_state["patient_source_split"] = split
                _clear_downstream_state()
                st.success(f"Da nap demo sample #{int(sample_index)} tu split `{split}`.")
        except Exception as exc:
            st.error(f"Khong the nap demo sample tu split `{split}`: {type(exc).__name__}: {exc}")
    else:
        default_lab_size = int(runtime.get("lab_feature_size", 0))
        default_vital_size = int(runtime.get("vital_feature_size", 0))
        diagnosis_size = int(len(runtime.get("diagnosis_vocab", {}).get("idx_to_token", [])))
        procedure_size = int(len(runtime.get("procedure_vocab", {}).get("idx_to_token", [])))
        drug_size = int(len(runtime.get("drug_vocab", {}).get("idx_to_token", [])))

        col_a, col_b, col_c = st.columns(3)
        subject_id = col_a.number_input("subject_id", min_value=1, value=999001, step=1, key="manual_subject_id")
        hadm_id = col_b.number_input("hadm_id", min_value=1, value=999101, step=1, key="manual_hadm_id")
        stay_id = col_c.number_input("stay_id", min_value=1, value=999201, step=1, key="manual_stay_id")
        delta_hours = st.number_input("delta_hours", min_value=0.0, value=0.0, step=1.0, key="manual_delta_hours")

        diagnosis_text = st.text_area(
            "Diagnosis indices",
            value="",
            help="Nhap danh sach chi so diagnosis, cach nhau bang dau phay. Vi du: 3,5,10",
            key="manual_diagnosis",
        )
        procedure_text = st.text_area(
            "Procedure indices",
            value="",
            help="Nhap danh sach chi so procedure, cach nhau bang dau phay.",
            key="manual_procedure",
        )
        med_history_text = st.text_area(
            "Medication history indices",
            value="",
            help="Nhap danh sach chi so thuoc lich su, cach nhau bang dau phay.",
            key="manual_med_history",
        )
        lab_text = st.text_area(
            "Lab values",
            value="",
            help="Nhap vector lab rut gon dang so thuc, cach nhau bang dau phay. App se pad them 0 den du kich thuoc mo hinh.",
            key="manual_lab_values",
        )
        vital_text = st.text_area(
            "Vital values",
            value="",
            help="Nhap vector vital rut gon dang so thuc, cach nhau bang dau phay. App se pad them 0 den du kich thuoc mo hinh.",
            key="manual_vital_values",
        )

        with st.expander("Lab / Vital feature preview", expanded=False):
            st.write("First lab features")
            st.dataframe(_feature_preview_rows(runtime.get("lab_metadata", {})), use_container_width=True, hide_index=True)
            st.write("First vital features")
            st.dataframe(_feature_preview_rows(runtime.get("vital_metadata", {})), use_container_width=True, hide_index=True)

        if st.button("Build Patient Input", use_container_width=True):
            try:
                diagnosis_ids, invalid_diag = _sanitize_indices(_parse_int_list(diagnosis_text), diagnosis_size)
                procedure_ids, invalid_proc = _sanitize_indices(_parse_int_list(procedure_text), procedure_size)
                med_history_ids, invalid_med = _sanitize_indices(_parse_int_list(med_history_text), drug_size)

                lab_dense, lab_mask = _build_dense_numeric(_parse_float_list(lab_text), default_lab_size)
                vital_dense, vital_mask = _build_dense_numeric(_parse_float_list(vital_text), default_vital_size)

                record = {
                    "subject_id": int(subject_id),
                    "hadm_id": int(hadm_id),
                    "stay_id": int(stay_id),
                    "intime": "2020-01-01 00:00:00",
                    "outtime": "2020-01-01 12:00:00",
                    "num_steps": 1,
                    "drug_vocab_size": int(drug_size),
                    "lab_feature_size": int(default_lab_size),
                    "vital_feature_size": int(default_vital_size),
                    "steps": [
                        {
                            "diagnosis_ids": diagnosis_ids,
                            "procedure_ids": procedure_ids,
                            "med_history_ids": med_history_ids,
                            "target_drugs": [],
                            "lab_values": lab_dense,
                            "lab_mask": lab_mask,
                            "vital_values": vital_dense,
                            "vital_mask": vital_mask,
                            "delta_hours": float(delta_hours),
                        }
                    ],
                }
                batch = collate_batch([record])
                st.session_state["patient_record"] = record
                st.session_state["patient_batch"] = batch
                st.session_state["patient_source"] = "manual"
                st.session_state["patient_source_split"] = None
                _clear_downstream_state()
                if invalid_diag or invalid_proc or invalid_med:
                    st.warning(
                        "Da bo qua cac index nam ngoai vocabulary: "
                        f"diag={invalid_diag} proc={invalid_proc} med_history={invalid_med}"
                    )
                st.success("Da tao patient input thu cong.")
            except Exception as exc:
                st.error(f"Khong the tao patient input: {type(exc).__name__}: {exc}")

    current_record = st.session_state.get("patient_record")
    current_batch = st.session_state.get("patient_batch")
    current_source = st.session_state.get("patient_source")
    current_split = st.session_state.get("patient_source_split")

    if current_record is not None and current_batch is not None:
        with st.expander("Current Patient Preview", expanded=True):
            st.dataframe(
                [_record_summary(current_record, runtime)],
                use_container_width=True,
                hide_index=True,
            )
            st.caption(
                f"source={current_source or 'unknown'} | "
                f"split={current_split or 'manual'} | "
                f"batch_shape.visit_mask={tuple(current_batch['visit_mask'].shape)}"
            )

    return {
        "record": current_record,
        "batch": current_batch,
        "source": current_source,
        "source_split": current_split,
    }


__all__ = ["render_patient_form"]
