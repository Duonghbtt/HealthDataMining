from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - UI dependency
    import streamlit as st
except ImportError:  # pragma: no cover - UI dependency
    st = None

if st is None:  # pragma: no cover - UI dependency
    def _cache_resource(*args: Any, **kwargs: Any):
        def decorator(func):
            return func

        return decorator
else:
    _cache_resource = st.cache_resource

from src.data.dataset import collate_batch
from src.models.ddi_regularization import load_ddi_matrix
from src.training.train_core import build_core_model, build_dataset, resolve_device
from src.training.train_extended import (
    build_extended_model,
    build_memory_bank_from_dataloader,
    collate_batch_with_records,
)
from src.utils.io import ensure_dir, load_yaml_config, read_json, resolve_path


DEFAULT_SESSION_STATE = {
    "app_runtime": None,
    "patient_record": None,
    "patient_batch": None,
    "patient_source": None,
    "patient_source_split": None,
    "inference_outputs": None,
    "retrieval_payload": None,
    "counterfactual_payload": None,
    "nl_explanation": None,
    "safety_summary": None,
    "ui_threshold": None,
    "ui_top_k": None,
}


def _require_streamlit() -> None:
    if st is None:  # pragma: no cover - UI dependency
        raise RuntimeError("streamlit is required for the app components")


def initialize_app_state() -> None:
    _require_streamlit()
    for key, value in DEFAULT_SESSION_STATE.items():
        st.session_state.setdefault(key, value)


def clear_app_session() -> None:
    _require_streamlit()
    for key in DEFAULT_SESSION_STATE:
        st.session_state[key] = None


def clear_app_runtime_cache() -> None:
    _require_streamlit()
    load_app_runtime_resource.clear()


def _existing_path(candidates: Sequence[Path | None]) -> Path | None:
    for candidate in candidates:
        if candidate is not None and candidate.exists():
            return candidate.resolve()
    return None


def _read_json_if_exists(path: Path | None, default: Any) -> Any:
    if path is None or not path.exists():
        return default
    return read_json(path)


def _sanitize_token(token: str) -> str:
    text = str(token)
    if ":" in text:
        text = text.split(":", 1)[1]
    return text.replace("_", " ").strip() or str(token)


def format_vocab_name(idx_to_token: Mapping[Any, Any] | Sequence[str] | None, index: int) -> str:
    if idx_to_token is None:
        return str(index)
    if isinstance(idx_to_token, Mapping):
        token = idx_to_token.get(index, idx_to_token.get(str(index), f"{index}"))
        return _sanitize_token(str(token))
    if 0 <= int(index) < len(idx_to_token):
        return _sanitize_token(str(idx_to_token[int(index)]))
    return str(index)


def format_drug_name(runtime: Mapping[str, Any], drug_index: int) -> str:
    return format_vocab_name(runtime.get("drug_idx_to_token"), int(drug_index))


def _write_runtime_data_config(
    *,
    project_root: Path,
    data_config: Mapping[str, Any],
    processed_root: Path,
    vocab_root: Path,
) -> Path:
    runtime_config = copy.deepcopy(
        {key: value for key, value in dict(data_config).items() if not str(key).startswith("_")}
    )
    runtime_config.setdefault("paths", {})
    runtime_config["paths"]["processed_root"] = str(processed_root.resolve())
    runtime_config["paths"]["interim_root"] = str(vocab_root.resolve().parent)

    output_dir = ensure_dir(project_root / "outputs" / "app")
    runtime_config_path = output_dir / "runtime_data_app.yaml"
    runtime_config_path.write_text(
        yaml.safe_dump(runtime_config, sort_keys=False),
        encoding="utf-8",
    )
    return runtime_config_path


def _resolve_checkpoint_candidates(project_root: Path, train_cfg: Mapping[str, Any], eval_cfg: Mapping[str, Any]) -> list[Path]:
    checkpoint_dir = resolve_path(
        project_root,
        eval_cfg.get("paths", {}).get(
            "checkpoint_dir",
            train_cfg.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints"),
        ),
    ).resolve()
    return [
        checkpoint_dir / "train_extended_best.pt",
        checkpoint_dir / "train_core_best.pt",
    ]


def _resolve_artifact_paths(
    *,
    project_root: Path,
    train_cfg: Mapping[str, Any],
    data_cfg: Mapping[str, Any],
    eval_cfg: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any] | None,
) -> dict[str, Path | None]:
    checkpoint_paths = dict((checkpoint_payload or {}).get("resolved_paths", {}))
    data_paths = dict(data_cfg.get("paths", {}))
    train_paths = dict(train_cfg.get("paths", {}))
    eval_paths = dict(eval_cfg.get("paths", {}))

    processed_root = _existing_path(
        [
            None
            if checkpoint_paths.get("processed_root") is None
            else Path(checkpoint_paths["processed_root"]).resolve(),
            None
            if data_paths.get("processed_root") is None
            else resolve_path(project_root, data_paths["processed_root"]).resolve(),
            (project_root / "handover_data" / "processed").resolve(),
            (project_root / "handover_data" / "handover_data" / "processed").resolve(),
        ]
    )
    vocab_root = _existing_path(
        [
            None
            if checkpoint_paths.get("vocab_root") is None
            else Path(checkpoint_paths["vocab_root"]).resolve(),
            None
            if train_paths.get("vocab_root") is None
            else resolve_path(project_root, train_paths["vocab_root"]).resolve(),
            None
            if data_paths.get("interim_root") is None
            else (resolve_path(project_root, data_paths["interim_root"]).resolve() / "vocab"),
            (project_root / "handover_data" / "vocab").resolve(),
            (project_root / "handover_data" / "handover_data" / "vocab").resolve(),
        ]
    )
    ddi_matrix_path = _existing_path(
        [
            None
            if checkpoint_paths.get("ddi_matrix_path") is None
            else Path(checkpoint_paths["ddi_matrix_path"]).resolve(),
            None
            if eval_paths.get("ddi_matrix_path") is None
            else resolve_path(project_root, eval_paths["ddi_matrix_path"]).resolve(),
            None
            if train_paths.get("ddi_matrix_path") is None
            else resolve_path(project_root, train_paths["ddi_matrix_path"]).resolve(),
            (project_root / "handover_data" / "processed" / "ddi" / "drug_ddi.pt").resolve(),
            (project_root / "handover_data" / "handover_data" / "processed" / "ddi" / "drug_ddi.pt").resolve(),
        ]
    )
    return {
        "processed_root": processed_root,
        "vocab_root": vocab_root,
        "ddi_matrix_path": ddi_matrix_path,
    }


def _build_model_for_app(
    *,
    checkpoint_path: Path,
    checkpoint_payload: Mapping[str, Any],
    train_config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    runtime_data_config_path: Path,
    vocab_root: Path,
    ddi_matrix_path: Path,
    processed_root: Path,
) -> tuple[torch.nn.Module | None, str, list[str]]:
    notes: list[str] = []
    train_cfg = copy.deepcopy(dict(train_config))
    train_cfg["_resolved_paths"] = {"processed_root": str(processed_root)}
    preferred_mode = "extended" if "extended" in checkpoint_path.name.lower() or checkpoint_payload.get("train_mode") == "extended" else "core"
    state_dict = checkpoint_payload.get("model_state_dict")
    if not isinstance(state_dict, Mapping):
        return None, "none", ["Checkpoint does not contain `model_state_dict`."]

    attempts = ["extended", "core"] if preferred_mode == "extended" else ["core"]
    for attempt in attempts:
        try:
            if attempt == "extended":
                model, _ = build_extended_model(
                    train_config=train_cfg,
                    model_config=model_config,
                    runtime_data_config_path=runtime_data_config_path,
                    vocab_root=vocab_root,
                    ddi_matrix_path=ddi_matrix_path,
                )
                model.load_state_dict(state_dict, strict=True)
                notes.append("Loaded checkpoint in extended mode.")
                return model, "extended", notes

            model, _ = build_core_model(
                train_config=train_cfg,
                model_config=model_config,
                runtime_data_config_path=runtime_data_config_path,
                vocab_root=vocab_root,
                ddi_matrix_path=ddi_matrix_path,
            )
            incompat = model.load_state_dict(state_dict, strict=False)
            if incompat.missing_keys or incompat.unexpected_keys:
                notes.append(
                    "Core fallback ignored incompatible keys: "
                    f"missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}"
                )
            else:
                notes.append("Loaded checkpoint in core mode.")
            return model, "core", notes
        except Exception as exc:
            notes.append(f"{attempt} load failed: {type(exc).__name__}: {exc}")
    return None, "none", notes


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


@_cache_resource(show_spinner=False)
def load_app_runtime_resource() -> dict[str, Any]:
    if st is None:  # pragma: no cover - UI dependency
        raise RuntimeError("streamlit is required to load app resources")

    project_root = PROJECT_ROOT.resolve()
    runtime: dict[str, Any] = {
        "project_root": project_root,
        "model": None,
        "model_available": False,
        "checkpoint_path": None,
        "checkpoint_kind": "none",
        "checkpoint_candidates": [],
        "checkpoint_notes": [],
        "paths": {},
        "errors": [],
        "memory_banks": {},
        "datasets": {},
    }

    train_cfg = load_yaml_config(project_root / "configs" / "train.yaml")
    eval_cfg = load_yaml_config(project_root / "configs" / "eval.yaml")
    data_cfg = load_yaml_config(project_root / "configs" / "data.yaml")
    model_cfg = load_yaml_config(project_root / "configs" / "model.yaml")
    runtime["checkpoint_candidates"] = _resolve_checkpoint_candidates(project_root, train_cfg, eval_cfg)

    checkpoint_path = _existing_path(runtime["checkpoint_candidates"])
    checkpoint_payload: Mapping[str, Any] | None = None
    if checkpoint_path is not None:
        try:
            checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            runtime["checkpoint_path"] = checkpoint_path
        except Exception as exc:
            runtime["errors"].append(f"Failed to load checkpoint {checkpoint_path}: {type(exc).__name__}: {exc}")

    runtime["train_config"] = copy.deepcopy(dict(checkpoint_payload.get("train_config"))) if checkpoint_payload and isinstance(checkpoint_payload.get("train_config"), Mapping) else train_cfg
    runtime["data_config"] = copy.deepcopy(dict(checkpoint_payload.get("data_config"))) if checkpoint_payload and isinstance(checkpoint_payload.get("data_config"), Mapping) else data_cfg
    runtime["model_config"] = copy.deepcopy(dict(checkpoint_payload.get("model_config"))) if checkpoint_payload and isinstance(checkpoint_payload.get("model_config"), Mapping) else model_cfg
    runtime["eval_config"] = eval_cfg

    resolved_paths = _resolve_artifact_paths(
        project_root=project_root,
        train_cfg=runtime["train_config"],
        data_cfg=runtime["data_config"],
        eval_cfg=runtime["eval_config"],
        checkpoint_payload=checkpoint_payload,
    )
    runtime["paths"] = resolved_paths

    vocab_root = resolved_paths["vocab_root"]
    ddi_matrix_path = resolved_paths["ddi_matrix_path"]
    processed_root = resolved_paths["processed_root"]

    runtime["drug_vocab"] = _read_json_if_exists(None if vocab_root is None else vocab_root / "drug_vocab.json", {"idx_to_token": [], "token_to_idx": {}})
    runtime["diagnosis_vocab"] = _read_json_if_exists(None if vocab_root is None else vocab_root / "diagnosis_vocab.json", {"idx_to_token": [], "token_to_idx": {}})
    runtime["procedure_vocab"] = _read_json_if_exists(None if vocab_root is None else vocab_root / "procedure_vocab.json", {"idx_to_token": [], "token_to_idx": {}})
    runtime["lab_metadata"] = _read_json_if_exists(None if vocab_root is None else vocab_root / "lab_metadata.json", {})
    runtime["vital_metadata"] = _read_json_if_exists(None if vocab_root is None else vocab_root / "vital_metadata.json", {})
    runtime["drug_idx_to_token"] = {index: token for index, token in enumerate(runtime["drug_vocab"].get("idx_to_token", []))}
    runtime["diagnosis_idx_to_token"] = {index: token for index, token in enumerate(runtime["diagnosis_vocab"].get("idx_to_token", []))}
    runtime["procedure_idx_to_token"] = {index: token for index, token in enumerate(runtime["procedure_vocab"].get("idx_to_token", []))}

    default_runtime_cfg = dict(eval_cfg.get("runtime", {}))
    default_prediction_cfg = dict(eval_cfg.get("prediction", {}))
    runtime["device"] = resolve_device(default_runtime_cfg.get("device", "cpu"))
    runtime["default_threshold"] = float(default_prediction_cfg.get("threshold", 0.5))
    runtime["default_top_k"] = int(default_prediction_cfg.get("top_k", 10))

    if ddi_matrix_path is not None:
        try:
            runtime["ddi_matrix"] = load_ddi_matrix(ddi_matrix_path, device="cpu")
        except Exception as exc:
            runtime["ddi_matrix"] = None
            runtime["errors"].append(f"Failed to load ddi matrix: {type(exc).__name__}: {exc}")
    else:
        runtime["ddi_matrix"] = None
        runtime["errors"].append("DDI matrix path could not be resolved.")

    lab_indices = [int(payload.get("index", -1)) for payload in runtime["lab_metadata"].values()]
    vital_indices = [int(payload.get("index", -1)) for payload in runtime["vital_metadata"].values()]
    runtime["lab_feature_size"] = max(lab_indices, default=-1) + 1
    runtime["vital_feature_size"] = max(vital_indices, default=-1) + 1

    if checkpoint_path is None:
        runtime["errors"].append("No checkpoint found. Expected train_extended_best.pt or train_core_best.pt.")
        return runtime
    if vocab_root is None or processed_root is None or ddi_matrix_path is None:
        runtime["errors"].append("Missing processed/vocab/ddi artifacts required to build the model.")
        return runtime

    runtime_data_config_path = _write_runtime_data_config(
        project_root=project_root,
        data_config=runtime["data_config"],
        processed_root=processed_root,
        vocab_root=vocab_root,
    )
    runtime["runtime_data_config_path"] = runtime_data_config_path

    try:
        model, checkpoint_kind, notes = _build_model_for_app(
            checkpoint_path=checkpoint_path,
            checkpoint_payload=checkpoint_payload or {},
            train_config=runtime["train_config"],
            model_config=runtime["model_config"],
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=vocab_root,
            ddi_matrix_path=ddi_matrix_path,
            processed_root=processed_root,
        )
        runtime["checkpoint_kind"] = checkpoint_kind
        runtime["checkpoint_notes"] = notes
        if model is None:
            runtime["errors"].extend(notes)
            return runtime

        runtime["model"] = model.to(runtime["device"])
        runtime["model"].eval()
        runtime["model_available"] = True
        runtime["lab_feature_size"] = int(getattr(runtime["model"].encoder.lab_projection, "in_features", runtime["lab_feature_size"]))
        runtime["vital_feature_size"] = int(getattr(runtime["model"].encoder.vital_projection, "in_features", runtime["vital_feature_size"]))
        runtime["drug_vocab_size"] = int(len(runtime["drug_vocab"].get("idx_to_token", [])))
    except Exception as exc:
        runtime["errors"].append(f"Failed to build app model: {type(exc).__name__}: {exc}")

    return runtime


def get_app_runtime(*, force_refresh: bool = False) -> dict[str, Any]:
    _require_streamlit()
    initialize_app_state()
    if force_refresh:
        clear_app_runtime_cache()
    runtime = load_app_runtime_resource()
    st.session_state["app_runtime"] = runtime
    st.session_state.setdefault("ui_threshold", runtime.get("default_threshold"))
    st.session_state.setdefault("ui_top_k", runtime.get("default_top_k"))
    return runtime


def render_runtime_status(runtime: Mapping[str, Any], *, compact: bool = False) -> None:
    _require_streamlit()

    checkpoint_path = runtime.get("checkpoint_path")
    processed_root = runtime.get("paths", {}).get("processed_root")
    vocab_root = runtime.get("paths", {}).get("vocab_root")
    ddi_matrix_path = runtime.get("paths", {}).get("ddi_matrix_path")

    st.subheader("Artifact Status" if not compact else "Runtime Status")
    cols = st.columns(4)
    cols[0].metric("Model", "Ready" if runtime.get("model_available") else "Unavailable")
    cols[1].metric("Checkpoint", runtime.get("checkpoint_kind", "none"))
    cols[2].metric("Default Top-K", int(runtime.get("default_top_k", 0)))
    cols[3].metric("Threshold", f"{float(runtime.get('default_threshold', 0.5)):.2f}")

    artifact_rows = [
        {"artifact": "checkpoint", "path": "" if checkpoint_path is None else str(checkpoint_path), "exists": checkpoint_path is not None and Path(checkpoint_path).exists()},
        {"artifact": "processed_root", "path": "" if processed_root is None else str(processed_root), "exists": processed_root is not None and Path(processed_root).exists()},
        {"artifact": "vocab_root", "path": "" if vocab_root is None else str(vocab_root), "exists": vocab_root is not None and Path(vocab_root).exists()},
        {"artifact": "ddi_matrix", "path": "" if ddi_matrix_path is None else str(ddi_matrix_path), "exists": ddi_matrix_path is not None and Path(ddi_matrix_path).exists()},
    ]
    st.dataframe(artifact_rows, use_container_width=True, hide_index=True)

    if runtime.get("checkpoint_notes"):
        with st.expander("Checkpoint Notes", expanded=not compact):
            for note in runtime["checkpoint_notes"]:
                st.write(f"- {note}")
    if runtime.get("errors"):
        with st.expander("Runtime Errors", expanded=True):
            for error in runtime["errors"]:
                st.error(error)


def get_dataset(runtime: Mapping[str, Any], split: str):
    split_name = str(split)
    cached = runtime.get("datasets", {}).get(split_name)
    if cached is not None:
        return cached

    processed_root = runtime.get("paths", {}).get("processed_root")
    runtime_data_config_path = runtime.get("runtime_data_config_path")
    if processed_root is None or runtime_data_config_path is None:
        raise RuntimeError("Runtime data configuration is not available for dataset loading.")

    drug_vocab_size = int(len(runtime.get("drug_vocab", {}).get("idx_to_token", [])))
    dataset = build_dataset(
        split=split_name,
        runtime_data_config_path=Path(runtime_data_config_path),
        processed_root=Path(processed_root),
        drug_vocab_size=drug_vocab_size,
    )
    runtime["datasets"][split_name] = dataset
    return dataset


def load_demo_record(runtime: Mapping[str, Any], split: str, index: int) -> dict[str, Any]:
    dataset = get_dataset(runtime, split)
    if len(dataset) <= 0:
        raise ValueError(f"Split `{split}` is empty.")
    safe_index = min(max(int(index), 0), len(dataset) - 1)
    return dict(dataset[safe_index])


def ensure_memory_bank(runtime: Mapping[str, Any], split: str) -> tuple[Any | None, str | None]:
    split_name = str(split)
    cached = runtime.get("memory_banks", {}).get(split_name)
    if cached is not None:
        return cached, None

    if not runtime.get("model_available", False):
        return None, "model_unavailable"

    try:
        dataset = get_dataset(runtime, split_name)
        dataloader = DataLoader(
            dataset,
            batch_size=16,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_batch_with_records,
        )
        bank = build_memory_bank_from_dataloader(
            model=runtime["model"],
            dataloader=dataloader,
            device=runtime["device"],
            split=split_name,
        )
        runtime["memory_banks"][split_name] = bank
        return bank, None
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"


def run_patient_inference(
    runtime: Mapping[str, Any],
    *,
    batch: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]] | None,
    use_retrieval: bool,
    source_split: str | None,
    top_k: int | None = None,
) -> dict[str, Any]:
    if not runtime.get("model_available", False):
        raise RuntimeError("Model is unavailable because no valid checkpoint has been loaded.")

    model = runtime["model"]
    device = runtime["device"]
    batch_on_device = _move_batch_to_device(batch, device)
    resolved_top_k = int(top_k if top_k is not None else runtime.get("default_top_k", 10))
    messages: list[str] = []
    requested_mode = "core"
    memory_bank = None
    memory_bank_split: str | None = None

    if use_retrieval and str(runtime.get("checkpoint_kind", "core")) == "extended":
        requested_mode = "extended"
        memory_bank_split = str(source_split or "train")
        memory_bank, bank_error = ensure_memory_bank(runtime, memory_bank_split)
        if memory_bank is None:
            requested_mode = "core"
            messages.append(
                "Retrieval khong san sang, app fallback sang core mode: "
                f"{bank_error or 'unknown memory bank error'}"
            )

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            try:
                outputs = model(
                    batch_on_device,
                    mode=requested_mode,
                    memory_bank=memory_bank if requested_mode == "extended" else None,
                    records=list(records) if requested_mode == "extended" and records is not None else None,
                    decoder_top_k=resolved_top_k,
                )
                effective_mode = requested_mode
            except Exception as exc:
                if requested_mode != "extended":
                    raise
                messages.append(f"Extended inference failed, fallback sang core mode: {type(exc).__name__}: {exc}")
                outputs = model(
                    batch_on_device,
                    mode="core",
                    memory_bank=None,
                    records=None,
                    decoder_top_k=resolved_top_k,
                )
                effective_mode = "core"

        resolved_outputs = _to_cpu(outputs)
        resolved_outputs["effective_mode"] = effective_mode
        resolved_outputs["messages"] = messages
        resolved_outputs["memory_bank_split"] = memory_bank_split
        return resolved_outputs
    finally:
        model.train(was_training)


def get_top_recommendation_rows(
    runtime: Mapping[str, Any],
    outputs: Mapping[str, Any],
    *,
    top_k: int | None = None,
) -> list[dict[str, Any]]:
    drug_probs = outputs.get("drug_probs")
    if not isinstance(drug_probs, torch.Tensor):
        return []
    if drug_probs.ndim != 2 or drug_probs.shape[0] <= 0:
        return []

    resolved_top_k = int(top_k if top_k is not None else runtime.get("default_top_k", 10))
    recommendation_metadata = outputs.get("recommendation_metadata", {})
    topk_indices = recommendation_metadata.get("topk_indices")
    topk_scores = recommendation_metadata.get("topk_scores")

    rows: list[dict[str, Any]] = []
    if isinstance(topk_indices, torch.Tensor) and isinstance(topk_scores, torch.Tensor):
        indices = topk_indices[0].tolist()
        scores = topk_scores[0].tolist()
    else:
        values, indices_tensor = torch.topk(drug_probs[0], k=min(resolved_top_k, drug_probs.shape[1]), dim=-1)
        indices = indices_tensor.tolist()
        scores = values.tolist()

    for rank, (index, score) in enumerate(zip(indices, scores), start=1):
        rows.append(
            {
                "rank": rank,
                "drug_index": int(index),
                "drug_name": format_drug_name(runtime, int(index)),
                "score": float(score),
            }
        )
    return rows


def compute_patient_safety_summary(
    runtime: Mapping[str, Any],
    outputs: Mapping[str, Any],
    *,
    threshold: float,
) -> dict[str, Any]:
    drug_probs = outputs.get("drug_probs")
    if not isinstance(drug_probs, torch.Tensor):
        raise RuntimeError("Inference outputs do not contain `drug_probs`.")
    if drug_probs.ndim != 2 or int(drug_probs.shape[0]) != 1:
        raise ValueError(f"App safety view expects a single patient, got shape {tuple(drug_probs.shape)}")

    probs = drug_probs[0].detach().cpu()
    predicted_indices = torch.nonzero(probs >= float(threshold), as_tuple=False).flatten().tolist()
    predicted_indices = [int(index) for index in predicted_indices]

    interacting_pairs: list[dict[str, Any]] = []
    ddi_matrix = runtime.get("ddi_matrix")
    if isinstance(ddi_matrix, torch.Tensor) and len(predicted_indices) >= 2:
        ddi_upper = torch.triu(torch.logical_or(ddi_matrix > 0, (ddi_matrix > 0).transpose(0, 1)), diagonal=1)
        for row_offset, left in enumerate(predicted_indices):
            for right in predicted_indices[row_offset + 1 :]:
                if bool(ddi_upper[left, right].item()):
                    interacting_pairs.append(
                        {
                            "left_index": int(left),
                            "left_name": format_drug_name(runtime, int(left)),
                            "right_index": int(right),
                            "right_name": format_drug_name(runtime, int(right)),
                        }
                    )

    ddi_penalty_value = None
    ddi_penalty_mean = outputs.get("ddi_penalty_mean")
    if isinstance(ddi_penalty_mean, torch.Tensor) and ddi_penalty_mean.numel() == 1:
        ddi_penalty_value = float(ddi_penalty_mean.item())

    return {
        "predicted_drug_count": len(predicted_indices),
        "predicted_indices": predicted_indices,
        "predicted_names": [format_drug_name(runtime, index) for index in predicted_indices],
        "interacting_pairs": interacting_pairs,
        "has_ddi": bool(interacting_pairs),
        "ddi_penalty": ddi_penalty_value,
        "retrieval_used": bool(outputs.get("retrieval_used", False)),
        "effective_mode": str(outputs.get("effective_mode", outputs.get("retrieval_mode", "core"))),
    }


def render_recommendation_panel(
    runtime: Mapping[str, Any],
    outputs: Mapping[str, Any],
    *,
    top_k: int | None = None,
    threshold: float | None = None,
) -> None:
    _require_streamlit()
    recommendation_rows = get_top_recommendation_rows(runtime, outputs, top_k=top_k)
    resolved_threshold = float(threshold if threshold is not None else runtime.get("default_threshold", 0.5))
    safety_summary = compute_patient_safety_summary(runtime, outputs, threshold=resolved_threshold)

    st.subheader("Recommendation")
    metrics = st.columns(4)
    metrics[0].metric("Effective Mode", str(outputs.get("effective_mode", "core")).upper())
    metrics[1].metric("Retrieval Used", "Yes" if bool(outputs.get("retrieval_used", False)) else "No")
    metrics[2].metric("Predicted Drugs", int(safety_summary["predicted_drug_count"]))
    metrics[3].metric("DDI Warning", "Yes" if safety_summary["has_ddi"] else "No")

    if outputs.get("messages"):
        for message in outputs["messages"]:
            st.info(message)

    if recommendation_rows:
        st.dataframe(recommendation_rows, use_container_width=True, hide_index=True)
    else:
        st.warning("Khong co top recommendation nao de hien thi.")

    st.caption(
        f"Threshold hien tai: {resolved_threshold:.2f} | "
        f"retrieval_mode={outputs.get('retrieval_mode', 'disabled')} | "
        f"decoder_available={outputs.get('recommendation_metadata', {}).get('decoder_available', False)}"
    )


__all__ = [
    "_move_batch_to_device",
    "clear_app_runtime_cache",
    "clear_app_session",
    "collate_batch",
    "compute_patient_safety_summary",
    "ensure_memory_bank",
    "format_drug_name",
    "format_vocab_name",
    "get_app_runtime",
    "get_dataset",
    "get_top_recommendation_rows",
    "initialize_app_state",
    "load_demo_record",
    "render_recommendation_panel",
    "render_runtime_status",
    "run_patient_inference",
]
