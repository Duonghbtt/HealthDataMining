from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from metrics import (  # type: ignore[import-not-found]
        compute_core_metrics,
        compute_core_metrics_for_mask,
        compute_ddi_flags,
        compute_samplewise_f1,
        compute_samplewise_jaccard,
        select_binary_predictions,
    )
else:
    from .metrics import (
        compute_core_metrics,
        compute_core_metrics_for_mask,
        compute_ddi_flags,
        compute_samplewise_f1,
        compute_samplewise_jaccard,
        select_binary_predictions,
    )

from src.data.dataset import validate_patient_level_splits
from src.models.ddi_regularization import load_ddi_matrix
from src.training.runtime_builder import (
    build_core_model,
    build_dataset,
    build_runtime_data_config_file,
    resolve_device,
    select_collate_fn,
)
from src.utils.io import ensure_dir, load_yaml_config, read_json, resolve_path, write_json

THRESHOLD_CANDIDATES: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
_SUBGROUP_ORDER: tuple[str, ...] = ("all_visits", "first_visit", "short_history", "long_history")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the core ClinRec medication recommendation model.")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to configs/eval.yaml")
    parser.add_argument("--data-config", default=None, help="Optional override for configs/data.yaml")
    parser.add_argument("--model-config", default=None, help="Optional override for configs/model.yaml")
    parser.add_argument("--train-config", default=None, help="Optional override for configs/train.yaml")
    parser.add_argument("--checkpoint", default=None, help="Optional override for best checkpoint path")
    parser.add_argument("--split", default=None, help="Optional override for evaluation split")
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Optional override for prediction threshold; skips validation threshold tuning",
    )
    parser.add_argument("--device", default=None, help="Optional override for runtime device")
    parser.add_argument("--processed-root", default=None, help="Optional override for processed data root")
    parser.add_argument("--vocab-root", default=None, help="Optional override for vocab directory")
    parser.add_argument("--ddi-matrix-path", default=None, help="Optional override for DDI matrix artifact")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a short evaluation path with limited batches for integration checking",
    )
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Optional cap for evaluation batches")
    return parser.parse_args()


def _load_embedded_or_yaml_config(
    *,
    explicit_path: str | None,
    embedded_payload: Mapping[str, Any] | None,
    fallback_path: Path,
) -> dict[str, Any]:
    if explicit_path is not None:
        return load_yaml_config(explicit_path)
    if embedded_payload is not None:
        return copy.deepcopy(dict(embedded_payload))
    return load_yaml_config(fallback_path)


def _existing_path_candidates_to_path(candidates: Sequence[str | Path | None]) -> list[Path]:
    resolved: list[Path] = []
    for candidate in candidates:
        if candidate is None:
            continue
        resolved.append(Path(candidate).resolve())
    return resolved


def _resolve_existing_path(
    *,
    kind: str,
    candidates: Sequence[str | Path | None],
) -> Path:
    checked: list[str] = []
    for candidate in _existing_path_candidates_to_path(candidates):
        checked.append(str(candidate))
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        f"Unable to resolve {kind}. Checked candidates: {checked if checked else ['<none>']}"
    )


def _write_plain_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    normalized_rows = [dict(row) for row in rows]
    fieldnames: list[str] = []
    for row in normalized_rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in normalized_rows:
            writer.writerow(row)
    return destination


def _merge_nested_dicts(base: Mapping[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    if override is None:
        return merged
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_nested_dicts(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _resolve_retrieval_policy(model: torch.nn.Module) -> dict[str, Any]:
    if hasattr(model, "get_retrieval_policy"):
        policy = model.get_retrieval_policy()
        if isinstance(policy, Mapping):
            return dict(policy)
    return {
        "memory_bank_split": None,
        "has_absolute_time": False,
        "all_visits_have_absolute_time": False,
        "exact_match_blocked": False,
        "same_patient_future_blocked": False,
        "cross_patient_absolute_temporal_filter": False,
        "notes": "Retrieval policy is unavailable because the model does not expose it.",
    }


def _flatten_report(prefix: str, payload: Mapping[str, Any], sink: dict[str, Any]) -> None:
    for key, value in payload.items():
        resolved_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            _flatten_report(resolved_key, value, sink)
        else:
            sink[resolved_key] = value


def _move_batch_to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.detach().cpu().reshape(-1).tolist()]
    return [int(item) for item in value]


def _resolve_checkpoint_path(project_root: Path, eval_config: Mapping[str, Any], args: argparse.Namespace) -> Path:
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint path does not exist: {checkpoint_path}")
        return checkpoint_path

    checkpoint_dir = resolve_path(
        project_root,
        eval_config.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints"),
    ).resolve()
    checkpoint_path = checkpoint_dir / "train_core_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found at {checkpoint_path}")
    return checkpoint_path


def _resolve_eval_paths(
    *,
    project_root: Path,
    eval_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Path]:
    eval_paths = dict(eval_config.get("paths", {}))
    train_paths = dict(train_config.get("paths", {}))
    data_paths = dict(data_config.get("paths", {}))
    checkpoint_paths = dict(checkpoint_payload.get("resolved_paths", {}))

    processed_root = _resolve_existing_path(
        kind="processed_root",
        candidates=[
            args.processed_root,
            checkpoint_paths.get("processed_root"),
            None if data_paths.get("processed_root") is None else resolve_path(project_root, data_paths["processed_root"]),
            project_root / "handover_data" / "processed",
        ],
    )
    vocab_root = _resolve_existing_path(
        kind="vocab_root",
        candidates=[
            args.vocab_root,
            checkpoint_paths.get("vocab_root"),
            None if train_paths.get("vocab_root") is None else resolve_path(project_root, train_paths["vocab_root"]),
            None if data_paths.get("interim_root") is None else resolve_path(project_root, data_paths["interim_root"]) / "vocab",
            project_root / "handover_data" / "vocab",
        ],
    )
    ddi_matrix_path = _resolve_existing_path(
        kind="ddi_matrix_path",
        candidates=[
            args.ddi_matrix_path,
            checkpoint_paths.get("ddi_matrix_path"),
            None if eval_paths.get("ddi_matrix_path") is None else resolve_path(project_root, eval_paths["ddi_matrix_path"]),
            None if train_paths.get("ddi_matrix_path") is None else resolve_path(project_root, train_paths["ddi_matrix_path"]),
            project_root / "handover_data" / "processed" / "ddi" / "drug_ddi.pt",
        ],
    )

    report_dir = ensure_dir(resolve_path(project_root, eval_paths.get("report_dir", "outputs/reports")).resolve())
    prediction_dir = ensure_dir(
        resolve_path(project_root, eval_paths.get("prediction_dir", "outputs/predictions")).resolve()
    )

    return {
        "processed_root": processed_root,
        "vocab_root": vocab_root,
        "ddi_matrix_path": ddi_matrix_path,
        "report_dir": report_dir,
        "prediction_dir": prediction_dir,
    }


def _stringify_indices(indices: torch.Tensor) -> str:
    if indices.numel() == 0:
        return ""
    return ";".join(str(int(index)) for index in indices.tolist())


def build_eval_dataloader(
    *,
    split: str,
    runtime_data_config_path: Path,
    processed_root: Path,
    drug_vocab_size: int,
    batch_size: int,
) -> DataLoader:
    split_validation = validate_patient_level_splits(runtime_data_config_path)
    if bool(split_validation.get("validated")):
        print(f"Validated patient-level split manifests: {split_validation['counts']}")
    dataset = build_dataset(
        split=split,
        runtime_data_config_path=runtime_data_config_path,
        processed_root=processed_root,
        drug_vocab_size=drug_vocab_size,
    )
    if len(dataset) <= 0:
        raise ValueError(f"Evaluation dataset for split `{split}` is empty")
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=False,
        num_workers=0,
        collate_fn=select_collate_fn(dataset),
    )


def _collect_core_outputs(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, Any]:
    collected_probs: list[torch.Tensor] = []
    collected_targets: list[torch.Tensor] = []
    patient_ids: list[int] = []
    subject_ids: list[int] = []
    hadm_ids: list[int] = []
    stay_ids: list[int] = []
    visit_index: list[int] = []
    visit_position: list[int] = []
    history_length: list[int] = []
    retrieval_valid_candidate_counts: list[torch.Tensor] = []
    retrieved_scores: list[torch.Tensor] = []
    retrieved_indices: list[torch.Tensor] = []

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= int(max_batches):
                break
            batch_on_device = _move_batch_to_device(batch, device)
            outputs = model(batch_on_device)
            drug_probs = outputs.get("drug_probs")
            final_target_drugs = outputs.get("final_target_drugs")
            if final_target_drugs is None:
                final_target_drugs = outputs.get("target_current")
            if drug_probs is None:
                raise RuntimeError("Model did not return `drug_probs` during evaluation.")
            if final_target_drugs is None:
                raise RuntimeError("Model did not return current-visit targets during evaluation.")

            collected_probs.append(drug_probs.detach().cpu())
            collected_targets.append(final_target_drugs.detach().cpu())
            patient_ids.extend(_as_int_list(batch.get("patient_ids", batch.get("subject_ids", []))))
            subject_ids.extend(int(value) for value in batch.get("subject_ids", []))
            hadm_ids.extend(int(value) for value in batch.get("hadm_ids", []))
            stay_ids.extend(int(value) for value in batch.get("stay_ids", []))
            visit_index.extend(_as_int_list(batch.get("visit_index")))
            visit_position.extend(_as_int_list(batch.get("visit_position")))
            history_length.extend(_as_int_list(batch.get("history_length")))
            if isinstance(outputs.get("retrieval_valid_candidate_counts"), torch.Tensor):
                retrieval_valid_candidate_counts.append(outputs["retrieval_valid_candidate_counts"].detach().cpu())
            if isinstance(outputs.get("retrieved_scores"), torch.Tensor):
                retrieved_scores.append(outputs["retrieved_scores"].detach().cpu())
            if isinstance(outputs.get("retrieved_indices"), torch.Tensor):
                retrieved_indices.append(outputs["retrieved_indices"].detach().cpu())

    if not collected_probs or not collected_targets:
        raise ValueError("Evaluation dataloader produced no batches")

    return {
        "drug_probs": torch.cat(collected_probs, dim=0),
        "targets": torch.cat(collected_targets, dim=0),
        "patient_ids": patient_ids,
        "subject_ids": subject_ids,
        "hadm_ids": hadm_ids,
        "stay_ids": stay_ids,
        "visit_index": visit_index,
        "visit_position": visit_position,
        "history_length": history_length,
        "retrieval_valid_candidate_counts": None
        if not retrieval_valid_candidate_counts
        else torch.cat(retrieval_valid_candidate_counts, dim=0),
        "retrieved_scores": None if not retrieved_scores else torch.cat(retrieved_scores, dim=0),
        "retrieved_indices": None if not retrieved_indices else torch.cat(retrieved_indices, dim=0),
    }


def _summarize_core_evaluation(
    *,
    collected_outputs: Mapping[str, Any],
    selection_config: Mapping[str, Any],
    ddi_matrix: torch.Tensor,
) -> dict[str, Any]:
    resolved_selection = _normalize_prediction_selection(selection_config)
    all_probs = collected_outputs["drug_probs"]
    all_targets = collected_outputs["targets"]
    patient_ids = [int(value) for value in collected_outputs.get("patient_ids", [])]
    subject_ids = [int(value) for value in collected_outputs.get("subject_ids", [])]
    hadm_ids = [int(value) for value in collected_outputs.get("hadm_ids", [])]
    stay_ids = [int(value) for value in collected_outputs.get("stay_ids", [])]
    visit_index = [int(value) for value in collected_outputs.get("visit_index", [])]
    visit_position = [int(value) for value in collected_outputs.get("visit_position", [])]
    history_length = [int(value) for value in collected_outputs.get("history_length", [])]
    retrieval_valid_candidate_counts = collected_outputs.get("retrieval_valid_candidate_counts")
    retrieved_scores = collected_outputs.get("retrieved_scores")
    retrieved_indices = collected_outputs.get("retrieved_indices")

    selection_kwargs = _selection_metric_kwargs(resolved_selection)
    binary_predictions = select_binary_predictions(
        all_probs,
        **selection_kwargs,
    ).cpu()
    ddi_matrix_cpu = ddi_matrix.detach().cpu()

    metrics = compute_core_metrics(
        all_targets,
        all_probs,
        threshold=float(selection_kwargs["threshold"]),
        ddi_matrix=ddi_matrix_cpu,
        prediction_method=str(selection_kwargs["prediction_method"]),
        top_k=selection_kwargs["top_k"],
        percentile=selection_kwargs["percentile"],
    )
    sample_jaccard = compute_samplewise_jaccard(all_targets, binary_predictions).cpu()
    sample_f1 = compute_samplewise_f1(all_targets, binary_predictions).cpu()
    ddi_flags = compute_ddi_flags(binary_predictions, ddi_matrix_cpu).cpu()

    prediction_rows: list[dict[str, Any]] = []
    for row_index in range(all_probs.shape[0]):
        predicted_indices = torch.nonzero(binary_predictions[row_index], as_tuple=False).flatten()
        prediction_rows.append(
            {
                "subject_id": subject_ids[row_index] if row_index < len(subject_ids) else -1,
                "patient_id": patient_ids[row_index] if row_index < len(patient_ids) else -1,
                "hadm_id": hadm_ids[row_index] if row_index < len(hadm_ids) else -1,
                "stay_id": stay_ids[row_index] if row_index < len(stay_ids) else -1,
                "visit_index": visit_index[row_index] if row_index < len(visit_index) else -1,
                "visit_position": visit_position[row_index] if row_index < len(visit_position) else -1,
                "history_length": history_length[row_index] if row_index < len(history_length) else -1,
                "true_count": int(all_targets[row_index].sum().item()),
                "pred_count": int(binary_predictions[row_index].sum().item()),
                "sample_jaccard": float(sample_jaccard[row_index].item()),
                "sample_f1": float(sample_f1[row_index].item()),
                "has_ddi": bool(ddi_flags[row_index].item()),
                "predicted_drug_indices": _stringify_indices(predicted_indices),
            }
        )

    prediction_summary = {
        "avg_predicted_drugs": float(binary_predictions.sum(dim=1, dtype=torch.float32).mean().item()),
        "avg_true_drugs": float(all_targets.sum(dim=1, dtype=torch.float32).mean().item()),
    }
    ddi_summary = {
        key: metrics[key]
        for key in ("ddi_rate", "total_predicted_pairs", "total_interacting_pairs", "patients_with_ddi", "num_samples")
    }
    metric_summary = {
        key: metrics[key]
        for key in ("jaccard", "f1", "prauc", "ddi_rate", "avg_predicted_drugs", "avg_true_drugs")
    }
    retrieval_summary = {
        "avg_valid_candidates": 0.0,
        "avg_retrieved_score": 0.0,
        "fraction_with_retrieval_context": 0.0,
    }
    if isinstance(retrieval_valid_candidate_counts, torch.Tensor):
        retrieval_summary["avg_valid_candidates"] = float(
            retrieval_valid_candidate_counts.to(dtype=torch.float32).mean().item()
        )
        retrieval_summary["fraction_with_retrieval_context"] = float(
            (retrieval_valid_candidate_counts > 0).to(dtype=torch.float32).mean().item()
        )
    if isinstance(retrieved_scores, torch.Tensor):
        if isinstance(retrieved_indices, torch.Tensor):
            valid_score_mask = retrieved_indices >= 0
        else:
            valid_score_mask = torch.ones_like(retrieved_scores, dtype=torch.bool)
        if bool(valid_score_mask.any().item()):
            retrieval_summary["avg_retrieved_score"] = float(retrieved_scores[valid_score_mask].mean().item())

    return {
        "drug_probs": all_probs,
        "targets": all_targets,
        "patient_ids": patient_ids,
        "subject_ids": subject_ids,
        "hadm_ids": hadm_ids,
        "stay_ids": stay_ids,
        "visit_index": visit_index,
        "visit_position": visit_position,
        "history_length": history_length,
        "prediction_rows": prediction_rows,
        "prediction_summary": prediction_summary,
        "retrieval_summary": retrieval_summary,
        "ddi_summary": ddi_summary,
        "metrics": metric_summary,
        "selection_config": resolved_selection,
    }


def _normalize_prediction_selection(selection_config: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(selection_config or {})
    method = str(payload.get("method", "global")).strip().lower()
    if method == "global":
        threshold = float(payload.get("threshold", 0.5))
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"Global prediction threshold must be in [0, 1], got {threshold!r}")
        return {
            "method": "global",
            "threshold": threshold,
            "top_k": None,
            "percentile": None,
        }
    if method == "topk":
        top_k = int(payload.get("top_k", 0))
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k!r}")
        return {
            "method": "topk",
            "threshold": None,
            "top_k": top_k,
            "percentile": None,
        }
    if method == "percentile":
        percentile = float(payload.get("percentile", 0.0))
        if not 0.0 <= percentile <= 100.0:
            raise ValueError(f"percentile must be in [0, 100], got {percentile!r}")
        return {
            "method": "percentile",
            "threshold": None,
            "top_k": None,
            "percentile": percentile,
        }
    raise ValueError(f"Unsupported prediction selection method: {method!r}")


def _selection_threshold_value(selection_config: Mapping[str, Any]) -> float:
    threshold = selection_config.get("threshold")
    if threshold is None:
        return 0.5
    return float(threshold)


def _selection_metric_kwargs(selection_config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "threshold": _selection_threshold_value(selection_config),
        "prediction_method": str(selection_config["method"]),
        "top_k": selection_config.get("top_k"),
        "percentile": selection_config.get("percentile"),
    }


def _selection_value(selection_config: Mapping[str, Any]) -> float:
    if selection_config["method"] == "global":
        return float(selection_config["threshold"])
    if selection_config["method"] == "topk":
        return float(selection_config["top_k"])
    return float(selection_config["percentile"])


def _selection_label(selection_config: Mapping[str, Any]) -> str:
    method = str(selection_config["method"])
    if method == "global":
        return f"threshold:{float(selection_config['threshold']):.2f}"
    if method == "topk":
        return f"topk:{int(selection_config['top_k'])}"
    return f"percentile:{float(selection_config['percentile']):.1f}"


def _build_threshold_search_candidates(eval_config: Mapping[str, Any]) -> list[dict[str, Any]]:
    search_cfg = dict(eval_config.get("threshold_search", {}))
    methods = [str(value).strip().lower() for value in search_cfg.get("methods", ["global"])]
    candidates: list[dict[str, Any]] = []
    if "global" in methods:
        for threshold in search_cfg.get("global_thresholds", list(THRESHOLD_CANDIDATES)):
            candidates.append({"method": "global", "threshold": float(threshold)})
    if "topk" in methods:
        for top_k in search_cfg.get("topk_values", []):
            candidates.append({"method": "topk", "top_k": int(top_k)})
    if "percentile" in methods:
        for percentile in search_cfg.get("percentile_values", []):
            candidates.append({"method": "percentile", "percentile": float(percentile)})
    if not candidates:
        raise ValueError("threshold_search is enabled but no threshold search candidates were configured.")
    return [_normalize_prediction_selection(candidate) for candidate in candidates]


def _score_threshold_candidate(
    *,
    drug_probs: torch.Tensor,
    y_true: torch.Tensor,
    ddi_matrix: torch.Tensor,
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    resolved_candidate = _normalize_prediction_selection(candidate)
    selection_kwargs = _selection_metric_kwargs(resolved_candidate)
    metrics = compute_core_metrics(
        y_true=y_true,
        y_score=drug_probs,
        threshold=float(selection_kwargs["threshold"]),
        ddi_matrix=ddi_matrix,
        prediction_method=str(selection_kwargs["prediction_method"]),
        top_k=selection_kwargs["top_k"],
        percentile=selection_kwargs["percentile"],
    )
    return {
        "method": str(resolved_candidate["method"]),
        "threshold": resolved_candidate.get("threshold"),
        "top_k": resolved_candidate.get("top_k"),
        "percentile": resolved_candidate.get("percentile"),
        "threshold_or_k": _selection_value(resolved_candidate),
        "selection_label": _selection_label(resolved_candidate),
        "jaccard": float(metrics["jaccard"]),
        "f1": float(metrics["f1"]),
        "prauc": float(metrics["prauc"]),
        "ddi_rate": float(metrics["ddi_rate"]),
        "avg_drugs": float(metrics["avg_predicted_drugs"]),
        "avg_predicted_drugs": float(metrics["avg_predicted_drugs"]),
        "avg_true_drugs": float(metrics["avg_true_drugs"]),
    }


def _run_threshold_search(
    *,
    drug_probs: torch.Tensor,
    y_true: torch.Tensor,
    ddi_matrix: torch.Tensor,
    eval_config: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    search_cfg = dict(eval_config.get("threshold_search", {}))
    selection_metric = str(search_cfg.get("selection_metric", "jaccard")).strip().lower()
    if selection_metric not in {"jaccard", "f1"}:
        raise ValueError(
            f"threshold_search.selection_metric must be 'jaccard' or 'f1', got {selection_metric!r}"
        )
    prefer_lower_ddi_as_tiebreaker = bool(search_cfg.get("prefer_lower_ddi_as_tiebreaker", True))
    comparison_rows = [
        _score_threshold_candidate(
            drug_probs=drug_probs,
            y_true=y_true,
            ddi_matrix=ddi_matrix,
            candidate=candidate,
        )
        for candidate in _build_threshold_search_candidates(eval_config)
    ]

    best_row = comparison_rows[0]
    for row in comparison_rows[1:]:
        current_metric = float(row[selection_metric])
        best_metric = float(best_row[selection_metric])
        if current_metric > best_metric:
            best_row = row
            continue
        if (
            abs(current_metric - best_metric) <= 1.0e-12
            and prefer_lower_ddi_as_tiebreaker
            and float(row["ddi_rate"]) < float(best_row["ddi_rate"])
        ):
            best_row = row

    best_config = _normalize_prediction_selection(best_row)
    return {
        "tuned_on_split": "val",
        "used_for_split": split,
        "selection_metric": selection_metric,
        "prefer_lower_ddi_as_tiebreaker": prefer_lower_ddi_as_tiebreaker,
        "best_config": best_config,
        "best_row": best_row,
        "comparison_rows": comparison_rows,
    }


def _build_target_diagnostics(
    *,
    all_targets: torch.Tensor,
    prediction_summary: Mapping[str, Any],
) -> dict[str, Any]:
    if all_targets.ndim != 2:
        raise ValueError(f"Expected evaluation targets with shape (N, D), got {tuple(all_targets.shape)}")
    if int(all_targets.shape[1]) <= 1:
        raise ValueError("Evaluation targets must have width > 1 to inspect UNK at index 1.")

    unk_positive_count = float(all_targets[:, 1].sum().item())
    return {
        "avg_predicted_drugs": float(prediction_summary["avg_predicted_drugs"]),
        "avg_true_drugs": float(prediction_summary["avg_true_drugs"]),
        "unk_positive_count": unk_positive_count,
        "unk_present_in_targets": bool(unk_positive_count > 0.0),
    }


def _build_subgroup_masks(collected_outputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    num_samples = int(collected_outputs["targets"].shape[0])
    visit_position = torch.as_tensor(collected_outputs.get("visit_position", []), dtype=torch.long)
    history_length = torch.as_tensor(collected_outputs.get("history_length", []), dtype=torch.long)
    if int(visit_position.numel()) != num_samples or int(history_length.numel()) != num_samples:
        raise ValueError(
            "Collected evaluation metadata is missing visit_position/history_length for subgroup evaluation."
        )

    all_mask = torch.ones(num_samples, dtype=torch.bool)
    first_visit_mask = visit_position <= 1
    short_history_mask = history_length <= 2
    long_history_mask = history_length > 2
    return {
        "all_visits": all_mask,
        "first_visit": first_visit_mask,
        "short_history": short_history_mask,
        "long_history": long_history_mask,
    }


def _summarize_subgroup_metrics(
    *,
    collected_outputs: Mapping[str, Any],
    selection_config: Mapping[str, Any],
    ddi_matrix: torch.Tensor,
) -> dict[str, dict[str, Any]]:
    resolved_selection = _normalize_prediction_selection(selection_config)
    selection_kwargs = _selection_metric_kwargs(resolved_selection)
    subgroup_masks = _build_subgroup_masks(collected_outputs)
    subgroup_metrics: dict[str, dict[str, Any]] = {}
    for subgroup_name in _SUBGROUP_ORDER:
        subgroup_mask = subgroup_masks[subgroup_name]
        metrics = compute_core_metrics_for_mask(
            y_true=collected_outputs["targets"],
            y_score=collected_outputs["drug_probs"],
            threshold=float(selection_kwargs["threshold"]),
            ddi_matrix=ddi_matrix,
            sample_mask=subgroup_mask,
            prediction_method=str(selection_kwargs["prediction_method"]),
            top_k=selection_kwargs["top_k"],
            percentile=selection_kwargs["percentile"],
        )
        subgroup_metrics[subgroup_name] = {
            "num_samples": int(subgroup_mask.sum().item()),
            **metrics,
        }
    return subgroup_metrics


def _print_subgroup_metrics(subgroup_metrics: Mapping[str, Mapping[str, Any]]) -> None:
    print("Subgroup metrics:")
    for subgroup_name in _SUBGROUP_ORDER:
        payload = subgroup_metrics[subgroup_name]
        print(
            f"  {subgroup_name}: "
            f"n={int(payload['num_samples'])} "
            f"jaccard={float(payload['jaccard']):.4f} "
            f"f1={float(payload['f1']):.4f} "
            f"prauc={float(payload['prauc']):.4f} "
            f"ddi_rate={float(payload['ddi_rate']):.4f} "
            f"avg_predicted_drugs={float(payload['avg_predicted_drugs']):.4f}"
        )


def run_core_evaluation(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    selection_config: Mapping[str, Any],
    ddi_matrix: torch.Tensor,
    max_eval_batches: int | None = None,
) -> dict[str, Any]:
    collected_outputs = _collect_core_outputs(
        model=model,
        dataloader=dataloader,
        device=device,
        max_batches=max_eval_batches,
    )
    return _summarize_core_evaluation(
        collected_outputs=collected_outputs,
        selection_config=selection_config,
        ddi_matrix=ddi_matrix,
    )


def _resolve_prediction_selection(
    *,
    explicit_threshold: float | None,
    checkpoint_payload: Mapping[str, Any],
    eval_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    split: str,
    model: torch.nn.Module,
    val_dataloader: DataLoader,
    device: torch.device,
    ddi_matrix: torch.Tensor,
    max_eval_batches: int | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any] | None, dict[str, Any] | None]:
    prediction_cfg = dict(eval_config.get("prediction", {}))
    train_prediction_cfg = dict(train_config.get("prediction", {}))
    threshold_search_cfg = dict(eval_config.get("threshold_search", {}))

    if explicit_threshold is not None:
        return (
            _normalize_prediction_selection({"method": "global", "threshold": float(explicit_threshold)}),
            "cli_or_caller_override",
            None,
            None,
        )

    if bool(threshold_search_cfg.get("enabled", False)):
        val_outputs = _collect_core_outputs(
            model=model,
            dataloader=val_dataloader,
            device=device,
            max_batches=max_eval_batches,
        )
        search_report = _run_threshold_search(
            drug_probs=val_outputs["drug_probs"],
            y_true=val_outputs["targets"],
            ddi_matrix=ddi_matrix.detach().cpu(),
            eval_config=eval_config,
            split=split,
        )
        return (
            _normalize_prediction_selection(search_report["best_config"]),
            "validation_threshold_search",
            search_report,
            val_outputs,
        )

    if prediction_cfg.get("method") is not None or prediction_cfg.get("threshold") is not None:
        return _normalize_prediction_selection(prediction_cfg), "eval_config", None, None

    checkpoint_threshold = checkpoint_payload.get("threshold")
    if checkpoint_threshold is not None:
        return (
            _normalize_prediction_selection({"method": "global", "threshold": float(checkpoint_threshold)}),
            "checkpoint",
            None,
            None,
        )

    if train_prediction_cfg.get("method") is not None or train_prediction_cfg.get("threshold") is not None:
        return _normalize_prediction_selection(train_prediction_cfg), "train_config", None, None

    raise ValueError(
        "No prediction selection was provided and threshold_search is disabled. "
        "Set prediction.threshold, prediction.method, or enable threshold_search."
    )


def evaluate_checkpoint(
    *,
    checkpoint_path: str | Path,
    eval_config_path: str | Path = "configs/eval.yaml",
    split: str | None = None,
    threshold: float | None = None,
    device: torch.device | str | None = None,
    data_config_path: str | None = None,
    model_config_path: str | None = None,
    train_config_path: str | None = None,
    processed_root: str | None = None,
    vocab_root: str | None = None,
    ddi_matrix_path: str | None = None,
    max_eval_batches: int | None = None,
    eval_config_override: Mapping[str, Any] | None = None,
    model_config_override: Mapping[str, Any] | None = None,
    report_stem_override: str | None = None,
) -> dict[str, Any]:
    eval_config = _merge_nested_dicts(load_yaml_config(eval_config_path), eval_config_override)
    project_root = Path(eval_config["_project_root"]).resolve()
    checkpoint_path = Path(checkpoint_path).resolve()
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config_refs = dict(eval_config.get("config_refs", {}))
    train_config = _load_embedded_or_yaml_config(
        explicit_path=train_config_path,
        embedded_payload=checkpoint_payload.get("train_config"),
        fallback_path=resolve_path(project_root, config_refs.get("train", "configs/train.yaml")),
    )
    data_config = _load_embedded_or_yaml_config(
        explicit_path=data_config_path,
        embedded_payload=checkpoint_payload.get("data_config"),
        fallback_path=resolve_path(project_root, config_refs.get("data", "configs/data.yaml")),
    )
    model_config = _load_embedded_or_yaml_config(
        explicit_path=model_config_path,
        embedded_payload=checkpoint_payload.get("model_config"),
        fallback_path=resolve_path(project_root, config_refs.get("model", "configs/model.yaml")),
    )
    model_config = _merge_nested_dicts(model_config, model_config_override)

    namespace = argparse.Namespace(
        checkpoint=str(checkpoint_path),
        split=split,
        threshold=threshold,
        device=None if device is None else str(device),
        processed_root=processed_root,
        vocab_root=vocab_root,
        ddi_matrix_path=ddi_matrix_path,
    )
    resolved_paths = _resolve_eval_paths(
        project_root=project_root,
        eval_config=eval_config,
        train_config=train_config,
        data_config=data_config,
        checkpoint_payload=checkpoint_payload,
        args=namespace,
    )
    print("Resolved evaluation paths:")
    for key, value in resolved_paths.items():
        print(f"  {key}: {value}")

    runtime_cfg = dict(eval_config.get("runtime", {}))
    run_cfg = dict(eval_config.get("run", {}))
    evaluation_cfg = dict(eval_config.get("evaluation", {}))
    baseline_cfg = dict(train_config.get("baseline", {}))

    resolved_split = str(split or evaluation_cfg.get("split", "test"))
    resolved_device = resolve_device(str(device) if device is not None else runtime_cfg.get("device", "cpu"))
    batch_size = int(runtime_cfg.get("batch_size", 32))
    resolved_max_eval_batches = (
        int(max_eval_batches)
        if max_eval_batches is not None
        else int(run_cfg["max_eval_batches"])
        if run_cfg.get("max_eval_batches") is not None
        else None
    )

    ddi_matrix = load_ddi_matrix(resolved_paths["ddi_matrix_path"], device="cpu")
    med_vocab_path = resolved_paths["vocab_root"] / "med_vocab_main.json"
    legacy_drug_vocab_path = resolved_paths["vocab_root"] / "drug_vocab.json"
    resolved_drug_vocab_path = med_vocab_path if med_vocab_path.exists() else legacy_drug_vocab_path
    drug_vocab_size = int(read_json(resolved_drug_vocab_path)["size"])
    if ddi_matrix.shape[0] != drug_vocab_size:
        raise ValueError(
            "DDI matrix width must match drug vocabulary size: "
            f"got ddi={int(ddi_matrix.shape[0])}, vocab={drug_vocab_size}"
        )

    print(f"Using device: {resolved_device}")
    print(f"Evaluating split: {resolved_split}")
    print(f"Loading checkpoint: {checkpoint_path}")

    with tempfile.TemporaryDirectory(prefix="clinrec_eval_runtime_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        runtime_data_config_path = build_runtime_data_config_file(
            project_root=project_root,
            data_config=data_config,
            processed_root=resolved_paths["processed_root"],
            vocab_root=resolved_paths["vocab_root"],
            temp_dir=temp_dir,
            retrieval_cache_config=train_config.get("retrieval_cache"),
        )

        need_val_dataloader = (
            resolved_split == "val"
            or bool(eval_config.get("threshold_search", {}).get("enabled", False))
        )
        train_retrieval_dataloader: DataLoader | None = None
        val_dataloader: DataLoader | None = None
        if need_val_dataloader:
            val_dataloader = build_eval_dataloader(
                split="val",
                runtime_data_config_path=runtime_data_config_path,
                processed_root=resolved_paths["processed_root"],
                drug_vocab_size=drug_vocab_size,
                batch_size=batch_size,
            )
        dataloader = (
            val_dataloader
            if resolved_split == "val" and val_dataloader is not None
            else build_eval_dataloader(
                split=resolved_split,
                runtime_data_config_path=runtime_data_config_path,
                processed_root=resolved_paths["processed_root"],
                drug_vocab_size=drug_vocab_size,
                batch_size=batch_size,
            )
        )
        model = build_core_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            processed_root=resolved_paths["processed_root"],
            vocab_root=resolved_paths["vocab_root"],
            ddi_matrix_path=resolved_paths["ddi_matrix_path"],
        )
        if bool(getattr(model, "use_retrieval", False)) and not bool(
            getattr(model, "uses_precomputed_retrieval_cache", False)
        ):
            train_retrieval_dataloader = build_eval_dataloader(
                split="train",
                runtime_data_config_path=runtime_data_config_path,
                processed_root=resolved_paths["processed_root"],
                drug_vocab_size=drug_vocab_size,
                batch_size=batch_size,
            )

    model_state_dict = checkpoint_payload.get("model_state_dict")
    if not isinstance(model_state_dict, Mapping):
        raise KeyError("Checkpoint does not contain `model_state_dict`.")
    model.load_state_dict(model_state_dict, strict=True)
    if (
        bool(getattr(model, "use_retrieval", False))
        and not bool(getattr(model, "uses_precomputed_retrieval_cache", False))
        and train_retrieval_dataloader is not None
    ):
        retrieval_bank = model.refresh_retrieval_memory_bank(
            train_retrieval_dataloader,
            split_name="train",
            device=resolved_device,
        )
        if retrieval_bank is not None:
            print(
                "Refreshed retrieval memory bank for evaluation "
                f"(visits={retrieval_bank.num_visits}, history_mode={getattr(model, 'history_mode', 'self_only')})"
            )
    retrieval_policy = _resolve_retrieval_policy(model)
    print(
        "Retrieval policy: "
        f"absolute_time={bool(retrieval_policy.get('has_absolute_time', False))} "
        f"same_patient_future_blocked={bool(retrieval_policy.get('same_patient_future_blocked', False))} "
        f"cross_patient_absolute_temporal_filter="
        f"{bool(retrieval_policy.get('cross_patient_absolute_temporal_filter', False))}"
    )
    print(str(retrieval_policy.get("notes", "")))

    threshold_search_report: dict[str, Any] | None = None
    val_outputs: dict[str, Any] | None = None
    resolved_selection_config, threshold_source, threshold_search_report, val_outputs = _resolve_prediction_selection(
        explicit_threshold=threshold,
        checkpoint_payload=checkpoint_payload,
        eval_config=eval_config,
        train_config=train_config,
        split=resolved_split,
        model=model,
        val_dataloader=val_dataloader if val_dataloader is not None else dataloader,
        device=resolved_device,
        ddi_matrix=ddi_matrix,
        max_eval_batches=resolved_max_eval_batches,
    )
    if threshold_search_report is not None:
        best_row = dict(threshold_search_report["best_row"])
        if str(best_row["method"]) == "global":
            print(
                f"Best threshold on val: {float(best_row['threshold']):.2f}, "
                f"val Jaccard: {float(best_row['jaccard']):.4f}"
            )
        else:
            print(
                "Best threshold config on val: "
                f"{best_row['selection_label']}, "
                f"val_{threshold_search_report['selection_metric']}="
                f"{float(best_row[threshold_search_report['selection_metric']]):.4f}"
            )
    print(f"Using prediction selection: {_selection_label(resolved_selection_config)} ({threshold_source})")

    if resolved_split == "val" and val_outputs is not None:
        evaluation_result = _summarize_core_evaluation(
            collected_outputs=val_outputs,
            selection_config=resolved_selection_config,
            ddi_matrix=ddi_matrix,
        )
    else:
        evaluation_result = run_core_evaluation(
            model=model,
            dataloader=dataloader,
            device=resolved_device,
            selection_config=resolved_selection_config,
            ddi_matrix=ddi_matrix,
            max_eval_batches=resolved_max_eval_batches,
        )
    subgroup_metrics = _summarize_subgroup_metrics(
        collected_outputs=evaluation_result,
        selection_config=resolved_selection_config,
        ddi_matrix=ddi_matrix,
    )
    _print_subgroup_metrics(subgroup_metrics)

    diagnostics = _build_target_diagnostics(
        all_targets=evaluation_result["targets"],
        prediction_summary=evaluation_result["prediction_summary"],
    )
    print(f"Average predicted drugs per patient: {float(diagnostics['avg_predicted_drugs']):.4f}")
    print(f"Average true drugs per patient: {float(diagnostics['avg_true_drugs']):.4f}")
    print(f"UNK positive count in targets: {float(diagnostics['unk_positive_count']):.4f}")
    print(f"UNK present in targets: {bool(diagnostics['unk_present_in_targets'])}")
    print(
        "Retrieval diagnostics: "
        f"avg_valid_candidates={float(evaluation_result['retrieval_summary']['avg_valid_candidates']):.4f} "
        f"avg_retrieved_score={float(evaluation_result['retrieval_summary']['avg_retrieved_score']):.4f}"
    )

    report: dict[str, Any] = {
        "split": resolved_split,
        "baseline_mode": str(
            baseline_cfg.get("mode")
            or checkpoint_payload.get("baseline_mode")
            or "current_self_history_ddi"
        ),
        "history_mode": str(getattr(model, "history_mode", "self_only")),
        "use_self_history": bool(baseline_cfg.get("use_self_history", True)),
        "use_retrieval": bool(getattr(model, "use_retrieval", False)),
        "use_ddi": bool(baseline_cfg.get("use_ddi", True)),
        "num_samples": int(evaluation_result["targets"].shape[0]),
        "threshold": None if resolved_selection_config.get("threshold") is None else float(resolved_selection_config["threshold"]),
        "top_k": None if resolved_selection_config.get("top_k") is None else int(resolved_selection_config["top_k"]),
        "percentile": None if resolved_selection_config.get("percentile") is None else float(resolved_selection_config["percentile"]),
        "selection_config": dict(resolved_selection_config),
        "selection_method": str(resolved_selection_config["method"]),
        "selection_value": float(_selection_value(resolved_selection_config)),
        "threshold_source": threshold_source,
        "checkpoint_path": str(checkpoint_path),
        "device": str(resolved_device),
        "metrics": evaluation_result["metrics"],
        "ddi_summary": evaluation_result["ddi_summary"],
        "prediction_summary": evaluation_result["prediction_summary"],
        "retrieval_summary": evaluation_result["retrieval_summary"],
        "retrieval_policy": retrieval_policy,
        "subgroup_metrics": subgroup_metrics,
        "diagnostics": diagnostics,
        "artifacts": {},
    }
    if threshold_search_report is not None:
        report["threshold_search"] = threshold_search_report

    save_reports = bool(evaluation_cfg.get("save_reports", True))
    save_predictions = bool(evaluation_cfg.get("save_predictions", True))
    report_cfg = dict(eval_config.get("report", {}))
    report_stem = str(report_stem_override or f"evaluate_core_{resolved_split}")

    if save_reports:
        if threshold_search_report is not None and bool(report_cfg.get("save_threshold_comparison", True)):
            threshold_comparison_json = write_json(
                resolved_paths["report_dir"] / f"{report_stem}_threshold_comparison.json",
                threshold_search_report["comparison_rows"],
            )
            threshold_comparison_csv = _write_plain_csv(
                resolved_paths["report_dir"] / f"{report_stem}_threshold_comparison.csv",
                threshold_search_report["comparison_rows"],
            )
            best_threshold_json = write_json(
                resolved_paths["report_dir"] / f"{report_stem}_best_threshold_config.json",
                threshold_search_report["best_config"],
            )
            report["artifacts"]["threshold_comparison_json"] = str(threshold_comparison_json)
            report["artifacts"]["threshold_comparison_csv"] = str(threshold_comparison_csv)
            report["artifacts"]["best_threshold_config_json"] = str(best_threshold_json)
        if threshold_search_report is not None and bool(report_cfg.get("save_tradeoff_curve_data", True)):
            tradeoff_json = write_json(
                resolved_paths["report_dir"] / f"{report_stem}_tradeoff_accuracy_safety.json",
                threshold_search_report["comparison_rows"],
            )
            tradeoff_csv = _write_plain_csv(
                resolved_paths["report_dir"] / f"{report_stem}_tradeoff_accuracy_safety.csv",
                threshold_search_report["comparison_rows"],
            )
            report["artifacts"]["tradeoff_accuracy_safety_json"] = str(tradeoff_json)
            report["artifacts"]["tradeoff_accuracy_safety_csv"] = str(tradeoff_csv)
        subgroup_json_path = write_json(
            resolved_paths["report_dir"] / f"{report_stem}_subgroup_metrics.json",
            subgroup_metrics,
        )
        subgroup_rows = [
            {"subgroup": subgroup_name, **dict(subgroup_metrics[subgroup_name])}
            for subgroup_name in _SUBGROUP_ORDER
        ]
        subgroup_csv_path = _write_plain_csv(
            resolved_paths["report_dir"] / f"{report_stem}_subgroup_metrics.csv",
            subgroup_rows,
        )
        retrieval_policy_json_path = write_json(
            resolved_paths["report_dir"] / f"{report_stem}_retrieval_policy.json",
            retrieval_policy,
        )
        json_path = write_json(resolved_paths["report_dir"] / f"{report_stem}.json", report)
        flat_report: dict[str, Any] = {}
        _flatten_report("", report, flat_report)
        csv_path = _write_plain_csv(resolved_paths["report_dir"] / f"{report_stem}.csv", [flat_report])
        report["artifacts"]["json"] = str(json_path)
        report["artifacts"]["csv"] = str(csv_path)
        report["artifacts"]["subgroup_metrics_json"] = str(subgroup_json_path)
        report["artifacts"]["subgroup_metrics_csv"] = str(subgroup_csv_path)
        report["artifacts"]["retrieval_policy_json"] = str(retrieval_policy_json_path)

    if save_predictions:
        prediction_csv_path = _write_plain_csv(
            resolved_paths["prediction_dir"] / f"{report_stem}_predictions.csv",
            evaluation_result["prediction_rows"],
        )
        report["artifacts"]["predictions_csv"] = str(prediction_csv_path)

    if save_reports:
        write_json(resolved_paths["report_dir"] / f"{report_stem}.json", report)

    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
    return report


def main() -> None:
    args = parse_args()
    eval_config = load_yaml_config(args.config)
    project_root = Path(eval_config["_project_root"]).resolve()
    checkpoint_path = _resolve_checkpoint_path(project_root, eval_config, args)
    run_cfg = dict(eval_config.get("run", {}))
    smoke_test = bool(args.smoke_test or run_cfg.get("smoke_test", False))
    max_eval_batches = (
        args.max_eval_batches
        if args.max_eval_batches is not None
        else run_cfg.get("max_eval_batches")
    )
    if smoke_test and max_eval_batches is None:
        max_eval_batches = 2

    evaluate_checkpoint(
        checkpoint_path=checkpoint_path,
        eval_config_path=args.config,
        split=args.split,
        threshold=args.threshold,
        device=args.device,
        data_config_path=args.data_config,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        processed_root=args.processed_root,
        vocab_root=args.vocab_root,
        ddi_matrix_path=args.ddi_matrix_path,
        max_eval_batches=max_eval_batches,
    )


if __name__ == "__main__":
    main()

