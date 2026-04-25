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
        binarize_predictions,
        compute_core_metrics,
        compute_core_metrics_from_binary_predictions,
        compute_ddi_flags,
        compute_samplewise_f1,
        compute_samplewise_jaccard,
        multilabel_prauc,
    )
    from prediction_control import (  # type: ignore[import-not-found]
        collect_model_predictions,
        compute_average_target_cardinality_from_dataloader,
        normalize_prediction_config,
        prediction_mode_requires_calibration,
        prediction_mode_requires_train_cardinality,
        resolve_prediction_control,
    )
else:
    from .metrics import (
        binarize_predictions,
        compute_core_metrics,
        compute_core_metrics_from_binary_predictions,
        compute_ddi_flags,
        compute_samplewise_f1,
        compute_samplewise_jaccard,
        multilabel_prauc,
    )
    from .prediction_control import (
        collect_model_predictions,
        compute_average_target_cardinality_from_dataloader,
        normalize_prediction_config,
        prediction_mode_requires_calibration,
        prediction_mode_requires_train_cardinality,
        resolve_prediction_control,
    )

from src.data.build_vocab import load_vocab_bundle
from src.data.dataset import build_collate_fn, collate_batch
from src.models.ddi_regularization import load_ddi_artifact
from src.training.train_core import (
    build_core_model,
    build_dataset,
    build_core_memory_bank,
    build_runtime_data_config_file,
    resolve_device,
    validate_core_runtime_config,
)
from src.utils.io import ensure_dir, load_yaml_config, read_json, resolve_path, write_json
from src.utils.runtime_truth import (
    build_core_runtime_truth,
    normalize_ddi_context,
    normalize_initialization_context,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the core ClinRec medication recommendation model.")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to configs/eval.yaml")
    parser.add_argument("--data-config", default=None, help="Optional override for configs/data.yaml")
    parser.add_argument("--model-config", default=None, help="Optional override for configs/model.yaml")
    parser.add_argument("--train-config", default=None, help="Optional override for configs/train.yaml")
    parser.add_argument("--checkpoint", default=None, help="Optional override for best checkpoint path")
    parser.add_argument("--split", default=None, help="Optional override for evaluation split")
    parser.add_argument("--threshold", type=float, default=None, help="Optional override for prediction threshold")
    parser.add_argument(
        "--prediction-mode",
        default=None,
        choices=("threshold", "calibrated_threshold", "top_k"),
        help="Optional override for evaluation prediction control mode",
    )
    parser.add_argument("--prediction-top-k", type=int, default=None, help="Optional override for top-k prediction mode")
    parser.add_argument("--device", default=None, help="Optional override for runtime device")
    parser.add_argument("--processed-root", default=None, help="Optional override for processed data root")
    parser.add_argument("--vocab-root", default=None, help="Optional override for vocab directory")
    parser.add_argument("--ddi-matrix-path", default=None, help="Optional override for DDI matrix artifact")
    parser.add_argument("--report-dir", default=None, help="Optional override for report output directory")
    parser.add_argument("--prediction-dir", default=None, help="Optional override for prediction output directory")
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
    candidates: Sequence[tuple[str, str | Path | None]],
) -> tuple[Path, str]:
    checked: list[str] = []
    for label, raw_candidate in candidates:
        for candidate in _existing_path_candidates_to_path([raw_candidate]):
            checked.append(f"{label}={candidate}")
            if candidate.exists():
                return candidate, label
    raise FileNotFoundError(
        f"Unable to resolve {kind}. Checked candidates: {checked if checked else ['<none>']}"
    )


def _core_checkpoint_help(train_config_path: str | Path) -> str:
    return (
        "Pass --checkpoint /path/to/train_core_best.pt or run "
        f"`python -m src.training.train_core --config {train_config_path}` first."
    )


def _stringify_path_source(path: Path, source: str) -> str:
    return f"{path} [{source}]"


def _compatibility_fallback_used(*, sources: Sequence[str]) -> bool:
    return any(source.startswith("compat:") for source in sources)


def _resolve_checkpoint_path(project_root: Path, eval_config: Mapping[str, Any], args: argparse.Namespace) -> Path:
    train_config_ref = resolve_path(
        project_root,
        dict(eval_config.get("config_refs", {})).get("train", "configs/train.yaml"),
    ).resolve()
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint).resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint path does not exist: {checkpoint_path}. {_core_checkpoint_help(train_config_ref)}"
            )
        return checkpoint_path

    checkpoint_dir = resolve_path(
        project_root,
        eval_config.get("paths", {}).get("checkpoint_dir", "outputs/checkpoints"),
    ).resolve()
    checkpoint_path = checkpoint_dir / "train_core_best.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Default core checkpoint not found at {checkpoint_path}. {_core_checkpoint_help(train_config_ref)}"
        )
    return checkpoint_path


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


def _build_ddi_summary(ddi_artifact: Mapping[str, Any], metrics: Mapping[str, Any] | None = None) -> dict[str, Any]:
    normalized = normalize_ddi_context(ddi_artifact)
    summary = {
        "available": bool(normalized["active"]),
        "status": str(normalized["status"]),
        "reason": normalized.get("reason"),
        "source": normalized.get("source"),
        "requested_source": normalized.get("requested_source"),
        "requested_source_format": normalized.get("requested_source_format"),
        "effective_source": normalized.get("effective_source"),
        "effective_source_format": normalized.get("effective_source_format"),
        "source_format": normalized.get("source_format"),
        "matched_pairs": normalized.get("matched_pairs"),
        "nonzero_pairs": normalized.get("nonzero_pairs"),
        "vocab_size": normalized.get("vocab_size"),
        "fallback_reason": normalized.get("fallback_reason"),
        "source_metadata": copy.deepcopy(dict(normalized.get("source_metadata") or {})),
        "ddi_active": bool(normalized["active"]),
        "ddi_type": str(normalized.get("ddi_type") or "unknown"),
        "ddi_research_grade": bool(normalized.get("ddi_research_grade", False)),
        "ddi_source": str(normalized.get("ddi_source") or normalized.get("effective_source") or ""),
        "ddi_rate": None,
        "total_predicted_pairs": None,
        "total_interacting_pairs": None,
        "patients_with_ddi": None,
        "num_samples": None,
    }
    if metrics is not None:
        for key in ("ddi_rate", "total_predicted_pairs", "total_interacting_pairs", "patients_with_ddi", "num_samples"):
            summary[key] = metrics.get(key)
    return summary


def _resolve_eval_paths(
    *,
    project_root: Path,
    eval_config: Mapping[str, Any],
    train_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    eval_paths = dict(eval_config.get("paths", {}))
    train_paths = dict(train_config.get("paths", {}))
    data_paths = dict(data_config.get("paths", {}))
    checkpoint_paths = dict(checkpoint_payload.get("resolved_paths", {}))

    processed_root, processed_root_source = _resolve_existing_path(
        kind="processed_root",
        candidates=[
            ("arg:processed_root", args.processed_root),
            ("checkpoint.resolved_paths.processed_root", checkpoint_paths.get("processed_root")),
            (
                "eval.paths.processed_root",
                None if eval_paths.get("processed_root") is None else resolve_path(project_root, eval_paths["processed_root"]),
            ),
            (
                "train.paths.processed_root",
                None if train_paths.get("processed_root") is None else resolve_path(project_root, train_paths["processed_root"]),
            ),
            (
                "data.paths.processed_root",
                None if data_paths.get("processed_root") is None else resolve_path(project_root, data_paths["processed_root"]),
            ),
            ("compat:handover_data/processed", project_root / "handover_data" / "processed"),
        ],
    )
    vocab_root, vocab_root_source = _resolve_existing_path(
        kind="vocab_root",
        candidates=[
            ("arg:vocab_root", args.vocab_root),
            ("checkpoint.resolved_paths.vocab_root", checkpoint_paths.get("vocab_root")),
            (
                "eval.paths.vocab_root",
                None if eval_paths.get("vocab_root") is None else resolve_path(project_root, eval_paths["vocab_root"]),
            ),
            (
                "train.paths.vocab_root",
                None if train_paths.get("vocab_root") is None else resolve_path(project_root, train_paths["vocab_root"]),
            ),
            (
                "data.paths.interim_root/vocab",
                None if data_paths.get("interim_root") is None else resolve_path(project_root, data_paths["interim_root"]) / "vocab",
            ),
            ("compat:handover_data/vocab", project_root / "handover_data" / "vocab"),
        ],
    )
    ddi_matrix_path, ddi_matrix_source = _resolve_existing_path(
        kind="ddi_matrix_path",
        candidates=[
            ("arg:ddi_matrix_path", args.ddi_matrix_path),
            ("checkpoint.resolved_paths.ddi_matrix_path", checkpoint_paths.get("ddi_matrix_path")),
            (
                "eval.paths.ddi_matrix_path",
                None if eval_paths.get("ddi_matrix_path") is None else resolve_path(project_root, eval_paths["ddi_matrix_path"]),
            ),
            (
                "train.paths.ddi_matrix_path",
                None if train_paths.get("ddi_matrix_path") is None else resolve_path(project_root, train_paths["ddi_matrix_path"]),
            ),
            ("compat:handover_data/processed/ddi/drug_ddi.pt", project_root / "handover_data" / "processed" / "ddi" / "drug_ddi.pt"),
        ],
    )

    report_dir = ensure_dir(
        (
            Path(args.report_dir).resolve()
            if args.report_dir is not None
            else resolve_path(project_root, eval_paths.get("report_dir", "outputs/reports")).resolve()
        )
    )
    prediction_dir = ensure_dir(
        (
            Path(args.prediction_dir).resolve()
            if args.prediction_dir is not None
            else resolve_path(project_root, eval_paths.get("prediction_dir", "outputs/predictions")).resolve()
        )
    )

    return {
        "processed_root": processed_root,
        "vocab_root": vocab_root,
        "ddi_matrix_path": ddi_matrix_path,
        "report_dir": report_dir,
        "prediction_dir": prediction_dir,
        "processed_root_source": processed_root_source,
        "vocab_root_source": vocab_root_source,
        "ddi_matrix_path_source": ddi_matrix_source,
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
    include_records: bool = False,
) -> DataLoader:
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
        collate_fn=build_collate_fn(
            include_full_targets=True,
            include_final_target=True,
            include_records=include_records,
        )
        if include_records
        else collate_batch,
    )


def _resolve_decoder_top_k(prediction_cfg: Mapping[str, Any]) -> int:
    decoder_top_k = prediction_cfg.get("decoder_top_k", prediction_cfg.get("top_k", 10))
    return int(decoder_top_k)


def _summarize_core_evaluation_payload(
    *,
    payload: Mapping[str, Any],
    binary_predictions: torch.Tensor,
    ddi_artifact: Mapping[str, Any],
) -> dict[str, Any]:
    ddi_artifact = normalize_ddi_context(ddi_artifact)
    all_probs = torch.as_tensor(payload["probs"], dtype=torch.float32)
    all_targets = torch.as_tensor(payload["targets"], dtype=torch.float32)
    subject_ids = list(payload.get("subject_ids", []))
    hadm_ids = list(payload.get("hadm_ids", []))
    stay_ids = list(payload.get("stay_ids", []))
    resolved_predictions = torch.as_tensor(binary_predictions, dtype=torch.bool).cpu()

    sample_jaccard = compute_samplewise_jaccard(all_targets, resolved_predictions).cpu()
    sample_f1 = compute_samplewise_f1(all_targets, resolved_predictions).cpu()
    ddi_active = bool(ddi_artifact.get("active", False))
    ddi_matrix_cpu = ddi_artifact["matrix"].detach().cpu()
    if ddi_active:
        metrics = compute_core_metrics_from_binary_predictions(
            all_targets,
            all_probs,
            resolved_predictions,
            ddi_matrix=ddi_matrix_cpu,
        )
        ddi_flags = compute_ddi_flags(resolved_predictions, ddi_matrix_cpu).cpu()
        ddi_summary = _build_ddi_summary(ddi_artifact, metrics=metrics)
        metric_summary: dict[str, Any] = {
            key: metrics[key]
            for key in ("jaccard", "f1", "prauc", "ddi_rate")
        }
    else:
        ddi_flags = torch.zeros(resolved_predictions.shape[0], dtype=torch.bool)
        ddi_summary = _build_ddi_summary(
            ddi_artifact,
            metrics={"num_samples": float(all_targets.shape[0])},
        )
        metric_summary = {
            "jaccard": float(sample_jaccard.mean().item()),
            "f1": float(sample_f1.mean().item()),
            "prauc": multilabel_prauc(all_targets, all_probs),
            "ddi_rate": None,
        }

    prediction_rows: list[dict[str, Any]] = []
    for row_index in range(all_probs.shape[0]):
        predicted_indices = torch.nonzero(resolved_predictions[row_index], as_tuple=False).flatten()
        prediction_rows.append(
            {
                "subject_id": subject_ids[row_index] if row_index < len(subject_ids) else -1,
                "hadm_id": hadm_ids[row_index] if row_index < len(hadm_ids) else -1,
                "stay_id": stay_ids[row_index] if row_index < len(stay_ids) else -1,
                "true_count": int(all_targets[row_index].sum().item()),
                "pred_count": int(resolved_predictions[row_index].sum().item()),
                "sample_jaccard": float(sample_jaccard[row_index].item()),
                "sample_f1": float(sample_f1[row_index].item()),
                "has_ddi": None if not ddi_active else bool(ddi_flags[row_index].item()),
                "predicted_drug_indices": _stringify_indices(predicted_indices),
            }
        )

    prediction_summary = {
        "avg_predicted_drugs": float(resolved_predictions.sum(dim=1, dtype=torch.float32).mean().item()),
        "avg_true_drugs": float(all_targets.sum(dim=1, dtype=torch.float32).mean().item()),
    }

    return {
        "drug_probs": all_probs,
        "targets": all_targets,
        "binary_predictions": resolved_predictions,
        "prediction_rows": prediction_rows,
        "prediction_summary": prediction_summary,
        "ddi_summary": ddi_summary,
        "metrics": metric_summary,
    }


def run_core_evaluation(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    ddi_artifact: Mapping[str, Any],
    decoder_top_k: int | None,
    prediction_config: Mapping[str, Any],
    checkpoint_payload: Mapping[str, Any] | None = None,
    cli_threshold: float | None = None,
    cli_prediction_mode: str | None = None,
    cli_prediction_top_k: int | None = None,
    calibration_dataloader: DataLoader | None = None,
    train_cardinality_dataloader: DataLoader | None = None,
    memory_bank: Any | None = None,
) -> dict[str, Any]:
    ddi_artifact = normalize_ddi_context(ddi_artifact)
    evaluation_payload = collect_model_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        decoder_top_k=decoder_top_k,
        memory_bank=memory_bank,
    )

    calibration_payload = None
    if calibration_dataloader is not None:
        calibration_payload = collect_model_predictions(
            model=model,
            dataloader=calibration_dataloader,
            device=device,
            decoder_top_k=decoder_top_k,
            memory_bank=memory_bank,
        )

    avg_train_drugs = None
    if train_cardinality_dataloader is not None:
        avg_train_drugs = compute_average_target_cardinality_from_dataloader(train_cardinality_dataloader)

    prediction_resolution = resolve_prediction_control(
        prediction_config=prediction_config,
        checkpoint_payload=checkpoint_payload,
        cli_threshold=cli_threshold,
        cli_prediction_mode=cli_prediction_mode,
        cli_prediction_top_k=cli_prediction_top_k,
        eval_probs=evaluation_payload["probs"],
        eval_targets=evaluation_payload["targets"],
        ddi_matrix=ddi_artifact["matrix"].detach().cpu() if ddi_artifact.get("active", False) else None,
        calibration_probs=None if calibration_payload is None else calibration_payload["probs"],
        calibration_targets=None if calibration_payload is None else calibration_payload["targets"],
        avg_train_drugs=avg_train_drugs,
    )
    summarized = _summarize_core_evaluation_payload(
        payload=evaluation_payload,
        binary_predictions=prediction_resolution["binary_predictions"],
        ddi_artifact=ddi_artifact,
    )
    summarized["prediction_mode"] = prediction_resolution["prediction_mode"]
    summarized["prediction_control"] = prediction_resolution["prediction_control"]
    summarized["threshold"] = prediction_resolution["threshold"]
    summarized["threshold_source"] = prediction_resolution["threshold_source"]
    summarized["threshold_selection"] = prediction_resolution["threshold_selection"]
    return summarized


def main() -> None:
    args = parse_args()
    eval_config = load_yaml_config(args.config)
    validate_core_runtime_config(
        runtime_cfg=dict(eval_config.get("runtime", {})),
        core_cfg=dict(eval_config.get("core", {})),
        context_label="evaluate_core.py",
    )
    project_root = Path(eval_config["_project_root"]).resolve()
    checkpoint_path = _resolve_checkpoint_path(project_root, eval_config, args)
    checkpoint_payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    config_refs = dict(eval_config.get("config_refs", {}))
    train_config = _load_embedded_or_yaml_config(
        explicit_path=args.train_config,
        embedded_payload=checkpoint_payload.get("train_config"),
        fallback_path=resolve_path(project_root, config_refs.get("train", "configs/train.yaml")),
    )
    data_config = _load_embedded_or_yaml_config(
        explicit_path=args.data_config,
        embedded_payload=checkpoint_payload.get("data_config"),
        fallback_path=resolve_path(project_root, config_refs.get("data", "configs/data.yaml")),
    )
    model_config = _load_embedded_or_yaml_config(
        explicit_path=args.model_config,
        embedded_payload=checkpoint_payload.get("model_config"),
        fallback_path=resolve_path(project_root, config_refs.get("model", "configs/model.yaml")),
    )

    resolved_paths = _resolve_eval_paths(
        project_root=project_root,
        eval_config=eval_config,
        train_config=train_config,
        data_config=data_config,
        checkpoint_payload=checkpoint_payload,
        args=args,
    )
    print("Resolved evaluation paths:")
    for key in ("processed_root", "vocab_root", "ddi_matrix_path", "report_dir", "prediction_dir"):
        source_key = f"{key}_source"
        if source_key in resolved_paths:
            print(f"  {key}: {_stringify_path_source(resolved_paths[key], str(resolved_paths[source_key]))}")
        else:
            print(f"  {key}: {resolved_paths[key]}")
    if _compatibility_fallback_used(
        sources=[
            str(resolved_paths.get("processed_root_source", "")),
            str(resolved_paths.get("vocab_root_source", "")),
            str(resolved_paths.get("ddi_matrix_path_source", "")),
        ]
    ):
        print("Compatibility fallback is active: evaluation is using handover_data artifacts instead of canonical data/... paths.")

    runtime_cfg = dict(eval_config.get("runtime", {}))
    evaluation_cfg = dict(eval_config.get("evaluation", {}))
    prediction_cfg = normalize_prediction_config(eval_config.get("prediction", {}))
    core_cfg = dict(eval_config.get("core", train_config.get("core", {})))
    retrieval_enabled = bool(core_cfg.get("use_retrieval", False))

    split = str(args.split or evaluation_cfg.get("split", "test"))
    device = resolve_device(args.device or runtime_cfg.get("device", "cpu"))
    batch_size = int(runtime_cfg.get("batch_size", 32))
    decoder_top_k = _resolve_decoder_top_k(prediction_cfg)

    ddi_artifact = load_ddi_artifact(resolved_paths["ddi_matrix_path"], device="cpu")
    ddi_artifact = normalize_ddi_context(ddi_artifact)
    drug_vocab_size = int(read_json(resolved_paths["vocab_root"] / "drug_vocab.json")["size"])
    if ddi_artifact["matrix"].shape[0] != drug_vocab_size:
        raise ValueError(
            "DDI matrix width must match drug vocabulary size: "
            f"got ddi={int(ddi_artifact['matrix'].shape[0])}, vocab={drug_vocab_size}"
        )

    print(f"Using device: {device}")
    print(f"Evaluating split: {split}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(
        "Evaluation DDI state: "
        f"status={ddi_artifact['status']} "
        f"reason={ddi_artifact['reason']} "
        f"source={ddi_artifact['source']} "
        f"matched_pairs={ddi_artifact['matched_pairs']} "
        f"nonzero_pairs={ddi_artifact['nonzero_pairs']} "
        f"kind={dict(ddi_artifact.get('source_metadata') or {}).get('kind', '')} "
        f"research_grade={dict(ddi_artifact.get('source_metadata') or {}).get('research_grade')} "
        f"purpose={dict(ddi_artifact.get('source_metadata') or {}).get('purpose', '')}"
    )

    train_config = copy.deepcopy(train_config)
    train_config["core"] = copy.deepcopy(core_cfg)
    train_config["_resolved_paths"] = {"processed_root": str(resolved_paths["processed_root"])}

    with tempfile.TemporaryDirectory(prefix="clinrec_eval_runtime_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        runtime_data_config_path = build_runtime_data_config_file(
            data_config=data_config,
            processed_root=resolved_paths["processed_root"],
            vocab_root=resolved_paths["vocab_root"],
            temp_dir=temp_dir,
        )
        _ = load_vocab_bundle(runtime_data_config_path)

        dataloader = build_eval_dataloader(
            split=split,
            runtime_data_config_path=runtime_data_config_path,
            processed_root=resolved_paths["processed_root"],
            drug_vocab_size=drug_vocab_size,
            batch_size=batch_size,
            include_records=retrieval_enabled,
        )
        calibration_dataloader = None
        if prediction_mode_requires_calibration(
            cli_prediction_mode=args.prediction_mode,
            cli_prediction_top_k=args.prediction_top_k,
            cli_threshold=args.threshold,
            prediction_config=prediction_cfg,
        ):
            calibration_split = str(prediction_cfg["calibration"]["split"])
            calibration_dataloader = dataloader
            if calibration_split != split:
                calibration_dataloader = build_eval_dataloader(
                    split=calibration_split,
                    runtime_data_config_path=runtime_data_config_path,
                    processed_root=resolved_paths["processed_root"],
                    drug_vocab_size=drug_vocab_size,
                    batch_size=batch_size,
                    include_records=retrieval_enabled,
                )
        train_cardinality_dataloader = None
        if prediction_mode_requires_train_cardinality(
            cli_prediction_mode=args.prediction_mode,
            cli_prediction_top_k=args.prediction_top_k,
            cli_threshold=args.threshold,
            prediction_config=prediction_cfg,
        ):
            train_cardinality_dataloader = build_eval_dataloader(
                split="train",
                runtime_data_config_path=runtime_data_config_path,
                processed_root=resolved_paths["processed_root"],
                drug_vocab_size=drug_vocab_size,
                batch_size=batch_size,
            )
        retrieval_bank_dataloader = None
        if retrieval_enabled:
            retrieval_bank_dataloader = build_eval_dataloader(
                split="train",
                runtime_data_config_path=runtime_data_config_path,
                processed_root=resolved_paths["processed_root"],
                drug_vocab_size=drug_vocab_size,
                batch_size=batch_size,
                include_records=True,
            )
        model, _ = build_core_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=resolved_paths["vocab_root"],
            ddi_matrix_path=resolved_paths["ddi_matrix_path"],
        )

    model_state_dict = checkpoint_payload.get("model_state_dict")
    if not isinstance(model_state_dict, Mapping):
        raise KeyError(f"Checkpoint at {checkpoint_path} does not contain `model_state_dict`.")
    try:
        model.load_state_dict(model_state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load checkpoint state from {checkpoint_path}: {exc}") from exc

    memory_bank = None
    if retrieval_enabled:
        if retrieval_bank_dataloader is None:
            raise RuntimeError("Retrieval evaluation requires a train memory-bank dataloader.")
        memory_bank = build_core_memory_bank(
            model=model,
            dataloader=retrieval_bank_dataloader,
            device=device,
            split="train",
        )
        print(
            "Evaluation retrieval memory bank: "
            f"split={memory_bank.split} "
            f"states={len(memory_bank)} "
            f"top_k={model.retrieval_top_k} "
            f"cross_split_policy={model.cross_split_policy or ('allow_all' if model.allow_cross_split else 'same_split')} "
            f"leakage_safe={model.retrieval_leakage_safe}"
        )

    evaluation_result = run_core_evaluation(
        model=model,
        dataloader=dataloader,
        device=device,
        ddi_artifact=ddi_artifact,
        decoder_top_k=decoder_top_k,
        prediction_config=prediction_cfg,
        checkpoint_payload=checkpoint_payload,
        cli_threshold=args.threshold,
        cli_prediction_mode=args.prediction_mode,
        cli_prediction_top_k=args.prediction_top_k,
        calibration_dataloader=calibration_dataloader,
        train_cardinality_dataloader=train_cardinality_dataloader,
        memory_bank=memory_bank,
    )
    print(f"Prediction mode: {evaluation_result['prediction_mode']}")
    if evaluation_result["threshold"] is not None:
        print(
            f"Using threshold: {evaluation_result['threshold']} "
            f"[{evaluation_result['threshold_source']}]"
        )
    else:
        prediction_control = dict(evaluation_result["prediction_control"])
        print(
            f"Using top-k: {prediction_control['top_k']} "
            f"[{prediction_control['top_k_source']}]"
        )
    runtime_truth = build_core_runtime_truth(
        fusion_strategy=str(model.fusion_module.strategy),
        ddi_context=ddi_artifact,
        retrieval_active=retrieval_enabled,
        retrieval_status="active" if retrieval_enabled else "disabled",
        retrieval_top_k=model.retrieval_top_k if retrieval_enabled else None,
        retrieval_scoring_mode=model.retrieval_scoring_mode if retrieval_enabled else None,
        retrieval_cross_split_policy=(
            model.cross_split_policy or ("allow_all" if model.allow_cross_split else "same_split")
        )
        if retrieval_enabled
        else None,
        retrieval_leakage_safe=model.retrieval_leakage_safe if retrieval_enabled else None,
    )
    print(
        "Evaluation runtime truth: "
        f"pipeline_level={runtime_truth['pipeline_level']} "
        f"history_active={runtime_truth['history_active']} "
        f"retrieval_active={runtime_truth['retrieval_active']} "
        f"fusion_strategy={runtime_truth['fusion_strategy']} "
        f"ddi_type={runtime_truth['ddi_type']} "
        f"ddi_research_grade={runtime_truth['ddi_research_grade']}"
    )
    training_ddi_context = checkpoint_payload.get("ddi_context")
    if isinstance(training_ddi_context, Mapping):
        training_ddi_context = normalize_ddi_context(training_ddi_context)
    initialization_context = normalize_initialization_context(checkpoint_payload)

    report: dict[str, Any] = {
        **runtime_truth,
        **initialization_context,
        "split": split,
        "num_samples": int(evaluation_result["targets"].shape[0]),
        "prediction_mode": evaluation_result["prediction_mode"],
        "prediction_control": evaluation_result["prediction_control"],
        "threshold": evaluation_result["threshold"],
        "threshold_source": evaluation_result["threshold_source"],
        "threshold_selection": evaluation_result["threshold_selection"],
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "metrics": evaluation_result["metrics"],
        "ddi_summary": evaluation_result["ddi_summary"],
        "ddi_context": {
            "training": training_ddi_context,
            "evaluation": {key: value for key, value in ddi_artifact.items() if key != "matrix"},
        },
        "prediction_summary": evaluation_result["prediction_summary"],
        "artifacts": {},
    }

    save_reports = bool(evaluation_cfg.get("save_reports", True))
    save_predictions = bool(evaluation_cfg.get("save_predictions", True))
    report_stem = f"evaluate_core_{split}"

    if save_reports:
        json_path = write_json(resolved_paths["report_dir"] / f"{report_stem}.json", report)
        flat_report: dict[str, Any] = {}
        _flatten_report("", report, flat_report)
        csv_path = _write_plain_csv(resolved_paths["report_dir"] / f"{report_stem}.csv", [flat_report])
        report["artifacts"]["json"] = str(json_path)
        report["artifacts"]["csv"] = str(csv_path)

    if save_predictions:
        prediction_csv_path = _write_plain_csv(
            resolved_paths["prediction_dir"] / f"{report_stem}_predictions.csv",
            evaluation_result["prediction_rows"],
        )
        report["artifacts"]["predictions_csv"] = str(prediction_csv_path)
        if save_reports:
            write_json(resolved_paths["report_dir"] / f"{report_stem}.json", report)

    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
