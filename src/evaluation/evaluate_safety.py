from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from evaluate_core import (  # type: ignore[import-not-found]
        _build_ddi_summary,
        _compatibility_fallback_used,
        _load_embedded_or_yaml_config,
        _flatten_report,
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _stringify_path_source,
        _write_plain_csv,
        build_eval_dataloader,
        build_runtime_data_config_file,
        parse_args,
        run_core_evaluation,
    )
else:
    from .evaluate_core import (
        _build_ddi_summary,
        _compatibility_fallback_used,
        _load_embedded_or_yaml_config,
        _flatten_report,
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _stringify_path_source,
        _write_plain_csv,
        build_eval_dataloader,
        build_runtime_data_config_file,
        parse_args,
        run_core_evaluation,
    )
    from .prediction_control import (
        normalize_prediction_config,
        prediction_mode_requires_calibration,
        prediction_mode_requires_train_cardinality,
    )

from src.data.build_vocab import load_vocab_bundle
from src.models.ddi_regularization import load_ddi_artifact
from src.training.train_core import build_core_model, resolve_device, validate_core_runtime_config
from src.utils.io import load_yaml_config, read_json, resolve_path, write_json
from src.utils.runtime_truth import (
    build_core_runtime_truth,
    normalize_ddi_context,
    normalize_initialization_context,
)

if __package__ in {None, ""}:
    from prediction_control import (  # type: ignore[import-not-found]
        normalize_prediction_config,
        prediction_mode_requires_calibration,
        prediction_mode_requires_train_cardinality,
    )


POLYPHARMACY_THRESHOLD = 5
HIGH_POLYPHARMACY_THRESHOLD = 10


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def build_safety_warnings(*, ddi_rate: float, avg_predicted_drugs: float) -> list[str]:
    warnings: list[str] = []
    if ddi_rate >= 0.05:
        warnings.append("high_ddi_rate")
    elif ddi_rate >= 0.01:
        warnings.append("moderate_ddi_rate")
    elif ddi_rate > 0.0:
        warnings.append("nonzero_ddi_rate")

    if avg_predicted_drugs >= HIGH_POLYPHARMACY_THRESHOLD:
        warnings.append("high_polypharmacy_burden")
    elif avg_predicted_drugs >= POLYPHARMACY_THRESHOLD:
        warnings.append("polypharmacy_burden")
    return warnings


def build_patient_safety_rows(prediction_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    patient_rows: list[dict[str, Any]] = []
    polypharmacy_count = 0.0
    high_polypharmacy_count = 0.0
    patient_ddi_count = 0.0

    for row in prediction_rows:
        pred_count = int(row["pred_count"])
        has_ddi = bool(row["has_ddi"])
        is_polypharmacy = pred_count >= POLYPHARMACY_THRESHOLD
        is_high_polypharmacy = pred_count >= HIGH_POLYPHARMACY_THRESHOLD

        patient_rows.append(
            {
                "subject_id": row["subject_id"],
                "hadm_id": row["hadm_id"],
                "stay_id": row["stay_id"],
                "true_count": row["true_count"],
                "pred_count": pred_count,
                "sample_jaccard": row["sample_jaccard"],
                "sample_f1": row["sample_f1"],
                "has_ddi": has_ddi,
                "polypharmacy": is_polypharmacy,
                "high_polypharmacy": is_high_polypharmacy,
                "predicted_drug_indices": row["predicted_drug_indices"],
            }
        )

        polypharmacy_count += float(is_polypharmacy)
        high_polypharmacy_count += float(is_high_polypharmacy)
        patient_ddi_count += float(has_ddi)

    num_samples = float(len(patient_rows))
    if num_samples <= 0.0:
        raise ValueError("No patient-level prediction rows found for safety evaluation")

    return patient_rows, {
        "polypharmacy_rate": polypharmacy_count / num_samples,
        "high_polypharmacy_rate": high_polypharmacy_count / num_samples,
        "patients_with_ddi_ratio": patient_ddi_count / num_samples,
    }


def main() -> None:
    args = parse_args()
    eval_config = load_yaml_config(args.config)
    validate_core_runtime_config(
        runtime_cfg=dict(eval_config.get("runtime", {})),
        core_cfg=dict(eval_config.get("core", {})),
        context_label="evaluate_safety.py",
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
    print("Resolved safety evaluation paths:")
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

    split = str(args.split or evaluation_cfg.get("split", "test"))
    device = resolve_device(args.device or runtime_cfg.get("device", "cpu"))
    batch_size = int(runtime_cfg.get("batch_size", 32))
    decoder_top_k = int(prediction_cfg.get("decoder_top_k", prediction_cfg.get("top_k", 10)))

    ddi_artifact = normalize_ddi_context(load_ddi_artifact(resolved_paths["ddi_matrix_path"], device="cpu"))
    drug_vocab_size = int(read_json(resolved_paths["vocab_root"] / "drug_vocab.json")["size"])
    if ddi_artifact["matrix"].shape[0] != drug_vocab_size:
        raise ValueError(
            "DDI matrix width must match drug vocabulary size: "
            f"got ddi={int(ddi_artifact['matrix'].shape[0])}, vocab={drug_vocab_size}"
        )

    train_config = dict(train_config)
    train_config["_resolved_paths"] = {"processed_root": str(resolved_paths["processed_root"])}

    print(f"Using device: {device}")
    print(f"Evaluating safety split: {split}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(
        "Safety DDI state: "
        f"status={ddi_artifact['status']} "
        f"source={ddi_artifact['source']} "
        f"kind={dict(ddi_artifact.get('source_metadata') or {}).get('kind', '')} "
        f"research_grade={dict(ddi_artifact.get('source_metadata') or {}).get('research_grade')}"
    )

    with tempfile.TemporaryDirectory(prefix="clinrec_safety_eval_") as temp_dir_name:
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
        model, _ = build_core_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=resolved_paths["vocab_root"],
            ddi_matrix_path=resolved_paths["ddi_matrix_path"],
        )

    model_state_dict = checkpoint_payload.get("model_state_dict")
    if not isinstance(model_state_dict, Mapping):
        raise KeyError("Checkpoint does not contain `model_state_dict`.")
    model.load_state_dict(model_state_dict, strict=True)

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
    )
    print(
        "Safety runtime truth: "
        f"pipeline_level={runtime_truth['pipeline_level']} "
        f"history_active={runtime_truth['history_active']} "
        f"retrieval_active={runtime_truth['retrieval_active']} "
        f"fusion_strategy={runtime_truth['fusion_strategy']} "
        f"ddi_type={runtime_truth['ddi_type']} "
        f"ddi_research_grade={runtime_truth['ddi_research_grade']}"
    )
    patient_rows, rate_summary = build_patient_safety_rows(evaluation_result["prediction_rows"])
    training_ddi_context = checkpoint_payload.get("ddi_context")
    if isinstance(training_ddi_context, Mapping):
        training_ddi_context = normalize_ddi_context(training_ddi_context)
    initialization_context = normalize_initialization_context(checkpoint_payload)

    avg_predicted_drugs = float(evaluation_result["prediction_summary"]["avg_predicted_drugs"])
    safety_report: dict[str, Any] = {
        **runtime_truth,
        **initialization_context,
        "report_type": "safety_smoke",
        "split": split,
        "num_samples": int(evaluation_result["targets"].shape[0]),
        "prediction_mode": evaluation_result["prediction_mode"],
        "prediction_control": evaluation_result["prediction_control"],
        "threshold": evaluation_result["threshold"],
        "threshold_source": evaluation_result["threshold_source"],
        "threshold_selection": evaluation_result["threshold_selection"],
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "ddi_summary": _build_ddi_summary(
            ddi_artifact,
            metrics={
                "ddi_rate": evaluation_result["ddi_summary"]["ddi_rate"],
                "total_predicted_pairs": evaluation_result["ddi_summary"]["total_predicted_pairs"],
                "total_interacting_pairs": evaluation_result["ddi_summary"]["total_interacting_pairs"],
                "patients_with_ddi": evaluation_result["ddi_summary"]["patients_with_ddi"],
                "num_samples": evaluation_result["ddi_summary"]["num_samples"],
            },
        ),
        "ddi_context": {
            "training": training_ddi_context,
            "evaluation": {key: value for key, value in ddi_artifact.items() if key != "matrix"},
        },
        "safety_metrics": {
            "ddi_rate": _safe_float(evaluation_result["ddi_summary"]["ddi_rate"]),
            "patients_with_ddi": _safe_float(evaluation_result["ddi_summary"]["patients_with_ddi"]),
            "patients_with_ddi_ratio": float(rate_summary["patients_with_ddi_ratio"]),
            "total_interacting_pairs": _safe_float(evaluation_result["ddi_summary"]["total_interacting_pairs"]),
            "total_predicted_pairs": _safe_float(evaluation_result["ddi_summary"]["total_predicted_pairs"]),
            "avg_predicted_drugs": avg_predicted_drugs,
            "avg_true_drugs": float(evaluation_result["prediction_summary"]["avg_true_drugs"]),
            "polypharmacy_rate": float(rate_summary["polypharmacy_rate"]),
            "high_polypharmacy_rate": float(rate_summary["high_polypharmacy_rate"]),
        },
        "safety_warnings": build_safety_warnings(
            ddi_rate=float(evaluation_result["ddi_summary"]["ddi_rate"] or 0.0),
            avg_predicted_drugs=avg_predicted_drugs,
        ),
        "artifacts": {},
    }

    report_stem = f"evaluate_safety_{split}"
    json_path = write_json(resolved_paths["report_dir"] / f"{report_stem}.json", safety_report)
    flat_report: dict[str, Any] = {}
    _flatten_report("", safety_report, flat_report)
    csv_path = _write_plain_csv(resolved_paths["report_dir"] / f"{report_stem}.csv", [flat_report])
    patient_csv_path = _write_plain_csv(
        resolved_paths["prediction_dir"] / f"{report_stem}_patient_safety.csv",
        patient_rows,
    )

    safety_report["artifacts"]["json"] = str(json_path)
    safety_report["artifacts"]["csv"] = str(csv_path)
    safety_report["artifacts"]["patient_safety_csv"] = str(patient_csv_path)
    write_json(resolved_paths["report_dir"] / f"{report_stem}.json", safety_report)

    print(json.dumps(safety_report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
