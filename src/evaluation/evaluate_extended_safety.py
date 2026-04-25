from __future__ import annotations

import copy
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
        _compatibility_fallback_used,
        _flatten_report,
        _load_embedded_or_yaml_config,
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _stringify_path_source,
        _write_plain_csv,
        parse_args,
    )
    from evaluate_extended_core import (  # type: ignore[import-not-found]
        build_extended_eval_dataloader,
        run_extended_evaluation,
    )
    from thresholding import resolve_effective_threshold  # type: ignore[import-not-found]
else:
    from .evaluate_core import (
        _compatibility_fallback_used,
        _flatten_report,
        _load_embedded_or_yaml_config,
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _stringify_path_source,
        _write_plain_csv,
        parse_args,
    )
    from .evaluate_extended_core import build_extended_eval_dataloader, run_extended_evaluation
    from .thresholding import resolve_effective_threshold

from src.data.build_vocab import load_vocab_bundle
from src.models.ddi_regularization import load_ddi_artifact
from src.training.train_core import build_runtime_data_config_file, resolve_device
from src.training.train_extended import build_extended_model, build_memory_bank_from_dataloader
from src.utils.io import load_yaml_config, read_json, resolve_path, write_json
from src.utils.runtime_truth import normalize_ddi_context, normalize_initialization_context


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
    print("Resolved extended safety evaluation paths:")
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
    prediction_cfg = dict(eval_config.get("prediction", {}))
    extended_cfg = dict(train_config.get("extended", {}))

    split = str(args.split or evaluation_cfg.get("split", "test"))
    threshold, threshold_source, threshold_selection = resolve_effective_threshold(
        cli_threshold=args.threshold,
        checkpoint_payload=checkpoint_payload,
        config_threshold=float(prediction_cfg.get("threshold", 0.5)),
    )
    device = resolve_device(args.device or runtime_cfg.get("device", "cpu"))
    batch_size = int(runtime_cfg.get("batch_size", 32))
    decoder_top_k = int(prediction_cfg.get("top_k", 10))

    ddi_artifact = normalize_ddi_context(load_ddi_artifact(resolved_paths["ddi_matrix_path"], device="cpu"))
    drug_vocab_size = int(read_json(resolved_paths["vocab_root"] / "drug_vocab.json")["size"])
    if ddi_artifact["matrix"].shape[0] != drug_vocab_size:
        raise ValueError(
            "DDI matrix width must match drug vocabulary size: "
            f"got ddi={int(ddi_artifact['matrix'].shape[0])}, vocab={drug_vocab_size}"
        )

    train_config = copy.deepcopy(train_config)
    train_config["_resolved_paths"] = {"processed_root": str(resolved_paths["processed_root"])}

    with tempfile.TemporaryDirectory(prefix="clinrec_safety_extended_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        runtime_data_config_path = build_runtime_data_config_file(
            data_config=data_config,
            processed_root=resolved_paths["processed_root"],
            vocab_root=resolved_paths["vocab_root"],
            temp_dir=temp_dir,
        )
        _ = load_vocab_bundle(runtime_data_config_path)

        dataloader = build_extended_eval_dataloader(
            split=split,
            runtime_data_config_path=runtime_data_config_path,
            processed_root=resolved_paths["processed_root"],
            drug_vocab_size=drug_vocab_size,
            batch_size=batch_size,
        )
        train_bank_loader = build_extended_eval_dataloader(
            split="train",
            runtime_data_config_path=runtime_data_config_path,
            processed_root=resolved_paths["processed_root"],
            drug_vocab_size=drug_vocab_size,
            batch_size=batch_size,
        )
        model, _ = build_extended_model(
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

    retrieval_active = bool(checkpoint_payload.get("retrieval_active", extended_cfg.get("use_retrieval", True)))
    memory_bank = None
    if retrieval_active:
        memory_bank = build_memory_bank_from_dataloader(
            model=model,
            dataloader=train_bank_loader,
            device=device,
            split="train",
        )

    evaluation_result = run_extended_evaluation(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=threshold,
        ddi_artifact=ddi_artifact,
        decoder_top_k=decoder_top_k,
        memory_bank=memory_bank,
    )
    patient_rows, patient_summary = build_patient_safety_rows(evaluation_result["prediction_rows"])

    ddi_rate = _safe_float(evaluation_result["ddi_summary"].get("ddi_rate"))
    avg_predicted_drugs = float(evaluation_result["prediction_summary"]["avg_predicted_drugs"])
    warnings = build_safety_warnings(
        ddi_rate=0.0 if ddi_rate is None else ddi_rate,
        avg_predicted_drugs=avg_predicted_drugs,
    )
    runtime_truth = dict(checkpoint_payload.get("runtime_truth") or getattr(model, "runtime_truth", {}) or {})
    runtime_truth["retrieval_active"] = retrieval_active
    runtime_truth["retrieval_bank_policy"] = "train_only"
    initialization_context = normalize_initialization_context(checkpoint_payload)

    report = {
        **runtime_truth,
        **initialization_context,
        "split": split,
        "num_samples": len(patient_rows),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_selection": threshold_selection,
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "metrics": evaluation_result["metrics"],
        "prediction_summary": evaluation_result["prediction_summary"],
        "ddi_summary": evaluation_result["ddi_summary"],
        "patient_summary": patient_summary,
        "warnings": warnings,
        "artifacts": {},
    }

    save_reports = bool(evaluation_cfg.get("save_reports", True))
    save_predictions = bool(evaluation_cfg.get("save_predictions", True))
    report_stem = f"evaluate_extended_safety_{split}"

    if save_reports:
        json_path = write_json(resolved_paths["report_dir"] / f"{report_stem}.json", report)
        flat_report: dict[str, Any] = {}
        _flatten_report("", report, flat_report)
        csv_path = _write_plain_csv(resolved_paths["report_dir"] / f"{report_stem}.csv", [flat_report])
        report["artifacts"]["json"] = str(json_path)
        report["artifacts"]["csv"] = str(csv_path)

    if save_predictions:
        prediction_csv_path = _write_plain_csv(
            resolved_paths["prediction_dir"] / f"{report_stem}_patients.csv",
            patient_rows,
        )
        report["artifacts"]["patient_predictions_csv"] = str(prediction_csv_path)
        if save_reports:
            write_json(resolved_paths["report_dir"] / f"{report_stem}.json", report)

    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
