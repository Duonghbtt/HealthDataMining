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
        _load_embedded_or_yaml_config,
        _flatten_report,
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _write_plain_csv,
        build_eval_dataloader,
        build_runtime_data_config_file,
        parse_args,
        run_core_evaluation,
    )
else:
    from .evaluate_core import (
        _load_embedded_or_yaml_config,
        _flatten_report,
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _write_plain_csv,
        build_eval_dataloader,
        build_runtime_data_config_file,
        parse_args,
        run_core_evaluation,
    )

from src.data.build_vocab import load_vocab_bundle
from src.models.ddi_regularization import load_ddi_matrix
from src.training.train_core import build_core_model, resolve_device
from src.utils.io import load_yaml_config, read_json, resolve_path, write_json


POLYPHARMACY_THRESHOLD = 5
HIGH_POLYPHARMACY_THRESHOLD = 10


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
    print("Resolved safety evaluation paths:")
    for key, value in resolved_paths.items():
        print(f"  {key}: {value}")

    runtime_cfg = dict(eval_config.get("runtime", {}))
    evaluation_cfg = dict(eval_config.get("evaluation", {}))
    prediction_cfg = dict(eval_config.get("prediction", {}))

    split = str(args.split or evaluation_cfg.get("split", "test"))
    threshold = float(args.threshold if args.threshold is not None else prediction_cfg.get("threshold", 0.5))
    device = resolve_device(args.device or runtime_cfg.get("device", "cpu"))
    batch_size = int(runtime_cfg.get("batch_size", 32))
    decoder_top_k = int(prediction_cfg.get("top_k", 10))

    ddi_matrix = load_ddi_matrix(resolved_paths["ddi_matrix_path"], device="cpu")
    drug_vocab_size = int(read_json(resolved_paths["vocab_root"] / "drug_vocab.json")["size"])
    if ddi_matrix.shape[0] != drug_vocab_size:
        raise ValueError(
            "DDI matrix width must match drug vocabulary size: "
            f"got ddi={int(ddi_matrix.shape[0])}, vocab={drug_vocab_size}"
        )

    train_config = dict(train_config)
    train_config["_resolved_paths"] = {"processed_root": str(resolved_paths["processed_root"])}

    print(f"Using device: {device}")
    print(f"Evaluating safety split: {split}")
    print(f"Using threshold: {threshold}")
    print(f"Loading checkpoint: {checkpoint_path}")

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
        threshold=threshold,
        ddi_matrix=ddi_matrix,
        decoder_top_k=decoder_top_k,
    )
    patient_rows, rate_summary = build_patient_safety_rows(evaluation_result["prediction_rows"])

    avg_predicted_drugs = float(evaluation_result["prediction_summary"]["avg_predicted_drugs"])
    safety_report: dict[str, Any] = {
        "split": split,
        "num_samples": int(evaluation_result["targets"].shape[0]),
        "threshold": threshold,
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "safety_metrics": {
            "ddi_rate": float(evaluation_result["ddi_summary"]["ddi_rate"]),
            "patients_with_ddi": float(evaluation_result["ddi_summary"]["patients_with_ddi"]),
            "patients_with_ddi_ratio": float(rate_summary["patients_with_ddi_ratio"]),
            "total_interacting_pairs": float(evaluation_result["ddi_summary"]["total_interacting_pairs"]),
            "total_predicted_pairs": float(evaluation_result["ddi_summary"]["total_predicted_pairs"]),
            "avg_predicted_drugs": avg_predicted_drugs,
            "avg_true_drugs": float(evaluation_result["prediction_summary"]["avg_true_drugs"]),
            "polypharmacy_rate": float(rate_summary["polypharmacy_rate"]),
            "high_polypharmacy_rate": float(rate_summary["high_polypharmacy_rate"]),
        },
        "safety_warnings": build_safety_warnings(
            ddi_rate=float(evaluation_result["ddi_summary"]["ddi_rate"]),
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
