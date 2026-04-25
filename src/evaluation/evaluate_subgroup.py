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
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _stringify_path_source,
        _write_plain_csv,
        build_eval_dataloader,
        build_runtime_data_config_file,
        parse_args,
    )
    from metrics import compute_core_metrics_from_binary_predictions  # type: ignore[import-not-found]
    from prediction_control import (  # type: ignore[import-not-found]
        collect_model_predictions,
        compute_average_target_cardinality_from_dataloader,
        normalize_prediction_config,
        prediction_mode_requires_calibration,
        prediction_mode_requires_train_cardinality,
        resolve_prediction_control,
    )
else:
    from .evaluate_core import (
        _build_ddi_summary,
        _compatibility_fallback_used,
        _load_embedded_or_yaml_config,
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _stringify_path_source,
        _write_plain_csv,
        build_eval_dataloader,
        build_runtime_data_config_file,
        parse_args,
    )
    from .metrics import compute_core_metrics_from_binary_predictions
    from .prediction_control import (
        collect_model_predictions,
        compute_average_target_cardinality_from_dataloader,
        normalize_prediction_config,
        prediction_mode_requires_calibration,
        prediction_mode_requires_train_cardinality,
        resolve_prediction_control,
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


def _trajectory_length_bucket(num_steps: int) -> str:
    if num_steps <= 1:
        return "short"
    if num_steps <= 3:
        return "medium"
    return "long"


def _medication_burden_bucket(count: int) -> str:
    if count <= 1:
        return "low"
    if count <= 4:
        return "medium"
    return "high"


def _collect_subgroup_payload(
    *,
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    decoder_top_k: int | None,
) -> dict[str, torch.Tensor]:
    payload = collect_model_predictions(
        model=model,
        dataloader=dataloader,
        device=device,
        decoder_top_k=decoder_top_k,
        include_num_steps=True,
    )
    return {
        "probs": payload["probs"],
        "targets": payload["targets"],
        "num_steps": payload["num_steps"],
    }


def _build_subgroup_rows(
    *,
    family: str,
    labels: list[str],
    subgroup_assignments: list[str],
    targets: torch.Tensor,
    probs: torch.Tensor,
    predictions: torch.Tensor,
    pred_counts: torch.Tensor,
    true_counts: torch.Tensor,
    ddi_matrix: torch.Tensor,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    ddi_matrix_cpu = ddi_matrix.detach().cpu()
    for subgroup_label in labels:
        indices = [index for index, label in enumerate(subgroup_assignments) if label == subgroup_label]
        if not indices:
            continue
        subset_index = torch.as_tensor(indices, dtype=torch.long)
        subset_targets = targets.index_select(0, subset_index)
        subset_probs = probs.index_select(0, subset_index)
        subset_predictions = predictions.index_select(0, subset_index)
        subset_pred_counts = pred_counts.index_select(0, subset_index).to(dtype=torch.float32)
        subset_true_counts = true_counts.index_select(0, subset_index).to(dtype=torch.float32)

        subset_metrics = compute_core_metrics_from_binary_predictions(
            subset_targets,
            subset_probs,
            subset_predictions,
            ddi_matrix=ddi_matrix_cpu,
        )
        num_samples = int(subset_targets.shape[0])
        patients_with_ddi_ratio = (
            0.0
            if num_samples <= 0
            else float(subset_metrics["patients_with_ddi"]) / float(num_samples)
        )
        rows.append(
            {
                "family": family,
                "subgroup": subgroup_label,
                "num_samples": num_samples,
                "jaccard": float(subset_metrics["jaccard"]),
                "f1": float(subset_metrics["f1"]),
                "prauc": float(subset_metrics["prauc"]),
                "ddi_rate": float(subset_metrics["ddi_rate"]),
                "avg_predicted_drugs": float(subset_pred_counts.mean().item()),
                "avg_true_drugs": float(subset_true_counts.mean().item()),
                "patients_with_ddi_ratio": patients_with_ddi_ratio,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    eval_config = load_yaml_config(args.config)
    validate_core_runtime_config(
        runtime_cfg=dict(eval_config.get("runtime", {})),
        core_cfg=dict(eval_config.get("core", {})),
        context_label="evaluate_subgroup.py",
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
    print("Resolved subgroup evaluation paths:")
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
    ddi_matrix = ddi_artifact["matrix"]
    drug_vocab_size = int(read_json(resolved_paths["vocab_root"] / "drug_vocab.json")["size"])
    if ddi_matrix.shape[0] != drug_vocab_size:
        raise ValueError(
            "DDI matrix width must match drug vocabulary size: "
            f"got ddi={int(ddi_matrix.shape[0])}, vocab={drug_vocab_size}"
        )

    train_config = dict(train_config)
    train_config["_resolved_paths"] = {"processed_root": str(resolved_paths["processed_root"])}

    print(f"Using device: {device}")
    print(f"Evaluating subgroup split: {split}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print(
        "Subgroup DDI state: "
        f"status={ddi_artifact['status']} "
        f"source={ddi_artifact['source']} "
        f"kind={dict(ddi_artifact.get('source_metadata') or {}).get('kind', '')} "
        f"research_grade={dict(ddi_artifact.get('source_metadata') or {}).get('research_grade')}"
    )

    with tempfile.TemporaryDirectory(prefix="clinrec_subgroup_eval_") as temp_dir_name:
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
    runtime_truth = build_core_runtime_truth(
        fusion_strategy=str(model.fusion_module.strategy),
        ddi_context=ddi_artifact,
    )
    print(
        "Subgroup runtime truth: "
        f"pipeline_level={runtime_truth['pipeline_level']} "
        f"history_active={runtime_truth['history_active']} "
        f"retrieval_active={runtime_truth['retrieval_active']} "
        f"fusion_strategy={runtime_truth['fusion_strategy']} "
        f"ddi_type={runtime_truth['ddi_type']} "
        f"ddi_research_grade={runtime_truth['ddi_research_grade']}"
    )

    subgroup_payload = _collect_subgroup_payload(
        model=model,
        dataloader=dataloader,
        device=device,
        decoder_top_k=decoder_top_k,
    )
    calibration_payload = None
    if calibration_dataloader is not None:
        calibration_payload = _collect_subgroup_payload(
            model=model,
            dataloader=calibration_dataloader,
            device=device,
            decoder_top_k=decoder_top_k,
        )
    avg_train_drugs = None
    if train_cardinality_dataloader is not None:
        avg_train_drugs = compute_average_target_cardinality_from_dataloader(train_cardinality_dataloader)
    prediction_resolution = resolve_prediction_control(
        prediction_config=prediction_cfg,
        checkpoint_payload=checkpoint_payload,
        cli_threshold=args.threshold,
        cli_prediction_mode=args.prediction_mode,
        cli_prediction_top_k=args.prediction_top_k,
        eval_probs=subgroup_payload["probs"],
        eval_targets=subgroup_payload["targets"],
        ddi_matrix=ddi_matrix if ddi_artifact.get("active", False) else None,
        calibration_probs=None if calibration_payload is None else calibration_payload["probs"],
        calibration_targets=None if calibration_payload is None else calibration_payload["targets"],
        avg_train_drugs=avg_train_drugs,
    )
    subgroup_payload["predictions"] = prediction_resolution["binary_predictions"].cpu()
    subgroup_payload["pred_counts"] = subgroup_payload["predictions"].sum(dim=1).to(dtype=torch.long)
    subgroup_payload["true_counts"] = subgroup_payload["targets"].sum(dim=1).to(dtype=torch.long)
    print(f"Prediction mode: {prediction_resolution['prediction_mode']}")
    if prediction_resolution["threshold"] is not None:
        print(
            f"Using threshold: {prediction_resolution['threshold']} "
            f"[{prediction_resolution['threshold_source']}]"
        )
    else:
        prediction_control = dict(prediction_resolution["prediction_control"])
        print(
            f"Using top-k: {prediction_control['top_k']} "
            f"[{prediction_control['top_k_source']}]"
        )

    num_steps = subgroup_payload["num_steps"]
    true_counts = subgroup_payload["true_counts"]
    pred_counts = subgroup_payload["pred_counts"]

    family_rows = {
        "trajectory_length": _build_subgroup_rows(
            family="trajectory_length",
            labels=["short", "medium", "long"],
            subgroup_assignments=[_trajectory_length_bucket(int(value)) for value in num_steps.tolist()],
            targets=subgroup_payload["targets"],
            probs=subgroup_payload["probs"],
            predictions=subgroup_payload["predictions"],
            pred_counts=pred_counts,
            true_counts=true_counts,
            ddi_matrix=ddi_matrix,
        ),
        "true_medication_burden": _build_subgroup_rows(
            family="true_medication_burden",
            labels=["low", "medium", "high"],
            subgroup_assignments=[_medication_burden_bucket(int(value)) for value in true_counts.tolist()],
            targets=subgroup_payload["targets"],
            probs=subgroup_payload["probs"],
            predictions=subgroup_payload["predictions"],
            pred_counts=pred_counts,
            true_counts=true_counts,
            ddi_matrix=ddi_matrix,
        ),
        "predicted_medication_burden": _build_subgroup_rows(
            family="predicted_medication_burden",
            labels=["low", "medium", "high"],
            subgroup_assignments=[_medication_burden_bucket(int(value)) for value in pred_counts.tolist()],
            targets=subgroup_payload["targets"],
            probs=subgroup_payload["probs"],
            predictions=subgroup_payload["predictions"],
            pred_counts=pred_counts,
            true_counts=true_counts,
            ddi_matrix=ddi_matrix,
        ),
    }
    csv_rows = [row for rows in family_rows.values() for row in rows]
    initialization_context = normalize_initialization_context(checkpoint_payload)

    subgroup_report: dict[str, Any] = {
        **runtime_truth,
        **initialization_context,
        "report_type": "subgroup_core",
        "split": split,
        "num_samples": int(subgroup_payload["targets"].shape[0]),
        "prediction_mode": prediction_resolution["prediction_mode"],
        "prediction_control": prediction_resolution["prediction_control"],
        "threshold": prediction_resolution["threshold"],
        "threshold_source": prediction_resolution["threshold_source"],
        "threshold_selection": prediction_resolution["threshold_selection"],
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "ddi_summary": _build_ddi_summary(ddi_artifact),
        "subgroup_rules": {
            "trajectory_length": {
                "short": "num_steps == 1",
                "medium": "2 <= num_steps <= 3",
                "long": "num_steps >= 4",
            },
            "true_medication_burden": {
                "low": "true_count <= 1",
                "medium": "2 <= true_count <= 4",
                "high": "true_count >= 5",
            },
            "predicted_medication_burden": {
                "low": "pred_count <= 1",
                "medium": "2 <= pred_count <= 4",
                "high": "pred_count >= 5",
            },
        },
        "families": family_rows,
        "artifacts": {},
    }

    report_stem = f"evaluate_subgroup_{split}"
    json_path = write_json(resolved_paths["report_dir"] / f"{report_stem}.json", subgroup_report)
    csv_path = _write_plain_csv(
        resolved_paths["report_dir"] / f"{report_stem}.csv",
        csv_rows,
    )

    subgroup_report["artifacts"]["json"] = str(json_path)
    subgroup_report["artifacts"]["csv"] = str(csv_path)
    write_json(resolved_paths["report_dir"] / f"{report_stem}.json", subgroup_report)

    print(json.dumps(subgroup_report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
