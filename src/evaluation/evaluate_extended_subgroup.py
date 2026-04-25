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
        _build_ddi_summary,
        _compatibility_fallback_used,
        _load_embedded_or_yaml_config,
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _stringify_path_source,
        _write_plain_csv,
        parse_args,
    )
    from evaluate_extended_core import (  # type: ignore[import-not-found]
        build_extended_eval_dataloader,
        collect_extended_prediction_payload,
    )
    from metrics import binarize_predictions, compute_core_metrics  # type: ignore[import-not-found]
    from thresholding import resolve_effective_threshold  # type: ignore[import-not-found]
else:
    from .evaluate_core import (
        _build_ddi_summary,
        _compatibility_fallback_used,
        _load_embedded_or_yaml_config,
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _stringify_path_source,
        _write_plain_csv,
        parse_args,
    )
    from .evaluate_extended_core import build_extended_eval_dataloader, collect_extended_prediction_payload
    from .metrics import binarize_predictions, compute_core_metrics
    from .thresholding import resolve_effective_threshold

from src.data.build_vocab import load_vocab_bundle
from src.models.ddi_regularization import load_ddi_artifact
from src.training.train_core import build_runtime_data_config_file, resolve_device
from src.training.train_extended import build_extended_model, build_memory_bank_from_dataloader
from src.utils.io import load_yaml_config, read_json, resolve_path, write_json
from src.utils.runtime_truth import normalize_ddi_context, normalize_initialization_context


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


def _build_subgroup_rows(
    *,
    family: str,
    labels: list[str],
    subgroup_assignments: list[str],
    targets: torch.Tensor,
    probs: torch.Tensor,
    pred_counts: torch.Tensor,
    true_counts: torch.Tensor,
    threshold: float,
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
        subset_pred_counts = pred_counts.index_select(0, subset_index).to(dtype=torch.float32)
        subset_true_counts = true_counts.index_select(0, subset_index).to(dtype=torch.float32)

        subset_metrics = compute_core_metrics(
            subset_targets,
            subset_probs,
            threshold=threshold,
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
    print("Resolved extended subgroup evaluation paths:")
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
    ddi_matrix = ddi_artifact["matrix"]
    drug_vocab_size = int(read_json(resolved_paths["vocab_root"] / "drug_vocab.json")["size"])
    if ddi_matrix.shape[0] != drug_vocab_size:
        raise ValueError(
            "DDI matrix width must match drug vocabulary size: "
            f"got ddi={int(ddi_matrix.shape[0])}, vocab={drug_vocab_size}"
        )

    train_config = copy.deepcopy(train_config)
    train_config["_resolved_paths"] = {"processed_root": str(resolved_paths["processed_root"])}

    with tempfile.TemporaryDirectory(prefix="clinrec_subgroup_extended_") as temp_dir_name:
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

    payload = collect_extended_prediction_payload(
        model=model,
        dataloader=dataloader,
        device=device,
        decoder_top_k=decoder_top_k,
        memory_bank=memory_bank,
    )
    all_probs = payload["drug_probs"]
    all_targets = payload["targets"]
    all_predictions = binarize_predictions(all_probs, threshold).cpu()
    pred_counts = all_predictions.sum(dim=1).to(dtype=torch.long)
    true_counts = all_targets.sum(dim=1).to(dtype=torch.long)

    trajectory_labels = [
        _trajectory_length_bucket(int(len(record.get("steps", []))))
        for batch in dataloader
        for record in batch.get("records", [])
    ]
    medication_labels = [_medication_burden_bucket(int(value)) for value in true_counts.tolist()]
    subgroup_rows = []
    subgroup_rows.extend(
        _build_subgroup_rows(
            family="trajectory_length",
            labels=["short", "medium", "long"],
            subgroup_assignments=trajectory_labels,
            targets=all_targets,
            probs=all_probs,
            pred_counts=pred_counts,
            true_counts=true_counts,
            threshold=threshold,
            ddi_matrix=ddi_matrix,
        )
    )
    subgroup_rows.extend(
        _build_subgroup_rows(
            family="medication_burden",
            labels=["low", "medium", "high"],
            subgroup_assignments=medication_labels,
            targets=all_targets,
            probs=all_probs,
            pred_counts=pred_counts,
            true_counts=true_counts,
            threshold=threshold,
            ddi_matrix=ddi_matrix,
        )
    )

    overall_metrics = compute_core_metrics(
        all_targets,
        all_probs,
        threshold=threshold,
        ddi_matrix=ddi_matrix.detach().cpu(),
    )
    runtime_truth = dict(checkpoint_payload.get("runtime_truth") or getattr(model, "runtime_truth", {}) or {})
    runtime_truth["retrieval_active"] = retrieval_active
    runtime_truth["retrieval_bank_policy"] = "train_only"
    initialization_context = normalize_initialization_context(checkpoint_payload)
    report = {
        **runtime_truth,
        **initialization_context,
        "split": split,
        "num_samples": int(all_targets.shape[0]),
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_selection": threshold_selection,
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "metrics": {
            "jaccard": float(overall_metrics["jaccard"]),
            "f1": float(overall_metrics["f1"]),
            "prauc": float(overall_metrics["prauc"]),
            "ddi_rate": float(overall_metrics["ddi_rate"]),
        },
        "ddi_summary": _build_ddi_summary(ddi_artifact, metrics=overall_metrics),
        "artifacts": {},
    }

    save_reports = bool(evaluation_cfg.get("save_reports", True))
    report_stem = f"evaluate_extended_subgroup_{split}"
    if save_reports:
        json_path = write_json(
            resolved_paths["report_dir"] / f"{report_stem}.json",
            {"report": report, "subgroups": subgroup_rows},
        )
        report["artifacts"]["json"] = str(json_path)
    subgroup_csv_path = _write_plain_csv(
        resolved_paths["report_dir"] / f"{report_stem}.csv",
        subgroup_rows,
    )
    report["artifacts"]["csv"] = str(subgroup_csv_path)

    print(
        json.dumps(
            {
                "report": report,
                "subgroups": subgroup_rows,
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
