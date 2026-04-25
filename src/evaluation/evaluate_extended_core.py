from __future__ import annotations

import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch
from torch.utils.data import DataLoader

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from evaluate_core import (  # type: ignore[import-not-found]
        _build_ddi_summary,
        _compatibility_fallback_used,
        _flatten_report,
        _load_embedded_or_yaml_config,
        _move_batch_to_device,
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _stringify_indices,
        _stringify_path_source,
        _write_plain_csv,
        parse_args,
    )
    from metrics import (  # type: ignore[import-not-found]
        binarize_predictions,
        compute_core_metrics,
        compute_ddi_flags,
        compute_samplewise_f1,
        compute_samplewise_jaccard,
        multilabel_prauc,
    )
    from thresholding import resolve_effective_threshold  # type: ignore[import-not-found]
else:
    from .evaluate_core import (
        _build_ddi_summary,
        _compatibility_fallback_used,
        _flatten_report,
        _load_embedded_or_yaml_config,
        _move_batch_to_device,
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _stringify_indices,
        _stringify_path_source,
        _write_plain_csv,
        parse_args,
    )
    from .metrics import (
        binarize_predictions,
        compute_core_metrics,
        compute_ddi_flags,
        compute_samplewise_f1,
        compute_samplewise_jaccard,
        multilabel_prauc,
    )
    from .thresholding import resolve_effective_threshold

from src.data.build_vocab import load_vocab_bundle
from src.explainability.attention_export import save_attention_artifacts
from src.models.ddi_regularization import load_ddi_artifact
from src.training.train_core import build_dataset, build_runtime_data_config_file, resolve_device
from src.training.train_extended import (
    build_extended_model,
    build_memory_bank_from_dataloader,
    collate_batch_with_records,
)
from src.utils.io import load_yaml_config, read_json, resolve_path, write_json
from src.utils.runtime_truth import normalize_ddi_context, normalize_initialization_context


def build_extended_eval_dataloader(
    *,
    split: str,
    runtime_data_config_path: Path,
    processed_root: Path,
    drug_vocab_size: int,
    batch_size: int,
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
        collate_fn=collate_batch_with_records,
    )


def collect_extended_prediction_payload(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    decoder_top_k: int | None,
    memory_bank: Any | None,
) -> dict[str, Any]:
    collected_probs: list[torch.Tensor] = []
    collected_targets: list[torch.Tensor] = []
    subject_ids: list[int] = []
    hadm_ids: list[int] = []
    stay_ids: list[int] = []
    preview_outputs: dict[str, Any] | None = None

    model = model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            batch_on_device = _move_batch_to_device(batch, device)
            outputs = model(
                batch_on_device,
                mode="extended",
                memory_bank=memory_bank,
                records=batch_on_device.get("records"),
                decoder_top_k=decoder_top_k,
            )
            drug_probs = outputs.get("drug_probs")
            final_target_drugs = outputs.get("final_target_drugs")
            if drug_probs is None:
                raise RuntimeError("Model did not return `drug_probs` during extended evaluation.")
            if final_target_drugs is None:
                raise RuntimeError("Model did not return `final_target_drugs` during extended evaluation.")

            collected_probs.append(drug_probs.detach().cpu())
            collected_targets.append(final_target_drugs.detach().cpu())
            subject_ids.extend(int(value) for value in batch.get("subject_ids", []))
            hadm_ids.extend(int(value) for value in batch.get("hadm_ids", []))
            stay_ids.extend(int(value) for value in batch.get("stay_ids", []))
            if preview_outputs is None:
                preview_outputs = {
                    "selection_outputs": outputs,
                    "fusion_outputs": outputs,
                    "batch_size": int(drug_probs.shape[0]),
                }

    if not collected_probs or not collected_targets:
        raise ValueError("Evaluation dataloader produced no batches")

    return {
        "drug_probs": torch.cat(collected_probs, dim=0),
        "targets": torch.cat(collected_targets, dim=0),
        "subject_ids": subject_ids,
        "hadm_ids": hadm_ids,
        "stay_ids": stay_ids,
        "preview_outputs": preview_outputs,
    }


def run_extended_evaluation(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float,
    ddi_artifact: Mapping[str, Any],
    decoder_top_k: int | None,
    memory_bank: Any | None,
) -> dict[str, Any]:
    ddi_artifact = normalize_ddi_context(ddi_artifact)
    payload = collect_extended_prediction_payload(
        model=model,
        dataloader=dataloader,
        device=device,
        decoder_top_k=decoder_top_k,
        memory_bank=memory_bank,
    )
    all_probs = payload["drug_probs"]
    all_targets = payload["targets"]
    binary_predictions = binarize_predictions(all_probs, threshold).cpu()
    sample_jaccard = compute_samplewise_jaccard(all_targets, binary_predictions).cpu()
    sample_f1 = compute_samplewise_f1(all_targets, binary_predictions).cpu()
    ddi_active = bool(ddi_artifact.get("active", False))
    ddi_matrix_cpu = ddi_artifact["matrix"].detach().cpu()
    if ddi_active:
        metrics = compute_core_metrics(
            all_targets,
            all_probs,
            threshold=threshold,
            ddi_matrix=ddi_matrix_cpu,
        )
        ddi_flags = compute_ddi_flags(binary_predictions, ddi_matrix_cpu).cpu()
        ddi_summary = _build_ddi_summary(ddi_artifact, metrics=metrics)
        metric_summary: dict[str, Any] = {
            key: metrics[key]
            for key in ("jaccard", "f1", "prauc", "ddi_rate")
        }
    else:
        ddi_flags = torch.zeros(binary_predictions.shape[0], dtype=torch.bool)
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
        predicted_indices = torch.nonzero(binary_predictions[row_index], as_tuple=False).flatten()
        prediction_rows.append(
            {
                "subject_id": payload["subject_ids"][row_index] if row_index < len(payload["subject_ids"]) else -1,
                "hadm_id": payload["hadm_ids"][row_index] if row_index < len(payload["hadm_ids"]) else -1,
                "stay_id": payload["stay_ids"][row_index] if row_index < len(payload["stay_ids"]) else -1,
                "true_count": int(all_targets[row_index].sum().item()),
                "pred_count": int(binary_predictions[row_index].sum().item()),
                "sample_jaccard": float(sample_jaccard[row_index].item()),
                "sample_f1": float(sample_f1[row_index].item()),
                "has_ddi": None if not ddi_active else bool(ddi_flags[row_index].item()),
                "predicted_drug_indices": _stringify_indices(predicted_indices),
            }
        )

    prediction_summary = {
        "avg_predicted_drugs": float(binary_predictions.sum(dim=1, dtype=torch.float32).mean().item()),
        "avg_true_drugs": float(all_targets.sum(dim=1, dtype=torch.float32).mean().item()),
    }

    return {
        "drug_probs": all_probs,
        "targets": all_targets,
        "prediction_rows": prediction_rows,
        "prediction_summary": prediction_summary,
        "ddi_summary": ddi_summary,
        "metrics": metric_summary,
        "preview_outputs": payload["preview_outputs"],
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
    print("Resolved extended evaluation paths:")
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

    print(f"Using device: {device}")
    print(f"Evaluating extended split: {split}")
    print(f"Using threshold: {threshold} [{threshold_source}]")
    print(f"Loading checkpoint: {checkpoint_path}")

    with tempfile.TemporaryDirectory(prefix="clinrec_eval_extended_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        runtime_data_config_path = build_runtime_data_config_file(
            data_config=data_config,
            processed_root=resolved_paths["processed_root"],
            vocab_root=resolved_paths["vocab_root"],
            temp_dir=temp_dir,
        )
        _ = load_vocab_bundle(runtime_data_config_path)

        eval_loader = build_extended_eval_dataloader(
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
        raise KeyError(f"Checkpoint at {checkpoint_path} does not contain `model_state_dict`.")
    try:
        model.load_state_dict(model_state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(f"Failed to load checkpoint state from {checkpoint_path}: {exc}") from exc

    retrieval_active = bool(checkpoint_payload.get("retrieval_active", extended_cfg.get("use_retrieval", True)))
    memory_bank = None
    if retrieval_active:
        memory_bank = build_memory_bank_from_dataloader(
            model=model,
            dataloader=train_bank_loader,
            device=device,
            split="train",
        )
        print(f"Built train-bank-only evaluation memory bank with {len(memory_bank)} visits.")
    else:
        print("Extended retrieval is disabled for this evaluation run; proceeding without a memory bank.")

    evaluation_result = run_extended_evaluation(
        model=model,
        dataloader=eval_loader,
        device=device,
        threshold=threshold,
        ddi_artifact=ddi_artifact,
        decoder_top_k=decoder_top_k,
        memory_bank=memory_bank,
    )
    runtime_truth = dict(checkpoint_payload.get("runtime_truth") or getattr(model, "runtime_truth", {}) or {})
    if not runtime_truth:
        runtime_truth = dict(getattr(model, "runtime_truth", {}))
    runtime_truth["retrieval_active"] = retrieval_active
    runtime_truth["retrieval_bank_policy"] = "train_only"
    runtime_truth["evaluation_mode"] = "extended"
    print(
        "Extended evaluation runtime truth: "
        f"pipeline_level={runtime_truth.get('pipeline_level', 'unknown')} "
        f"history_active={runtime_truth.get('history_active')} "
        f"retrieval_active={runtime_truth.get('retrieval_active')} "
        f"retrieval_mode={runtime_truth.get('retrieval_mode', 'unknown')} "
        f"fusion_strategy={runtime_truth.get('fusion_strategy', 'unknown')}"
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
        "threshold": threshold,
        "threshold_source": threshold_source,
        "threshold_selection": threshold_selection,
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

    preview_outputs = evaluation_result.get("preview_outputs")
    if isinstance(preview_outputs, Mapping):
        attention_paths = save_attention_artifacts(
            project_root,
            name=f"evaluate_extended_core_{split}",
            selection_outputs=preview_outputs["selection_outputs"],
            fusion_outputs=preview_outputs["fusion_outputs"],
            output_dir=resolved_paths["report_dir"],
        )
        report["artifacts"]["attention_json"] = str(attention_paths["json"])
        report["artifacts"]["attention_csv"] = str(attention_paths["csv"])
        report["attention_preview_batch_size"] = int(preview_outputs.get("batch_size", 0))

    save_reports = bool(evaluation_cfg.get("save_reports", True))
    save_predictions = bool(evaluation_cfg.get("save_predictions", True))
    report_stem = f"evaluate_extended_core_{split}"

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


__all__ = [
    "build_extended_eval_dataloader",
    "collect_extended_prediction_payload",
    "run_extended_evaluation",
]


if __name__ == "__main__":
    main()
