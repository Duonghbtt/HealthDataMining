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
        _move_batch_to_device,
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _write_plain_csv,
        build_eval_dataloader,
        build_runtime_data_config_file,
        parse_args,
    )
    from metrics import binarize_predictions, compute_core_metrics  # type: ignore[import-not-found]
else:
    from .evaluate_core import (
        _load_embedded_or_yaml_config,
        _move_batch_to_device,
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _write_plain_csv,
        build_eval_dataloader,
        build_runtime_data_config_file,
        parse_args,
    )
    from .metrics import binarize_predictions, compute_core_metrics

from src.data.build_vocab import load_vocab_bundle
from src.models.ddi_regularization import load_ddi_matrix
from src.training.train_core import build_core_model, resolve_device
from src.utils.io import load_yaml_config, read_json, resolve_path, write_json


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
    threshold: float,
    decoder_top_k: int | None,
) -> dict[str, torch.Tensor]:
    model = model.to(device)
    model.eval()

    collected_probs: list[torch.Tensor] = []
    collected_targets: list[torch.Tensor] = []
    collected_num_steps: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in dataloader:
            batch_on_device = _move_batch_to_device(batch, device)
            outputs = model(
                batch_on_device,
                mode="core",
                decoder_top_k=decoder_top_k,
            )
            drug_probs = outputs.get("drug_probs")
            final_target_drugs = outputs.get("final_target_drugs")
            if drug_probs is None:
                raise RuntimeError("Model did not return `drug_probs` during subgroup evaluation.")
            if final_target_drugs is None:
                raise RuntimeError("Model did not return `final_target_drugs` during subgroup evaluation.")

            collected_probs.append(drug_probs.detach().cpu())
            collected_targets.append(final_target_drugs.detach().cpu())
            collected_num_steps.append(batch_on_device["visit_mask"].sum(dim=1).detach().cpu())

    if not collected_probs or not collected_targets:
        raise ValueError("Evaluation dataloader produced no batches")

    all_probs = torch.cat(collected_probs, dim=0)
    all_targets = torch.cat(collected_targets, dim=0)
    all_num_steps = torch.cat(collected_num_steps, dim=0).to(dtype=torch.long)
    all_predictions = binarize_predictions(all_probs, threshold).cpu()

    return {
        "probs": all_probs,
        "targets": all_targets,
        "num_steps": all_num_steps,
        "predictions": all_predictions,
        "pred_counts": all_predictions.sum(dim=1).to(dtype=torch.long),
        "true_counts": all_targets.sum(dim=1).to(dtype=torch.long),
    }


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
    print("Resolved subgroup evaluation paths:")
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
    print(f"Evaluating subgroup split: {split}")
    print(f"Using threshold: {threshold}")
    print(f"Loading checkpoint: {checkpoint_path}")

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

    subgroup_payload = _collect_subgroup_payload(
        model=model,
        dataloader=dataloader,
        device=device,
        threshold=threshold,
        decoder_top_k=decoder_top_k,
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
            pred_counts=pred_counts,
            true_counts=true_counts,
            threshold=threshold,
            ddi_matrix=ddi_matrix,
        ),
        "true_medication_burden": _build_subgroup_rows(
            family="true_medication_burden",
            labels=["low", "medium", "high"],
            subgroup_assignments=[_medication_burden_bucket(int(value)) for value in true_counts.tolist()],
            targets=subgroup_payload["targets"],
            probs=subgroup_payload["probs"],
            pred_counts=pred_counts,
            true_counts=true_counts,
            threshold=threshold,
            ddi_matrix=ddi_matrix,
        ),
        "predicted_medication_burden": _build_subgroup_rows(
            family="predicted_medication_burden",
            labels=["low", "medium", "high"],
            subgroup_assignments=[_medication_burden_bucket(int(value)) for value in pred_counts.tolist()],
            targets=subgroup_payload["targets"],
            probs=subgroup_payload["probs"],
            pred_counts=pred_counts,
            true_counts=true_counts,
            threshold=threshold,
            ddi_matrix=ddi_matrix,
        ),
    }
    csv_rows = [row for rows in family_rows.values() for row in rows]

    subgroup_report: dict[str, Any] = {
        "split": split,
        "num_samples": int(subgroup_payload["targets"].shape[0]),
        "threshold": threshold,
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
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
