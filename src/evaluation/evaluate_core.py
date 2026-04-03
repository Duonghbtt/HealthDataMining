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
        compute_ddi_flags,
        compute_samplewise_f1,
        compute_samplewise_jaccard,
    )
else:
    from .metrics import (
        binarize_predictions,
        compute_core_metrics,
        compute_ddi_flags,
        compute_samplewise_f1,
        compute_samplewise_jaccard,
    )

from src.data.build_vocab import load_vocab_bundle
from src.data.dataset import collate_batch
from src.models.ddi_regularization import load_ddi_matrix
from src.training.train_core import (
    build_core_model,
    build_dataset,
    build_runtime_data_config_file,
    resolve_device,
)
from src.utils.io import ensure_dir, load_yaml_config, read_json, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate the core ClinRec medication recommendation model.")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to configs/eval.yaml")
    parser.add_argument("--data-config", default=None, help="Optional override for configs/data.yaml")
    parser.add_argument("--model-config", default=None, help="Optional override for configs/model.yaml")
    parser.add_argument("--train-config", default=None, help="Optional override for configs/train.yaml")
    parser.add_argument("--checkpoint", default=None, help="Optional override for best checkpoint path")
    parser.add_argument("--split", default=None, help="Optional override for evaluation split")
    parser.add_argument("--threshold", type=float, default=None, help="Optional override for prediction threshold")
    parser.add_argument("--device", default=None, help="Optional override for runtime device")
    parser.add_argument("--processed-root", default=None, help="Optional override for processed data root")
    parser.add_argument("--vocab-root", default=None, help="Optional override for vocab directory")
    parser.add_argument("--ddi-matrix-path", default=None, help="Optional override for DDI matrix artifact")
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
        collate_fn=collate_batch,
    )


def run_core_evaluation(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    threshold: float,
    ddi_matrix: torch.Tensor,
    decoder_top_k: int | None,
) -> dict[str, Any]:
    collected_probs: list[torch.Tensor] = []
    collected_targets: list[torch.Tensor] = []
    subject_ids: list[int] = []
    hadm_ids: list[int] = []
    stay_ids: list[int] = []

    model = model.to(device)
    model.eval()
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
                raise RuntimeError("Model did not return `drug_probs` during evaluation.")
            if final_target_drugs is None:
                raise RuntimeError("Model did not return `final_target_drugs` during evaluation.")

            collected_probs.append(drug_probs.detach().cpu())
            collected_targets.append(final_target_drugs.detach().cpu())
            subject_ids.extend(int(value) for value in batch.get("subject_ids", []))
            hadm_ids.extend(int(value) for value in batch.get("hadm_ids", []))
            stay_ids.extend(int(value) for value in batch.get("stay_ids", []))

    if not collected_probs or not collected_targets:
        raise ValueError("Evaluation dataloader produced no batches")

    all_probs = torch.cat(collected_probs, dim=0)
    all_targets = torch.cat(collected_targets, dim=0)
    binary_predictions = binarize_predictions(all_probs, threshold).cpu()
    ddi_matrix_cpu = ddi_matrix.detach().cpu()

    metrics = compute_core_metrics(
        all_targets,
        all_probs,
        threshold=threshold,
        ddi_matrix=ddi_matrix_cpu,
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
                "hadm_id": hadm_ids[row_index] if row_index < len(hadm_ids) else -1,
                "stay_id": stay_ids[row_index] if row_index < len(stay_ids) else -1,
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
        for key in ("jaccard", "f1", "prauc", "ddi_rate")
    }

    return {
        "drug_probs": all_probs,
        "targets": all_targets,
        "prediction_rows": prediction_rows,
        "prediction_summary": prediction_summary,
        "ddi_summary": ddi_summary,
        "metrics": metric_summary,
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
    print("Resolved evaluation paths:")
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

    print(f"Using device: {device}")
    print(f"Evaluating split: {split}")
    print(f"Using threshold: {threshold}")
    print(f"Loading checkpoint: {checkpoint_path}")

    train_config = copy.deepcopy(train_config)
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

    report: dict[str, Any] = {
        "split": split,
        "num_samples": int(evaluation_result["targets"].shape[0]),
        "threshold": threshold,
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "metrics": evaluation_result["metrics"],
        "ddi_summary": evaluation_result["ddi_summary"],
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
