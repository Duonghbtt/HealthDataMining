from __future__ import annotations

import argparse
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
        _normalize_prediction_selection,
        _resolve_prediction_selection,
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _selection_label,
        _selection_metric_kwargs,
        _write_plain_csv,
        build_eval_dataloader,
        build_runtime_data_config_file,
        run_core_evaluation,
    )
    from metrics import compute_core_metrics, compute_ddi_flags, select_binary_predictions  # type: ignore[import-not-found]
else:
    from .evaluate_core import (
        _load_embedded_or_yaml_config,
        _normalize_prediction_selection,
        _resolve_prediction_selection,
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _selection_label,
        _selection_metric_kwargs,
        _write_plain_csv,
        build_eval_dataloader,
        build_runtime_data_config_file,
        run_core_evaluation,
    )
    from .metrics import compute_core_metrics, compute_ddi_flags, select_binary_predictions

from src.models.ddi_regularization import load_ddi_matrix
from src.training.runtime_builder import build_core_model, resolve_device
from src.utils.io import ensure_dir, load_yaml_config, read_json, resolve_path, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ablation settings for the core ClinRec medication model.")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to configs/eval.yaml")
    parser.add_argument("--data-config", default=None, help="Optional override for configs/data.yaml")
    parser.add_argument("--model-config", default=None, help="Optional override for configs/model.yaml")
    parser.add_argument("--train-config", default=None, help="Optional override for configs/train.yaml")
    parser.add_argument("--checkpoint", default=None, help="Optional override for best checkpoint path")
    parser.add_argument("--split", default=None, help="Optional override for evaluation split")
    parser.add_argument("--threshold", type=float, default=None, help="Optional override for base prediction threshold")
    parser.add_argument("--device", default=None, help="Optional override for runtime device")
    parser.add_argument("--processed-root", default=None, help="Optional override for processed data root")
    parser.add_argument("--vocab-root", default=None, help="Optional override for vocab directory")
    parser.add_argument("--ddi-matrix-path", default=None, help="Optional override for DDI matrix artifact")
    return parser.parse_args()


def _clamp_threshold(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _resolve_ablation_thresholds(
    *,
    base_threshold: float,
    eval_config: Mapping[str, Any],
) -> tuple[float, float]:
    ablation_cfg = dict(eval_config.get("ablation", {}))
    higher_threshold = _clamp_threshold(
        float(ablation_cfg.get("higher_threshold", min(float(base_threshold) + 0.1, 0.95)))
    )
    lower_threshold = _clamp_threshold(
        float(ablation_cfg.get("lower_threshold", max(float(base_threshold) - 0.1, 0.05)))
    )
    return higher_threshold, lower_threshold


def _build_ablation_settings(
    *,
    base_selection: Mapping[str, Any],
    eval_config: Mapping[str, Any],
) -> list[dict[str, Any]]:
    resolved_base_selection = _normalize_prediction_selection(base_selection)
    settings: list[dict[str, Any]] = [
        {
            "name": "full_model",
            "selection_config": dict(resolved_base_selection),
            "ddi_regularizer_enabled": True,
        },
        {
            "name": "no_ddi_regularizer",
            "selection_config": dict(resolved_base_selection),
            "ddi_regularizer_enabled": False,
            "notes": "Inference probabilities are reused; disabling the DDI regularizer only affects training loss.",
        },
    ]
    method = str(resolved_base_selection["method"])
    ablation_cfg = dict(eval_config.get("ablation", {}))
    if method == "global":
        higher_threshold, lower_threshold = _resolve_ablation_thresholds(
            base_threshold=float(resolved_base_selection["threshold"]),
            eval_config=eval_config,
        )
        settings.extend(
            [
                {
                    "name": "higher_threshold",
                    "selection_config": {"method": "global", "threshold": float(higher_threshold)},
                    "ddi_regularizer_enabled": True,
                },
                {
                    "name": "lower_threshold",
                    "selection_config": {"method": "global", "threshold": float(lower_threshold)},
                    "ddi_regularizer_enabled": True,
                },
            ]
        )
    elif method == "topk":
        top_k = int(resolved_base_selection["top_k"])
        topk_delta = max(int(ablation_cfg.get("topk_delta", 5)), 1)
        settings.extend(
            [
                {
                    "name": "higher_topk",
                    "selection_config": {"method": "topk", "top_k": top_k + topk_delta},
                    "ddi_regularizer_enabled": True,
                },
                {
                    "name": "lower_topk",
                    "selection_config": {"method": "topk", "top_k": max(1, top_k - topk_delta)},
                    "ddi_regularizer_enabled": True,
                },
            ]
        )
    elif method == "percentile":
        percentile = float(resolved_base_selection["percentile"])
        percentile_delta = max(float(ablation_cfg.get("percentile_delta", 5.0)), 1.0)
        settings.extend(
            [
                {
                    "name": "higher_percentile",
                    "selection_config": {
                        "method": "percentile",
                        "percentile": _clamp_threshold((percentile + percentile_delta) / 100.0) * 100.0,
                    },
                    "ddi_regularizer_enabled": True,
                },
                {
                    "name": "lower_percentile",
                    "selection_config": {
                        "method": "percentile",
                        "percentile": _clamp_threshold((percentile - percentile_delta) / 100.0) * 100.0,
                    },
                    "ddi_regularizer_enabled": True,
                },
            ]
        )
    return settings


def _summarize_ablation_setting(
    *,
    setting: Mapping[str, Any],
    drug_probs: torch.Tensor,
    targets: torch.Tensor,
    ddi_matrix: torch.Tensor,
) -> dict[str, Any]:
    selection_config = _normalize_prediction_selection(setting.get("selection_config", setting))
    selection_kwargs = _selection_metric_kwargs(selection_config)
    binary_predictions = select_binary_predictions(
        drug_probs,
        **selection_kwargs,
    ).cpu()
    metrics = compute_core_metrics(
        targets,
        drug_probs,
        threshold=float(selection_kwargs["threshold"]),
        ddi_matrix=ddi_matrix,
        prediction_method=str(selection_kwargs["prediction_method"]),
        top_k=selection_kwargs["top_k"],
        percentile=selection_kwargs["percentile"],
    )
    ddi_flags = compute_ddi_flags(binary_predictions, ddi_matrix).cpu()
    avg_predicted_drugs = float(binary_predictions.sum(dim=1, dtype=torch.float32).mean().item())

    row = {
        "setting": str(setting["name"]),
        "method": str(selection_config["method"]),
        "selection_label": _selection_label(selection_config),
        "threshold": None if selection_config.get("threshold") is None else float(selection_config["threshold"]),
        "top_k": None if selection_config.get("top_k") is None else int(selection_config["top_k"]),
        "percentile": (
            None if selection_config.get("percentile") is None else float(selection_config["percentile"])
        ),
        "ddi_regularizer_enabled": bool(setting.get("ddi_regularizer_enabled", True)),
        "jaccard": float(metrics["jaccard"]),
        "f1": float(metrics["f1"]),
        "prauc": float(metrics["prauc"]),
        "ddi_rate": float(metrics["ddi_rate"]),
        "avg_predicted_drugs": avg_predicted_drugs,
        "patients_with_ddi": int(metrics["patients_with_ddi"]),
        "num_samples": int(targets.shape[0]),
    }
    if "notes" in setting:
        row["notes"] = str(setting["notes"])
    row["patients_with_ddi_ratio"] = (
        0.0 if row["num_samples"] <= 0 else float(ddi_flags.sum(dtype=torch.float32).item() / float(row["num_samples"]))
    )
    return row


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
    print("Resolved ablation evaluation paths:")
    for key, value in resolved_paths.items():
        print(f"  {key}: {value}")

    runtime_cfg = dict(eval_config.get("runtime", {}))
    evaluation_cfg = dict(eval_config.get("evaluation", {}))

    split = str(args.split or evaluation_cfg.get("split", "test"))
    device = resolve_device(args.device or runtime_cfg.get("device", "cpu"))
    batch_size = int(runtime_cfg.get("batch_size", 32))
    max_eval_batches = eval_config.get("run", {}).get("max_eval_batches")
    max_eval_batches = None if max_eval_batches is None else int(max_eval_batches)

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

    print(f"Using device: {device}")
    print(f"Evaluating ablation split: {split}")
    print(f"Loading checkpoint: {checkpoint_path}")

    with tempfile.TemporaryDirectory(prefix="clinrec_ablation_eval_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        runtime_data_config_path = build_runtime_data_config_file(
            project_root=project_root,
            data_config=data_config,
            processed_root=resolved_paths["processed_root"],
            vocab_root=resolved_paths["vocab_root"],
            temp_dir=temp_dir,
            retrieval_cache_config=train_config.get("retrieval_cache"),
        )
        dataloader = build_eval_dataloader(
            split=split,
            runtime_data_config_path=runtime_data_config_path,
            processed_root=resolved_paths["processed_root"],
            drug_vocab_size=drug_vocab_size,
            batch_size=batch_size,
        )
        val_dataloader = build_eval_dataloader(
            split="val",
            runtime_data_config_path=runtime_data_config_path,
            processed_root=resolved_paths["processed_root"],
            drug_vocab_size=drug_vocab_size,
            batch_size=batch_size,
        )
        model = build_core_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            processed_root=resolved_paths["processed_root"],
            vocab_root=resolved_paths["vocab_root"],
            ddi_matrix_path=resolved_paths["ddi_matrix_path"],
        )

    model_state_dict = checkpoint_payload.get("model_state_dict")
    if not isinstance(model_state_dict, Mapping):
        raise KeyError("Checkpoint does not contain `model_state_dict`.")
    model.load_state_dict(model_state_dict, strict=True)

    resolved_selection_config, selection_source, threshold_search_report, _ = _resolve_prediction_selection(
        explicit_threshold=args.threshold,
        checkpoint_payload=checkpoint_payload,
        eval_config=eval_config,
        train_config=train_config,
        split=split,
        model=model,
        val_dataloader=val_dataloader,
        device=device,
        ddi_matrix=ddi_matrix,
        max_eval_batches=max_eval_batches,
    )
    print(f"Using ablation base selection: {_selection_label(resolved_selection_config)} ({selection_source})")
    if str(resolved_selection_config["method"]) == "global":
        print(f"Using base threshold: {float(resolved_selection_config['threshold']):.2f}")

    full_result = run_core_evaluation(
        model=model,
        dataloader=dataloader,
        device=device,
        selection_config=resolved_selection_config,
        ddi_matrix=ddi_matrix,
        max_eval_batches=max_eval_batches,
    )

    ablation_rows = [
        _summarize_ablation_setting(
            setting=setting,
            drug_probs=full_result["drug_probs"],
            targets=full_result["targets"],
            ddi_matrix=ddi_matrix,
        )
        for setting in _build_ablation_settings(
            base_selection=resolved_selection_config,
            eval_config=eval_config,
        )
    ]

    report: dict[str, Any] = {
        "split": split,
        "num_samples": int(full_result["targets"].shape[0]),
        "base_selection": {
            "method": str(resolved_selection_config["method"]),
            "threshold": (
                None
                if resolved_selection_config.get("threshold") is None
                else float(resolved_selection_config["threshold"])
            ),
            "top_k": (
                None if resolved_selection_config.get("top_k") is None else int(resolved_selection_config["top_k"])
            ),
            "percentile": (
                None
                if resolved_selection_config.get("percentile") is None
                else float(resolved_selection_config["percentile"])
            ),
            "selection_label": _selection_label(resolved_selection_config),
            "source": selection_source,
        },
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "settings": ablation_rows,
        "artifacts": {},
    }
    if threshold_search_report is not None:
        report["threshold_search"] = threshold_search_report

    report_dir = ensure_dir(resolved_paths["report_dir"])
    report_stem = f"evaluate_ablation_{split}"
    json_path = write_json(report_dir / f"{report_stem}.json", report)
    csv_path = _write_plain_csv(report_dir / f"{report_stem}.csv", ablation_rows)
    report["artifacts"]["json"] = str(json_path)
    report["artifacts"]["csv"] = str(csv_path)
    write_json(json_path, report)

    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
