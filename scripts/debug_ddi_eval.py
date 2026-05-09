from __future__ import annotations

import argparse
import copy
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.evaluate_core import (  # noqa: E402
    _collect_core_outputs,
    _load_embedded_or_yaml_config,
    _resolve_eval_paths,
    build_eval_dataloader,
)
from src.evaluation.metrics import (  # noqa: E402
    compute_avg_predicted_drugs,
    compute_avg_true_drugs,
    compute_ddi_flags,
    compute_ddi_rate,
    compute_prauc,
    multilabel_f1,
    multilabel_jaccard,
    select_binary_predictions,
)
from src.models.ddi_regularization import load_ddi_matrix  # noqa: E402
from src.training.runtime_builder import (  # noqa: E402
    build_core_model,
    build_runtime_data_config_file,
    resolve_device,
)
from src.utils.io import load_yaml_config, read_json, resolve_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug ClinRec DDI evaluation and decode behavior.")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to configs/eval.yaml")
    parser.add_argument("--checkpoint", default="outputs/checkpoints/train_core_best.pt", help="Checkpoint path")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"), help="Split to inspect")
    parser.add_argument("--max-batches", type=int, default=100, help="Maximum batches to collect")
    parser.add_argument("--batch-size", type=int, default=None, help="Optional batch-size override")
    parser.add_argument("--device", default=None, help="Optional device override")
    parser.add_argument("--data-config", default=None, help="Optional configs/data.yaml override")
    parser.add_argument("--model-config", default=None, help="Optional configs/model.yaml override")
    parser.add_argument("--train-config", default=None, help="Optional configs/train.yaml override")
    parser.add_argument("--processed-root", default=None, help="Optional processed root override")
    parser.add_argument("--vocab-root", default=None, help="Optional vocab root override")
    parser.add_argument("--ddi-matrix-path", default=None, help="Optional DDI matrix path override")
    parser.add_argument("--top-k", type=int, default=16, help="Top-k value to compare")
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.90)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    return parser.parse_args()


def _merge_nested_dicts(base: Mapping[str, Any], override: Mapping[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    if override is None:
        return merged
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _merge_nested_dicts(dict(merged[key]), value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _vocab_tokens(vocab_payload: Mapping[str, Any]) -> list[str]:
    idx_to_token = vocab_payload.get("idx_to_token")
    if isinstance(idx_to_token, list):
        return [str(token) for token in idx_to_token]
    if isinstance(idx_to_token, Mapping):
        return [
            str(idx_to_token[str(index)] if str(index) in idx_to_token else idx_to_token[index])
            for index in range(int(vocab_payload["size"]))
        ]
    raise ValueError("Vocabulary payload is missing idx_to_token")


def _strict_upper_ddi(ddi_matrix: torch.Tensor) -> torch.Tensor:
    ddi_bool = (torch.as_tensor(ddi_matrix) > 0).to(dtype=torch.bool)
    ddi_bool = torch.logical_or(ddi_bool, ddi_bool.transpose(0, 1))
    ddi_bool.fill_diagonal_(False)
    return torch.triu(ddi_bool, diagonal=1)


def _ddi_degrees(ddi_matrix: torch.Tensor) -> torch.Tensor:
    ddi_bool = (torch.as_tensor(ddi_matrix) > 0).to(dtype=torch.bool)
    ddi_bool = torch.logical_or(ddi_bool, ddi_bool.transpose(0, 1))
    ddi_bool.fill_diagonal_(False)
    return ddi_bool.sum(dim=1, dtype=torch.long)


def _token(tokens: list[str], index: int) -> str:
    if 0 <= int(index) < len(tokens):
        return tokens[int(index)]
    return "<out-of-range>"


def _json_print(title: str, payload: Any) -> None:
    print(f"\n## {title}")
    print(json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=True))


def _format_float(value: float) -> float:
    return float(f"{float(value):.8g}")


def _threshold_values(start: float, end: float, step: float) -> list[float]:
    if step <= 0:
        raise ValueError("--threshold-step must be positive")
    values: list[float] = []
    current = float(start)
    while current <= float(end) + 1.0e-9:
        values.append(round(current, 10))
        current += float(step)
    return values


def _metric_row(
    *,
    name: str,
    y_true: torch.Tensor,
    probs: torch.Tensor,
    ddi_matrix: torch.Tensor,
    prauc: float,
    threshold: float = 0.5,
    prediction_method: str = "global",
    top_k: int | None = None,
    percentile: float | None = None,
) -> dict[str, Any]:
    binary = select_binary_predictions(
        probs,
        threshold=float(threshold),
        prediction_method=prediction_method,
        top_k=top_k,
        percentile=percentile,
    ).cpu()
    ddi_summary = compute_ddi_rate(binary, ddi_matrix)
    ddi_flags = compute_ddi_flags(binary, ddi_matrix)
    return {
        "name": name,
        "method": prediction_method,
        "threshold": None if prediction_method != "global" else _format_float(threshold),
        "top_k": top_k,
        "percentile": percentile,
        "jaccard": _format_float(multilabel_jaccard(y_true, binary)),
        "f1": _format_float(multilabel_f1(y_true, binary)),
        "prauc": _format_float(prauc),
        "ddi_rate": _format_float(ddi_summary["ddi_rate"]),
        "avg_predicted_drugs": _format_float(compute_avg_predicted_drugs(binary)),
        "avg_true_drugs": _format_float(compute_avg_true_drugs(y_true)),
        "total_predicted_pairs": int(ddi_summary["total_predicted_pairs"]),
        "total_interacting_pairs": int(ddi_summary["total_interacting_pairs"]),
        "samples_with_any_ddi_pair": int(ddi_flags.sum().item()),
        "num_samples": int(binary.shape[0]),
    }


def _ddi_matrix_report(ddi_matrix: torch.Tensor, tokens: list[str]) -> dict[str, Any]:
    upper = _strict_upper_ddi(ddi_matrix)
    degrees = _ddi_degrees(ddi_matrix)
    n = int(ddi_matrix.shape[0])
    real_start = 2 if n > 2 else 0
    real_n = max(0, n - real_start)
    total_pairs = n * (n - 1) // 2
    real_total_pairs = real_n * (real_n - 1) // 2
    upper_real = upper[real_start:, real_start:] if real_n > 0 else upper[:0, :0]
    nonzero_pairs = torch.nonzero(upper, as_tuple=False)
    first_pairs = []
    for row, col in nonzero_pairs[:20].tolist():
        first_pairs.append(
            {
                "i": int(row),
                "j": int(col),
                "token_i": _token(tokens, int(row)),
                "token_j": _token(tokens, int(col)),
                "degree_i": int(degrees[int(row)].item()),
                "degree_j": int(degrees[int(col)].item()),
            }
        )
    return {
        "shape": [int(dim) for dim in ddi_matrix.shape],
        "sum_full_symmetric": _format_float(float((ddi_matrix > 0).sum().item())),
        "nonzero_upper_pairs": int(upper.sum().item()),
        "density_upper_all_tokens": 0.0 if total_pairs <= 0 else _format_float(float(upper.sum().item()) / total_pairs),
        "density_upper_real_labels_excluding_pad_unk": 0.0
        if real_total_pairs <= 0
        else _format_float(float(upper_real.sum().item()) / real_total_pairs),
        "nonzero_rows": int((degrees > 0).sum().item()),
        "zero_degree_rows": int((degrees == 0).sum().item()),
        "pad_degree": int(degrees[0].item()) if n > 0 else None,
        "unk_degree": int(degrees[1].item()) if n > 1 else None,
        "first_20_nonzero_pairs": first_pairs,
    }


def _prediction_report(
    *,
    label: str,
    binary: torch.Tensor,
    ddi_matrix: torch.Tensor,
    tokens: list[str],
    pad_idx: int | None,
    unk_idx: int | None,
) -> dict[str, Any]:
    binary = binary.cpu().to(dtype=torch.bool)
    counts_per_sample = binary.sum(dim=1, dtype=torch.float32)
    col_counts = binary.sum(dim=0, dtype=torch.long)
    degrees = _ddi_degrees(ddi_matrix)
    ddi_flags = compute_ddi_flags(binary, ddi_matrix)
    nonzero_degree_mask = (degrees > 0).cpu()
    zero_degree_mask = ~nonzero_degree_mask
    selected_total = int(binary.sum().item())
    selected_zero_degree = int(binary[:, zero_degree_mask].sum().item()) if binary.shape[1] == zero_degree_mask.numel() else 0
    selected_nonzero_degree = (
        int(binary[:, nonzero_degree_mask].sum().item()) if binary.shape[1] == nonzero_degree_mask.numel() else 0
    )
    samples_with_predictions = counts_per_sample > 0
    samples_all_zero_degree = 0
    if bool(samples_with_predictions.any().item()) and binary.shape[1] == nonzero_degree_mask.numel():
        selected_nonzero_by_sample = binary[:, nonzero_degree_mask].sum(dim=1)
        samples_all_zero_degree = int((samples_with_predictions & (selected_nonzero_by_sample == 0)).sum().item())

    top_k = min(30, int(col_counts.numel()))
    top_values, top_indices = torch.topk(col_counts, k=top_k)
    top_predicted = []
    for count, index in zip(top_values.tolist(), top_indices.tolist()):
        if int(count) <= 0:
            continue
        top_predicted.append(
            {
                "drug_id": int(index),
                "token": _token(tokens, int(index)),
                "predicted_count": int(count),
                "ddi_degree": int(degrees[int(index)].item()) if int(index) < int(degrees.numel()) else None,
                "is_pad": pad_idx is not None and int(index) == int(pad_idx),
                "is_unk": unk_idx is not None and int(index) == int(unk_idx),
            }
        )

    special_counts: dict[str, Any] = {}
    for name, index in (("pad", pad_idx), ("unk", unk_idx)):
        if index is not None and 0 <= int(index) < int(binary.shape[1]):
            idx = int(index)
            special_counts[f"{name}_idx"] = idx
            special_counts[f"{name}_predicted_occurrences"] = int(col_counts[idx].item())
            special_counts[f"samples_with_{name}"] = int(binary[:, idx].sum().item())

    return {
        "label": label,
        "num_samples": int(binary.shape[0]),
        "avg_predicted_drugs_per_sample": _format_float(float(counts_per_sample.mean().item())),
        "min_predicted_drugs_per_sample": int(counts_per_sample.min().item()) if counts_per_sample.numel() else 0,
        "max_predicted_drugs_per_sample": int(counts_per_sample.max().item()) if counts_per_sample.numel() else 0,
        "samples_with_at_least_2_drugs": int((counts_per_sample >= 2).sum().item()),
        "samples_with_any_ddi_pair": int(ddi_flags.sum().item()),
        "selected_total": selected_total,
        "selected_zero_degree_drugs": selected_zero_degree,
        "selected_nonzero_degree_drugs": selected_nonzero_degree,
        "selected_zero_degree_fraction": 0.0
        if selected_total <= 0
        else _format_float(selected_zero_degree / float(selected_total)),
        "samples_all_predictions_zero_degree": samples_all_zero_degree,
        "special_token_predictions": special_counts,
        "top_30_predicted_drug_ids": top_predicted,
    }


def main() -> None:
    args = parse_args()
    eval_config = load_yaml_config(args.config)
    project_root = Path(eval_config["_project_root"]).resolve()
    checkpoint_path = resolve_path(project_root, args.checkpoint).resolve()
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

    path_args = argparse.Namespace(
        checkpoint=str(checkpoint_path),
        split=args.split,
        threshold=None,
        device=args.device,
        processed_root=args.processed_root,
        vocab_root=args.vocab_root,
        ddi_matrix_path=args.ddi_matrix_path,
    )
    resolved_paths = _resolve_eval_paths(
        project_root=project_root,
        eval_config=eval_config,
        train_config=train_config,
        data_config=data_config,
        checkpoint_payload=checkpoint_payload,
        args=path_args,
    )

    runtime_cfg = dict(eval_config.get("runtime", {}))
    batch_size = int(args.batch_size if args.batch_size is not None else runtime_cfg.get("batch_size", 32))
    device = resolve_device(str(args.device or runtime_cfg.get("device", "cpu")))

    med_vocab_path = resolved_paths["vocab_root"] / "med_vocab_main.json"
    if not med_vocab_path.exists():
        med_vocab_path = resolved_paths["vocab_root"] / "drug_vocab.json"
    vocab_payload = read_json(med_vocab_path)
    tokens = _vocab_tokens(vocab_payload)
    drug_vocab_size = int(vocab_payload["size"])
    pad_idx = vocab_payload.get("pad_idx")
    unk_idx = vocab_payload.get("unk_idx")

    ddi_matrix = load_ddi_matrix(resolved_paths["ddi_matrix_path"], device="cpu")

    print("ClinRec DDI eval debug")
    print(f"checkpoint_loaded_path: {checkpoint_path}")
    print(f"split: {args.split}")
    print(f"max_batches: {args.max_batches}")
    print(f"batch_size: {batch_size}")
    print(f"device: {device}")
    print(f"vocab_path: {med_vocab_path}")
    print(f"vocab_size: {drug_vocab_size}")
    print(f"ddi_matrix_path: {resolved_paths['ddi_matrix_path']}")

    _json_print("DDI matrix stats", _ddi_matrix_report(ddi_matrix, tokens))

    warnings: list[str] = []
    if int(ddi_matrix.shape[0]) != drug_vocab_size or int(ddi_matrix.shape[1]) != drug_vocab_size:
        warnings.append(
            f"DDI/vocab mismatch: ddi_shape={tuple(ddi_matrix.shape)} vocab_size={drug_vocab_size}"
        )
    if float((ddi_matrix > 0).sum().item()) <= 0.0:
        warnings.append("DDI matrix is all-zero.")
    if pad_idx is not None and int(pad_idx) < int(ddi_matrix.shape[0]) and bool((ddi_matrix[int(pad_idx)] > 0).any().item()):
        warnings.append(f"PAD row/col is non-zero: pad_idx={pad_idx}")
    if unk_idx is not None and int(unk_idx) < int(ddi_matrix.shape[0]) and bool((ddi_matrix[int(unk_idx)] > 0).any().item()):
        warnings.append(f"UNK row/col is non-zero: unk_idx={unk_idx}")

    temp_dir_manager = tempfile.TemporaryDirectory(prefix="clinrec_debug_ddi_")
    try:
        runtime_data_config_path = build_runtime_data_config_file(
            project_root=project_root,
            data_config=data_config,
            processed_root=resolved_paths["processed_root"],
            vocab_root=resolved_paths["vocab_root"],
            temp_dir=Path(temp_dir_manager.name),
        )
        dataloader = build_eval_dataloader(
            split=args.split,
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
            raise KeyError("Checkpoint does not contain model_state_dict")
        model.load_state_dict(model_state_dict, strict=True)

        outputs = _collect_core_outputs(
            model=model,
            dataloader=dataloader,
            device=device,
            max_batches=args.max_batches,
        )
    finally:
        temp_dir_manager.cleanup()
    probs = outputs["drug_probs"].detach().cpu()
    targets = outputs["targets"].detach().cpu()

    if int(probs.shape[1]) != drug_vocab_size:
        warnings.append(f"Prediction width/vocab mismatch: probs_width={int(probs.shape[1])} vocab_size={drug_vocab_size}")
    if int(targets.shape[1]) != drug_vocab_size:
        warnings.append(f"Target width/vocab mismatch: target_width={int(targets.shape[1])} vocab_size={drug_vocab_size}")
    if int(probs.shape[1]) != int(ddi_matrix.shape[0]):
        warnings.append(f"Prediction width/DDI mismatch: probs_width={int(probs.shape[1])} ddi_width={int(ddi_matrix.shape[0])}")

    prauc = compute_prauc(probs.numpy(), targets.numpy())

    target_ddi_summary = compute_ddi_rate(targets > 0, ddi_matrix)
    target_flags = compute_ddi_flags(targets > 0, ddi_matrix)
    _json_print(
        "Ground-truth target sanity",
        {
            "num_samples": int(targets.shape[0]),
            "target_width": int(targets.shape[1]),
            "avg_true_drugs": _format_float(compute_avg_true_drugs(targets)),
            "ddi_rate": _format_float(target_ddi_summary["ddi_rate"]),
            "total_target_pairs": int(target_ddi_summary["total_predicted_pairs"]),
            "total_interacting_pairs": int(target_ddi_summary["total_interacting_pairs"]),
            "samples_with_any_ddi_pair": int(target_flags.sum().item()),
            "pad_positive_count": int(targets[:, int(pad_idx)].sum().item())
            if pad_idx is not None and int(pad_idx) < int(targets.shape[1])
            else None,
            "unk_positive_count": int(targets[:, int(unk_idx)].sum().item())
            if unk_idx is not None and int(unk_idx) < int(targets.shape[1])
            else None,
        },
    )

    raw_binary = select_binary_predictions(probs, threshold=0.5, prediction_method="global").cpu()
    topk_binary = select_binary_predictions(probs, threshold=0.5, prediction_method="topk", top_k=args.top_k).cpu()
    _json_print(
        "Prediction stats raw threshold 0.5",
        _prediction_report(
            label="raw_threshold_0.5",
            binary=raw_binary,
            ddi_matrix=ddi_matrix,
            tokens=tokens,
            pad_idx=None if pad_idx is None else int(pad_idx),
            unk_idx=None if unk_idx is None else int(unk_idx),
        ),
    )
    _json_print(
        f"Prediction stats top-k {args.top_k}",
        _prediction_report(
            label=f"topk_{args.top_k}",
            binary=topk_binary,
            ddi_matrix=ddi_matrix,
            tokens=tokens,
            pad_idx=None if pad_idx is None else int(pad_idx),
            unk_idx=None if unk_idx is None else int(unk_idx),
        ),
    )

    mode_rows = [
        _metric_row(
            name="raw_threshold_0.5",
            y_true=targets,
            probs=probs,
            ddi_matrix=ddi_matrix,
            prauc=prauc,
            threshold=0.5,
            prediction_method="global",
        ),
        _metric_row(
            name=f"topk_{args.top_k}",
            y_true=targets,
            probs=probs,
            ddi_matrix=ddi_matrix,
            prauc=prauc,
            threshold=0.5,
            prediction_method="topk",
            top_k=args.top_k,
        ),
    ]
    _json_print("Decode mode comparison", mode_rows)

    sweep_rows = [
        _metric_row(
            name=f"threshold_{threshold:.2f}",
            y_true=targets,
            probs=probs,
            ddi_matrix=ddi_matrix,
            prauc=prauc,
            threshold=threshold,
            prediction_method="global",
        )
        for threshold in _threshold_values(args.threshold_start, args.threshold_end, args.threshold_step)
    ]
    _json_print("Threshold sweep", sweep_rows)

    if warnings:
        _json_print("Warnings", warnings)
    else:
        _json_print("Warnings", [])


if __name__ == "__main__":
    main()
