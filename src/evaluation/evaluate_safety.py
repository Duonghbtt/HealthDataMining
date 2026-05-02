from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch.utils.data import DataLoader

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - tqdm is optional at runtime
    class _NullTqdm:
        def __init__(self, iterable: Any = None, *args: Any, **kwargs: Any) -> None:
            self._iterable = iterable

        def __iter__(self):
            if self._iterable is None:
                return iter(())
            return iter(self._iterable)

        def set_postfix(self, *args: Any, **kwargs: Any) -> None:
            return None

        def set_postfix_str(self, *args: Any, **kwargs: Any) -> None:
            return None

        def update(self, *args: Any, **kwargs: Any) -> None:
            return None

        def close(self) -> None:
            return None

    def tqdm(iterable: Any = None, *args: Any, **kwargs: Any) -> _NullTqdm:  # type: ignore[no-redef]
        return _NullTqdm(iterable=iterable, *args, **kwargs)

    def _tqdm_write(message: str) -> None:
        print(message)

else:
    def _tqdm_write(message: str) -> None:
        tqdm.write(message)

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))
    from evaluate_core import (  # type: ignore[import-not-found]
        _collect_core_outputs,
        _flatten_report,
        _load_embedded_or_yaml_config,
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _resolve_retrieval_policy,
        _write_plain_csv,
        build_eval_dataloader,
    )
else:
    from .evaluate_core import (
        _collect_core_outputs,
        _flatten_report,
        _load_embedded_or_yaml_config,
        _resolve_checkpoint_path,
        _resolve_eval_paths,
        _resolve_retrieval_policy,
        _write_plain_csv,
        build_eval_dataloader,
    )

from src.evaluation.metrics import (
    compute_avg_predicted_drugs,
    compute_avg_true_drugs,
    compute_ddi_flags,
    compute_ddi_rate,
    compute_prauc,
    compute_samplewise_f1,
    compute_samplewise_jaccard,
    multilabel_f1,
    multilabel_jaccard,
)
from src.models.ddi_regularization import load_ddi_matrix, rerank_prediction_set
from src.training.runtime_builder import build_core_model, build_runtime_data_config_file, resolve_device
from src.utils.io import load_yaml_config, read_json, resolve_path, write_json


POLYPHARMACY_THRESHOLD = 5
HIGH_POLYPHARMACY_THRESHOLD = 10
GROUP_ORDER: tuple[str, ...] = ("all_visits", "first_visit", "not_first_visit", "short_history", "long_history")
COUNT_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("0-2", 0, 2),
    ("3-5", 3, 5),
    ("6-8", 6, 8),
    ("9+", 9, None),
)
VALID_DECODE_MODES = {"threshold", "topk", "soft_constrained_rerank"}
_SOFT_RERANK_PROGRESS_CHUNK_SIZE = 128


def _log_progress(message: str) -> None:
    _tqdm_write(message)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run safety-aware evaluation and decode-policy comparison.")
    parser.add_argument("--config", default="configs/eval.yaml", help="Path to configs/eval.yaml")
    parser.add_argument("--data-config", default=None, help="Optional override for configs/data.yaml")
    parser.add_argument("--model-config", default=None, help="Optional override for configs/model.yaml")
    parser.add_argument("--train-config", default=None, help="Optional override for configs/train.yaml")
    parser.add_argument("--checkpoint", default=None, help="Optional override for best checkpoint path")
    parser.add_argument("--split", default=None, help="Optional override for evaluation split")
    parser.add_argument("--threshold", type=float, default=None, help="Optional threshold override")
    parser.add_argument("--device", default=None, help="Optional override for runtime device")
    parser.add_argument("--processed-root", default=None, help="Optional override for processed data root")
    parser.add_argument("--vocab-root", default=None, help="Optional override for vocab directory")
    parser.add_argument("--ddi-matrix-path", default=None, help="Optional override for DDI matrix artifact")
    parser.add_argument("--smoke-test", action="store_true", help="Run a short smoke evaluation path")
    parser.add_argument("--max-eval-batches", type=int, default=None, help="Optional cap for evaluation batches")
    parser.add_argument("--decode-mode", default=None, choices=sorted(VALID_DECODE_MODES))
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-m", type=int, default=None)
    parser.add_argument("--alpha-utility", type=float, default=None)
    parser.add_argument("--beta-ddi", type=float, default=None)
    parser.add_argument("--gamma-size", type=float, default=None)
    parser.add_argument("--min-drugs", type=int, default=None)
    parser.add_argument("--max-drugs", type=int, default=None)
    parser.add_argument("--target-avg-drugs", type=float, default=None)
    return parser.parse_args()


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


def _has_cli_policy_override(args: argparse.Namespace) -> bool:
    return any(
        value is not None
        for value in (
            args.decode_mode,
            args.threshold,
            args.top_k,
            args.top_m,
            args.alpha_utility,
            args.beta_ddi,
            args.gamma_size,
            args.min_drugs,
            args.max_drugs,
            args.target_avg_drugs,
        )
    )


def _metadata_at(values: Sequence[int], index: int, default: int = -1) -> int:
    return int(values[index]) if index < len(values) else int(default)


def _safe_mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / float(len(values)))


def _resolve_target_avg_drugs(
    *,
    safety_cfg: Mapping[str, Any],
    cli_target_avg_drugs: float | None,
    val_outputs: Mapping[str, Any] | None,
    target_outputs: Mapping[str, Any],
) -> float | None:
    if cli_target_avg_drugs is not None:
        return float(cli_target_avg_drugs)
    if safety_cfg.get("target_avg_drugs") is not None:
        return float(safety_cfg["target_avg_drugs"])
    if val_outputs is not None:
        return float(compute_avg_true_drugs(torch.as_tensor(val_outputs["targets"], dtype=torch.float32)))
    return float(compute_avg_true_drugs(torch.as_tensor(target_outputs["targets"], dtype=torch.float32)))


def _resolve_policy_value(
    *,
    policy: Mapping[str, Any],
    default_cfg: Mapping[str, Any],
    key: str,
    fallback: Any = None,
) -> Any:
    value = policy.get(key)
    if value is not None:
        return value
    default_value = default_cfg.get(key)
    if default_value is not None:
        return default_value
    return fallback


def _optional_float(value: Any) -> float | None:
    return None if value is None else float(value)


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _resolve_progress_total(dataloader: DataLoader, max_batches: int | None) -> int | None:
    try:
        total_batches = int(len(dataloader))
    except TypeError:
        return None
    if max_batches is not None:
        total_batches = min(total_batches, int(max_batches))
    return total_batches


def _collect_core_outputs_with_progress(
    *,
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    split_name: str,
    max_batches: int | None = None,
) -> dict[str, Any]:
    progress = tqdm(
        dataloader,
        total=_resolve_progress_total(dataloader, max_batches),
        desc=f"Collecting safety predictions ({split_name})",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
    )
    try:
        return _collect_core_outputs(
            model=model,
            dataloader=progress,  # type: ignore[arg-type]
            device=device,
            max_batches=max_batches,
        )
    finally:
        progress.close()


def _normalize_safety_policy(
    *,
    policy: Mapping[str, Any],
    default_cfg: Mapping[str, Any],
    target_avg_drugs: float | None,
) -> dict[str, Any]:
    mode = str(policy.get("mode", default_cfg.get("mode", "threshold"))).strip().lower()
    if mode not in VALID_DECODE_MODES:
        raise ValueError(f"Unsupported safety decode mode: {mode!r}")

    threshold_value = float(
        _resolve_policy_value(
            policy=policy,
            default_cfg=default_cfg,
            key="threshold",
            fallback=0.5,
        )
    )
    top_k_value = _resolve_policy_value(policy=policy, default_cfg=default_cfg, key="top_k")
    top_m_value = int(
        _resolve_policy_value(
            policy=policy,
            default_cfg=default_cfg,
            key="top_m",
            fallback=20,
        )
    )
    alpha_utility = float(
        _resolve_policy_value(
            policy=policy,
            default_cfg=default_cfg,
            key="alpha_utility",
            fallback=1.0,
        )
    )
    beta_ddi = float(
        _resolve_policy_value(
            policy=policy,
            default_cfg=default_cfg,
            key="beta_ddi",
            fallback=0.5,
        )
    )
    gamma_size = float(
        _resolve_policy_value(
            policy=policy,
            default_cfg=default_cfg,
            key="gamma_size",
            fallback=0.1,
        )
    )
    min_drugs = int(
        _resolve_policy_value(
            policy=policy,
            default_cfg=default_cfg,
            key="min_drugs",
            fallback=1,
        )
    )
    max_drugs = _optional_int(
        _resolve_policy_value(
            policy=policy,
            default_cfg=default_cfg,
            key="max_drugs",
        )
    )
    resolved_target_avg = _optional_float(
        _resolve_policy_value(
            policy=policy,
            default_cfg=default_cfg,
            key="target_avg_drugs",
            fallback=target_avg_drugs,
        )
    )

    if mode == "topk":
        if top_k_value is None:
            raise ValueError("top_k must be provided for top-k decoding.")
        top_k_value = int(top_k_value)
        if top_k_value <= 0:
            raise ValueError(f"top_k must be positive, got {top_k_value!r}")
        label = f"topk:{top_k_value}"
    elif mode == "threshold":
        top_k_value = None
        label = f"threshold:{threshold_value:.2f}"
    else:
        top_k_value = None
        label = f"soft_constrained_rerank:beta={beta_ddi:.2f}:topm={top_m_value}"

    return {
        "mode": mode,
        "policy_name": label,
        "threshold": threshold_value,
        "top_k": top_k_value,
        "top_m": top_m_value,
        "alpha_utility": alpha_utility,
        "beta_ddi": beta_ddi,
        "gamma_size": gamma_size,
        "min_drugs": min_drugs,
        "max_drugs": max_drugs,
        "target_avg_drugs": resolved_target_avg,
    }


def _policy_from_cli(
    *,
    args: argparse.Namespace,
    default_cfg: Mapping[str, Any],
    target_avg_drugs: float | None,
) -> dict[str, Any]:
    return _normalize_safety_policy(
        policy={
            "mode": args.decode_mode or default_cfg.get("mode", "threshold"),
            "threshold": args.threshold if args.threshold is not None else default_cfg.get("threshold", 0.5),
            "top_k": args.top_k if args.top_k is not None else default_cfg.get("top_k"),
            "top_m": args.top_m if args.top_m is not None else default_cfg.get("top_m", 20),
            "alpha_utility": (
                args.alpha_utility if args.alpha_utility is not None else default_cfg.get("alpha_utility", 1.0)
            ),
            "beta_ddi": args.beta_ddi if args.beta_ddi is not None else default_cfg.get("beta_ddi", 0.5),
            "gamma_size": args.gamma_size if args.gamma_size is not None else default_cfg.get("gamma_size", 0.1),
            "min_drugs": args.min_drugs if args.min_drugs is not None else default_cfg.get("min_drugs", 1),
            "max_drugs": args.max_drugs if args.max_drugs is not None else default_cfg.get("max_drugs"),
            "target_avg_drugs": (
                args.target_avg_drugs
                if args.target_avg_drugs is not None
                else default_cfg.get("target_avg_drugs", target_avg_drugs)
            ),
        },
        default_cfg=default_cfg,
        target_avg_drugs=target_avg_drugs,
    )


def _build_policy_candidates(
    *,
    eval_config: Mapping[str, Any],
    args: argparse.Namespace,
    target_avg_drugs: float | None,
) -> tuple[list[dict[str, Any]], str]:
    safety_cfg = dict(eval_config.get("safety_decoding", {}))
    prediction_cfg = dict(eval_config.get("prediction", {}))
    if _has_cli_policy_override(args):
        return [_policy_from_cli(args=args, default_cfg=safety_cfg, target_avg_drugs=target_avg_drugs)], "cli_override"

    candidates: list[dict[str, Any]] = []
    base_threshold = float(prediction_cfg.get("threshold", 0.5) or 0.5)
    for threshold in safety_cfg.get("search_thresholds", [base_threshold]):
        candidates.append(
            _normalize_safety_policy(
                policy={"mode": "threshold", "threshold": float(threshold)},
                default_cfg=safety_cfg,
                target_avg_drugs=target_avg_drugs,
            )
        )
    for top_k in safety_cfg.get("search_topk_values", []):
        candidates.append(
            _normalize_safety_policy(
                policy={"mode": "topk", "top_k": int(top_k)},
                default_cfg=safety_cfg,
                target_avg_drugs=target_avg_drugs,
            )
        )
    for beta_ddi in safety_cfg.get("search_beta_values", []):
        candidates.append(
            _normalize_safety_policy(
                policy={"mode": "soft_constrained_rerank", "beta_ddi": float(beta_ddi)},
                default_cfg=safety_cfg,
                target_avg_drugs=target_avg_drugs,
            )
        )

    if not candidates:
        candidates.append(
            _normalize_safety_policy(
                policy={"mode": str(safety_cfg.get("mode", "threshold"))},
                default_cfg=safety_cfg,
                target_avg_drugs=target_avg_drugs,
            )
        )

    deduped: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in candidates:
        if candidate["policy_name"] in seen:
            continue
        seen.add(str(candidate["policy_name"]))
        deduped.append(candidate)
    return deduped, "config_search"


def _resolve_retrieval_summary(collected_outputs: Mapping[str, Any]) -> dict[str, float]:
    summary = {
        "avg_valid_candidates": 0.0,
        "avg_retrieved_score": 0.0,
        "fraction_with_retrieval_context": 0.0,
    }
    valid_candidate_counts = collected_outputs.get("retrieval_valid_candidate_counts")
    retrieved_scores = collected_outputs.get("retrieved_scores")
    retrieved_indices = collected_outputs.get("retrieved_indices")
    if isinstance(valid_candidate_counts, torch.Tensor) and valid_candidate_counts.numel() > 0:
        counts = valid_candidate_counts.to(dtype=torch.float32)
        summary["avg_valid_candidates"] = float(counts.mean().item())
        summary["fraction_with_retrieval_context"] = float((counts > 0).float().mean().item())
    if isinstance(retrieved_scores, torch.Tensor) and retrieved_scores.numel() > 0:
        scores = retrieved_scores.to(dtype=torch.float32)
        if isinstance(retrieved_indices, torch.Tensor) and retrieved_indices.shape == retrieved_scores.shape:
            valid_mask = retrieved_indices >= 0
        else:
            valid_mask = torch.ones_like(scores, dtype=torch.bool)
        if bool(valid_mask.any().item()):
            summary["avg_retrieved_score"] = float(scores[valid_mask].mean().item())
    return summary


def _compute_metric_bundle(
    *,
    y_true: torch.Tensor,
    drug_probs: torch.Tensor,
    y_pred_binary: torch.Tensor,
    ddi_matrix: torch.Tensor,
) -> dict[str, float]:
    ddi_summary = compute_ddi_rate(y_pred_binary, ddi_matrix)
    return {
        "jaccard": float(multilabel_jaccard(y_true, y_pred_binary)),
        "f1": float(multilabel_f1(y_true, y_pred_binary)),
        "prauc": float(compute_prauc(drug_probs.detach().cpu().numpy(), y_true.detach().cpu().numpy())),
        "ddi_rate": float(ddi_summary["ddi_rate"]),
        "total_predicted_pairs": float(ddi_summary["total_predicted_pairs"]),
        "total_interacting_pairs": float(ddi_summary["total_interacting_pairs"]),
        "patients_with_ddi": float(ddi_summary["patients_with_ddi"]),
        "num_samples": float(ddi_summary["num_samples"]),
        "avg_predicted_drugs": float(compute_avg_predicted_drugs(y_pred_binary)),
        "avg_true_drugs": float(compute_avg_true_drugs(y_true)),
    }


def _merge_decode_outputs(chunk_outputs: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not chunk_outputs:
        raise ValueError("Cannot merge empty decode outputs.")
    tensor_keys = (
        "prediction_mask",
        "selected_mask",
        "utility_score",
        "ddi_penalty",
        "size_penalty",
        "total_score",
        "ddi_rate_per_sample",
        "num_ddi_pairs",
        "num_predicted_drugs",
        "drug_probs",
    )
    merged: dict[str, Any] = {
        key: torch.cat([torch.as_tensor(output[key]) for output in chunk_outputs], dim=0)
        for key in tensor_keys
    }
    merged["selected_indices"] = [
        sample_indices
        for output in chunk_outputs
        for sample_indices in list(output["selected_indices"])
    ]
    merged["decode_mode"] = str(chunk_outputs[0]["decode_mode"])
    return merged


def _decode_policy_outputs(
    *,
    drug_probs: torch.Tensor,
    ddi_matrix: torch.Tensor,
    policy: Mapping[str, Any],
    split: str,
) -> dict[str, Any]:
    if str(policy["mode"]) != "soft_constrained_rerank":
        return rerank_prediction_set(
            drug_probs,
            ddi_matrix,
            input_is_logits=False,
            decode_mode=str(policy["mode"]),
            threshold=float(policy["threshold"]),
            top_k=policy.get("top_k"),
            top_m=int(policy["top_m"]),
            alpha_utility=float(policy["alpha_utility"]),
            beta_ddi=float(policy["beta_ddi"]),
            gamma_size=float(policy["gamma_size"]),
            min_drugs=int(policy["min_drugs"]),
            max_drugs=policy.get("max_drugs"),
            target_avg_drugs=policy.get("target_avg_drugs"),
        )

    batch_size = int(drug_probs.shape[0])
    if batch_size <= 0:
        raise ValueError("Cannot decode an empty batch of drug probabilities.")
    chunk_size = max(1, min(_SOFT_RERANK_PROGRESS_CHUNK_SIZE, batch_size))
    progress = tqdm(
        range(0, batch_size, chunk_size),
        total=((batch_size + chunk_size - 1) // chunk_size),
        desc=f"Soft safety rerank ({split})",
        unit="chunk",
        dynamic_ncols=True,
        leave=False,
    )
    chunk_outputs: list[dict[str, Any]] = []
    policy_name = str(policy["policy_name"])
    try:
        for start in progress:
            end = min(start + chunk_size, batch_size)
            progress.set_postfix_str(f"{policy_name} | samples={end}/{batch_size}")
            chunk_outputs.append(
                rerank_prediction_set(
                    drug_probs[start:end],
                    ddi_matrix,
                    input_is_logits=False,
                    decode_mode=str(policy["mode"]),
                    threshold=float(policy["threshold"]),
                    top_k=policy.get("top_k"),
                    top_m=int(policy["top_m"]),
                    alpha_utility=float(policy["alpha_utility"]),
                    beta_ddi=float(policy["beta_ddi"]),
                    gamma_size=float(policy["gamma_size"]),
                    min_drugs=int(policy["min_drugs"]),
                    max_drugs=policy.get("max_drugs"),
                    target_avg_drugs=policy.get("target_avg_drugs"),
                )
            )
    finally:
        progress.close()
    return _merge_decode_outputs(chunk_outputs)


def _evaluate_policy(
    *,
    collected_outputs: Mapping[str, Any],
    policy: Mapping[str, Any],
    ddi_matrix: torch.Tensor,
    split: str,
) -> dict[str, Any]:
    drug_probs = torch.as_tensor(collected_outputs["drug_probs"], dtype=torch.float32).cpu()
    targets = torch.as_tensor(collected_outputs["targets"], dtype=torch.float32).cpu()
    ddi_matrix_cpu = ddi_matrix.cpu()
    decode_output = _decode_policy_outputs(
        drug_probs=drug_probs,
        ddi_matrix=ddi_matrix_cpu,
        policy=policy,
        split=split,
    )
    prediction_mask = torch.as_tensor(decode_output["prediction_mask"], dtype=torch.float32).cpu()
    metric_bundle = _compute_metric_bundle(
        y_true=targets,
        drug_probs=drug_probs,
        y_pred_binary=prediction_mask,
        ddi_matrix=ddi_matrix_cpu,
    )
    sample_jaccard = compute_samplewise_jaccard(targets, prediction_mask).cpu()
    sample_f1 = compute_samplewise_f1(targets, prediction_mask).cpu()
    ddi_flags = compute_ddi_flags(prediction_mask, ddi_matrix_cpu).cpu()
    retrieval_summary = _resolve_retrieval_summary(collected_outputs)

    patient_ids = [int(value) for value in collected_outputs.get("patient_ids", [])]
    subject_ids = [int(value) for value in collected_outputs.get("subject_ids", [])]
    hadm_ids = [int(value) for value in collected_outputs.get("hadm_ids", [])]
    stay_ids = [int(value) for value in collected_outputs.get("stay_ids", [])]
    visit_index = [int(value) for value in collected_outputs.get("visit_index", [])]
    visit_position = [int(value) for value in collected_outputs.get("visit_position", [])]
    history_length = [int(value) for value in collected_outputs.get("history_length", [])]

    utility_scores = torch.as_tensor(decode_output["utility_score"], dtype=torch.float32).cpu()
    ddi_penalties = torch.as_tensor(decode_output["ddi_penalty"], dtype=torch.float32).cpu()
    size_penalties = torch.as_tensor(decode_output["size_penalty"], dtype=torch.float32).cpu()
    total_scores = torch.as_tensor(decode_output["total_score"], dtype=torch.float32).cpu()
    ddi_rates = torch.as_tensor(decode_output["ddi_rate_per_sample"], dtype=torch.float32).cpu()
    num_ddi_pairs = torch.as_tensor(decode_output["num_ddi_pairs"], dtype=torch.float32).cpu()
    num_predicted_drugs = torch.as_tensor(decode_output["num_predicted_drugs"], dtype=torch.float32).cpu()
    selected_indices = list(decode_output["selected_indices"])

    sample_rows: list[dict[str, Any]] = []
    for row_index in range(targets.shape[0]):
        sample_rows.append(
            {
                "split": split,
                "policy_name": str(policy["policy_name"]),
                "decode_mode": str(policy["mode"]),
                "patient_id": _metadata_at(patient_ids, row_index),
                "subject_id": _metadata_at(subject_ids, row_index),
                "hadm_id": _metadata_at(hadm_ids, row_index),
                "stay_id": _metadata_at(stay_ids, row_index),
                "visit_index": _metadata_at(visit_index, row_index),
                "visit_position": _metadata_at(visit_position, row_index),
                "history_length": _metadata_at(history_length, row_index),
                "true_count": int(targets[row_index].sum().item()),
                "pred_count": int(num_predicted_drugs[row_index].item()),
                "sample_jaccard": float(sample_jaccard[row_index].item()),
                "sample_f1": float(sample_f1[row_index].item()),
                "has_ddi": bool(ddi_flags[row_index].item()),
                "ddi_penalty": float(ddi_penalties[row_index].item()),
                "ddi_rate_per_sample": float(ddi_rates[row_index].item()),
                "num_ddi_pairs": float(num_ddi_pairs[row_index].item()),
                "utility_score": float(utility_scores[row_index].item()),
                "size_penalty": float(size_penalties[row_index].item()),
                "total_score": float(total_scores[row_index].item()),
                "predicted_drug_indices": json.dumps(selected_indices[row_index]),
            }
        )

    summary_row = {
        "split": split,
        "policy_name": str(policy["policy_name"]),
        "decode_mode": str(policy["mode"]),
        "threshold": float(policy["threshold"]),
        "top_k": policy.get("top_k"),
        "top_m": int(policy["top_m"]),
        "alpha_utility": float(policy["alpha_utility"]),
        "beta_ddi": float(policy["beta_ddi"]),
        "gamma_size": float(policy["gamma_size"]),
        "min_drugs": int(policy["min_drugs"]),
        "max_drugs": policy.get("max_drugs"),
        "target_avg_drugs": _optional_float(policy.get("target_avg_drugs")),
        "non_empty_fraction": float((prediction_mask.sum(dim=1) > 0).float().mean().item()),
        "avg_ddi_penalty": float(ddi_penalties.mean().item()),
        "avg_valid_candidates": float(retrieval_summary["avg_valid_candidates"]),
        "avg_retrieved_score": float(retrieval_summary["avg_retrieved_score"]),
        "fraction_with_retrieval_context": float(retrieval_summary["fraction_with_retrieval_context"]),
        **metric_bundle,
    }
    return {
        "policy": dict(policy),
        "summary_row": summary_row,
        "sample_rows": sample_rows,
        "prediction_mask": prediction_mask,
    }


def _evaluate_policy_candidates(
    *,
    collected_outputs: Mapping[str, Any],
    policies: Sequence[Mapping[str, Any]],
    ddi_matrix: torch.Tensor,
    split: str,
    progress_desc: str,
    summary_only: bool = False,
) -> list[Any]:
    progress = tqdm(
        policies,
        total=int(len(policies)),
        desc=progress_desc,
        unit="policy",
        dynamic_ncols=True,
        leave=False,
    )
    evaluated_rows: list[Any] = []
    try:
        for policy in progress:
            policy_name = str(policy["policy_name"])
            progress.set_postfix_str(policy_name)
            result = _evaluate_policy(
                collected_outputs=collected_outputs,
                policy=policy,
                ddi_matrix=ddi_matrix,
                split=split,
            )
            evaluated_rows.append(dict(result["summary_row"]) if summary_only else result)
    finally:
        progress.close()
    return evaluated_rows


def _select_policy(
    *,
    validation_rows: Sequence[Mapping[str, Any]],
    safety_cfg: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not validation_rows:
        raise ValueError("Cannot select a safety policy from an empty validation comparison.")
    selection_metric = str(safety_cfg.get("selection_metric", "jaccard")).strip().lower()
    if selection_metric not in {"jaccard", "f1"}:
        raise ValueError(f"safety_decoding.selection_metric must be 'jaccard' or 'f1', got {selection_metric!r}")
    selection_strategy = str(safety_cfg.get("selection_strategy", "maximize_metric")).strip().lower()
    max_ddi_rate = safety_cfg.get("max_ddi_rate")
    utility_floor_ratio = float(safety_cfg.get("utility_floor_relative_to_baseline", 0.98))

    baseline_candidates = [dict(row) for row in validation_rows if str(row.get("decode_mode")) == "threshold"]
    baseline_reference = baseline_candidates if baseline_candidates else [dict(row) for row in validation_rows]
    baseline_row = max(baseline_reference, key=lambda row: (float(row[selection_metric]), -float(row["ddi_rate"])))
    baseline_metric = float(baseline_row[selection_metric])

    filtered_rows = [dict(row) for row in validation_rows]
    filter_reason = "none"
    if selection_strategy == "maximize_metric_under_ddi_constraint":
        if max_ddi_rate is not None:
            under_constraint = [row for row in filtered_rows if float(row["ddi_rate"]) <= float(max_ddi_rate)]
            if under_constraint:
                filtered_rows = under_constraint
                filter_reason = f"ddi_rate<={float(max_ddi_rate):.4f}"
    elif selection_strategy == "minimize_ddi_with_metric_floor":
        metric_floor = baseline_metric * float(utility_floor_ratio)
        above_floor = [row for row in filtered_rows if float(row[selection_metric]) >= metric_floor]
        if above_floor:
            filtered_rows = above_floor
            filter_reason = f"{selection_metric}>={metric_floor:.4f}"
    elif selection_strategy != "maximize_metric":
        raise ValueError(
            "Unsupported safety_decoding.selection_strategy: "
            f"{selection_strategy!r}. Expected maximize_metric, "
            "maximize_metric_under_ddi_constraint, or minimize_ddi_with_metric_floor."
        )

    if selection_strategy == "minimize_ddi_with_metric_floor":
        selected = min(filtered_rows, key=lambda row: (float(row["ddi_rate"]), -float(row[selection_metric])))
    else:
        selected = max(filtered_rows, key=lambda row: (float(row[selection_metric]), -float(row["ddi_rate"])))

    return dict(selected), {
        "selection_metric": selection_metric,
        "selection_strategy": selection_strategy,
        "max_ddi_rate": None if max_ddi_rate is None else float(max_ddi_rate),
        "utility_floor_relative_to_baseline": float(utility_floor_ratio),
        "baseline_reference_policy": str(baseline_row["policy_name"]),
        "baseline_reference_metric": float(baseline_metric),
        "filter_reason": filter_reason,
    }


def _build_group_masks(collected_outputs: Mapping[str, Any]) -> dict[str, torch.Tensor]:
    num_samples = int(torch.as_tensor(collected_outputs["targets"]).shape[0])
    all_mask = torch.ones(num_samples, dtype=torch.bool)
    visit_position = torch.as_tensor(collected_outputs.get("visit_position", []), dtype=torch.long)
    history_length = torch.as_tensor(collected_outputs.get("history_length", []), dtype=torch.long)
    if int(visit_position.numel()) != num_samples or int(history_length.numel()) != num_samples:
        return {"all_visits": all_mask}
    return {
        "all_visits": all_mask,
        "first_visit": visit_position <= 1,
        "not_first_visit": visit_position > 1,
        "short_history": history_length <= 2,
        "long_history": history_length > 2,
    }


def _summarize_groups(
    *,
    collected_outputs: Mapping[str, Any],
    prediction_mask: torch.Tensor,
    ddi_matrix: torch.Tensor,
) -> list[dict[str, Any]]:
    targets = torch.as_tensor(collected_outputs["targets"], dtype=torch.float32).cpu()
    drug_probs = torch.as_tensor(collected_outputs["drug_probs"], dtype=torch.float32).cpu()
    masks = _build_group_masks(collected_outputs)
    group_rows: list[dict[str, Any]] = []
    for group_name in GROUP_ORDER:
        if group_name not in masks:
            continue
        mask = masks[group_name]
        if int(mask.sum().item()) <= 0:
            continue
        metrics = _compute_metric_bundle(
            y_true=targets[mask],
            drug_probs=drug_probs[mask],
            y_pred_binary=prediction_mask[mask],
            ddi_matrix=ddi_matrix.cpu(),
        )
        group_rows.append(
            {
                "group": group_name,
                "num_samples": int(mask.sum().item()),
                "jaccard": float(metrics["jaccard"]),
                "f1": float(metrics["f1"]),
                "prauc": float(metrics["prauc"]),
                "ddi_rate": float(metrics["ddi_rate"]),
                "avg_drugs": float(metrics["avg_predicted_drugs"]),
                "avg_true_drugs": float(metrics["avg_true_drugs"]),
            }
        )
    return group_rows


def _summarize_ddi_by_predicted_count(
    *,
    sample_rows: Sequence[Mapping[str, Any]],
    prediction_mask: torch.Tensor,
    targets: torch.Tensor,
    drug_probs: torch.Tensor,
    ddi_matrix: torch.Tensor,
) -> list[dict[str, Any]]:
    pred_counts = torch.as_tensor([int(row["pred_count"]) for row in sample_rows], dtype=torch.long)
    ddi_penalties = [float(row["ddi_penalty"]) for row in sample_rows]
    bucket_rows: list[dict[str, Any]] = []
    for label, lower, upper in COUNT_BUCKETS:
        mask = pred_counts >= int(lower) if upper is None else (pred_counts >= int(lower)) & (pred_counts <= int(upper))
        if int(mask.sum().item()) <= 0:
            continue
        metrics = _compute_metric_bundle(
            y_true=targets[mask],
            drug_probs=drug_probs[mask],
            y_pred_binary=prediction_mask[mask],
            ddi_matrix=ddi_matrix.cpu(),
        )
        selected_indices = torch.nonzero(mask, as_tuple=False).flatten().tolist()
        bucket_rows.append(
            {
                "predicted_drug_count_bucket": label,
                "num_samples": int(mask.sum().item()),
                "ddi_rate": float(metrics["ddi_rate"]),
                "avg_ddi_penalty": _safe_mean([ddi_penalties[index] for index in selected_indices]),
                "jaccard": float(metrics["jaccard"]),
                "f1": float(metrics["f1"]),
                "avg_drugs": float(metrics["avg_predicted_drugs"]),
            }
        )
    return bucket_rows


def build_patient_safety_rows(sample_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    patient_rows: list[dict[str, Any]] = []
    polypharmacy_count = 0.0
    high_polypharmacy_count = 0.0
    patient_ddi_count = 0.0
    for row in sample_rows:
        pred_count = int(row["pred_count"])
        has_ddi = bool(row["has_ddi"])
        is_polypharmacy = pred_count >= POLYPHARMACY_THRESHOLD
        is_high_polypharmacy = pred_count >= HIGH_POLYPHARMACY_THRESHOLD
        patient_rows.append(
            {
                "patient_id": int(row["patient_id"]),
                "subject_id": int(row["subject_id"]),
                "hadm_id": int(row["hadm_id"]),
                "stay_id": int(row["stay_id"]),
                "visit_index": int(row["visit_index"]),
                "visit_position": int(row["visit_position"]),
                "history_length": int(row["history_length"]),
                "true_count": int(row["true_count"]),
                "pred_count": pred_count,
                "sample_jaccard": float(row["sample_jaccard"]),
                "sample_f1": float(row["sample_f1"]),
                "has_ddi": has_ddi,
                "ddi_penalty": float(row["ddi_penalty"]),
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


def _print_policy_summary(title: str, rows: Sequence[Mapping[str, Any]]) -> None:
    print(title)
    for row in rows:
        print(
            f"  {str(row['policy_name'])}: "
            f"jaccard={float(row['jaccard']):.4f} "
            f"f1={float(row['f1']):.4f} "
            f"prauc={float(row['prauc']):.4f} "
            f"ddi_rate={float(row['ddi_rate']):.4f} "
            f"avg_drugs={float(row['avg_predicted_drugs']):.4f}"
        )


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
    run_cfg = dict(eval_config.get("run", {}))
    evaluation_cfg = dict(eval_config.get("evaluation", {}))
    safety_cfg = dict(eval_config.get("safety_decoding", {}))

    split = str(args.split or evaluation_cfg.get("split", "test"))
    device = resolve_device(args.device or runtime_cfg.get("device", "cpu"))
    batch_size = int(runtime_cfg.get("batch_size", 32))
    resolved_max_eval_batches = (
        int(args.max_eval_batches)
        if args.max_eval_batches is not None
        else int(run_cfg["max_eval_batches"])
        if run_cfg.get("max_eval_batches") is not None
        else 2
        if bool(args.smoke_test or run_cfg.get("smoke_test", False))
        else None
    )

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
    print(f"Evaluating safety split: {split}")
    print(f"Loading checkpoint: {checkpoint_path}")

    with tempfile.TemporaryDirectory(prefix="clinrec_safety_eval_") as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        runtime_data_config_path = build_runtime_data_config_file(
            project_root=project_root,
            data_config=data_config,
            processed_root=resolved_paths["processed_root"],
            vocab_root=resolved_paths["vocab_root"],
            temp_dir=temp_dir,
        )
        dataloader = build_eval_dataloader(
            split=split,
            runtime_data_config_path=runtime_data_config_path,
            processed_root=resolved_paths["processed_root"],
            drug_vocab_size=drug_vocab_size,
            batch_size=batch_size,
        )
        val_dataloader: DataLoader | None = None
        if split != "val":
            val_dataloader = build_eval_dataloader(
                split="val",
                runtime_data_config_path=runtime_data_config_path,
                processed_root=resolved_paths["processed_root"],
                drug_vocab_size=drug_vocab_size,
                batch_size=batch_size,
            )
        train_retrieval_dataloader: DataLoader | None = None
        model = build_core_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            processed_root=resolved_paths["processed_root"],
            vocab_root=resolved_paths["vocab_root"],
            ddi_matrix_path=resolved_paths["ddi_matrix_path"],
        )
        if bool(getattr(model, "use_retrieval", False)):
            train_retrieval_dataloader = build_eval_dataloader(
                split="train",
                runtime_data_config_path=runtime_data_config_path,
                processed_root=resolved_paths["processed_root"],
                drug_vocab_size=drug_vocab_size,
                batch_size=batch_size,
            )

    model_state_dict = checkpoint_payload.get("model_state_dict")
    if not isinstance(model_state_dict, Mapping):
        raise KeyError("Checkpoint does not contain `model_state_dict`.")
    model.load_state_dict(model_state_dict, strict=True)
    if bool(getattr(model, "use_retrieval", False)) and train_retrieval_dataloader is not None:
        retrieval_bank = model.refresh_retrieval_memory_bank(
            train_retrieval_dataloader,
            split_name="train",
            device=device,
        )
        if retrieval_bank is not None:
            print(
                "Refreshed retrieval memory bank for safety evaluation "
                f"(visits={retrieval_bank.num_visits}, history_mode={getattr(model, 'history_mode', 'self_only')})"
            )

    retrieval_policy = _resolve_retrieval_policy(model)
    print(
        "Retrieval policy: "
        f"absolute_time={bool(retrieval_policy.get('has_absolute_time', False))} "
        f"same_patient_future_blocked={bool(retrieval_policy.get('same_patient_future_blocked', False))} "
        f"cross_patient_absolute_temporal_filter="
        f"{bool(retrieval_policy.get('cross_patient_absolute_temporal_filter', False))}"
    )
    print(str(retrieval_policy.get("notes", "")))

    _log_progress(f"Collecting safety predictions on {split} split...")
    target_outputs = _collect_core_outputs_with_progress(
        model=model,
        dataloader=dataloader,
        device=device,
        split_name=split,
        max_batches=resolved_max_eval_batches,
    )
    val_outputs = None
    if val_dataloader is not None:
        _log_progress("Collecting safety predictions on val split...")
        val_outputs = _collect_core_outputs_with_progress(
            model=model,
            dataloader=val_dataloader,
            device=device,
            split_name="val",
            max_batches=resolved_max_eval_batches,
        )

    target_avg_drugs = _resolve_target_avg_drugs(
        safety_cfg=safety_cfg,
        cli_target_avg_drugs=args.target_avg_drugs,
        val_outputs=val_outputs,
        target_outputs=target_outputs,
    )
    _log_progress("Building safety policy candidates...")
    policy_candidates, candidate_source = _build_policy_candidates(
        eval_config=eval_config,
        args=args,
        target_avg_drugs=target_avg_drugs,
    )
    print(
        "Safety decoding candidates: "
        + ", ".join(str(candidate["policy_name"]) for candidate in policy_candidates)
        + f" ({candidate_source})"
    )

    _log_progress(f"Evaluating candidate policies on {split} split...")
    target_policy_results = _evaluate_policy_candidates(
        collected_outputs=target_outputs,
        policies=policy_candidates,
        ddi_matrix=ddi_matrix,
        split=split,
        progress_desc=f"Scoring safety policies ({split})",
        summary_only=False,
    )
    target_comparison_rows = [dict(result["summary_row"]) for result in target_policy_results]
    if val_outputs is not None:
        _log_progress("Evaluating candidate policies on validation split...")
    validation_policy_rows = (
        _evaluate_policy_candidates(
            collected_outputs=val_outputs,
            policies=policy_candidates,
            ddi_matrix=ddi_matrix,
            split="val",
            progress_desc="Scoring safety policies (val)",
            summary_only=True,
        )
        if val_outputs is not None
        else [dict(row) for row in target_comparison_rows]
    )

    selected_row, selection_summary = _select_policy(validation_rows=validation_policy_rows, safety_cfg=safety_cfg)
    selected_policy_name = str(selected_row["policy_name"])
    _log_progress(f"Running selected safety policy on {split} split using cached candidate evaluation...")
    selected_target_result = next(
        result for result in target_policy_results if str(result["summary_row"]["policy_name"]) == selected_policy_name
    )
    selected_prediction_mask = torch.as_tensor(selected_target_result["prediction_mask"], dtype=torch.float32).cpu()
    selected_sample_rows = list(selected_target_result["sample_rows"])
    patient_rows, rate_summary = build_patient_safety_rows(selected_sample_rows)
    subgroup_rows = _summarize_groups(
        collected_outputs=target_outputs,
        prediction_mask=selected_prediction_mask,
        ddi_matrix=ddi_matrix,
    )
    ddi_bucket_rows = _summarize_ddi_by_predicted_count(
        sample_rows=selected_sample_rows,
        prediction_mask=selected_prediction_mask,
        targets=torch.as_tensor(target_outputs["targets"], dtype=torch.float32).cpu(),
        drug_probs=torch.as_tensor(target_outputs["drug_probs"], dtype=torch.float32).cpu(),
        ddi_matrix=ddi_matrix,
    )

    diagnostics = {
        "avg_predicted_drugs": float(selected_row["avg_predicted_drugs"]),
        "avg_true_drugs": float(selected_row["avg_true_drugs"]),
        "avg_valid_candidates": float(selected_row["avg_valid_candidates"]),
        "avg_retrieved_score": float(selected_row["avg_retrieved_score"]),
        "fraction_with_retrieval_context": float(selected_row["fraction_with_retrieval_context"]),
        "non_empty_fraction": float(selected_row["non_empty_fraction"]),
        "target_avg_drugs_prior": _optional_float(target_avg_drugs),
    }
    safety_report: dict[str, Any] = {
        "split": split,
        "num_samples": int(torch.as_tensor(target_outputs["targets"]).shape[0]),
        "checkpoint_path": str(checkpoint_path),
        "device": str(device),
        "selection_source": "validation_split" if val_outputs is not None else "current_split",
        "selected_policy": dict(selected_row),
        "selection_summary": selection_summary,
        "safety_metrics": {
            "ddi_rate": float(selected_row["ddi_rate"]),
            "patients_with_ddi": float(selected_row["patients_with_ddi"]),
            "patients_with_ddi_ratio": float(rate_summary["patients_with_ddi_ratio"]),
            "polypharmacy_rate": float(rate_summary["polypharmacy_rate"]),
            "high_polypharmacy_rate": float(rate_summary["high_polypharmacy_rate"]),
            "avg_predicted_drugs": float(selected_row["avg_predicted_drugs"]),
            "avg_true_drugs": float(selected_row["avg_true_drugs"]),
            "jaccard": float(selected_row["jaccard"]),
            "f1": float(selected_row["f1"]),
            "prauc": float(selected_row["prauc"]),
            "avg_ddi_penalty": float(selected_row["avg_ddi_penalty"]),
        },
        "diagnostics": diagnostics,
        "retrieval_policy": retrieval_policy,
        "subgroup_metrics": subgroup_rows,
        "ddi_by_predicted_drug_count": ddi_bucket_rows,
        "warnings": build_safety_warnings(
            ddi_rate=float(selected_row["ddi_rate"]),
            avg_predicted_drugs=float(selected_row["avg_predicted_drugs"]),
        ),
        "artifacts": {},
    }
    print(
        "Selected safety policy: "
        f"{selected_policy_name} | jaccard={float(selected_row['jaccard']):.4f} "
        f"f1={float(selected_row['f1']):.4f} "
        f"prauc={float(selected_row['prauc']):.4f} "
        f"ddi_rate={float(selected_row['ddi_rate']):.4f} "
        f"avg_drugs={float(selected_row['avg_predicted_drugs']):.4f}"
    )
    _print_policy_summary("Safety policy comparison:", target_comparison_rows)

    save_reports = bool(evaluation_cfg.get("save_reports", True))
    save_predictions = bool(evaluation_cfg.get("save_predictions", True))
    report_stem = f"evaluate_safety_{split}"

    if save_reports:
        report_dir = resolved_paths["report_dir"]
        comparison_json_path = write_json(report_dir / f"{report_stem}_policy_comparison.json", target_comparison_rows)
        comparison_csv_path = _write_plain_csv(report_dir / f"{report_stem}_policy_comparison.csv", target_comparison_rows)
        val_rows_json_path = write_json(report_dir / f"{report_stem}_policy_selection_rows.json", validation_policy_rows)
        val_rows_csv_path = _write_plain_csv(report_dir / f"{report_stem}_policy_selection_rows.csv", validation_policy_rows)
        curve_json_path = write_json(report_dir / f"{report_stem}_accuracy_safety_curve.json", target_comparison_rows)
        curve_csv_path = _write_plain_csv(report_dir / f"{report_stem}_accuracy_safety_curve.csv", target_comparison_rows)
        subgroup_json_path = write_json(report_dir / f"{report_stem}_ddi_by_group.json", subgroup_rows)
        subgroup_csv_path = _write_plain_csv(report_dir / f"{report_stem}_ddi_by_group.csv", subgroup_rows)
        bucket_json_path = write_json(report_dir / f"{report_stem}_ddi_by_predicted_drug_count.json", ddi_bucket_rows)
        bucket_csv_path = _write_plain_csv(report_dir / f"{report_stem}_ddi_by_predicted_drug_count.csv", ddi_bucket_rows)
        retrieval_policy_json_path = write_json(report_dir / f"{report_stem}_retrieval_policy.json", retrieval_policy)
        json_path = write_json(report_dir / f"{report_stem}.json", safety_report)
        flat_report: dict[str, Any] = {}
        _flatten_report("", safety_report, flat_report)
        csv_path = _write_plain_csv(report_dir / f"{report_stem}.csv", [flat_report])
        safety_report["artifacts"].update(
            {
                "json": str(json_path),
                "csv": str(csv_path),
                "safety_policy_comparison_json": str(comparison_json_path),
                "safety_policy_comparison_csv": str(comparison_csv_path),
                "policy_selection_rows_json": str(val_rows_json_path),
                "policy_selection_rows_csv": str(val_rows_csv_path),
                "accuracy_safety_curve_json": str(curve_json_path),
                "accuracy_safety_curve_csv": str(curve_csv_path),
                "ddi_by_group_json": str(subgroup_json_path),
                "ddi_by_group_csv": str(subgroup_csv_path),
                "ddi_by_predicted_drug_count_json": str(bucket_json_path),
                "ddi_by_predicted_drug_count_csv": str(bucket_csv_path),
                "retrieval_policy_json": str(retrieval_policy_json_path),
            }
        )
        write_json(report_dir / f"{report_stem}.json", safety_report)

    if save_predictions:
        rows_path = _write_plain_csv(
            resolved_paths["prediction_dir"] / f"{report_stem}_patients.csv",
            patient_rows,
        )
        safety_report["artifacts"]["patients_csv"] = str(rows_path)
        sample_rows_path = _write_plain_csv(
            resolved_paths["prediction_dir"] / f"{report_stem}_samples.csv",
            selected_sample_rows,
        )
        safety_report["artifacts"]["samples_csv"] = str(sample_rows_path)
        if save_reports:
            write_json(resolved_paths["report_dir"] / f"{report_stem}.json", safety_report)

    print(json.dumps(safety_report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
