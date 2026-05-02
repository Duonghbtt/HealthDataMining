from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    def tqdm(iterable, *args, **kwargs):  # type: ignore[no-redef]
        return iterable

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from src.models.ddi_regularization import load_ddi_matrix
from src.training.runtime_builder import build_dataset
from src.utils.io import ensure_dir, load_yaml_config, read_json, resolve_path, write_json


COUNT_BUCKETS: tuple[tuple[str, int, int | None], ...] = (
    ("0", 0, 0),
    ("1-2", 1, 2),
    ("3-5", 3, 5),
    ("6-8", 6, 8),
    ("9-12", 9, 12),
    ("13-20", 13, 20),
    ("21+", 21, None),
)
SPLITS: tuple[str, ...] = ("train", "val", "test")
SPECIAL_DRUG_IDS = {0, 1}
DEFAULT_DRUG_REPRESENTATION = "med_vocab_main"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze medication targets, vocab coverage, and ground-truth DDI."
    )
    parser.add_argument("--config", default="configs/data.yaml", help="Path to configs/data.yaml")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional override for report directory. Defaults to outputs/reports under the project root.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=30,
        help="How many top medication rows to keep in the summary tables.",
    )
    return parser.parse_args()


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> Path:
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


def _resolve_report_dir(config: Mapping[str, Any], output_dir: str | None) -> Path:
    project_root = Path(config["_project_root"]).resolve()
    if output_dir:
        return ensure_dir(Path(resolve_path(project_root, output_dir)))
    return ensure_dir(project_root / "outputs" / "reports")


def _resolve_trajectory_root(config: Mapping[str, Any]) -> Path:
    paths_cfg = dict(config.get("paths", {}))
    trajectory_root = paths_cfg.get("trajectory_interim_root")
    if trajectory_root:
        return Path(resolve_path(config["_project_root"], trajectory_root)).resolve()
    interim_root = Path(resolve_path(config["_project_root"], paths_cfg["interim_root"])).resolve()
    return interim_root / "trajectories"


def _resolve_drug_representation(
    *,
    processed_metadata: Mapping[str, Any],
) -> str:
    return str(processed_metadata.get("drug_representation") or DEFAULT_DRUG_REPRESENTATION)


def _load_med_vocab_main(vocab_root: Path) -> tuple[Path, dict[str, Any]]:
    med_vocab_path = vocab_root / "med_vocab_main.json"
    if not med_vocab_path.exists():
        raise FileNotFoundError(
            f"Missing med_vocab_main artifact at {med_vocab_path}. "
            "Run build_vocab.py before analyze_medication_targets.py."
        )
    return med_vocab_path, dict(read_json(med_vocab_path))


def _bucket_label(count: int) -> str:
    resolved = int(count)
    for label, lower, upper in COUNT_BUCKETS:
        if resolved < lower:
            continue
        if upper is None or resolved <= upper:
            return label
    return COUNT_BUCKETS[-1][0]


def _empty_bucket_stats() -> dict[str, float]:
    return {
        "num_samples": 0.0,
        "samples_with_pairs": 0.0,
        "patients_with_ddi": 0.0,
        "total_predicted_pairs": 0.0,
        "total_interacting_pairs": 0.0,
        "sum_true_drugs": 0.0,
    }


def _init_split_state() -> dict[str, Any]:
    return {
        "num_visits": 0,
        "target_counts": [],
        "empty_visits": 0,
        "single_drug_visits": 0,
        "visits_over_20": 0,
        "visits_over_30": 0,
        "unk_positive_count": 0,
        "visits_with_unk": 0,
        "pad_positive_count": 0,
        "visits_with_pad": 0,
        "out_of_vocab_target_count": 0,
        "visits_with_out_of_vocab_targets": 0,
        "top_drug_counter": Counter(),
        "ddi_drug_counter": Counter(),
        "count_buckets": Counter(),
        "ddi": {
            "samples_with_pairs": 0.0,
            "patients_with_ddi": 0.0,
            "total_predicted_pairs": 0.0,
            "total_interacting_pairs": 0.0,
        },
        "bucket_ddi": {label: _empty_bucket_stats() for label, _, _ in COUNT_BUCKETS},
    }


def _merge_split_states(states: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    merged = _init_split_state()
    for state in states:
        merged["num_visits"] += int(state["num_visits"])
        merged["target_counts"].extend(int(value) for value in state["target_counts"])
        merged["empty_visits"] += int(state["empty_visits"])
        merged["single_drug_visits"] += int(state["single_drug_visits"])
        merged["visits_over_20"] += int(state["visits_over_20"])
        merged["visits_over_30"] += int(state["visits_over_30"])
        merged["unk_positive_count"] += int(state["unk_positive_count"])
        merged["visits_with_unk"] += int(state["visits_with_unk"])
        merged["pad_positive_count"] += int(state["pad_positive_count"])
        merged["visits_with_pad"] += int(state["visits_with_pad"])
        merged["out_of_vocab_target_count"] += int(state["out_of_vocab_target_count"])
        merged["visits_with_out_of_vocab_targets"] += int(state["visits_with_out_of_vocab_targets"])
        merged["top_drug_counter"].update(state["top_drug_counter"])
        merged["ddi_drug_counter"].update(state["ddi_drug_counter"])
        merged["count_buckets"].update(state["count_buckets"])
        merged["ddi"]["patients_with_ddi"] += float(state["ddi"]["patients_with_ddi"])
        merged["ddi"]["samples_with_pairs"] += float(state["ddi"]["samples_with_pairs"])
        merged["ddi"]["total_predicted_pairs"] += float(state["ddi"]["total_predicted_pairs"])
        merged["ddi"]["total_interacting_pairs"] += float(state["ddi"]["total_interacting_pairs"])
        for label in merged["bucket_ddi"]:
            merged_bucket = merged["bucket_ddi"][label]
            source_bucket = state["bucket_ddi"][label]
            for key in merged_bucket:
                merged_bucket[key] += float(source_bucket[key])
    return merged


def _resolve_ddi_upper(ddi_matrix: torch.Tensor) -> torch.Tensor:
    matrix = torch.as_tensor(ddi_matrix, dtype=torch.float32).cpu()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"ddi_matrix must be square, got {tuple(matrix.shape)}")
    ddi_bool = (matrix > 0).to(dtype=torch.bool)
    ddi_bool = torch.logical_or(ddi_bool, ddi_bool.transpose(0, 1))
    ddi_bool.fill_diagonal_(False)
    return torch.triu(ddi_bool, diagonal=1)


def _safe_ratio(numerator: float, denominator: float) -> float:
    return 0.0 if float(denominator) <= 0.0 else float(numerator) / float(denominator)


def _resolve_quantile(counts: Sequence[int], quantile: float) -> float:
    if not counts:
        return 0.0
    tensor = torch.as_tensor(counts, dtype=torch.float32)
    return float(torch.quantile(tensor, torch.tensor(float(quantile), dtype=torch.float32)).item())


def _count_summary(counts: Sequence[int]) -> dict[str, float]:
    if not counts:
        return {"mean": 0.0, "median": 0.0, "p90": 0.0, "p95": 0.0, "max": 0.0}
    tensor = torch.as_tensor(counts, dtype=torch.float32)
    return {
        "mean": float(tensor.mean().item()),
        "median": float(torch.median(tensor).item()),
        "p90": _resolve_quantile(counts, 0.90),
        "p95": _resolve_quantile(counts, 0.95),
        "max": float(tensor.max().item()),
    }


def _iter_target_lists(record: Mapping[str, Any]) -> Iterable[list[int]]:
    steps = record.get("steps")
    if isinstance(steps, list):
        for step in steps:
            yield [int(value) for value in step.get("target_drugs", []) if value is not None]
        return

    target_matrix = record.get("target_drugs")
    visit_mask = record.get("visit_mask")
    if not isinstance(target_matrix, torch.Tensor) or not isinstance(visit_mask, torch.Tensor):
        raise TypeError("Record must contain either `steps` or tensorized `target_drugs` and `visit_mask`.")
    if target_matrix.ndim != 2:
        raise ValueError(f"Tensorized target_drugs must have shape (T, D), got {tuple(target_matrix.shape)}")
    if visit_mask.ndim != 1:
        raise ValueError(f"Tensorized visit_mask must have shape (T,), got {tuple(visit_mask.shape)}")

    valid_steps = int(min(target_matrix.shape[0], visit_mask.shape[0]))
    for step_index in range(valid_steps):
        if not bool(visit_mask[step_index].item()):
            continue
        yield torch.nonzero(target_matrix[step_index] > 0, as_tuple=False).flatten().tolist()


def _update_split_state(
    *,
    state: dict[str, Any],
    raw_target_ids: Sequence[int],
    ddi_upper: torch.Tensor,
    drug_vocab_size: int,
) -> None:
    raw_ids = [int(value) for value in raw_target_ids]
    state["num_visits"] += 1
    invalid_ids = [value for value in raw_ids if value < 0 or value >= int(drug_vocab_size)]
    if invalid_ids:
        state["out_of_vocab_target_count"] += len(invalid_ids)
        state["visits_with_out_of_vocab_targets"] += 1

    valid_ids = sorted({value for value in raw_ids if 0 <= value < int(drug_vocab_size)})
    target_count = int(len(valid_ids))
    state["target_counts"].append(target_count)
    state["count_buckets"][_bucket_label(target_count)] += 1
    if target_count <= 0:
        state["empty_visits"] += 1
    if target_count == 1:
        state["single_drug_visits"] += 1
    if target_count > 20:
        state["visits_over_20"] += 1
    if target_count > 30:
        state["visits_over_30"] += 1
    if 1 in valid_ids:
        state["unk_positive_count"] += 1
        state["visits_with_unk"] += 1
    if 0 in valid_ids:
        state["pad_positive_count"] += 1
        state["visits_with_pad"] += 1

    state["top_drug_counter"].update(value for value in valid_ids if value not in SPECIAL_DRUG_IDS)

    bucket_stats = state["bucket_ddi"][_bucket_label(target_count)]
    bucket_stats["num_samples"] += 1.0
    bucket_stats["sum_true_drugs"] += float(target_count)

    total_pairs = 0.0
    interacting_pairs = 0.0
    if target_count >= 2:
        state["ddi"]["samples_with_pairs"] += 1.0
        bucket_stats["samples_with_pairs"] += 1.0
        index_tensor = torch.as_tensor(valid_ids, dtype=torch.long)
        sample_ddi = ddi_upper.index_select(0, index_tensor).index_select(1, index_tensor)
        total_pairs = float(target_count * (target_count - 1) // 2)
        interacting_pairs = float(sample_ddi.sum(dtype=torch.float32).item())
        if interacting_pairs > 0.0:
            state["ddi"]["patients_with_ddi"] += 1.0
            bucket_stats["patients_with_ddi"] += 1.0
            per_drug_participation = sample_ddi.sum(dim=0, dtype=torch.float32) + sample_ddi.sum(
                dim=1,
                dtype=torch.float32,
            )
            for local_index, participation in enumerate(per_drug_participation.tolist()):
                global_id = int(valid_ids[local_index])
                if global_id in SPECIAL_DRUG_IDS or participation <= 0.0:
                    continue
                state["ddi_drug_counter"][global_id] += int(participation)

    state["ddi"]["total_predicted_pairs"] += total_pairs
    state["ddi"]["total_interacting_pairs"] += interacting_pairs
    bucket_stats["total_predicted_pairs"] += total_pairs
    bucket_stats["total_interacting_pairs"] += interacting_pairs


def _finalize_ddi_summary(state: Mapping[str, Any]) -> dict[str, float]:
    num_samples = float(state["num_visits"])
    samples_with_pairs = float(state["ddi"]["samples_with_pairs"])
    total_predicted_pairs = float(state["ddi"]["total_predicted_pairs"])
    total_interacting_pairs = float(state["ddi"]["total_interacting_pairs"])
    samples_with_ddi = float(state["ddi"]["patients_with_ddi"])
    return {
        "num_samples": num_samples,
        "samples_with_pairs": samples_with_pairs,
        "samples_with_pairs_ratio": _safe_ratio(samples_with_pairs, num_samples),
        "samples_with_ddi": samples_with_ddi,
        "samples_with_ddi_ratio": _safe_ratio(samples_with_ddi, num_samples),
        "samples_with_ddi_given_pairs_ratio": _safe_ratio(samples_with_ddi, samples_with_pairs),
        "patients_with_ddi": samples_with_ddi,
        "patients_with_ddi_ratio": _safe_ratio(samples_with_ddi, num_samples),
        "total_predicted_pairs": total_predicted_pairs,
        "total_interacting_pairs": total_interacting_pairs,
        "ddi_rate": _safe_ratio(total_interacting_pairs, total_predicted_pairs),
        "avg_true_drugs": _count_summary(state["target_counts"])["mean"],
    }


def _finalize_integrity_summary(state: Mapping[str, Any]) -> dict[str, float]:
    num_visits = float(state["num_visits"])
    return {
        "num_visits": num_visits,
        "empty_visits": float(state["empty_visits"]),
        "empty_visit_rate": _safe_ratio(float(state["empty_visits"]), num_visits),
        "single_drug_visits": float(state["single_drug_visits"]),
        "single_drug_visit_rate": _safe_ratio(float(state["single_drug_visits"]), num_visits),
        "visits_over_20": float(state["visits_over_20"]),
        "visit_rate_over_20": _safe_ratio(float(state["visits_over_20"]), num_visits),
        "visits_over_30": float(state["visits_over_30"]),
        "visit_rate_over_30": _safe_ratio(float(state["visits_over_30"]), num_visits),
        "unk_positive_count": float(state["unk_positive_count"]),
        "unk_present_in_targets": bool(state["unk_positive_count"] > 0),
        "visits_with_unk": float(state["visits_with_unk"]),
        "pad_positive_count": float(state["pad_positive_count"]),
        "visits_with_pad": float(state["visits_with_pad"]),
        "out_of_vocab_target_count": float(state["out_of_vocab_target_count"]),
        "visits_with_out_of_vocab_targets": float(state["visits_with_out_of_vocab_targets"]),
        "has_out_of_vocab_targets": bool(state["out_of_vocab_target_count"] > 0),
    }


def _distribution_rows(split: str, state: Mapping[str, Any]) -> list[dict[str, Any]]:
    total_visits = float(state["num_visits"])
    rows: list[dict[str, Any]] = []
    for label, _, _ in COUNT_BUCKETS:
        bucket_count = int(state["count_buckets"].get(label, 0))
        rows.append(
            {
                "split": split,
                "bucket": label,
                "num_visits": bucket_count,
                "fraction": _safe_ratio(bucket_count, total_visits),
            }
        )
    return rows


def _bucket_ddi_rows(split: str, state: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for label, _, _ in COUNT_BUCKETS:
        bucket_stats = state["bucket_ddi"][label]
        num_samples = float(bucket_stats["num_samples"])
        samples_with_pairs = float(bucket_stats["samples_with_pairs"])
        total_pairs = float(bucket_stats["total_predicted_pairs"])
        total_interacting_pairs = float(bucket_stats["total_interacting_pairs"])
        rows.append(
            {
                "split": split,
                "bucket": label,
                "num_samples": int(num_samples),
                "samples_with_pairs": int(samples_with_pairs),
                "samples_with_pairs_ratio": _safe_ratio(samples_with_pairs, num_samples),
                "avg_true_drugs": _safe_ratio(float(bucket_stats["sum_true_drugs"]), num_samples),
                "samples_with_ddi": float(bucket_stats["patients_with_ddi"]),
                "samples_with_ddi_ratio": _safe_ratio(float(bucket_stats["patients_with_ddi"]), num_samples),
                "samples_with_ddi_given_pairs_ratio": _safe_ratio(float(bucket_stats["patients_with_ddi"]), samples_with_pairs),
                "patients_with_ddi": float(bucket_stats["patients_with_ddi"]),
                "patients_with_ddi_ratio": _safe_ratio(float(bucket_stats["patients_with_ddi"]), num_samples),
                "total_predicted_pairs": total_pairs,
                "total_interacting_pairs": total_interacting_pairs,
                "ddi_rate": _safe_ratio(total_interacting_pairs, total_pairs),
            }
        )
    return rows


def _top_counter_rows(
    *,
    split: str,
    counter: Counter[int],
    idx_to_token: Sequence[str],
    top_k: int,
    value_key: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, (drug_id, value) in enumerate(counter.most_common(int(top_k)), start=1):
        rows.append(
            {
                "split": split,
                "rank": rank,
                "drug_id": int(drug_id),
                "drug_token": str(idx_to_token[int(drug_id)]),
                value_key: int(value),
            }
        )
    return rows


def _ddi_degree_rows(ddi_upper: torch.Tensor, idx_to_token: Sequence[str], top_k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    real_ids = list(range(2, int(ddi_upper.shape[0])))
    symmetric = torch.logical_or(ddi_upper, ddi_upper.transpose(0, 1))
    degrees = symmetric.sum(dim=1, dtype=torch.int32)
    rows: list[dict[str, Any]] = []
    for drug_id in real_ids:
        degree = int(degrees[drug_id].item())
        rows.append(
            {
                "drug_id": drug_id,
                "drug_token": str(idx_to_token[drug_id]),
                "degree": degree,
                "has_ddi_edge": bool(degree > 0),
            }
        )
    top_rows = sorted(rows, key=lambda row: (-int(row["degree"]), str(row["drug_token"])))[: int(top_k)]
    return rows, top_rows


def _load_mapping_diagnostics(trajectory_root: Path) -> dict[str, Any] | None:
    candidate_paths = (
        trajectory_root / "medication_target_mapping_diagnostics.json",
        trajectory_root / "trajectory_summary.json",
    )
    for diagnostics_path in candidate_paths:
        if diagnostics_path.exists():
            payload = dict(read_json(diagnostics_path))
            payload["path"] = str(diagnostics_path)
            return payload
    return None


def _extract_model_target_comparison(report_dir: Path) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for prefix in ("evaluate_core_", "evaluate_safety_"):
        candidates = sorted(report_dir.glob(f"{prefix}*.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if not candidates:
            continue
        payload = dict(read_json(candidates[0]))
        metrics = payload.get("metrics") or payload.get("safety_metrics")
        if not isinstance(metrics, Mapping):
            continue
        comparison[candidates[0].name] = {
            "avg_true_drugs": float(metrics.get("avg_true_drugs", 0.0)),
            "avg_predicted_drugs": float(metrics.get("avg_predicted_drugs", 0.0)),
            "path": str(candidates[0]),
        }
    return comparison


def _analyze_split(
    *,
    split: str,
    config_path: Path,
    processed_root: Path,
    drug_vocab_size: int,
    ddi_upper: torch.Tensor,
) -> dict[str, Any]:
    dataset = build_dataset(
        split=split,
        runtime_data_config_path=config_path,
        processed_root=processed_root,
        drug_vocab_size=drug_vocab_size,
    )
    state = _init_split_state()
    for record_index in tqdm(range(len(dataset)), desc=f"Analyze medication targets ({split})", unit="record"):
        record = dataset[record_index]
        for target_ids in _iter_target_lists(record):
            _update_split_state(
                state=state,
                raw_target_ids=target_ids,
                ddi_upper=ddi_upper,
                drug_vocab_size=drug_vocab_size,
            )
    return state


def main() -> None:
    args = parse_args()
    config = load_yaml_config(args.config)
    project_root = Path(config["_project_root"]).resolve()
    processed_root = Path(resolve_path(project_root, config["paths"]["processed_root"])).resolve()
    trajectory_root = _resolve_trajectory_root(config)
    vocab_root = Path(resolve_path(project_root, config["paths"]["vocab_root"])).resolve()
    ddi_matrix_path = Path(resolve_path(project_root, config["paths"]["ddi_root"])) / "drug_ddi.pt"
    report_dir = _resolve_report_dir(config, args.output_dir)
    processed_metadata_path = trajectory_root / "metadata.json"
    processed_metadata = dict(read_json(processed_metadata_path)) if processed_metadata_path.exists() else {}
    drug_representation = _resolve_drug_representation(processed_metadata=processed_metadata)

    resolved_vocab_path, med_vocab = _load_med_vocab_main(vocab_root)
    idx_to_token = [str(token) for token in med_vocab["idx_to_token"]]
    drug_vocab_size = int(med_vocab["size"])
    ddi_matrix = load_ddi_matrix(ddi_matrix_path, device="cpu")
    ddi_upper = _resolve_ddi_upper(ddi_matrix)
    if int(ddi_upper.shape[0]) != drug_vocab_size:
        raise ValueError(
            "DDI matrix width must match drug vocab size: "
            f"ddi={int(ddi_upper.shape[0])}, vocab={drug_vocab_size}"
        )

    print("Running medication target diagnostics...")
    print(f"  config: {args.config}")
    print(f"  processed_root: {processed_root}")
    print(f"  trajectory_root: {trajectory_root}")
    print(f"  vocab_root: {vocab_root}")
    print(f"  med_vocab_path: {resolved_vocab_path}")
    print(f"  ddi_matrix_path: {ddi_matrix_path}")
    print(f"  report_dir: {report_dir}")
    print(f"  drug_representation: {drug_representation}")

    split_states: dict[str, dict[str, Any]] = {}
    for split in SPLITS:
        split_states[split] = _analyze_split(
            split=split,
            config_path=Path(args.config),
            processed_root=processed_root,
            drug_vocab_size=drug_vocab_size,
            ddi_upper=ddi_upper,
        )

    overall_state = _merge_split_states(list(split_states.values()))
    distribution_rows: list[dict[str, Any]] = []
    bucket_ddi_rows: list[dict[str, Any]] = []
    top_drug_rows: list[dict[str, Any]] = []
    top_ddi_rows: list[dict[str, Any]] = []
    count_distribution_summary: dict[str, Any] = {"overall": _count_summary(overall_state["target_counts"]), "per_split": {}}
    integrity_summary: dict[str, Any] = {"overall": _finalize_integrity_summary(overall_state), "per_split": {}}
    ground_truth_ddi_summary: dict[str, Any] = {"overall": _finalize_ddi_summary(overall_state), "per_split": {}}

    for split in SPLITS:
        state = split_states[split]
        distribution_rows.extend(_distribution_rows(split, state))
        bucket_ddi_rows.extend(_bucket_ddi_rows(split, state))
        count_distribution_summary["per_split"][split] = _count_summary(state["target_counts"])
        integrity_summary["per_split"][split] = _finalize_integrity_summary(state)
        ground_truth_ddi_summary["per_split"][split] = _finalize_ddi_summary(state)
        top_drug_rows.extend(
            _top_counter_rows(
                split=split,
                counter=state["top_drug_counter"],
                idx_to_token=idx_to_token,
                top_k=int(args.top_k),
                value_key="frequency",
            )
        )
        top_ddi_rows.extend(
            _top_counter_rows(
                split=split,
                counter=state["ddi_drug_counter"],
                idx_to_token=idx_to_token,
                top_k=int(args.top_k),
                value_key="ddi_pair_participation",
            )
        )

    distribution_rows.extend(_distribution_rows("overall", overall_state))
    bucket_ddi_rows.extend(_bucket_ddi_rows("overall", overall_state))
    top_drug_rows.extend(
        _top_counter_rows(
            split="overall",
            counter=overall_state["top_drug_counter"],
            idx_to_token=idx_to_token,
            top_k=int(args.top_k),
            value_key="frequency",
        )
    )
    top_ddi_rows.extend(
        _top_counter_rows(
            split="overall",
            counter=overall_state["ddi_drug_counter"],
            idx_to_token=idx_to_token,
            top_k=int(args.top_k),
            value_key="ddi_pair_participation",
        )
    )

    possible_pairs_real = float(max((drug_vocab_size - 2) * (drug_vocab_size - 3) // 2, 0))
    ddi_interacting_pairs = float(ddi_upper[2:, 2:].sum(dtype=torch.float32).item()) if drug_vocab_size > 2 else 0.0
    degree_rows, top_degree_rows = _ddi_degree_rows(ddi_upper, idx_to_token, top_k=int(args.top_k))
    zero_degree_rows = [row for row in degree_rows if int(row["degree"]) <= 0]
    degree_summary = _count_summary([int(row["degree"]) for row in degree_rows])

    mapping_diagnostics = _load_mapping_diagnostics(trajectory_root)
    warnings: list[str] = []
    if mapping_diagnostics is None:
        warnings.append("trajectory_summary_missing")
        print(
            "WARNING: trajectory summary metadata is missing. "
            "Re-run build_trajectories.py to persist trajectory diagnostics."
        )

    model_target_comparison = _extract_model_target_comparison(report_dir)
    generated_at = datetime.now(timezone.utc).astimezone().isoformat()
    ground_truth_ddi_summary = {
        "drug_representation": drug_representation,
        **ground_truth_ddi_summary,
    }

    summary_payload: dict[str, Any] = {
        "generated_at": generated_at,
        "config_path": str(Path(args.config).resolve()),
        "processed_root": str(processed_root),
        "trajectory_root": str(trajectory_root),
        "vocab_root": str(vocab_root),
        "med_vocab_path": str(resolved_vocab_path),
        "ddi_matrix_path": str(ddi_matrix_path),
        "drug_representation": drug_representation,
        "medication_target_source_tables": list(processed_metadata.get("medication_target_source_tables", [])),
        "trajectory_metadata_path": str(processed_metadata_path),
        "mapping_coverage": mapping_diagnostics,
        "target_count_distribution": count_distribution_summary,
        "target_count_distribution_buckets": {
            split: [row for row in distribution_rows if row["split"] == split]
            for split in (*SPLITS, "overall")
        },
        "ground_truth_ddi": ground_truth_ddi_summary,
        "ground_truth_ddi_by_bucket": {
            split: [row for row in bucket_ddi_rows if row["split"] == split]
            for split in (*SPLITS, "overall")
        },
        "ddi_matrix_coverage": {
            "vocab_size_total": drug_vocab_size,
            "num_real_medication_labels": max(drug_vocab_size - 2, 0),
            "num_possible_pairs_real_labels": possible_pairs_real,
            "num_interacting_pairs_real_labels": ddi_interacting_pairs,
            "ddi_density_real_labels": _safe_ratio(ddi_interacting_pairs, possible_pairs_real),
            "num_zero_degree_real_labels": len(zero_degree_rows),
            "degree_distribution_real_labels": degree_summary,
            "top_degree_drugs": top_degree_rows,
        },
        "ddi_rate_definition": {
            "ddi_rate": "pair_based_ratio = total_interacting_pairs / total_predicted_pairs",
            "samples_with_ddi_ratio": "sample_based_ratio = samples_with_ddi / num_samples",
            "samples_with_ddi_given_pairs_ratio": "sample_based_ratio among visits with at least one possible drug pair",
        },
        "target_integrity": integrity_summary,
        "top_drugs_by_split": {
            split: [row for row in top_drug_rows if row["split"] == split]
            for split in (*SPLITS, "overall")
        },
        "top_ddi_participating_drugs_by_split": {
            split: [row for row in top_ddi_rows if row["split"] == split]
            for split in (*SPLITS, "overall")
        },
        "model_vs_target_comparison": model_target_comparison,
        "warnings": warnings,
        "artifacts": {},
    }

    diagnostics_json_path = write_json(report_dir / "medication_target_diagnostics.json", summary_payload)
    count_distribution_csv_path = _write_csv(report_dir / "medication_target_count_distribution.csv", distribution_rows)
    ground_truth_ddi_json_path = write_json(report_dir / "ground_truth_ddi_summary.json", ground_truth_ddi_summary)
    ground_truth_ddi_bucket_csv_path = _write_csv(report_dir / "ground_truth_ddi_by_bucket.csv", bucket_ddi_rows)
    ddi_degree_csv_path = _write_csv(report_dir / "ddi_vocab_degree_stats.csv", degree_rows)
    top_drug_csv_path = _write_csv(report_dir / "ground_truth_top_drugs_by_split.csv", top_drug_rows)
    top_ddi_csv_path = _write_csv(report_dir / "ground_truth_top_ddi_drugs.csv", top_ddi_rows)

    summary_payload["artifacts"] = {
        "medication_target_diagnostics_json": str(diagnostics_json_path),
        "medication_target_count_distribution_csv": str(count_distribution_csv_path),
        "ground_truth_ddi_summary_json": str(ground_truth_ddi_json_path),
        "ground_truth_ddi_by_bucket_csv": str(ground_truth_ddi_bucket_csv_path),
        "ddi_vocab_degree_stats_csv": str(ddi_degree_csv_path),
        "ground_truth_top_drugs_by_split_csv": str(top_drug_csv_path),
        "ground_truth_top_ddi_drugs_csv": str(top_ddi_csv_path),
    }
    write_json(report_dir / "medication_target_diagnostics.json", summary_payload)

    print("Medication target diagnostics complete.")
    print(f"  medication_target_diagnostics.json: {diagnostics_json_path}")
    print(f"  medication_target_count_distribution.csv: {count_distribution_csv_path}")
    print(f"  ground_truth_ddi_summary.json: {ground_truth_ddi_json_path}")
    print(f"  ground_truth_ddi_by_bucket.csv: {ground_truth_ddi_bucket_csv_path}")
    print(f"  ddi_vocab_degree_stats.csv: {ddi_degree_csv_path}")

    print(json.dumps(summary_payload, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
