from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from src.utils.io import load_pt, resolve_path


RETRIEVAL_RECORD_KEYS = (
    "retrieval_neighbor_ids",
    "retrieval_neighbor_patient_ids",
    "retrieval_neighbor_visit_indices",
    "retrieval_scores",
    "retrieval_mask",
    "retrieval_medication_ids",
)


def retrieval_cache_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return dict(config.get("retrieval_cache", {}))


def retrieval_cache_enabled(config: Mapping[str, Any]) -> bool:
    cfg = retrieval_cache_config(config)
    return bool(cfg.get("enabled", False) and cfg.get("use_precomputed", True))


def retrieval_cache_path(config: Mapping[str, Any], split: str) -> Path:
    cfg = retrieval_cache_config(config)
    cache_root = cfg.get("cache_root", "data/artifacts/retrieval_cache")
    return Path(resolve_path(config["_project_root"], cache_root)) / f"{split}_topk.pt"


def load_retrieval_cache_for_split(
    *,
    config: Mapping[str, Any],
    split: str,
    expected_rows: int,
    drug_vocab_size: int,
) -> dict[str, Any] | None:
    if not retrieval_cache_enabled(config):
        return None

    path = retrieval_cache_path(config, split)
    if not path.exists():
        if bool(retrieval_cache_config(config).get("build_if_missing", False)):
            raise FileNotFoundError(
                "Retrieval cache build_if_missing=True is not supported inside Dataset/DataLoader. "
                f"Build it offline first, missing path: {path}"
            )
        raise FileNotFoundError(
            f"Missing offline retrieval cache for split `{split}`: {path}. "
            "Run src/data/build_retrieval_cache.py before training/evaluation."
        )

    payload = load_pt(path)
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"Retrieval cache must be a dict payload: {path}")
    cache = dict(payload)
    num_queries = int(cache.get("num_queries", -1))
    if num_queries != int(expected_rows):
        raise RuntimeError(
            f"Retrieval cache row count mismatch for split `{split}`: "
            f"cache={num_queries}, dataset={expected_rows}, path={path}"
        )
    cache_drug_vocab_size = int(cache.get("drug_vocab_size", drug_vocab_size))
    if cache_drug_vocab_size != int(drug_vocab_size):
        raise RuntimeError(
            f"Retrieval cache drug vocab mismatch for split `{split}`: "
            f"cache={cache_drug_vocab_size}, dataset={drug_vocab_size}, path={path}"
        )
    for key in RETRIEVAL_RECORD_KEYS:
        if key not in cache:
            raise KeyError(f"Retrieval cache is missing `{key}`: {path}")
        value = cache[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Retrieval cache field `{key}` must be a tensor, got {type(value)!r}")
        if int(value.shape[0]) != num_queries:
            raise RuntimeError(
                f"Retrieval cache field `{key}` first dimension must be {num_queries}, "
                f"got {tuple(value.shape)} at {path}"
            )
    cache["_cache_path"] = str(path)
    return cache


def retrieval_record_from_cache(cache: Mapping[str, Any], index: int) -> dict[str, torch.Tensor]:
    return {
        key: torch.as_tensor(cache[key][int(index)]).clone()
        for key in RETRIEVAL_RECORD_KEYS
    }


def attach_retrieval_batch_fields(
    batch: dict[str, Any],
    records: list[Mapping[str, Any]],
    *,
    drug_vocab_size: int,
) -> None:
    if not records or not all("retrieval_mask" in record for record in records):
        return
    for key in RETRIEVAL_RECORD_KEYS:
        if not all(key in record for record in records):
            raise KeyError(f"Retrieval cache record field `{key}` must be present for every record.")

    batch["retrieval_neighbor_ids"] = torch.stack(
        [torch.as_tensor(record["retrieval_neighbor_ids"], dtype=torch.long) for record in records],
        dim=0,
    )
    batch["retrieval_neighbor_patient_ids"] = torch.stack(
        [torch.as_tensor(record["retrieval_neighbor_patient_ids"], dtype=torch.long) for record in records],
        dim=0,
    )
    batch["retrieval_neighbor_visit_indices"] = torch.stack(
        [torch.as_tensor(record["retrieval_neighbor_visit_indices"], dtype=torch.long) for record in records],
        dim=0,
    )
    batch["retrieval_scores"] = torch.stack(
        [torch.as_tensor(record["retrieval_scores"], dtype=torch.float32) for record in records],
        dim=0,
    )
    batch["retrieval_mask"] = torch.stack(
        [torch.as_tensor(record["retrieval_mask"], dtype=torch.bool) for record in records],
        dim=0,
    )
    medication_ids = torch.stack(
        [torch.as_tensor(record["retrieval_medication_ids"], dtype=torch.long) for record in records],
        dim=0,
    )
    batch["retrieval_medication_ids"] = medication_ids

    batch_size, top_k, max_medication_ids = medication_ids.shape
    multi_hot = torch.zeros(batch_size, top_k, int(drug_vocab_size), dtype=torch.float32)
    valid_med_mask = (
        batch["retrieval_mask"].unsqueeze(-1)
        & (medication_ids >= 0)
        & (medication_ids < int(drug_vocab_size))
    )
    if bool(valid_med_mask.any().item()):
        row_idx, neighbor_idx, med_slot_idx = torch.nonzero(valid_med_mask, as_tuple=True)
        drug_idx = medication_ids[row_idx, neighbor_idx, med_slot_idx]
        multi_hot[row_idx, neighbor_idx, drug_idx] = 1.0
    batch["retrieval_medication_multi_hot"] = multi_hot
    batch["retrieval_valid_candidate_counts"] = batch["retrieval_mask"].sum(dim=1, dtype=torch.long)


__all__ = [
    "RETRIEVAL_RECORD_KEYS",
    "attach_retrieval_batch_fields",
    "load_retrieval_cache_for_split",
    "retrieval_cache_enabled",
    "retrieval_cache_path",
    "retrieval_record_from_cache",
]
