from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F

if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover
    def tqdm(iterable, *args, **kwargs):
        return iterable

from src.data.tensorized_dataset import tensorized_manifest_path_from_config
from src.retrieval.faiss_index import VisitFaissIndex
from src.utils.io import ensure_dir, load_pt, load_yaml_config, read_json, resolve_path, write_json


_HASH_MULTIPLIERS = {
    "diag": 1000003,
    "proc": 1009837,
    "med": 1012927,
}
_HASH_OFFSETS = {
    "diag": 17,
    "proc": 7919,
    "med": 15485863,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline retrieval cache for ClinRec.")
    parser.add_argument("--config", default="configs/data.yaml", help="Path to configs/data.yaml")
    parser.add_argument("--train-config", default="configs/train_retrieval_cached.yaml")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], choices=("train", "val", "test"))
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--cache-root", default=None)
    parser.add_argument("--backend", default=None, choices=("faiss", "bruteforce"))
    parser.add_argument("--embedding-dim", type=int, default=128)
    parser.add_argument("--max-medication-ids", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--coarse-multiplier", type=int, default=50)
    parser.add_argument("--allow-same-patient", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _runtime_config_without_cache(data_config: Mapping[str, Any]) -> dict[str, Any]:
    runtime_config = copy.deepcopy(dict(data_config))
    runtime_config["retrieval_cache"] = {"enabled": False}
    return runtime_config


def _current_indices(visit_mask: torch.Tensor) -> torch.Tensor:
    return torch.clamp(visit_mask.to(dtype=torch.long).sum(dim=1) - 1, min=0)


def _hash_add(
    vector: torch.Tensor,
    ids: torch.Tensor,
    *,
    namespace: str,
    weight: float,
) -> None:
    if ids.numel() <= 0:
        return
    ids = ids.to(dtype=torch.long)
    ids = ids[(ids >= 2)]
    if ids.numel() <= 0:
        return
    dim = int(vector.shape[0])
    hashed = ((ids * _HASH_MULTIPLIERS[namespace]) + _HASH_OFFSETS[namespace]).remainder(dim)
    vector.index_add_(0, hashed, torch.full((int(hashed.numel()),), float(weight), dtype=vector.dtype))


def _sample_query_embedding(
    *,
    diag_codes: torch.Tensor,
    diag_mask: torch.Tensor,
    proc_codes: torch.Tensor,
    proc_mask: torch.Tensor,
    med_history: torch.Tensor,
    med_history_mask: torch.Tensor,
    current_index: int,
    embedding_dim: int,
) -> torch.Tensor:
    vector = torch.zeros(int(embedding_dim), dtype=torch.float32)
    _hash_add(
        vector,
        diag_codes[current_index][diag_mask[current_index].to(dtype=torch.bool)],
        namespace="diag",
        weight=1.0,
    )
    _hash_add(
        vector,
        proc_codes[current_index][proc_mask[current_index].to(dtype=torch.bool)],
        namespace="proc",
        weight=1.0,
    )
    _hash_add(
        vector,
        med_history[current_index][med_history_mask[current_index].to(dtype=torch.bool)],
        namespace="med",
        weight=0.5,
    )
    return F.normalize(vector.unsqueeze(0), dim=1).squeeze(0)


def _target_medication_ids(
    target_drugs: torch.Tensor,
    *,
    current_index: int,
    max_medication_ids: int,
) -> torch.Tensor:
    ids = torch.nonzero(target_drugs[current_index] > 0, as_tuple=False).flatten()
    ids = ids[(ids >= 2)]
    if int(ids.numel()) > int(max_medication_ids):
        ids = ids[: int(max_medication_ids)]
    output = torch.full((int(max_medication_ids),), -1, dtype=torch.int16)
    if ids.numel() > 0:
        output[: int(ids.numel())] = ids.to(dtype=torch.int16)
    return output


def _load_split_features(
    *,
    config: Mapping[str, Any],
    split: str,
    embedding_dim: int,
    max_medication_ids: int,
) -> dict[str, torch.Tensor]:
    manifest_path = tensorized_manifest_path_from_config(config)
    manifest = read_json(manifest_path)
    split_payload = manifest.get("splits", {}).get(split)
    if split_payload is None:
        raise FileNotFoundError(f"Split `{split}` is missing from tensorized manifest {manifest_path}")

    rows_total = int(sum(int(item["rows"]) for item in split_payload))
    embeddings = torch.zeros(rows_total, int(embedding_dim), dtype=torch.float32)
    medication_ids = torch.full((rows_total, int(max_medication_ids)), -1, dtype=torch.int16)
    patient_ids = torch.zeros(rows_total, dtype=torch.long)
    visit_indices = torch.zeros(rows_total, dtype=torch.long)
    sample_ids = torch.arange(rows_total, dtype=torch.long)

    row_offset = 0
    progress = tqdm(split_payload, desc=f"Encoding {split} retrieval features", unit="shard", leave=False)
    for shard in progress:
        shard_path = manifest_path.parent / str(shard["path"])
        payload = load_pt(shard_path)
        visit_mask = torch.as_tensor(payload["visit_mask"], dtype=torch.bool)
        current = _current_indices(visit_mask)
        shard_rows = int(visit_mask.shape[0])
        diag_codes = torch.as_tensor(payload["diag_codes"], dtype=torch.long)
        diag_mask = torch.as_tensor(payload.get("diag_mask", diag_codes.ne(0)), dtype=torch.bool)
        proc_codes = torch.as_tensor(payload["proc_codes"], dtype=torch.long)
        proc_mask = torch.as_tensor(payload.get("proc_mask", proc_codes.ne(0)), dtype=torch.bool)
        med_history = torch.as_tensor(payload["med_history"], dtype=torch.long)
        med_history_mask = torch.as_tensor(payload.get("med_history_mask", med_history.ne(0)), dtype=torch.bool)
        target_drugs = torch.as_tensor(payload["target_drugs"], dtype=torch.float32)

        subject_ids = torch.as_tensor(payload.get("subject_ids", payload.get("patient_ids")), dtype=torch.long)
        if subject_ids.ndim != 1:
            raise RuntimeError(f"Tensorized shard subject_ids must be 1D: {shard_path}")

        for local_index in range(shard_rows):
            global_index = row_offset + local_index
            current_index = int(current[local_index].item())
            embeddings[global_index] = _sample_query_embedding(
                diag_codes=diag_codes[local_index],
                diag_mask=diag_mask[local_index],
                proc_codes=proc_codes[local_index],
                proc_mask=proc_mask[local_index],
                med_history=med_history[local_index],
                med_history_mask=med_history_mask[local_index],
                current_index=current_index,
                embedding_dim=embedding_dim,
            )
            medication_ids[global_index] = _target_medication_ids(
                target_drugs[local_index],
                current_index=current_index,
                max_medication_ids=max_medication_ids,
            )
            patient_ids[global_index] = int(subject_ids[local_index].item())
            visit_indices[global_index] = current_index
        row_offset += shard_rows

    return {
        "embeddings": embeddings,
        "medication_ids": medication_ids,
        "patient_ids": patient_ids,
        "visit_indices": visit_indices,
        "sample_ids": sample_ids,
    }


def _filter_candidate(
    *,
    query_split: str,
    memory_split: str,
    query_sample_id: int,
    query_patient_id: int,
    query_visit_index: int,
    candidate_sample_id: int,
    candidate_patient_id: int,
    candidate_visit_index: int,
    allow_same_patient: bool,
    leakage_counts: dict[str, int],
) -> bool:
    same_split = query_split == memory_split
    same_patient = int(query_patient_id) == int(candidate_patient_id)
    if same_split and int(query_sample_id) == int(candidate_sample_id):
        leakage_counts["exact_self_excluded"] += 1
        return False
    if same_patient and not allow_same_patient:
        leakage_counts["same_patient_excluded"] += 1
        return False
    if same_patient and int(candidate_visit_index) > int(query_visit_index):
        leakage_counts["same_patient_future_excluded"] += 1
        return False
    if same_patient and int(candidate_visit_index) == int(query_visit_index) and same_split:
        leakage_counts["same_patient_same_visit_excluded"] += 1
        return False
    return True


def _build_split_cache(
    *,
    query_split: str,
    memory_split: str,
    query_features: Mapping[str, torch.Tensor],
    memory_features: Mapping[str, torch.Tensor],
    top_k: int,
    cache_path: Path,
    backend: str,
    chunk_size: int,
    coarse_multiplier: int,
    allow_same_patient: bool,
    drug_vocab_size: int,
    max_medication_ids: int,
    overwrite: bool,
) -> dict[str, Any]:
    if cache_path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing retrieval cache: {cache_path}")
    ensure_dir(cache_path.parent)

    num_queries = int(query_features["embeddings"].shape[0])
    top_k = int(top_k)
    neighbor_ids = torch.full((num_queries, top_k), -1, dtype=torch.long)
    neighbor_patient_ids = torch.full((num_queries, top_k), -1, dtype=torch.long)
    neighbor_visit_indices = torch.full((num_queries, top_k), -1, dtype=torch.long)
    neighbor_scores = torch.zeros(num_queries, top_k, dtype=torch.float32)
    neighbor_mask = torch.zeros(num_queries, top_k, dtype=torch.bool)
    neighbor_medication_ids = torch.full(
        (num_queries, top_k, int(max_medication_ids)),
        -1,
        dtype=torch.int16,
    )

    index = VisitFaissIndex(backend=backend, use_faiss_if_available=True)
    index.build_index(memory_features["embeddings"])
    backend_used = "faiss" if index._index is not None else "bruteforce"
    coarse_k = min(
        int(memory_features["embeddings"].shape[0]),
        max(top_k * max(int(coarse_multiplier), 1), top_k + 10),
    )

    leakage_counts = {
        "exact_self_excluded": 0,
        "same_patient_excluded": 0,
        "same_patient_future_excluded": 0,
        "same_patient_same_visit_excluded": 0,
        "selected_exact_self_matches": 0,
        "selected_same_patient_future": 0,
    }
    progress = tqdm(range(0, num_queries, int(chunk_size)), desc=f"Searching {query_split}", unit="chunk", leave=False)
    for start in progress:
        end = min(start + int(chunk_size), num_queries)
        search = index.search(query_features["embeddings"][start:end], top_k=max(coarse_k, top_k))
        for local_row in range(end - start):
            query_index = start + local_row
            write_pos = 0
            for score, candidate_index_tensor in zip(
                search["scores"][local_row].tolist(),
                search["indices"][local_row].tolist(),
            ):
                candidate_index = int(candidate_index_tensor)
                if candidate_index < 0:
                    continue
                keep = _filter_candidate(
                    query_split=query_split,
                    memory_split=memory_split,
                    query_sample_id=int(query_features["sample_ids"][query_index].item()),
                    query_patient_id=int(query_features["patient_ids"][query_index].item()),
                    query_visit_index=int(query_features["visit_indices"][query_index].item()),
                    candidate_sample_id=int(memory_features["sample_ids"][candidate_index].item()),
                    candidate_patient_id=int(memory_features["patient_ids"][candidate_index].item()),
                    candidate_visit_index=int(memory_features["visit_indices"][candidate_index].item()),
                    allow_same_patient=allow_same_patient,
                    leakage_counts=leakage_counts,
                )
                if not keep:
                    continue
                neighbor_ids[query_index, write_pos] = int(memory_features["sample_ids"][candidate_index].item())
                neighbor_patient_ids[query_index, write_pos] = int(memory_features["patient_ids"][candidate_index].item())
                neighbor_visit_indices[query_index, write_pos] = int(memory_features["visit_indices"][candidate_index].item())
                neighbor_scores[query_index, write_pos] = float(score)
                neighbor_mask[query_index, write_pos] = True
                neighbor_medication_ids[query_index, write_pos] = memory_features["medication_ids"][candidate_index]
                write_pos += 1
                if write_pos >= top_k:
                    break

    same_split = query_split == memory_split
    if same_split:
        selected_self = (neighbor_ids == query_features["sample_ids"].unsqueeze(1)) & neighbor_mask
        leakage_counts["selected_exact_self_matches"] = int(selected_self.sum().item())
    same_patient_selected = (neighbor_patient_ids == query_features["patient_ids"].unsqueeze(1)) & neighbor_mask
    future_selected = same_patient_selected & (neighbor_visit_indices > query_features["visit_indices"].unsqueeze(1))
    leakage_counts["selected_same_patient_future"] = int(future_selected.sum().item())

    payload = {
        "schema_version": 1,
        "split": query_split,
        "memory_split": memory_split,
        "top_k": top_k,
        "num_queries": num_queries,
        "drug_vocab_size": int(drug_vocab_size),
        "max_medication_ids": int(max_medication_ids),
        "allow_cross_split": False,
        "allow_same_patient": bool(allow_same_patient),
        "backend_requested": backend,
        "backend_used": backend_used,
        "query_sample_ids": query_features["sample_ids"].to(dtype=torch.long),
        "query_patient_ids": query_features["patient_ids"].to(dtype=torch.long),
        "query_visit_indices": query_features["visit_indices"].to(dtype=torch.long),
        "retrieval_neighbor_ids": neighbor_ids,
        "retrieval_neighbor_patient_ids": neighbor_patient_ids,
        "retrieval_neighbor_visit_indices": neighbor_visit_indices,
        "retrieval_scores": neighbor_scores,
        "retrieval_mask": neighbor_mask,
        "retrieval_medication_ids": neighbor_medication_ids,
        "leakage_counts": leakage_counts,
    }
    torch.save(payload, cache_path)

    valid_counts = neighbor_mask.sum(dim=1, dtype=torch.float32)
    valid_scores = neighbor_scores[neighbor_mask]
    return {
        "split": query_split,
        "memory_split": memory_split,
        "num_queries": int(num_queries),
        "avg_valid_neighbors": float(valid_counts.mean().item()),
        "fraction_with_neighbors": float((valid_counts > 0).to(dtype=torch.float32).mean().item()),
        "avg_score": 0.0 if valid_scores.numel() <= 0 else float(valid_scores.mean().item()),
        "leakage_check_counts": leakage_counts,
        "cache_path": str(cache_path),
        "backend_used": backend_used,
    }


def main() -> None:
    args = parse_args()
    data_config = load_yaml_config(args.config)
    train_config = load_yaml_config(args.train_config)
    project_root = Path(data_config["_project_root"]).resolve()
    runtime_config = _runtime_config_without_cache(data_config)
    retrieval_cfg = dict(train_config.get("retrieval_cache", {}))

    top_k = int(args.top_k if args.top_k is not None else retrieval_cfg.get("top_k", 3))
    cache_root_value = args.cache_root if args.cache_root is not None else retrieval_cfg.get("cache_root", "data/artifacts/retrieval_cache")
    cache_root = ensure_dir(resolve_path(project_root, cache_root_value))
    backend = str(args.backend or retrieval_cfg.get("backend", train_config.get("extended", {}).get("retrieval_backend", "faiss"))).lower()
    if backend not in {"faiss", "bruteforce"}:
        backend = "faiss"
    memory_split_for_eval = str(retrieval_cfg.get("memory_split_for_eval", "train"))
    allow_cross_split = bool(retrieval_cfg.get("allow_cross_split", False))
    if allow_cross_split:
        raise ValueError("allow_cross_split=true is not supported for leakage-safe offline cache builds.")
    if memory_split_for_eval != "train":
        raise ValueError("Only memory_split_for_eval=train is supported for leakage-safe offline cache builds.")

    manifest = read_json(tensorized_manifest_path_from_config(runtime_config))
    drug_vocab_size = int(manifest["drug_vocab_size"])
    requested_splits = [str(split) for split in args.splits]
    required_feature_splits = sorted(set(requested_splits + ["train"]))
    feature_by_split = {
        split: _load_split_features(
            config=runtime_config,
            split=split,
            embedding_dim=int(args.embedding_dim),
            max_medication_ids=int(args.max_medication_ids),
        )
        for split in required_feature_splits
    }

    report_rows = []
    for split in requested_splits:
        memory_split = "train"
        report_rows.append(
            _build_split_cache(
                query_split=split,
                memory_split=memory_split,
                query_features=feature_by_split[split],
                memory_features=feature_by_split[memory_split],
                top_k=top_k,
                cache_path=cache_root / f"{split}_topk.pt",
                backend=backend,
                chunk_size=int(args.chunk_size),
                coarse_multiplier=int(args.coarse_multiplier),
                allow_same_patient=bool(args.allow_same_patient or retrieval_cfg.get("allow_same_patient", False)),
                drug_vocab_size=drug_vocab_size,
                max_medication_ids=int(args.max_medication_ids),
                overwrite=bool(args.overwrite),
            )
        )

    report = {
        "retrieval_cache_root": str(cache_root),
        "top_k": top_k,
        "embedding_dim": int(args.embedding_dim),
        "max_medication_ids": int(args.max_medication_ids),
        "allow_cross_split": False,
        "memory_split_for_eval": "train",
        "warnings": [
            "Absolute visit timestamps are not required by this cache builder. "
            "Same-patient future leakage is blocked with visit order; val/test use train memory only."
        ],
        "splits": report_rows,
    }
    report_path = write_json(resolve_path(project_root, "outputs/reports/retrieval_cache_report.json"), report)
    print(json.dumps({**report, "report_path": str(report_path)}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
