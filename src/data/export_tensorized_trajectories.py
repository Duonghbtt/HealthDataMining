from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress dependency
    def tqdm(iterable, *args, **kwargs):
        return iterable

import torch

from src.data.tensorized_dataset import tensorized_root_from_config
from src.data.tensorization_utils import augment_record, collate_batch
from src.utils.io import (
    ensure_dir,
    iter_jsonl_gz,
    load_yaml_config,
    read_json,
    resolve_path,
    save_pt,
    write_json,
)


def _trajectory_root_from_config(config: Mapping[str, Any]) -> Path:
    paths_cfg = dict(config.get("paths", {}))
    trajectory_root = paths_cfg.get("trajectory_interim_root")
    if trajectory_root:
        return Path(resolve_path(config["_project_root"], trajectory_root))
    interim_root = resolve_path(config["_project_root"], paths_cfg["interim_root"])
    return Path(interim_root) / "trajectories"


def _trajectory_metadata_path(trajectory_root: Path) -> Path:
    return trajectory_root / "metadata.json"


def _vocab_root_from_config(config: Mapping[str, Any]) -> Path:
    return Path(resolve_path(config["_project_root"], config["paths"]["vocab_root"]))


def _current_med_vocab_payload(config: Mapping[str, Any]) -> dict[str, Any]:
    vocab_root = _vocab_root_from_config(config)
    med_vocab_path = vocab_root / "med_vocab_main.json"
    if not med_vocab_path.exists():
        raise FileNotFoundError(
            f"Canonical medication vocab is missing at {med_vocab_path}. "
            f"Run `python -m src.data.build_vocab --config {config['_config_path']}` first."
        )
    return dict(read_json(med_vocab_path))


def _resolve_output_root(
    config: Mapping[str, Any],
    *,
    output_root: str | None,
) -> Path:
    if output_root:
        return Path(resolve_path(config["_project_root"], output_root))
    return tensorized_root_from_config(config)


def _split_trajectory_path(trajectory_root: Path, split: str) -> Path:
    return trajectory_root / split / "trajectories.jsonl.gz"


def _chunk_records(records: Iterable[dict[str, Any]], chunk_size: int) -> Iterable[list[dict[str, Any]]]:
    if int(chunk_size) <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size!r}")
    batch: list[dict[str, Any]] = []
    for record in records:
        batch.append(dict(record))
        if len(batch) >= int(chunk_size):
            yield batch
            batch = []
    if batch:
        yield batch


def _tensorize_records(
    records: list[dict[str, Any]],
    *,
    drug_vocab_size: int,
    default_lab_feature_size: int,
    default_vital_feature_size: int,
) -> dict[str, torch.Tensor]:
    augmented_records = [
        augment_record(
            record,
            drug_vocab_size=drug_vocab_size,
            default_lab_feature_size=default_lab_feature_size,
            default_vital_feature_size=default_vital_feature_size,
        )
        for record in records
    ]
    batch = collate_batch(augmented_records)
    return {
        "diag_codes": batch["diag_codes"].contiguous(),
        "diag_mask": batch["diag_mask"].contiguous(),
        "proc_codes": batch["proc_codes"].contiguous(),
        "proc_mask": batch["proc_mask"].contiguous(),
        "lab_values": batch["lab_values"].contiguous(),
        "lab_mask": batch["lab_mask"].contiguous(),
        "vital_values": batch["vital_values"].contiguous(),
        "vital_mask": batch["vital_mask"].contiguous(),
        "med_history": batch["med_history"].contiguous(),
        "med_history_mask": batch["med_history_mask"].contiguous(),
        "time_delta_hours": batch["time_delta_hours"].contiguous(),
        "visit_mask": batch["visit_mask"].contiguous(),
        "target_drugs": batch["target_drugs"].contiguous(),
        "subject_ids": torch.as_tensor(batch["subject_ids"], dtype=torch.long).contiguous(),
        "hadm_ids": torch.as_tensor(batch["hadm_ids"], dtype=torch.long).contiguous(),
        "stay_ids": torch.as_tensor(batch["stay_ids"], dtype=torch.long).contiguous(),
    }


def export_tensorized_trajectories(
    config_path: str | Path,
    *,
    output_root: str | None = None,
    overwrite: bool = False,
    rows_per_pt_shard: int = 256,
) -> Path:
    if int(rows_per_pt_shard) <= 0:
        raise ValueError(f"rows_per_pt_shard must be positive, got {rows_per_pt_shard!r}")

    config = load_yaml_config(config_path)
    trajectory_root = _trajectory_root_from_config(config)
    trajectory_metadata_path = _trajectory_metadata_path(trajectory_root)
    if not trajectory_metadata_path.exists():
        raise FileNotFoundError(
            f"Trajectory metadata does not exist at {trajectory_metadata_path}. "
            f"Run `python -m src.data.build_trajectories --config {config['_config_path']}` first."
        )

    trajectory_metadata = dict(read_json(trajectory_metadata_path))
    current_med_vocab = _current_med_vocab_payload(config)
    current_drug_vocab_size = int(current_med_vocab.get("size", 0))
    current_drug_representation = str(
        trajectory_metadata.get("drug_representation") or "med_vocab_main"
    ).strip() or "med_vocab_main"

    tensorized_root = _resolve_output_root(config, output_root=output_root)
    tensorized_manifest_path = tensorized_root / "manifest.json"
    tensorized_metadata_path = tensorized_root / "metadata.json"
    if tensorized_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Tensorized output root already exists: {tensorized_root}. "
                "Use --overwrite to regenerate it."
            )
        shutil.rmtree(tensorized_root)

    ensure_dir(tensorized_root)
    print(f"Trajectory root: {trajectory_root}")
    print(f"Tensorized output root: {tensorized_root}")

    drug_vocab_size = int(trajectory_metadata.get("drug_vocab_size", 0)) or current_drug_vocab_size
    default_lab_feature_size = int(trajectory_metadata.get("lab_feature_size", 0))
    default_vital_feature_size = int(trajectory_metadata.get("vital_feature_size", 0))

    manifest_payload: dict[str, Any] = {
        "format": "tensorized_pt",
        "schema_version": 1,
        "drug_vocab_size": int(drug_vocab_size),
        "default_lab_feature_size": int(default_lab_feature_size),
        "default_vital_feature_size": int(default_vital_feature_size),
        "drug_representation": current_drug_representation,
        "source_trajectory_root": str(trajectory_root),
        "source_trajectory_metadata_path": str(trajectory_metadata_path),
        "splits": {},
    }

    for split_name in tqdm(("train", "val", "test"), desc="Export tensorized splits", unit="split"):
        split_dir = ensure_dir(tensorized_root / split_name)
        source_path = _split_trajectory_path(trajectory_root, split_name)
        manifest_payload["splits"][split_name] = []
        pt_shard_index = 0
        split_row_count = 0

        if source_path.exists():
            source_records = iter_jsonl_gz(source_path)
            for chunk in _chunk_records(source_records, int(rows_per_pt_shard)):
                tensorized_shard = _tensorize_records(
                    chunk,
                    drug_vocab_size=drug_vocab_size,
                    default_lab_feature_size=default_lab_feature_size,
                    default_vital_feature_size=default_vital_feature_size,
                )
                output_path = split_dir / f"part-{pt_shard_index:05d}.pt"
                save_pt(output_path, tensorized_shard)
                shard_rows = int(tensorized_shard["subject_ids"].shape[0])
                split_row_count += shard_rows
                manifest_payload["splits"][split_name].append(
                    {
                        "path": str(output_path.relative_to(tensorized_root)).replace("\\", "/"),
                        "rows": shard_rows,
                    }
                )
                pt_shard_index += 1

        print(
            f"Exported split={split_name}: pt_shards={len(manifest_payload['splits'][split_name])} "
            f"rows={split_row_count}"
        )

    tensorized_metadata = {
        "drug_vocab_size": manifest_payload["drug_vocab_size"],
        "default_lab_feature_size": manifest_payload["default_lab_feature_size"],
        "default_vital_feature_size": manifest_payload["default_vital_feature_size"],
        "drug_representation": manifest_payload["drug_representation"],
        "current_drug_vocab_size": current_drug_vocab_size,
        "current_drug_vocab_path": str(
            (_vocab_root_from_config(config) / "med_vocab_main.json").resolve()
        ),
        "source_trajectory_root": str(trajectory_root),
        "source_trajectory_metadata": trajectory_metadata,
    }
    write_json(tensorized_metadata_path, tensorized_metadata)
    write_json(tensorized_manifest_path, manifest_payload)
    print(f"Wrote tensorized metadata: {tensorized_metadata_path}")
    print(f"Wrote tensorized manifest: {tensorized_manifest_path}")
    return tensorized_root


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export interim JSONL trajectories to tensorized .pt shards."
    )
    parser.add_argument("--config", default="configs/data.yaml", help="Path to configs/data.yaml")
    parser.add_argument("--output-root", default=None, help="Optional override for tensorized output root")
    parser.add_argument("--rows-per-pt-shard", type=int, default=256, help="Maximum number of records per exported .pt shard")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing tensorized output root")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    export_tensorized_trajectories(
        args.config,
        output_root=args.output_root,
        overwrite=bool(args.overwrite),
        rows_per_pt_shard=int(args.rows_per_pt_shard),
    )


if __name__ == "__main__":
    main()
