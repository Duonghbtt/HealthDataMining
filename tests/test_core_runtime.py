from __future__ import annotations

import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest
import yaml

from src.data.build_ddi_matrix import build_ddi_matrix
from src.data.build_vocab import build_vocab
from src.data.dataset import ShardLengthBatchSampler, collate_batch, detect_trajectory_layout
from src.models.ddi_regularization import DDIRegularizer, load_ddi_artifact
from src.training.train_core import (
    TqdmCoreTrainer,
    apply_profile_overrides,
    build_core_model,
    build_dataloaders,
    build_dataset,
    build_optimizer,
    build_runtime_data_config_file,
    build_scheduler,
    resolve_profile_name,
)
from src.utils.io import (
    load_yaml_config,
    save_pt,
    write_csv_gz,
    write_json,
    write_parquet_pylist,
)


def _write_yaml(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _write_config_bundle(project_root: Path) -> dict[str, Path]:
    configs_dir = project_root / "configs"
    data_config = {
        "seed": 7,
        "paths": {
            "raw_root": "data/raw",
            "interim_root": "data/interim",
            "processed_root": "data/processed",
            "ddi_source_path": "",
        },
        "processed_format": "parquet",
        "split": {"train": 0.7, "val": 0.15, "test": 0.15},
        "cohort": {
            "require_diagnosis": True,
            "require_medication": True,
            "min_los_hours": 0.0,
        },
        "features": {
            "time_bucket_hours": 24,
            "top_k_labs": 8,
            "top_k_vitals": 8,
            "max_med_history": 8,
            "max_visits": None,
            "max_history": None,
            "normalization_eps": 1.0e-6,
        },
        "spark": {
            "enabled": False,
            "max_open_shards_per_dataset": 4,
        },
        "profiles": {
            "safe": {
                "features": {"max_visits": None, "max_history": None},
                "spark": {"max_open_shards_per_dataset": 4},
            },
            "balanced": {
                "features": {"max_visits": None, "max_history": None},
                "spark": {"max_open_shards_per_dataset": 4},
            },
            "fast": {
                "features": {"max_visits": 1, "max_history": 4},
                "spark": {"max_open_shards_per_dataset": 6},
            },
        },
    }
    train_config = {
        "config_refs": {
            "data": "configs/data.yaml",
            "model": "configs/model.yaml",
        },
        "paths": {
            "processed_root": "data/processed",
            "vocab_root": "data/interim/vocab",
            "ddi_matrix_path": "data/processed/ddi/drug_ddi.pt",
            "checkpoint_dir": "outputs/checkpoints",
            "log_dir": "outputs/logs",
        },
        "runtime": {
            "mode": "core",
            "profile": "balanced",
            "device": "cpu",
            "batch_size": 1,
            "num_workers": 0,
            "pin_memory": False,
            "amp": False,
            "grad_accum_steps": 1,
            "non_blocking_transfer": False,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "log_interval": 1,
            "profile_steps": None,
            "train_decoder_top_k": 0,
            "matmul_precision": "high",
            "length_bucket_window": 32,
        },
        "optimization": {
            "epochs": 1,
            "learning_rate": 1.0e-3,
            "optimizer": "adam",
            "scheduler": "none",
        },
        "loss": {"ddi_lambda": 0.05},
        "prediction": {"top_k": 2, "threshold": 0.5},
        "core": {"mode": "core", "use_retrieval": False, "use_group_encoder": False},
        "profiles": {
            "safe": {
                "runtime": {
                    "batch_size": 1,
                    "num_workers": 0,
                    "grad_accum_steps": 1,
                    "log_interval": 1,
                    "train_decoder_top_k": 0,
                    "length_bucket_window": 16,
                }
            },
            "balanced": {
                "runtime": {
                    "batch_size": 1,
                    "num_workers": 0,
                    "grad_accum_steps": 1,
                    "log_interval": 1,
                    "train_decoder_top_k": 0,
                    "length_bucket_window": 32,
                }
            },
            "fast": {
                "runtime": {
                    "batch_size": 2,
                    "num_workers": 0,
                    "grad_accum_steps": 1,
                    "log_interval": 1,
                    "profile_steps": 1,
                    "train_decoder_top_k": 0,
                    "length_bucket_window": 32,
                }
            },
        },
    }
    eval_config = {
        "config_refs": {
            "data": "configs/data.yaml",
            "model": "configs/model.yaml",
            "train": "configs/train.yaml",
        },
        "paths": {
            "processed_root": "data/processed",
            "vocab_root": "data/interim/vocab",
            "ddi_matrix_path": "data/processed/ddi/drug_ddi.pt",
            "checkpoint_dir": "outputs/checkpoints",
            "log_dir": "outputs/logs",
            "report_dir": "outputs/reports",
            "prediction_dir": "outputs/predictions",
        },
        "runtime": {
            "mode": "core",
            "device": "cpu",
            "batch_size": 1,
        },
        "loss": {"ddi_lambda": 0.05},
        "prediction": {"top_k": 2, "threshold": 0.5},
        "evaluation": {
            "split": "test",
            "save_predictions": False,
            "save_reports": True,
        },
        "core": {"mode": "core", "use_retrieval": False, "use_group_encoder": False},
    }
    model_config = {
        "model": {"hidden_dim": 16, "num_layers": 2, "dropout": 0.1},
        "embedding": {
            "diag_dim": 16,
            "proc_dim": 16,
            "drug_dim": 16,
            "lab_dim": 8,
            "vital_dim": 8,
            "time_dim": 4,
        },
        "sequence": {"rnn_type": "gru", "bidirectional": False},
        "history_selector": {
            "dropout": 0.1,
            "score_bias_weight": 0.5,
            "self_top_k": 1,
            "neighbor_top_k": 1,
            "use_retrieval_bias": True,
        },
        "fusion": {"dropout": 0.1, "strategy": "gated"},
        "profiles": {
            "safe": {
                "model": {"hidden_dim": 16, "num_layers": 2, "dropout": 0.1},
                "embedding": {
                    "diag_dim": 16,
                    "proc_dim": 16,
                    "drug_dim": 16,
                    "lab_dim": 8,
                    "vital_dim": 8,
                    "time_dim": 4,
                },
                "history_selector": {"self_top_k": 1},
            },
            "balanced": {
                "model": {"hidden_dim": 16, "num_layers": 2, "dropout": 0.1},
                "embedding": {
                    "diag_dim": 16,
                    "proc_dim": 16,
                    "drug_dim": 16,
                    "lab_dim": 8,
                    "vital_dim": 8,
                    "time_dim": 4,
                },
                "history_selector": {"self_top_k": 1},
            },
            "fast": {
                "model": {"hidden_dim": 12, "num_layers": 1, "dropout": 0.1},
                "embedding": {
                    "diag_dim": 12,
                    "proc_dim": 12,
                    "drug_dim": 12,
                    "lab_dim": 6,
                    "vital_dim": 6,
                    "time_dim": 2,
                },
                "history_selector": {"self_top_k": 1},
            },
        },
    }
    return {
        "data": _write_yaml(configs_dir / "data.yaml", data_config),
        "train": _write_yaml(configs_dir / "train.yaml", train_config),
        "eval": _write_yaml(configs_dir / "eval.yaml", eval_config),
        "model": _write_yaml(configs_dir / "model.yaml", model_config),
    }


def _update_data_ddi_source_path(config_path: Path, ddi_source_path: str) -> Path:
    payload = load_yaml_config(config_path)
    resolved = {key: value for key, value in payload.items() if not str(key).startswith("_")}
    resolved.setdefault("paths", {})
    resolved["paths"]["ddi_source_path"] = ddi_source_path
    return _write_yaml(config_path, resolved)


def _write_vocab_bundle(project_root: Path) -> Path:
    vocab_root = project_root / "data" / "interim" / "vocab"
    vocab_root.mkdir(parents=True, exist_ok=True)
    vocab_payloads = {
        "diagnosis": ["PAD", "UNK", "ICD9:4019", "ICD9:25000"],
        "procedure": ["PAD", "UNK", "PROC9:3893", "PROC9:8872"],
        "drug": ["PAD", "UNK", "NAME:ASPIRIN", "NAME:HEPARIN"],
        "lab": ["PAD", "UNK", "LAB:50912", "LAB:50931"],
        "vital": ["PAD", "UNK", "VITAL:220045", "VITAL:220046"],
    }
    for name, tokens in vocab_payloads.items():
        payload = {
            "name": name,
            "size": len(tokens),
            "pad_idx": 0,
            "unk_idx": 1,
            "idx_to_token": tokens,
            "token_to_idx": {token: index for index, token in enumerate(tokens)},
        }
        write_json(vocab_root / f"{name}_vocab.json", payload)

    write_json(vocab_root / "lab_metadata.json", {"LAB:50912": {"index": 2}, "LAB:50931": {"index": 3}})
    write_json(vocab_root / "vital_metadata.json", {"VITAL:220045": {"index": 2}, "VITAL:220046": {"index": 3}})
    write_json(
        vocab_root / "vocab_summary.json",
        {
            "diagnosis_size": 4,
            "procedure_size": 4,
            "drug_size": 4,
            "lab_size": 4,
            "vital_size": 4,
            "built_from_split": "train",
        },
    )
    return vocab_root


def _trajectory_record(*, split: str, subject_id: int, hadm_id: int, stay_id: int, drug_id: int) -> dict:
    return {
        "subject_id": subject_id,
        "hadm_id": hadm_id,
        "stay_id": stay_id,
        "split": split,
        "intime": "2020-01-01 00:00:00",
        "outtime": "2020-01-02 00:00:00",
        "num_steps": 1,
        "drug_vocab_size": 4,
        "lab_feature_size": 1,
        "vital_feature_size": 1,
        "steps": [
            {
                "step_index": 0,
                "diagnosis_ids": [2],
                "procedure_ids": [2],
                "lab_values": [1.0],
                "lab_mask": [True],
                "vital_values": [1.0],
                "vital_mask": [True],
                "med_history_ids": [drug_id],
                "delta_hours": 0.0,
                "target_drugs": [drug_id],
            }
        ],
    }


def _write_canonical_trajectories(project_root: Path) -> Path:
    processed_root = project_root / "data" / "processed"
    trajectory_root = processed_root / "trajectories"
    trajectory_root.mkdir(parents=True, exist_ok=True)
    records_by_split = {
        "train": [_trajectory_record(split="train", subject_id=1, hadm_id=11, stay_id=111, drug_id=2)],
        "val": [_trajectory_record(split="val", subject_id=2, hadm_id=22, stay_id=222, drug_id=2)],
        "test": [_trajectory_record(split="test", subject_id=3, hadm_id=33, stay_id=333, drug_id=2)],
    }
    manifest = {
        "format": "parquet",
        "schema_version": 1,
        "counts_by_split": {split: len(records) for split, records in records_by_split.items()},
        "splits": {},
    }
    for split_name, records in records_by_split.items():
        split_dir = trajectory_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        shard_path = split_dir / "part-00000.parquet"
        write_parquet_pylist(shard_path, records)
        manifest["splits"][split_name] = {
            "rows": len(records),
            "shards": [{"path": f"{split_name}/part-00000.parquet", "rows": len(records)}],
        }
    write_json(trajectory_root / "manifest.json", manifest)
    write_json(
        trajectory_root / "metadata.json",
        {
            "bucket_hours": 24,
            "counts_by_split": {split: len(records) for split, records in records_by_split.items()},
            "drug_vocab_size": 4,
            "lab_feature_size": 1,
            "max_med_history": 8,
            "processed_format": "parquet",
            "vital_feature_size": 1,
        },
    )
    write_json(trajectory_root / "normalization_stats.json", {"lab": [], "vital": [], "eps": 1.0e-6})
    return processed_root


def _write_direct_manifest_layout(project_root: Path) -> Path:
    processed_root = project_root / "data" / "processed"
    processed_root.mkdir(parents=True, exist_ok=True)
    records_by_split = {
        "train": [_trajectory_record(split="train", subject_id=1, hadm_id=11, stay_id=111, drug_id=2)],
        "val": [_trajectory_record(split="val", subject_id=2, hadm_id=22, stay_id=222, drug_id=2)],
        "test": [_trajectory_record(split="test", subject_id=3, hadm_id=33, stay_id=333, drug_id=2)],
    }
    manifest = {
        "format": "parquet",
        "schema_version": 1,
        "counts_by_split": {split: len(records) for split, records in records_by_split.items()},
        "splits": {},
    }
    for split_name, records in records_by_split.items():
        split_dir = processed_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        shard_path = split_dir / "part-00000.parquet"
        write_parquet_pylist(shard_path, records)
        manifest["splits"][split_name] = {
            "rows": len(records),
            "shards": [{"path": f"{split_name}/part-00000.parquet", "rows": len(records)}],
        }
    write_json(processed_root / "manifest.json", manifest)
    return processed_root


def _write_inactive_ddi_artifact(project_root: Path, *, drug_vocab_size: int = 4) -> Path:
    ddi_dir = project_root / "data" / "processed" / "ddi"
    ddi_dir.mkdir(parents=True, exist_ok=True)
    matrix = [[0 for _ in range(drug_vocab_size)] for _ in range(drug_vocab_size)]
    payload = {
        "matrix": matrix,
        "active": False,
        "reason": "fallback_zero",
        "source": "fallback_zero",
        "matched_pairs": 0,
        "nonzero_pairs": 0,
        "vocab_size": drug_vocab_size,
        "pad_idx": 0,
        "unk_idx": 1,
        "source_metadata": {
            "kind": "fallback_zero",
            "purpose": "no DDI source configured; inactive fallback artifact",
            "research_grade": False,
            "pair_schema": "none",
            "display_name": "Fallback Zero DDI",
        },
    }
    save_pt(ddi_dir / "drug_ddi.pt", payload)
    write_json(ddi_dir / "drug_ddi_report.json", {key: value for key, value in payload.items() if key != "matrix"})
    return ddi_dir / "drug_ddi.pt"


def _write_manual_smoke_ddi_source(
    project_root: Path,
    *,
    pairs: list[tuple[str, str]] | None = None,
) -> Path:
    ddi_root = project_root / "data" / "raw" / "ddi"
    ddi_root.mkdir(parents=True, exist_ok=True)
    source_path = ddi_root / "drug_ddi_smoke.csv"
    resolved_pairs = pairs or [("NAME:ASPIRIN", "NAME:HEPARIN")]
    source_path.write_text(
        "\n".join(
            [
                "drug_a,drug_b,comment",
                *[
                    f"{drug_a},{drug_b},manual smoke pair for local DDI wiring"
                    for drug_a, drug_b in resolved_pairs
                ],
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    write_json(
        ddi_root / "drug_ddi_smoke.metadata.json",
        {
            "kind": "manual_smoke",
            "purpose": "local wiring only",
            "research_grade": False,
            "pair_schema": "canonicalized_drug_token_pairs",
            "display_name": "Test Manual Smoke DDI",
        },
    )
    return source_path


def _prepare_runtime_project(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    project_root = tmp_path / "runtime_project"
    configs = _write_config_bundle(project_root)
    _write_vocab_bundle(project_root)
    _write_canonical_trajectories(project_root)
    _write_inactive_ddi_artifact(project_root)
    return project_root, configs


def _prepare_runtime_project_with_active_manual_smoke_ddi(tmp_path: Path) -> tuple[Path, dict[str, Path]]:
    project_root = tmp_path / "runtime_project_active_ddi"
    configs = _write_config_bundle(project_root)
    _write_vocab_bundle(project_root)
    _write_canonical_trajectories(project_root)
    _write_manual_smoke_ddi_source(project_root)
    _update_data_ddi_source_path(configs["data"], "data/raw/ddi/drug_ddi_smoke.csv")
    build_ddi_matrix(configs["data"])
    return project_root, configs


def test_build_core_model_disables_inactive_ddi(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pyarrow")

    project_root, configs = _prepare_runtime_project(tmp_path)
    train_config = load_yaml_config(configs["train"])
    data_config = load_yaml_config(configs["data"])
    model_config = load_yaml_config(configs["model"])
    train_config["_resolved_paths"] = {"processed_root": str((project_root / "data" / "processed").resolve())}

    with tempfile.TemporaryDirectory(prefix="core_runtime_smoke_") as temp_dir_name:
        runtime_data_config_path = build_runtime_data_config_file(
            data_config=data_config,
            processed_root=(project_root / "data" / "processed").resolve(),
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            temp_dir=Path(temp_dir_name),
        )
        model, loss_fn = build_core_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            ddi_matrix_path=(project_root / "data" / "processed" / "ddi" / "drug_ddi.pt").resolve(),
        )

    assert model.ddi_regularizer is None
    assert loss_fn.ddi_regularizer is None
    assert loss_fn.ddi_context["status"] == "inactive"
    assert loss_fn.ddi_context["reason"] == "fallback_zero"
    assert loss_fn.configured_lambda_ddi == pytest.approx(0.05)
    assert loss_fn.effective_lambda_ddi == pytest.approx(0.0)
    assert model.encoder.gru.num_layers == 2


def test_build_ddi_matrix_writes_inactive_artifact_for_empty_source(tmp_path: Path) -> None:
    project_root = tmp_path / "ddi_project"
    configs = _write_config_bundle(project_root)
    _write_vocab_bundle(project_root)

    with pytest.warns(UserWarning, match="inactive fallback_zero DDI artifact"):
        ddi_path = build_ddi_matrix(configs["data"])

    ddi_artifact = load_ddi_artifact(ddi_path)
    report = json.load((project_root / "data" / "processed" / "ddi" / "drug_ddi_report.json").open())

    assert ddi_artifact["active"] is False
    assert ddi_artifact["reason"] == "missing_ddi_source_path"
    assert ddi_artifact["source"] == "fallback_zero"
    assert ddi_artifact["source_metadata"]["kind"] == "fallback_zero"
    assert report["active"] is False
    assert report["matched_pairs"] == 0
    assert report["nonzero_pairs"] == 0


def test_build_ddi_matrix_writes_active_manual_smoke_artifact(tmp_path: Path) -> None:
    project_root = tmp_path / "ddi_manual_smoke_project"
    configs = _write_config_bundle(project_root)
    _write_vocab_bundle(project_root)
    _write_manual_smoke_ddi_source(project_root)
    _update_data_ddi_source_path(configs["data"], "data/raw/ddi/drug_ddi_smoke.csv")

    ddi_path = build_ddi_matrix(configs["data"])
    ddi_artifact = load_ddi_artifact(ddi_path)
    report = json.load((project_root / "data" / "processed" / "ddi" / "drug_ddi_report.json").open())

    assert ddi_artifact["active"] is True
    assert ddi_artifact["reason"] == "available"
    assert ddi_artifact["matched_pairs"] == 1
    assert ddi_artifact["nonzero_pairs"] == 1
    assert ddi_artifact["source_metadata"]["kind"] == "manual_smoke"
    assert ddi_artifact["source_metadata"]["research_grade"] is False
    assert ddi_artifact["source_metadata"]["purpose"] == "local wiring only"
    assert report["active"] is True
    assert report["matched_pairs"] == 1
    assert report["nonzero_pairs"] == 1
    assert report["source_metadata"]["kind"] == "manual_smoke"
    assert report["source_metadata"]["research_grade"] is False


def test_build_core_model_enables_manual_smoke_ddi_and_wires_loss(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pyarrow")

    project_root, configs = _prepare_runtime_project_with_active_manual_smoke_ddi(tmp_path)
    train_config = load_yaml_config(configs["train"])
    data_config = load_yaml_config(configs["data"])
    model_config = load_yaml_config(configs["model"])
    train_config["_resolved_paths"] = {"processed_root": str((project_root / "data" / "processed").resolve())}

    with tempfile.TemporaryDirectory(prefix="core_runtime_manual_smoke_") as temp_dir_name:
        runtime_data_config_path = build_runtime_data_config_file(
            data_config=data_config,
            processed_root=(project_root / "data" / "processed").resolve(),
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            temp_dir=Path(temp_dir_name),
        )
        model, loss_fn = build_core_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            ddi_matrix_path=(project_root / "data" / "processed" / "ddi" / "drug_ddi.pt").resolve(),
        )
        dataset = build_dataset(
            split="train",
            runtime_data_config_path=runtime_data_config_path,
            processed_root=(project_root / "data" / "processed").resolve(),
            drug_vocab_size=4,
        )
        batch = collate_batch([dataset[0]])
        outputs = model(batch, mode="core", decoder_top_k=2)
        loss_outputs = loss_fn(
            drug_logits=outputs["drug_logits"],
            drug_probs=outputs["drug_probs"],
            target_drugs=batch["target_drugs"],
            visit_mask=batch["visit_mask"],
        )

    assert model.ddi_regularizer is None
    assert loss_fn.ddi_regularizer is not None
    assert loss_fn.ddi_context["status"] == "active"
    assert loss_fn.ddi_context["source_metadata"]["kind"] == "manual_smoke"
    assert loss_fn.ddi_context["source_metadata"]["research_grade"] is False
    assert loss_fn.effective_lambda_ddi == pytest.approx(0.05)
    assert float(loss_outputs["ddi_loss"].item()) > 0.0
    assert float(loss_outputs["weighted_ddi_loss"].item()) > 0.0
    assert loss_outputs["total_loss"].item() == pytest.approx(
        loss_outputs["prediction_loss"].item() + loss_outputs["weighted_ddi_loss"].item()
    )


def test_build_vocab_uses_train_split_only(tmp_path: Path) -> None:
    project_root = tmp_path / "vocab_project"
    configs = _write_config_bundle(project_root)
    raw_hosp = project_root / "data" / "raw" / "hosp"
    raw_icu = project_root / "data" / "raw" / "icu"
    raw_hosp.mkdir(parents=True, exist_ok=True)
    raw_icu.mkdir(parents=True, exist_ok=True)

    write_csv_gz(
        project_root / "data" / "interim" / "cohort" / "cohort.csv.gz",
        [
            {"hadm_id": 11, "stay_id": 111, "split": "train"},
            {"hadm_id": 22, "stay_id": 222, "split": "val"},
        ],
        fieldnames=["hadm_id", "stay_id", "split"],
    )
    write_csv_gz(
        raw_hosp / "diagnoses_icd.csv.gz",
        [
            {"hadm_id": 11, "icd_code": "4019", "icd_version": 9},
            {"hadm_id": 22, "icd_code": "25000", "icd_version": 9},
        ],
        fieldnames=["hadm_id", "icd_code", "icd_version"],
    )
    write_csv_gz(
        raw_hosp / "procedures_icd.csv.gz",
        [
            {"hadm_id": 11, "icd_code": "3893", "icd_version": 9},
            {"hadm_id": 22, "icd_code": "8872", "icd_version": 9},
        ],
        fieldnames=["hadm_id", "icd_code", "icd_version"],
    )
    write_csv_gz(
        raw_hosp / "prescriptions.csv.gz",
        [
            {"hadm_id": 11, "drug": "Aspirin", "formulary_drug_cd": "ASP100"},
            {"hadm_id": 22, "drug": "Heparin", "formulary_drug_cd": "HEP5000"},
        ],
        fieldnames=["hadm_id", "drug", "formulary_drug_cd"],
    )
    write_csv_gz(raw_hosp / "emar.csv.gz", [], fieldnames=["hadm_id", "medication"])
    write_csv_gz(raw_hosp / "pharmacy.csv.gz", [], fieldnames=["hadm_id", "medication"])
    write_csv_gz(
        raw_hosp / "labevents.csv.gz",
        [
            {"hadm_id": 11, "itemid": 50912, "valuenum": 100},
            {"hadm_id": 22, "itemid": 50931, "valuenum": 140},
        ],
        fieldnames=["hadm_id", "itemid", "valuenum"],
    )
    write_csv_gz(
        raw_icu / "chartevents.csv.gz",
        [
            {"stay_id": 111, "itemid": 220045, "valuenum": 80},
            {"stay_id": 222, "itemid": 220046, "valuenum": 90},
        ],
        fieldnames=["stay_id", "itemid", "valuenum"],
    )
    write_csv_gz(
        raw_hosp / "d_labitems.csv.gz",
        [
            {"itemid": 50912, "label": "Creatinine", "category": "Chemistry", "fluid": "Blood"},
            {"itemid": 50931, "label": "Glucose", "category": "Chemistry", "fluid": "Blood"},
        ],
        fieldnames=["itemid", "label", "category", "fluid"],
    )
    write_csv_gz(
        raw_icu / "d_items.csv.gz",
        [
            {"itemid": 220045, "label": "Heart Rate", "category": "Vital", "unitname": "bpm"},
            {"itemid": 220046, "label": "Respiratory Rate", "category": "Vital", "unitname": "bpm"},
        ],
        fieldnames=["itemid", "label", "category", "unitname"],
    )

    build_vocab(configs["data"])

    vocab_root = project_root / "data" / "interim" / "vocab"
    diagnosis_vocab = json.load((vocab_root / "diagnosis_vocab.json").open())
    procedure_vocab = json.load((vocab_root / "procedure_vocab.json").open())
    drug_vocab = json.load((vocab_root / "drug_vocab.json").open())
    lab_vocab = json.load((vocab_root / "lab_vocab.json").open())
    vital_vocab = json.load((vocab_root / "vital_vocab.json").open())
    summary = json.load((vocab_root / "vocab_summary.json").open())

    assert "ICD9:4019" in diagnosis_vocab["token_to_idx"]
    assert "ICD9:25000" not in diagnosis_vocab["token_to_idx"]
    assert "PROC9:3893" in procedure_vocab["token_to_idx"]
    assert "PROC9:8872" not in procedure_vocab["token_to_idx"]
    assert "NAME:ASPIRIN" in drug_vocab["token_to_idx"]
    assert "NAME:HEPARIN" not in drug_vocab["token_to_idx"]
    assert "LAB:50912" in lab_vocab["token_to_idx"]
    assert "LAB:50931" not in lab_vocab["token_to_idx"]
    assert "VITAL:220045" in vital_vocab["token_to_idx"]
    assert "VITAL:220046" not in vital_vocab["token_to_idx"]
    assert summary["built_from_split"] == "train"


def test_detect_trajectory_layout_is_explicit(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    canonical_root = tmp_path / "canonical_project"
    configs = _write_config_bundle(canonical_root)
    _write_canonical_trajectories(canonical_root)
    canonical_layout = detect_trajectory_layout("train", configs["data"])
    assert canonical_layout["kind"] == "canonical_parquet"

    direct_root = tmp_path / "direct_project"
    direct_configs = _write_config_bundle(direct_root)
    _write_direct_manifest_layout(direct_root)
    direct_layout = detect_trajectory_layout("train", direct_configs["data"])
    assert direct_layout["kind"] == "direct_split_manifest"

    both_root = tmp_path / "both_project"
    both_configs = _write_config_bundle(both_root)
    _write_canonical_trajectories(both_root)
    _write_direct_manifest_layout(both_root)
    preferred_layout = detect_trajectory_layout("train", both_configs["data"])
    assert preferred_layout["kind"] == "canonical_parquet"


def test_core_entrypoints_import() -> None:
    assert importlib.import_module("src.training.train_core").__name__ == "src.training.train_core"
    assert importlib.import_module("src.evaluation.evaluate_core").__name__ == "src.evaluation.evaluate_core"


def test_evaluate_core_reports_ddi_unavailable_with_mock_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pyarrow")

    project_root, configs = _prepare_runtime_project(tmp_path)
    train_config = load_yaml_config(configs["train"])
    data_config = load_yaml_config(configs["data"])
    model_config = load_yaml_config(configs["model"])
    train_config["_resolved_paths"] = {"processed_root": str((project_root / "data" / "processed").resolve())}

    with tempfile.TemporaryDirectory(prefix="eval_runtime_smoke_") as temp_dir_name:
        runtime_data_config_path = build_runtime_data_config_file(
            data_config=data_config,
            processed_root=(project_root / "data" / "processed").resolve(),
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            temp_dir=Path(temp_dir_name),
        )
        model, loss_fn = build_core_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            ddi_matrix_path=(project_root / "data" / "processed" / "ddi" / "drug_ddi.pt").resolve(),
        )

    checkpoint_dir = project_root / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "train_core_best.pt"
    import torch

    torch.save(
        {
            "epoch": 1,
            "best_metric": 0.0,
            "monitor_metric": "val_total_loss",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "train_config": {key: value for key, value in train_config.items() if not str(key).startswith("_")},
            "data_config": {key: value for key, value in data_config.items() if not str(key).startswith("_")},
            "model_config": {key: value for key, value in model_config.items() if not str(key).startswith("_")},
            "resolved_paths": {
                "processed_root": str((project_root / "data" / "processed").resolve()),
                "vocab_root": str((project_root / "data" / "interim" / "vocab").resolve()),
                "ddi_matrix_path": str((project_root / "data" / "processed" / "ddi" / "drug_ddi.pt").resolve()),
            },
            "ddi_context": loss_fn.ddi_context,
            "configured_ddi_lambda": loss_fn.configured_lambda_ddi,
            "effective_ddi_lambda": loss_fn.effective_lambda_ddi,
        },
        checkpoint_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.evaluate_core",
            "--config",
            str(configs["eval"]),
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    report = json.load((project_root / "outputs" / "reports" / "evaluate_core_test.json").open())
    assert report["metrics"]["jaccard"] is not None
    assert report["metrics"]["ddi_rate"] is None
    assert report["ddi_summary"]["available"] is False
    assert report["ddi_summary"]["status"] == "inactive"
    assert report["ddi_context"]["training"]["status"] == "inactive"
    assert report["ddi_context"]["evaluation"]["status"] == "inactive"


def test_evaluate_core_reports_manual_smoke_ddi_with_mock_checkpoint(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pyarrow")

    project_root, configs = _prepare_runtime_project_with_active_manual_smoke_ddi(tmp_path)
    train_config = load_yaml_config(configs["train"])
    data_config = load_yaml_config(configs["data"])
    model_config = load_yaml_config(configs["model"])
    train_config["_resolved_paths"] = {"processed_root": str((project_root / "data" / "processed").resolve())}

    with tempfile.TemporaryDirectory(prefix="eval_runtime_manual_smoke_") as temp_dir_name:
        runtime_data_config_path = build_runtime_data_config_file(
            data_config=data_config,
            processed_root=(project_root / "data" / "processed").resolve(),
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            temp_dir=Path(temp_dir_name),
        )
        model, loss_fn = build_core_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            ddi_matrix_path=(project_root / "data" / "processed" / "ddi" / "drug_ddi.pt").resolve(),
        )

    checkpoint_dir = project_root / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "train_core_best.pt"
    import torch

    torch.save(
        {
            "epoch": 1,
            "best_metric": 0.0,
            "monitor_metric": "val_total_loss",
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "train_config": {key: value for key, value in train_config.items() if not str(key).startswith("_")},
            "data_config": {key: value for key, value in data_config.items() if not str(key).startswith("_")},
            "model_config": {key: value for key, value in model_config.items() if not str(key).startswith("_")},
            "resolved_paths": {
                "processed_root": str((project_root / "data" / "processed").resolve()),
                "vocab_root": str((project_root / "data" / "interim" / "vocab").resolve()),
                "ddi_matrix_path": str((project_root / "data" / "processed" / "ddi" / "drug_ddi.pt").resolve()),
            },
            "ddi_context": loss_fn.ddi_context,
            "configured_ddi_lambda": loss_fn.configured_lambda_ddi,
            "effective_ddi_lambda": loss_fn.effective_lambda_ddi,
        },
        checkpoint_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.evaluate_core",
            "--config",
            str(configs["eval"]),
            "--device",
            "cpu",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

    report = json.load((project_root / "outputs" / "reports" / "evaluate_core_test.json").open())
    assert report["metrics"]["ddi_rate"] is not None
    assert report["ddi_summary"]["available"] is True
    assert report["ddi_summary"]["status"] == "active"
    assert report["ddi_summary"]["source_metadata"]["kind"] == "manual_smoke"
    assert report["ddi_summary"]["source_metadata"]["research_grade"] is False
    assert report["ddi_context"]["training"]["status"] == "active"
    assert report["ddi_context"]["training"]["source_metadata"]["kind"] == "manual_smoke"
    assert report["ddi_context"]["training"]["source_metadata"]["research_grade"] is False
    assert report["ddi_context"]["evaluation"]["status"] == "active"
    assert report["ddi_context"]["evaluation"]["source_metadata"]["kind"] == "manual_smoke"
    assert report["ddi_context"]["evaluation"]["source_metadata"]["research_grade"] is False


def test_profile_overrides_merge_expected_runtime_model_and_data(tmp_path: Path) -> None:
    project_root, configs = _prepare_runtime_project(tmp_path)
    del project_root

    raw_train_config = load_yaml_config(configs["train"])
    raw_data_config = load_yaml_config(configs["data"])
    raw_model_config = load_yaml_config(configs["model"])

    assert resolve_profile_name(raw_train_config, None) == "balanced"

    balanced_train = apply_profile_overrides(raw_train_config, profile_name="balanced")
    fast_train = apply_profile_overrides(raw_train_config, profile_name="fast")
    fast_data = apply_profile_overrides(raw_data_config, profile_name="fast")
    fast_model = apply_profile_overrides(raw_model_config, profile_name="fast")

    assert balanced_train["_selected_profile"] == "balanced"
    assert balanced_train["runtime"]["batch_size"] == 1
    assert fast_train["runtime"]["batch_size"] == 2
    assert fast_train["runtime"]["profile_steps"] == 1
    assert fast_data["features"]["max_visits"] == 1
    assert fast_data["features"]["max_history"] == 4
    assert fast_model["model"]["hidden_dim"] == 12


def test_build_dataloaders_core_fast_path_emits_final_targets_only(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pyarrow")

    project_root, configs = _prepare_runtime_project_with_active_manual_smoke_ddi(tmp_path)
    train_config = apply_profile_overrides(load_yaml_config(configs["train"]), profile_name="fast")
    data_config = apply_profile_overrides(load_yaml_config(configs["data"]), profile_name="fast")

    with tempfile.TemporaryDirectory(prefix="core_fast_loader_") as temp_dir_name:
        runtime_data_config_path = build_runtime_data_config_file(
            data_config=data_config,
            processed_root=(project_root / "data" / "processed").resolve(),
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            temp_dir=Path(temp_dir_name),
        )
        train_loader, val_loader = build_dataloaders(
            runtime_data_config_path=runtime_data_config_path,
            processed_root=(project_root / "data" / "processed").resolve(),
            drug_vocab_size=4,
            batch_size=int(train_config["runtime"]["batch_size"]),
            num_workers=int(train_config["runtime"]["num_workers"]),
            pin_memory=bool(train_config["runtime"]["pin_memory"]),
            persistent_workers=bool(train_config["runtime"]["persistent_workers"]),
            prefetch_factor=train_config["runtime"]["prefetch_factor"],
            length_bucket_window=int(train_config["runtime"]["length_bucket_window"]),
            seed=int(data_config["seed"]),
            max_open_shards=int(data_config["spark"]["max_open_shards_per_dataset"]),
            max_visits=data_config["features"]["max_visits"],
            max_history=data_config["features"]["max_history"],
        )

    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))

    assert isinstance(train_loader.batch_sampler, ShardLengthBatchSampler)
    assert "target_drugs" not in train_batch
    assert "final_target_drugs" in train_batch
    assert "target_drugs" not in val_batch
    assert "final_target_drugs" in val_batch


def test_core_model_and_loss_support_final_targets_without_full_target_tensor(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("pyarrow")

    project_root, configs = _prepare_runtime_project_with_active_manual_smoke_ddi(tmp_path)
    train_config = apply_profile_overrides(load_yaml_config(configs["train"]), profile_name="fast")
    data_config = apply_profile_overrides(load_yaml_config(configs["data"]), profile_name="fast")
    model_config = apply_profile_overrides(load_yaml_config(configs["model"]), profile_name="fast")
    train_config["_resolved_paths"] = {"processed_root": str((project_root / "data" / "processed").resolve())}

    with tempfile.TemporaryDirectory(prefix="core_final_target_only_") as temp_dir_name:
        runtime_data_config_path = build_runtime_data_config_file(
            data_config=data_config,
            processed_root=(project_root / "data" / "processed").resolve(),
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            temp_dir=Path(temp_dir_name),
        )
        dataset = build_dataset(
            split="train",
            runtime_data_config_path=runtime_data_config_path,
            processed_root=(project_root / "data" / "processed").resolve(),
            drug_vocab_size=4,
            max_open_shards=int(data_config["spark"]["max_open_shards_per_dataset"]),
        )
        batch = collate_batch(
            [dataset[0]],
            include_full_targets=False,
            include_final_target=True,
            max_visits=data_config["features"]["max_visits"],
            max_history=data_config["features"]["max_history"],
        )
        model, loss_fn = build_core_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            ddi_matrix_path=(project_root / "data" / "processed" / "ddi" / "drug_ddi.pt").resolve(),
        )
        outputs = model(batch, mode="core", decoder_top_k=0, compute_ddi_metrics=False)
        loss_outputs = loss_fn(
            drug_logits=outputs["drug_logits"],
            drug_probs=outputs["drug_probs"],
            target_drugs=batch["final_target_drugs"],
            visit_mask=batch["visit_mask"],
        )

    assert outputs["target_drugs"] is None
    assert outputs["final_target_drugs"] is not None
    assert tuple(batch["final_target_drugs"].shape) == (1, 4)
    assert torch.isfinite(loss_outputs["total_loss"]).item()


def test_trainer_logs_timing_metrics_and_checkpoint_profile(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("pyarrow")

    project_root, configs = _prepare_runtime_project_with_active_manual_smoke_ddi(tmp_path)
    profile_name = "fast"
    train_config = apply_profile_overrides(load_yaml_config(configs["train"]), profile_name=profile_name)
    data_config = apply_profile_overrides(load_yaml_config(configs["data"]), profile_name=profile_name)
    model_config = apply_profile_overrides(load_yaml_config(configs["model"]), profile_name=profile_name)
    train_config["_resolved_paths"] = {"processed_root": str((project_root / "data" / "processed").resolve())}

    with tempfile.TemporaryDirectory(prefix="core_trainer_metrics_") as temp_dir_name:
        runtime_data_config_path = build_runtime_data_config_file(
            data_config=data_config,
            processed_root=(project_root / "data" / "processed").resolve(),
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            temp_dir=Path(temp_dir_name),
        )
        train_loader, val_loader = build_dataloaders(
            runtime_data_config_path=runtime_data_config_path,
            processed_root=(project_root / "data" / "processed").resolve(),
            drug_vocab_size=4,
            batch_size=int(train_config["runtime"]["batch_size"]),
            num_workers=int(train_config["runtime"]["num_workers"]),
            pin_memory=bool(train_config["runtime"]["pin_memory"]),
            persistent_workers=bool(train_config["runtime"]["persistent_workers"]),
            prefetch_factor=train_config["runtime"]["prefetch_factor"],
            length_bucket_window=int(train_config["runtime"]["length_bucket_window"]),
            seed=int(data_config["seed"]),
            max_open_shards=int(data_config["spark"]["max_open_shards_per_dataset"]),
            max_visits=data_config["features"]["max_visits"],
            max_history=data_config["features"]["max_history"],
        )
        model, loss_fn = build_core_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            ddi_matrix_path=(project_root / "data" / "processed" / "ddi" / "drug_ddi.pt").resolve(),
        )

    optimizer = build_optimizer(model=model, train_config=train_config)
    scheduler = build_scheduler(optimizer=optimizer, train_config=train_config)
    trainer = TqdmCoreTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        checkpoint_dir=project_root / "outputs" / "checkpoints",
        log_dir=project_root / "outputs" / "logs",
        monitor_metric="val_total_loss",
        monitor_mode="min",
        decoder_top_k=int(train_config["runtime"]["train_decoder_top_k"]),
        amp=bool(train_config["runtime"]["amp"]),
        grad_accum_steps=int(train_config["runtime"]["grad_accum_steps"]),
        non_blocking_transfer=bool(train_config["runtime"]["non_blocking_transfer"]),
        log_interval=int(train_config["runtime"]["log_interval"]),
        profile_steps=int(train_config["runtime"]["profile_steps"]),
        run_context={
            "selected_profile": profile_name,
            "ddi_context": loss_fn.ddi_context,
            "configured_ddi_lambda": loss_fn.configured_lambda_ddi,
            "effective_ddi_lambda": loss_fn.effective_lambda_ddi,
        },
    )
    fit_result = trainer.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=1,
        extra_checkpoint_state={
            "selected_profile": profile_name,
            "train_config": {key: value for key, value in train_config.items() if not str(key).startswith("_")},
            "data_config": {key: value for key, value in data_config.items() if not str(key).startswith("_")},
            "model_config": {key: value for key, value in model_config.items() if not str(key).startswith("_")},
        },
    )

    history_entry = fit_result["history"][0]
    assert history_entry["train_data_time"] >= 0.0
    assert history_entry["train_step_time"] >= 0.0
    assert history_entry["train_samples_per_sec"] >= 0.0
    assert history_entry["val_data_time"] >= 0.0
    assert history_entry["val_step_time"] >= 0.0
    assert history_entry["val_samples_per_sec"] >= 0.0

    log_entry = json.loads(trainer.metrics_log_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert "train_data_time" in log_entry
    assert "train_step_time" in log_entry
    assert "train_samples_per_sec" in log_entry

    checkpoint = torch.load(trainer.best_checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["selected_profile"] == profile_name


def test_sparse_ddi_regularizer_matches_dense_penalty() -> None:
    torch = pytest.importorskip("torch")

    matrix = torch.tensor(
        [
            [0.0, 1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )
    probs = torch.tensor(
        [
            [0.2, 0.8, 0.4, 0.6],
            [0.7, 0.1, 0.9, 0.3],
        ],
        dtype=torch.float32,
    )
    regularizer = DDIRegularizer({"matrix": matrix}, reduction="none")

    ddi_upper = torch.triu((matrix > 0).to(dtype=torch.float32), diagonal=1)
    dense_penalty = (
        (probs.unsqueeze(2) * probs.unsqueeze(1) * ddi_upper.unsqueeze(0)).sum(dim=(1, 2))
        / ddi_upper.sum().clamp(min=1.0)
    )

    assert torch.allclose(
        regularizer.compute_penalty_per_sample(probs),
        dense_penalty,
        atol=1.0e-6,
        rtol=1.0e-6,
    )
