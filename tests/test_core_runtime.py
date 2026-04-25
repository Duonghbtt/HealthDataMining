from __future__ import annotations

import csv
import gzip
import importlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest import mock

import pytest
import src.data.build_ddi_matrix as build_ddi_module
import yaml

from src.data.build_ddi_matrix import build_ddi_matrix
from src.data.build_vocab import build_vocab, load_vocab_bundle
from src.data.dataset import ShardLengthBatchSampler, collate_batch, detect_trajectory_layout
from src.evaluation.prediction_control import (
    binarize_top_k_predictions,
    normalize_prediction_config,
    resolve_prediction_control,
)
from src.models.ddi_regularization import DDIRegularizer, load_ddi_artifact
from src.training.train_core import (
    TqdmCoreTrainer,
    apply_model_initialization,
    apply_profile_overrides,
    build_positive_class_weight,
    build_core_model,
    build_dataloaders,
    build_dataset,
    build_optimizer,
    build_runtime_data_config_file,
    build_scheduler,
    resolve_core_monitor_config,
    resolve_profile_name,
)
from src.training.losses import MedicationRecommendationLoss
from src.training.train_extended import (
    ExtendedTrainer,
    build_extended_dataloaders,
    build_extended_model,
)
from src.training.trainer import Trainer, resolve_precision_policy
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
        "ddi": {
            "source_path": "",
            "source_format": "twosides_csv",
            "fallback_source_path": "data/raw/ddi/drug_ddi_smoke.csv",
            "fallback_source_format": "manual_smoke_csv",
            "canonical_pairs_path": "data/processed/ddi/drug_ddi_pairs.csv.gz",
            "min_support_a": 5,
            "min_prr_ci_lower_bound": 1.0,
        },
        "drugbank": {
            "source_path": "data/raw/drugbank/full database.xml",
            "summary_path": "data/processed/drugbank/drugbank_summary.json",
            "records_path": "data/processed/drugbank/drugbank_drugs.jsonl.gz",
            "vocab_metadata_path": "data/interim/vocab/drugbank_drug_metadata.json",
            "ddi_pairs_path": "data/processed/ddi/drugbank_ddi_pairs.jsonl.gz",
            "ddi_matrix_path": "data/processed/ddi/drug_ddi_drugbank.pt",
            "ddi_report_path": "data/processed/ddi/drug_ddi_drugbank_report.json",
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
            "max_grad_norm": 1.0,
        },
        "loss": {
            "ddi_lambda": 0.05,
            "pos_weight_mode": "log_balanced",
            "pos_weight_clip": 12.0,
            "ddi_lambda_candidates": [0.0, 0.05, 0.01],
        },
        "threshold_tuning": {
            "enabled": True,
            "metric": "f1",
            "tie_breaker": "jaccard",
            "split": "val",
            "candidates": [0.10, 0.25, 0.50],
        },
        "prediction": {"top_k": 2, "threshold": 0.5},
        "core": {"use_retrieval": False, "use_group_encoder": False},
        "extended": {
            "mode": "extended",
            "use_retrieval": True,
            "use_group_encoder": True,
            "retrieval_top_k": 2,
            "temporal_decay_alpha": 0.05,
            "retrieval_backend": "bruteforce",
            "use_faiss_if_available": True,
            "allow_cross_split": False,
        },
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
        "prediction": {
            "mode": "threshold",
            "top_k": 2,
            "top_k_strategy": "fixed",
            "decoder_top_k": 2,
            "threshold": 0.5,
            "calibration": {
                "enabled": False,
                "split": "val",
                "metric": "f1",
                "tie_breaker": "jaccard",
                "candidates": [0.5],
            },
        },
        "evaluation": {
            "split": "test",
            "save_predictions": False,
            "save_reports": True,
        },
        "core": {"use_retrieval": False, "use_group_encoder": False},
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
        "retrieval": {
            "top_k": 2,
            "temporal_decay_alpha": 0.05,
            "backend": "bruteforce",
            "use_faiss_if_available": True,
        },
        "history_selector": {
            "dropout": 0.1,
            "score_bias_weight": 0.5,
            "self_top_k": 1,
            "neighbor_top_k": 1,
            "use_retrieval_bias": True,
        },
        "fusion": {"dropout": 0.1, "strategy": "gated"},
        "hypergraph": {
            "num_layers": 1,
            "dropout": 0.1,
            "num_group_prototypes": 4,
            "use_semantic_edges": True,
            "use_weighted_edges": True,
            "prototype_top_k": 1,
        },
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
    resolved.setdefault("ddi", {})
    resolved["ddi"]["source_path"] = ddi_source_path
    source_name = Path(ddi_source_path).name.lower()
    resolved["ddi"]["source_format"] = "twosides_csv" if source_name == "twosides.csv" else "manual_smoke_csv"
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


def _write_twosides_ddi_source(
    project_root: Path,
    *,
    rows: list[dict[str, object]] | None = None,
) -> Path:
    ddi_root = project_root / "data" / "raw" / "ddi" / "twosides"
    ddi_root.mkdir(parents=True, exist_ok=True)
    source_path = ddi_root / "TWOSIDES.csv"
    resolved_rows = rows or [
        {
            "drug_1_rxnorn_id": "1191",
            "drug_1_concept_name": "Aspirin",
            "drug_2_rxnorm_id": "5224",
            "drug_2_concept_name": "Heparin",
            "condition_meddra_id": "10013993",
            "condition_concept_name": "Gastrointestinal haemorrhage",
            "A": 6,
            "B": 1,
            "C": 1,
            "D": 1,
            "PRR": 2.0,
            "PRR_error": 0.2,
            "mean_reporting_frequency": 0.01,
        }
    ]
    fieldnames = [
        "drug_1_rxnorn_id",
        "drug_1_concept_name",
        "drug_2_rxnorm_id",
        "drug_2_concept_name",
        "condition_meddra_id",
        "condition_concept_name",
        "A",
        "B",
        "C",
        "D",
        "PRR",
        "PRR_error",
        "mean_reporting_frequency",
    ]
    lines = [",".join(fieldnames)]
    for row in resolved_rows:
        lines.append(",".join(str(row.get(field, "")) for field in fieldnames))
    source_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    write_json(
        ddi_root / "TWOSIDES.metadata.json",
        {
            "kind": "twosides_real_condition_aggregated",
            "purpose": "TwoSIDES condition-aggregated real DDI source",
            "research_grade": False,
            "pair_schema": "twosides_condition_rows",
            "display_name": "Test TWOSIDES.csv",
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
    assert model.medication_decoder.decoder_mode == "independent"
    assert model.medication_decoder.label_correlation_enabled is False


def test_build_core_model_reads_label_correlation_decoder_mode(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pyarrow")

    project_root, configs = _prepare_runtime_project(tmp_path)
    train_config = load_yaml_config(configs["train"])
    data_config = load_yaml_config(configs["data"])
    model_config = load_yaml_config(configs["model"])
    model_config["decoder"] = {
        "mode": "label_correlation_residual",
        "label_correlation": {
            "enabled": True,
            "correlation_dim": 4,
            "patient_residual_weight": 0.2,
            "coprescription_residual_weight": 0.1,
            "dropout": 0.0,
        },
    }
    train_config["_resolved_paths"] = {"processed_root": str((project_root / "data" / "processed").resolve())}

    with tempfile.TemporaryDirectory(prefix="core_runtime_labelcorr_") as temp_dir_name:
        runtime_data_config_path = build_runtime_data_config_file(
            data_config=data_config,
            processed_root=(project_root / "data" / "processed").resolve(),
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            temp_dir=Path(temp_dir_name),
        )
        model, _ = build_core_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            ddi_matrix_path=(project_root / "data" / "processed" / "ddi" / "drug_ddi.pt").resolve(),
        )

    assert model.medication_decoder.decoder_mode == "label_correlation_residual"
    assert model.medication_decoder.label_correlation_enabled is True
    assert model.medication_decoder.correlation_dim == 4
    assert model.medication_decoder.patient_residual_weight == pytest.approx(0.2)
    assert model.medication_decoder.coprescription_residual_weight == pytest.approx(0.1)


def test_build_ddi_matrix_writes_inactive_artifact_for_empty_source(tmp_path: Path) -> None:
    project_root = tmp_path / "ddi_project"
    configs = _write_config_bundle(project_root)
    _write_vocab_bundle(project_root)

    with pytest.warns(UserWarning, match="inactive fallback_zero DDI artifact"):
        ddi_path = build_ddi_matrix(configs["data"])

    ddi_artifact = load_ddi_artifact(ddi_path)
    report = json.load((project_root / "data" / "processed" / "ddi" / "drug_ddi_report.json").open())

    assert ddi_artifact["active"] is False
    assert "missing_ddi_source_path" in ddi_artifact["reason"]
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


def test_build_ddi_matrix_writes_active_twosides_artifact_and_canonical_pairs(tmp_path: Path) -> None:
    project_root = tmp_path / "ddi_twosides_project"
    configs = _write_config_bundle(project_root)
    _write_vocab_bundle(project_root)
    _write_twosides_ddi_source(project_root)
    _update_data_ddi_source_path(configs["data"], "data/raw/ddi/twosides/TWOSIDES.csv")

    ddi_path = build_ddi_matrix(configs["data"])
    ddi_artifact = load_ddi_artifact(ddi_path)
    report_path = project_root / "data" / "processed" / "ddi" / "drug_ddi_report.json"
    canonical_pairs_path = project_root / "data" / "processed" / "ddi" / "drug_ddi_pairs.csv.gz"
    report = json.load(report_path.open())

    assert ddi_artifact["active"] is True
    assert ddi_artifact["ddi_type"] == "twosides_real_condition_aggregated"
    assert ddi_artifact["ddi_research_grade"] is True
    assert ddi_artifact["matched_pairs"] == 1
    assert ddi_artifact["nonzero_pairs"] == 1
    assert report["effective_source_format"] == "twosides_csv"
    assert report["requested_source_format"] == "twosides_csv"
    assert report["rows_scanned"] == 1
    assert report["rows_mapped"] == 1
    assert report["rows_retained"] == 1
    assert report["header_rows_skipped"] == 0
    assert report["invalid_numeric_rows"] == 0
    assert report["self_pair_rows_skipped"] == 0
    assert report["canonical_rows_written"] == 1
    assert report["source_metadata"]["kind"] == "twosides_real_condition_aggregated"
    assert report["source_metadata"]["research_grade"] is True
    assert canonical_pairs_path.exists()
    with gzip.open(canonical_pairs_path, "rt", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["drug_1_token"] == "NAME:ASPIRIN"
    assert rows[0]["drug_2_token"] == "NAME:HEPARIN"
    assert rows[0]["passes_statistical_filter"] in {"True", "true", "1"}


def test_build_ddi_matrix_falls_back_to_manual_smoke_when_twosides_has_no_matches(tmp_path: Path) -> None:
    project_root = tmp_path / "ddi_twosides_fallback_project"
    configs = _write_config_bundle(project_root)
    _write_vocab_bundle(project_root)
    _write_manual_smoke_ddi_source(project_root)
    _write_twosides_ddi_source(
        project_root,
        rows=[
            {
                "drug_1_rxnorn_id": "1",
                "drug_1_concept_name": "UnknownDrugA",
                "drug_2_rxnorm_id": "2",
                "drug_2_concept_name": "UnknownDrugB",
                "condition_meddra_id": "10000001",
                "condition_concept_name": "Placeholder condition",
                "A": 9,
                "B": 1,
                "C": 1,
                "D": 1,
                "PRR": 3.0,
                "PRR_error": 0.2,
                "mean_reporting_frequency": 0.02,
            }
        ],
    )
    _update_data_ddi_source_path(configs["data"], "data/raw/ddi/twosides/TWOSIDES.csv")

    ddi_path = build_ddi_matrix(configs["data"])
    ddi_artifact = load_ddi_artifact(ddi_path)
    report = json.load((project_root / "data" / "processed" / "ddi" / "drug_ddi_report.json").open())

    assert ddi_artifact["active"] is True
    assert ddi_artifact["ddi_type"] == "manual_smoke"
    assert ddi_artifact["ddi_research_grade"] is False
    assert report["requested_source_format"] == "twosides_csv"
    assert report["effective_source_format"] == "manual_smoke_csv"
    assert report["fallback_reason"] == "no_matched_pairs"
    assert report["source_metadata"]["kind"] == "manual_smoke"


def test_build_source_result_does_not_keep_row_level_canonical_data_in_memory(tmp_path: Path) -> None:
    project_root = tmp_path / "ddi_source_result_project"
    configs = _write_config_bundle(project_root)
    _write_vocab_bundle(project_root)
    source_path = _write_twosides_ddi_source(project_root)
    data_config = load_yaml_config(configs["data"])
    drug_vocab = load_vocab_bundle(data_config)["drug"]

    result = build_ddi_module._build_source_result(
        source_path=source_path,
        source_format="twosides_csv",
        drug_vocab=drug_vocab,
        min_support_a=5,
        min_prr_ci_lower_bound=1.0,
        canonical_pairs_path=None,
    )

    assert "canonical_rows" not in result
    assert result["canonical_rows_written"] == 0
    assert result["rows_scanned"] == 1
    assert result["rows_mapped"] == 1
    assert result["rows_retained"] == 1
    assert result["header_rows_skipped"] == 0
    assert result["invalid_numeric_rows"] == 0
    assert result["self_pair_rows_skipped"] == 0


def test_build_ddi_matrix_failure_keeps_final_artifacts_unchanged_and_only_touches_part_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project_root, configs = _prepare_runtime_project_with_active_manual_smoke_ddi(tmp_path)
    _write_twosides_ddi_source(project_root)
    _update_data_ddi_source_path(configs["data"], "data/raw/ddi/twosides/TWOSIDES.csv")

    ddi_dir = project_root / "data" / "processed" / "ddi"
    matrix_path = ddi_dir / "drug_ddi.pt"
    report_path = ddi_dir / "drug_ddi_report.json"
    canonical_pairs_path = ddi_dir / "drug_ddi_pairs.csv.gz"
    original_matrix_bytes = matrix_path.read_bytes()
    original_report_bytes = report_path.read_bytes()
    original_canonical_bytes = canonical_pairs_path.read_bytes()

    emit_calls = {"count": 0}
    original_emit = build_ddi_module._emit_canonical_row

    def _crash_after_first_emit(stats, canonical_row, *, canonical_row_writer=None):
        original_emit(stats, canonical_row, canonical_row_writer=canonical_row_writer)
        emit_calls["count"] += 1
        if emit_calls["count"] == 1:
            raise RuntimeError("simulated_twosides_failure")

    monkeypatch.setattr(build_ddi_module, "_emit_canonical_row", _crash_after_first_emit)

    with pytest.raises(RuntimeError, match="simulated_twosides_failure"):
        build_ddi_matrix(configs["data"])

    assert matrix_path.read_bytes() == original_matrix_bytes
    assert report_path.read_bytes() == original_report_bytes
    assert canonical_pairs_path.read_bytes() == original_canonical_bytes
    assert not (ddi_dir / "drug_ddi.pt.part").exists()
    assert not (ddi_dir / "drug_ddi_report.json.part").exists()
    assert (ddi_dir / "drug_ddi_pairs.csv.gz.part").exists()


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
    assert report["pipeline_level"] == "core+history"
    assert report["history_active"] is True
    assert report["retrieval_active"] is False
    assert report["fusion_strategy"] == "gated"
    assert report["ddi_type"] == "manual_smoke"
    assert report["ddi_research_grade"] is False
    assert report["metrics"]["ddi_rate"] is not None
    assert report["ddi_summary"]["available"] is True
    assert report["ddi_summary"]["status"] == "active"
    assert report["ddi_summary"]["ddi_type"] == "manual_smoke"
    assert report["ddi_summary"]["ddi_research_grade"] is False
    assert report["ddi_summary"]["source_metadata"]["kind"] == "manual_smoke"
    assert report["ddi_summary"]["source_metadata"]["research_grade"] is False
    assert report["ddi_context"]["training"]["status"] == "active"
    assert report["ddi_context"]["training"]["source_metadata"]["kind"] == "manual_smoke"
    assert report["ddi_context"]["training"]["source_metadata"]["research_grade"] is False
    assert report["ddi_context"]["evaluation"]["status"] == "active"
    assert report["ddi_context"]["evaluation"]["source_metadata"]["kind"] == "manual_smoke"
    assert report["ddi_context"]["evaluation"]["source_metadata"]["research_grade"] is False


def test_evaluate_safety_runs_and_reports_manual_smoke_truth(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    project_root, configs = _prepare_runtime_project_with_active_manual_smoke_ddi(tmp_path)
    train_config = load_yaml_config(configs["train"])
    data_config = load_yaml_config(configs["data"])
    model_config = load_yaml_config(configs["model"])
    train_config["_resolved_paths"] = {"processed_root": str((project_root / "data" / "processed").resolve())}

    with tempfile.TemporaryDirectory(prefix="safety_runtime_manual_smoke_") as temp_dir_name:
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
            "pipeline_level": "core+history",
            "history_active": True,
            "retrieval_active": False,
            "fusion_strategy": "gated",
            "train_mode": "core",
            "ddi_context": loss_fn.ddi_context,
            "ddi_type": "manual_smoke",
            "ddi_research_grade": False,
            "ddi_source": str((project_root / "data" / "processed" / "ddi" / "drug_ddi.pt").resolve()),
            "configured_ddi_lambda": loss_fn.configured_lambda_ddi,
            "effective_ddi_lambda": loss_fn.effective_lambda_ddi,
        },
        checkpoint_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.evaluate_safety",
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

    report = json.load((project_root / "outputs" / "reports" / "evaluate_safety_test.json").open())
    assert report["pipeline_level"] == "core+history"
    assert report["history_active"] is True
    assert report["retrieval_active"] is False
    assert report["fusion_strategy"] == "gated"
    assert report["ddi_type"] == "manual_smoke"
    assert report["ddi_research_grade"] is False
    assert report["report_type"] == "safety_smoke"
    assert report["ddi_summary"]["ddi_type"] == "manual_smoke"
    assert report["ddi_summary"]["ddi_research_grade"] is False
    assert report["ddi_context"]["training"]["source_metadata"]["kind"] == "manual_smoke"
    assert report["ddi_context"]["evaluation"]["source_metadata"]["kind"] == "manual_smoke"


def test_evaluate_subgroup_reports_core_truth_and_manual_smoke_ddi(tmp_path: Path) -> None:
    pytest.importorskip("pyarrow")

    project_root, configs = _prepare_runtime_project_with_active_manual_smoke_ddi(tmp_path)
    train_config = load_yaml_config(configs["train"])
    data_config = load_yaml_config(configs["data"])
    model_config = load_yaml_config(configs["model"])
    train_config["_resolved_paths"] = {"processed_root": str((project_root / "data" / "processed").resolve())}

    with tempfile.TemporaryDirectory(prefix="subgroup_runtime_manual_smoke_") as temp_dir_name:
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
            "pipeline_level": "core+history",
            "history_active": True,
            "retrieval_active": False,
            "fusion_strategy": "gated",
            "train_mode": "core",
            "ddi_context": loss_fn.ddi_context,
            "ddi_type": "manual_smoke",
            "ddi_research_grade": False,
            "ddi_source": str((project_root / "data" / "processed" / "ddi" / "drug_ddi.pt").resolve()),
            "configured_ddi_lambda": loss_fn.configured_lambda_ddi,
            "effective_ddi_lambda": loss_fn.effective_lambda_ddi,
        },
        checkpoint_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.evaluate_subgroup",
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

    report = json.load((project_root / "outputs" / "reports" / "evaluate_subgroup_test.json").open())
    assert report["pipeline_level"] == "core+history"
    assert report["history_active"] is True
    assert report["retrieval_active"] is False
    assert report["fusion_strategy"] == "gated"
    assert report["ddi_type"] == "manual_smoke"
    assert report["ddi_research_grade"] is False
    assert report["report_type"] == "subgroup_core"
    assert report["ddi_summary"]["ddi_type"] == "manual_smoke"
    assert report["ddi_summary"]["ddi_research_grade"] is False


def _prepare_core_eval_checkpoint(
    tmp_path: Path,
    *,
    checkpoint_overrides: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Path]]:
    torch = pytest.importorskip("torch")
    pytest.importorskip("pyarrow")

    project_root, configs = _prepare_runtime_project_with_active_manual_smoke_ddi(tmp_path)
    train_config = load_yaml_config(configs["train"])
    data_config = load_yaml_config(configs["data"])
    model_config = load_yaml_config(configs["model"])
    train_config["_resolved_paths"] = {"processed_root": str((project_root / "data" / "processed").resolve())}

    with tempfile.TemporaryDirectory(prefix="eval_runtime_checkpoint_") as temp_dir_name:
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
    checkpoint_payload: dict[str, Any] = {
        "epoch": 1,
        "best_metric": 0.0,
        "monitor_metric": "val_f1_tuned",
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
    }
    if checkpoint_overrides:
        checkpoint_payload.update(dict(checkpoint_overrides))
    torch.save(checkpoint_payload, checkpoint_path)
    return project_root, configs


def _run_core_family_evaluator(
    *,
    project_root: Path,
    configs: Mapping[str, Path],
    module_name: str,
    extra_args: list[str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    report_name = {
        "src.evaluation.evaluate_core": "evaluate_core_test.json",
        "src.evaluation.evaluate_safety": "evaluate_safety_test.json",
        "src.evaluation.evaluate_subgroup": "evaluate_subgroup_test.json",
    }[module_name]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module_name,
            "--config",
            str(configs["eval"]),
            "--device",
            "cpu",
            *(extra_args or []),
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    report = json.load((project_root / "outputs" / "reports" / report_name).open())
    return result, report


def test_evaluate_core_threshold_precedence_prefers_checkpoint_then_cli(tmp_path: Path) -> None:
    project_root, configs = _prepare_core_eval_checkpoint(
        tmp_path,
        checkpoint_overrides={
            "effective_threshold": 0.25,
            "threshold_selection": {
                "source": "validation_sweep",
                "split": "val",
                "metric": "f1",
                "tie_breaker": "jaccard",
                "candidates": [0.10, 0.25, 0.50],
                "best_threshold": 0.25,
            },
        },
    )

    _, report = _run_core_family_evaluator(
        project_root=project_root,
        configs=configs,
        module_name="src.evaluation.evaluate_core",
    )
    assert report["prediction_mode"] == "threshold"
    assert report["prediction_control"]["mode"] == "threshold"
    assert report["threshold"] == pytest.approx(0.25)
    assert report["threshold_source"] == "checkpoint.threshold_selection.best_threshold"
    assert report["threshold_selection"]["source"] == "validation_sweep"
    assert report["threshold_selection"]["best_threshold"] == pytest.approx(0.25)
    assert report["threshold_selection"]["effective_threshold"] == pytest.approx(0.25)

    _, report = _run_core_family_evaluator(
        project_root=project_root,
        configs=configs,
        module_name="src.evaluation.evaluate_core",
        extra_args=["--threshold", "0.4"],
    )
    assert report["prediction_mode"] == "threshold"
    assert report["prediction_control"]["mode"] == "threshold"
    assert report["threshold"] == pytest.approx(0.4)
    assert report["threshold_source"] == "cli"
    assert report["threshold_selection"]["source"] == "cli"
    assert report["threshold_selection"]["best_threshold"] == pytest.approx(0.4)


def test_evaluators_carry_checkpoint_threshold_ddi_and_initialization_metadata(tmp_path: Path) -> None:
    project_root, configs = _prepare_core_eval_checkpoint(
        tmp_path,
        checkpoint_overrides={
            "ddi_type": "manual_smoke",
            "ddi_research_grade": False,
            "initialization_mode": "warm_start_model_only",
            "warm_start_mode": "model_only",
            "warm_start_checkpoint": str((Path(tmp_path) / "seed.pt").resolve()),
            "train_budget_label": "warm_start_full_data_max5ep",
            "effective_threshold": 0.25,
            "threshold_selection": {
                "source": "validation_sweep",
                "split": "val",
                "metric": "f1",
                "tie_breaker": "jaccard",
                "candidates": [0.10, 0.25, 0.50],
                "best_threshold": 0.25,
            },
        },
    )

    for module_name in (
        "src.evaluation.evaluate_core",
        "src.evaluation.evaluate_safety",
        "src.evaluation.evaluate_subgroup",
    ):
        _, report = _run_core_family_evaluator(
            project_root=project_root,
            configs=configs,
            module_name=module_name,
        )
        assert report["prediction_mode"] == "threshold"
        assert report["prediction_control"]["mode"] == "threshold"
        assert report["threshold"] == pytest.approx(0.25)
        assert report["threshold_source"] == "checkpoint.threshold_selection.best_threshold"
        assert report["threshold_selection"]["source"] == "validation_sweep"
        assert report["ddi_type"] == "manual_smoke"
        assert report["ddi_research_grade"] is False
        assert report["ddi_summary"]["ddi_type"] == "manual_smoke"
        assert report["ddi_summary"]["ddi_research_grade"] is False
        assert report["ddi_summary"]["effective_source_format"] == "manual_smoke_csv"
        assert report["ddi_summary"]["matched_pairs"] == 1
        assert report["initialization_mode"] == "warm_start_model_only"
        assert report["warm_start_mode"] == "model_only"
        assert Path(report["warm_start_checkpoint"]).name == "seed.pt"
        assert report["train_budget_label"] == "warm_start_full_data_max5ep"


def test_evaluators_prefer_validation_sweep_threshold_when_checkpoint_metadata_conflicts(tmp_path: Path) -> None:
    project_root, configs = _prepare_core_eval_checkpoint(
        tmp_path,
        checkpoint_overrides={
            "effective_threshold": 0.5,
            "threshold_selection": {
                "source": "validation_sweep",
                "split": "val",
                "metric": "f1",
                "tie_breaker": "jaccard",
                "candidates": [0.25, 0.5],
                "best_threshold": 0.25,
            },
        },
    )

    for module_name in (
        "src.evaluation.evaluate_core",
        "src.evaluation.evaluate_safety",
        "src.evaluation.evaluate_subgroup",
    ):
        result, report = _run_core_family_evaluator(
            project_root=project_root,
            configs=configs,
            module_name=module_name,
        )
        assert "Checkpoint threshold metadata mismatch" in result.stderr
        assert report["prediction_mode"] == "threshold"
        assert report["prediction_control"]["mode"] == "threshold"
        assert report["threshold"] == pytest.approx(0.25)
        assert report["threshold_source"] == "checkpoint.threshold_selection.best_threshold"
        assert report["threshold_selection"]["source"] == "validation_sweep"
        assert report["threshold_selection"]["best_threshold"] == pytest.approx(0.25)
        assert report["threshold_selection"]["effective_threshold"] == pytest.approx(0.5)
        mismatch = report["threshold_selection"]["checkpoint_threshold_metadata_mismatch"]
        assert mismatch["effective_threshold"] == pytest.approx(0.5)
        assert mismatch["threshold_selection_best_threshold"] == pytest.approx(0.25)
        assert mismatch["resolved_threshold"] == pytest.approx(0.25)
        assert mismatch["resolved_threshold_source"] == "checkpoint.threshold_selection.best_threshold"


def test_evaluators_cli_threshold_override_wins_on_checkpoint_threshold_metadata_conflict(tmp_path: Path) -> None:
    project_root, configs = _prepare_core_eval_checkpoint(
        tmp_path,
        checkpoint_overrides={
            "effective_threshold": 0.5,
            "threshold_selection": {
                "source": "validation_sweep",
                "split": "val",
                "metric": "f1",
                "tie_breaker": "jaccard",
                "candidates": [0.25, 0.5],
                "best_threshold": 0.25,
            },
        },
    )

    for module_name in (
        "src.evaluation.evaluate_core",
        "src.evaluation.evaluate_safety",
        "src.evaluation.evaluate_subgroup",
    ):
        _, report = _run_core_family_evaluator(
            project_root=project_root,
            configs=configs,
            module_name=module_name,
            extra_args=["--threshold", "0.4"],
        )
        assert report["prediction_mode"] == "threshold"
        assert report["prediction_control"]["mode"] == "threshold"
        assert report["threshold"] == pytest.approx(0.4)
        assert report["threshold_source"] == "cli"
        assert report["threshold_selection"]["source"] == "cli"
        assert report["threshold_selection"]["best_threshold"] == pytest.approx(0.4)


def test_evaluators_fall_back_to_config_threshold_when_checkpoint_has_no_threshold_metadata(tmp_path: Path) -> None:
    project_root, configs = _prepare_core_eval_checkpoint(tmp_path)

    for module_name in (
        "src.evaluation.evaluate_core",
        "src.evaluation.evaluate_safety",
        "src.evaluation.evaluate_subgroup",
    ):
        _, report = _run_core_family_evaluator(
            project_root=project_root,
            configs=configs,
            module_name=module_name,
        )
        assert report["prediction_mode"] == "threshold"
        assert report["prediction_control"]["mode"] == "threshold"
        assert report["threshold"] == pytest.approx(0.5)
        assert report["threshold_source"] == "config.prediction.threshold"
        assert report["threshold_selection"]["source"] == "config.prediction.threshold"
        assert report["threshold_selection"]["best_threshold"] == pytest.approx(0.5)


def test_resolve_prediction_control_selects_validation_calibrated_threshold() -> None:
    torch = pytest.importorskip("torch")
    prediction_config = normalize_prediction_config(
        {
            "mode": "threshold",
            "top_k": 2,
            "top_k_strategy": "fixed",
            "threshold": 0.5,
            "calibration": {
                "enabled": True,
                "split": "val",
                "metric": "f1",
                "tie_breaker": "jaccard",
                "candidates": [0.5, 0.65, 0.8],
            },
        }
    )
    calibration_targets = torch.tensor(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    calibration_probs = torch.tensor(
        [
            [0.66, 0.55],
            [0.70, 0.10],
            [0.20, 0.67],
        ]
    )
    eval_probs = torch.tensor(
        [
            [0.68, 0.62],
            [0.30, 0.69],
        ]
    )
    eval_targets = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )

    resolved = resolve_prediction_control(
        prediction_config=prediction_config,
        checkpoint_payload={"effective_threshold": 0.25},
        cli_threshold=None,
        cli_prediction_mode=None,
        cli_prediction_top_k=None,
        eval_probs=eval_probs,
        eval_targets=eval_targets,
        ddi_matrix=None,
        calibration_probs=calibration_probs,
        calibration_targets=calibration_targets,
    )

    assert resolved["prediction_mode"] == "calibrated_threshold"
    assert resolved["threshold"] == pytest.approx(0.65)
    assert resolved["threshold_source"] == "evaluation.calibration.best_threshold"
    assert resolved["threshold_selection"]["source"] == "evaluation.calibration"
    assert resolved["threshold_selection"]["best_threshold"] == pytest.approx(0.65)
    assert resolved["binary_predictions"].sum(dim=1).tolist() == [1, 1]


def test_resolve_prediction_control_top_k_modes_keep_expected_cardinality() -> None:
    torch = pytest.importorskip("torch")
    eval_probs = torch.tensor(
        [
            [0.8, 0.7, 0.1, 0.2],
            [0.3, 0.9, 0.5, 0.4],
        ]
    )
    eval_targets = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 0.0],
        ]
    )
    fixed_config = normalize_prediction_config(
        {
            "mode": "top_k",
            "top_k": 2,
            "top_k_strategy": "fixed",
            "threshold": 0.5,
            "calibration": {"enabled": False, "candidates": [0.5]},
        }
    )
    fixed_resolved = resolve_prediction_control(
        prediction_config=fixed_config,
        checkpoint_payload=None,
        cli_threshold=None,
        cli_prediction_mode=None,
        cli_prediction_top_k=None,
        eval_probs=eval_probs,
        eval_targets=eval_targets,
        ddi_matrix=None,
    )
    assert fixed_resolved["prediction_mode"] == "top_k"
    assert fixed_resolved["prediction_control"]["top_k"] == 2
    assert fixed_resolved["prediction_control"]["top_k_source"] == "config.prediction.top_k"
    assert fixed_resolved["binary_predictions"].sum(dim=1).tolist() == [2, 2]

    avg_train_config = normalize_prediction_config(
        {
            "mode": "top_k",
            "top_k": 2,
            "top_k_strategy": "avg_train_drugs",
            "threshold": 0.5,
            "calibration": {"enabled": False, "candidates": [0.5]},
        }
    )
    avg_resolved = resolve_prediction_control(
        prediction_config=avg_train_config,
        checkpoint_payload=None,
        cli_threshold=None,
        cli_prediction_mode=None,
        cli_prediction_top_k=None,
        eval_probs=eval_probs,
        eval_targets=eval_targets,
        ddi_matrix=None,
        avg_train_drugs=2.6,
    )
    assert avg_resolved["prediction_mode"] == "top_k"
    assert avg_resolved["prediction_control"]["top_k"] == 3
    assert avg_resolved["prediction_control"]["top_k_source"] == "train.avg_true_drugs"
    assert avg_resolved["prediction_control"]["avg_train_drugs"] == pytest.approx(2.6)
    assert avg_resolved["binary_predictions"].sum(dim=1).tolist() == [3, 3]


def test_binarize_top_k_predictions_returns_exact_cardinality() -> None:
    torch = pytest.importorskip("torch")
    predictions = binarize_top_k_predictions(
        torch.tensor(
            [
                [0.8, 0.2, 0.5],
                [0.1, 0.6, 0.4],
                [0.9, 0.7, 0.3],
            ]
        ),
        top_k=2,
    )
    assert predictions.dtype == torch.bool
    assert predictions.sum(dim=1).tolist() == [2, 2, 2]


def test_cli_threshold_override_disables_calibrated_threshold_mode() -> None:
    torch = pytest.importorskip("torch")
    prediction_config = normalize_prediction_config(
        {
            "mode": "threshold",
            "top_k": 2,
            "top_k_strategy": "fixed",
            "threshold": 0.5,
            "calibration": {
                "enabled": True,
                "split": "val",
                "metric": "f1",
                "tie_breaker": "jaccard",
                "candidates": [0.5, 0.65],
            },
        }
    )
    resolved = resolve_prediction_control(
        prediction_config=prediction_config,
        checkpoint_payload={"effective_threshold": 0.25},
        cli_threshold=0.4,
        cli_prediction_mode=None,
        cli_prediction_top_k=None,
        eval_probs=torch.tensor([[0.45, 0.35], [0.41, 0.39]]),
        eval_targets=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        ddi_matrix=None,
        calibration_probs=torch.tensor([[0.9, 0.1]]),
        calibration_targets=torch.tensor([[1.0, 0.0]]),
    )
    assert resolved["prediction_mode"] == "threshold"
    assert resolved["threshold"] == pytest.approx(0.4)
    assert resolved["threshold_source"] == "cli"


def test_evaluators_support_calibrated_threshold_mode_consistently(tmp_path: Path) -> None:
    project_root, configs = _prepare_core_eval_checkpoint(tmp_path)
    eval_config = load_yaml_config(configs["eval"])
    eval_config["prediction"]["calibration"]["enabled"] = True
    eval_config["prediction"]["calibration"]["candidates"] = [0.25, 0.5, 0.75]
    _write_yaml(configs["eval"], {key: value for key, value in eval_config.items() if not str(key).startswith("_")})

    resolved_threshold = None
    for module_name in (
        "src.evaluation.evaluate_core",
        "src.evaluation.evaluate_safety",
        "src.evaluation.evaluate_subgroup",
    ):
        _, report = _run_core_family_evaluator(
            project_root=project_root,
            configs=configs,
            module_name=module_name,
        )
        assert report["prediction_mode"] == "calibrated_threshold"
        assert report["prediction_control"]["mode"] == "calibrated_threshold"
        assert report["threshold_source"] == "evaluation.calibration.best_threshold"
        assert report["threshold_selection"]["source"] == "evaluation.calibration"
        if resolved_threshold is None:
            resolved_threshold = float(report["threshold"])
        assert report["threshold"] == pytest.approx(resolved_threshold)


def test_evaluators_support_cli_top_k_override_consistently(tmp_path: Path) -> None:
    project_root, configs = _prepare_core_eval_checkpoint(tmp_path)

    for module_name in (
        "src.evaluation.evaluate_core",
        "src.evaluation.evaluate_safety",
        "src.evaluation.evaluate_subgroup",
    ):
        _, report = _run_core_family_evaluator(
            project_root=project_root,
            configs=configs,
            module_name=module_name,
            extra_args=["--prediction-mode", "top_k", "--prediction-top-k", "1"],
        )
        assert report["prediction_mode"] == "top_k"
        assert report["prediction_control"]["mode"] == "top_k"
        assert report["prediction_control"]["top_k"] == 1
        assert report["prediction_control"]["top_k_source"] == "cli"
        assert report["threshold"] is None
        assert report["threshold_source"] is None
        assert report["threshold_selection"] is None


def test_new_calibrated_benchmark_configs_parse_with_isolated_outputs() -> None:
    train_config = load_yaml_config("configs/benchmarks/warm_asym_focal_calibrated_safe.yaml")
    eval_config = load_yaml_config("configs/benchmarks/warm_asym_focal_calibrated_safe.eval.yaml")

    assert train_config["runtime"]["train_budget_label"] == "warm_asym_focal_calibrated_safe"
    assert train_config["threshold_tuning"]["candidates"][-1] == pytest.approx(0.9)
    assert eval_config["config_refs"]["train"] == "warm_asym_focal_calibrated_safe.yaml"
    assert eval_config["prediction"]["calibration"]["enabled"] is True
    assert eval_config["prediction"]["calibration"]["candidates"][-1] == pytest.approx(0.9)
    assert "warm_asym_focal_calibrated_safe" in eval_config["paths"]["report_dir"]
    assert "warm_asym_focal_calibrated_safe" in eval_config["paths"]["prediction_dir"]


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


def test_build_extended_dataloaders_honor_runtime_knobs_and_preserve_prepared_records(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pyarrow")

    project_root, configs = _prepare_runtime_project_with_active_manual_smoke_ddi(tmp_path)
    train_config = apply_profile_overrides(load_yaml_config(configs["train"]), profile_name="fast")
    data_config = apply_profile_overrides(load_yaml_config(configs["data"]), profile_name="fast")

    with tempfile.TemporaryDirectory(prefix="extended_fast_loader_") as temp_dir_name:
        runtime_data_config_path = build_runtime_data_config_file(
            data_config=data_config,
            processed_root=(project_root / "data" / "processed").resolve(),
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            temp_dir=Path(temp_dir_name),
        )
        configured_train_loader, configured_val_loader, configured_train_bank_loader, configured_val_bank_loader = build_extended_dataloaders(
            runtime_data_config_path=runtime_data_config_path,
            processed_root=(project_root / "data" / "processed").resolve(),
            drug_vocab_size=4,
            batch_size=int(train_config["runtime"]["batch_size"]),
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            length_bucket_window=int(train_config["runtime"]["length_bucket_window"]),
            seed=int(data_config["seed"]),
            max_open_shards=int(data_config["spark"]["max_open_shards_per_dataset"]),
            max_visits=data_config["features"]["max_visits"],
            max_history=data_config["features"]["max_history"],
        )
        train_loader, _, train_bank_loader, _ = build_extended_dataloaders(
            runtime_data_config_path=runtime_data_config_path,
            processed_root=(project_root / "data" / "processed").resolve(),
            drug_vocab_size=4,
            batch_size=int(train_config["runtime"]["batch_size"]),
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None,
            length_bucket_window=int(train_config["runtime"]["length_bucket_window"]),
            seed=int(data_config["seed"]),
            max_open_shards=int(data_config["spark"]["max_open_shards_per_dataset"]),
            max_visits=data_config["features"]["max_visits"],
            max_history=data_config["features"]["max_history"],
        )

    train_batch = next(iter(train_loader))
    bank_batch = next(iter(train_bank_loader))

    assert isinstance(configured_train_loader.batch_sampler, ShardLengthBatchSampler)
    assert configured_train_loader.num_workers == 1
    assert configured_train_loader.persistent_workers is True
    assert getattr(configured_train_loader, "prefetch_factor", None) == 4
    assert configured_val_loader.num_workers == 1
    assert configured_train_bank_loader.num_workers == 1
    assert configured_val_bank_loader.num_workers == 1
    assert "target_drugs" not in train_batch
    assert "final_target_drugs" in train_batch
    assert "records" in train_batch
    assert int(train_batch["visit_mask"].shape[1]) == 1
    assert all(int(record["num_steps"]) <= 1 for record in train_batch["records"])
    assert all(
        len(record["steps"][-1].get("med_history_ids", [])) <= 4
        for record in train_batch["records"]
        if record["steps"]
    )
    assert "target_drugs" not in bank_batch
    assert "records" in bank_batch


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
    assert outputs["runtime_truth"]["pipeline_level"] == "core+history"
    assert outputs["runtime_truth"]["history_active"] is True
    assert outputs["runtime_truth"]["retrieval_active"] is False
    assert outputs["runtime_truth"]["fusion_strategy"] == "gated"
    assert tuple(batch["final_target_drugs"].shape) == (1, 4)
    assert torch.isfinite(loss_outputs["total_loss"]).item()


def test_build_positive_class_weight_log_balanced_clips_expected_values() -> None:
    class TinyDataset:
        def __len__(self) -> int:
            return 4

        def __getitem__(self, index: int) -> dict[str, object]:
            target_rows = {
                0: [2],
                1: [2],
                2: [2, 3],
                3: [],
            }
            return {"steps": [{"target_drugs": list(target_rows[index])}]}

    weights, stats = build_positive_class_weight(
        dataset=TinyDataset(),
        drug_vocab_size=5,
        mode="log_balanced",
        clip=1.1,
    )

    assert weights is not None
    assert stats["mode"] == "log_balanced"
    assert stats["num_samples"] == 4
    assert stats["num_labels_with_positive"] == 2
    assert weights[0].item() == pytest.approx(1.0)
    assert weights[1].item() == pytest.approx(1.0)
    assert weights[2].item() == pytest.approx(1.0)
    assert weights[3].item() == pytest.approx(1.1)
    assert weights[4].item() == pytest.approx(1.0)


def test_apply_model_initialization_warm_start_model_only_loads_weights(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    source_model = torch.nn.Linear(3, 2)
    target_model = torch.nn.Linear(3, 2)

    with torch.no_grad():
        source_model.weight.copy_(torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=torch.float32))
        source_model.bias.copy_(torch.tensor([0.5, -0.5], dtype=torch.float32))
        target_model.weight.zero_()
        target_model.bias.zero_()

    checkpoint_path = tmp_path / "warm_start.pt"
    torch.save(
        {
            "model_state_dict": source_model.state_dict(),
            "monitor_metric": "val_f1_tuned",
            "ddi_type": "manual_smoke",
            "ddi_research_grade": False,
        },
        checkpoint_path,
    )

    initialization_context = apply_model_initialization(
        model=target_model,
        train_config={
            "_project_root": str(tmp_path),
            "initialization": {
                "warm_start_checkpoint": checkpoint_path.name,
                "warm_start_mode": "model_only",
                "strict": True,
            },
        },
    )

    assert initialization_context["initialization_mode"] == "warm_start_model_only"
    assert initialization_context["warm_start_mode"] == "model_only"
    assert Path(initialization_context["warm_start_checkpoint"]).resolve() == checkpoint_path.resolve()
    assert torch.allclose(target_model.weight, source_model.weight)
    assert torch.allclose(target_model.bias, source_model.bias)


def test_build_scheduler_supports_reduce_on_plateau_with_max_monitor() -> None:
    torch = pytest.importorskip("torch")

    parameter = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    optimizer = torch.optim.Adam([parameter], lr=1.0e-3)
    scheduler = build_scheduler(
        optimizer=optimizer,
        train_config={
            "optimization": {
                "scheduler": "reduce_on_plateau",
                "scheduler_factor": 0.5,
                "scheduler_patience": 1,
                "scheduler_min_lr": 1.0e-6,
            }
        },
        monitor_mode="max",
    )

    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert getattr(scheduler, "mode", None) == "max"
    assert getattr(scheduler, "factor", None) == pytest.approx(0.5)
    assert getattr(scheduler, "patience", None) == 1


def test_trainer_threshold_tuning_rejects_non_validation_split(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("pyarrow")

    project_root, configs = _prepare_runtime_project_with_active_manual_smoke_ddi(tmp_path)
    train_config = apply_profile_overrides(load_yaml_config(configs["train"]), profile_name="balanced")
    data_config = apply_profile_overrides(load_yaml_config(configs["data"]), profile_name="balanced")
    model_config = apply_profile_overrides(load_yaml_config(configs["model"]), profile_name="balanced")
    train_config["_resolved_paths"] = {"processed_root": str((project_root / "data" / "processed").resolve())}

    with tempfile.TemporaryDirectory(prefix="core_trainer_threshold_split_") as temp_dir_name:
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
        pos_weight, _ = build_positive_class_weight(
            dataset=train_loader.dataset,
            drug_vocab_size=4,
            mode="log_balanced",
            clip=12.0,
        )
        model, loss_fn = build_core_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            ddi_matrix_path=(project_root / "data" / "processed" / "ddi" / "drug_ddi.pt").resolve(),
            pos_weight=pos_weight,
        )

    optimizer = build_optimizer(model=model, train_config=train_config)
    trainer = TqdmCoreTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=torch.device("cpu"),
        checkpoint_dir=project_root / "outputs" / "checkpoints",
        log_dir=project_root / "outputs" / "logs",
        monitor_metric="val_f1_tuned",
        monitor_mode="max",
        decoder_top_k=int(train_config["runtime"]["train_decoder_top_k"]),
        run_context={
            "threshold_tuning": {
                "enabled": True,
                "metric": "f1",
                "tie_breaker": "jaccard",
                "split": "test",
                "candidates": [0.10, 0.25, 0.50],
            }
        },
    )

    with pytest.raises(ValueError, match="validation split only"):
        trainer.fit(train_dataloader=train_loader, val_dataloader=val_loader, epochs=1)


def test_trainer_reduce_on_plateau_steps_with_monitor_metric_and_stops_early(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class RecordingReduceLROnPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.seen_metrics: list[float] = []

        def step(self, metrics, epoch=None) -> None:
            self.seen_metrics.append(float(metrics))
            super().step(metrics, epoch=epoch)

    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0e-3)
    scheduler = RecordingReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1,
        min_lr=1.0e-6,
    )
    trainer = TqdmCoreTrainer(
        model=model,
        loss_fn=torch.nn.Identity(),
        optimizer=optimizer,
        scheduler=scheduler,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        monitor_metric="val_f1_tuned",
        monitor_mode="max",
        early_stopping_patience=1,
        run_context={
            "threshold_tuning": {
                "enabled": True,
                "metric": "f1",
                "tie_breaker": "jaccard",
                "split": "val",
                "candidates": [0.24, 0.25, 0.26],
            },
            "effective_threshold": 0.25,
            "threshold_selection": {
                "source": "config.prediction.threshold",
                "split": "config",
                "metric": "f1",
                "tie_breaker": "jaccard",
                "candidates": [0.24, 0.25, 0.26],
                "best_threshold": 0.25,
            },
            "ddi_context": {
                "active": True,
                "status": "active",
                "source_metadata": {"kind": "manual_smoke", "research_grade": False},
            },
            "configured_ddi_lambda": 0.05,
            "effective_ddi_lambda": 0.05,
            "pipeline_level": "core+history",
            "history_active": True,
            "retrieval_active": False,
            "fusion_strategy": "gated",
        },
    )

    train_epoch_metrics = {
        "train_total_loss": 1.0,
        "train_prediction_loss": 0.9,
        "train_ddi_loss": 0.1,
        "train_weighted_ddi_loss": 0.005,
    }
    val_epoch_metrics = {
        "val_total_loss": 0.8,
        "val_prediction_loss": 0.7,
        "val_ddi_loss": 0.1,
        "val_weighted_ddi_loss": 0.005,
    }
    tuning_payloads = [
        {
            "val_f1_tuned": 0.60,
            "val_jaccard_tuned": 0.40,
            "val_prauc_tuned": 0.10,
            "val_threshold_best": 0.24,
            "val_avg_predicted_drugs_tuned": 22.0,
            "val_avg_true_drugs": 23.0,
        },
        {
            "val_f1_tuned": 0.59,
            "val_jaccard_tuned": 0.39,
            "val_prauc_tuned": 0.10,
            "val_threshold_best": 0.25,
            "val_avg_predicted_drugs_tuned": 22.2,
            "val_avg_true_drugs": 23.0,
        },
    ]

    with (
        mock.patch.object(trainer, "train_one_epoch", side_effect=[dict(train_epoch_metrics), dict(train_epoch_metrics)]),
        mock.patch.object(trainer, "validate_one_epoch", side_effect=[dict(val_epoch_metrics), dict(val_epoch_metrics)]),
        mock.patch.object(trainer, "_run_threshold_tuning", side_effect=tuning_payloads),
    ):
        fit_result = trainer.fit(
            train_dataloader=[None],
            val_dataloader=[None],
            epochs=4,
            extra_checkpoint_state={"train_mode": "core"},
        )

    assert fit_result["monitor_metric"] == "val_f1_tuned"
    assert fit_result["epochs_completed"] == 2
    assert fit_result["stopped_early"] is True
    assert "val_f1_tuned" in str(fit_result["stop_reason"])
    assert scheduler.seen_metrics == pytest.approx([0.60, 0.59])

    checkpoint = torch.load(trainer.best_checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["monitor_metric"] == "val_f1_tuned"
    assert checkpoint["threshold_selection"]["best_threshold"] == pytest.approx(0.25)


def test_core_monitor_config_defaults_and_prauc_override() -> None:
    default_metric, default_mode = resolve_core_monitor_config(
        {"optimization": {}},
        {"enabled": True},
    )
    assert default_metric == "val_f1_tuned"
    assert default_mode == "max"

    loss_metric, loss_mode = resolve_core_monitor_config(
        {"optimization": {}},
        {"enabled": False},
    )
    assert loss_metric == "val_total_loss"
    assert loss_mode == "min"

    prauc_metric, prauc_mode = resolve_core_monitor_config(
        {"optimization": {"monitor_metric": "val_prauc_tuned", "monitor_mode": "max"}},
        {"enabled": True},
    )
    assert prauc_metric == "val_prauc_tuned"
    assert prauc_mode == "max"


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
        max_grad_norm=float(train_config["optimization"]["max_grad_norm"]),
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
    assert log_entry["run_context"]["pipeline_level"] == "core+history"
    assert log_entry["run_context"]["history_active"] is True
    assert log_entry["run_context"]["retrieval_active"] is False
    assert log_entry["run_context"]["fusion_strategy"] == "gated"
    assert log_entry["run_context"]["ddi_type"] == "manual_smoke"
    assert log_entry["run_context"]["ddi_research_grade"] is False
    assert log_entry["run_context"]["runtime"]["requested_amp"] is False
    assert log_entry["run_context"]["runtime"]["resolved_precision"] == "fp32"
    assert log_entry["run_context"]["runtime"]["grad_scaler_enabled"] is False
    assert log_entry["run_context"]["runtime"]["max_grad_norm"] == pytest.approx(1.0)

    checkpoint = torch.load(trainer.best_checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["selected_profile"] == profile_name
    assert checkpoint["pipeline_level"] == "core+history"
    assert checkpoint["history_active"] is True
    assert checkpoint["retrieval_active"] is False
    assert checkpoint["fusion_strategy"] == "gated"
    assert checkpoint["ddi_type"] == "manual_smoke"
    assert checkpoint["ddi_research_grade"] is False
    assert checkpoint["train_mode"] == "core"


def test_trainer_detailed_timing_metrics_are_opt_in(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class TimedCoreModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def forward(self, batch, *, mode, decoder_top_k, compute_ddi_metrics=False):
            _ = mode
            _ = decoder_top_k
            _ = compute_ddi_metrics
            batch_size = int(batch["visit_mask"].shape[0])
            logits = self.anchor.expand(batch_size, 2)
            return {
                "pooled_state": torch.zeros(batch_size, 2, dtype=torch.float32),
                "fused_repr": torch.zeros(batch_size, 2, dtype=torch.float32),
                "drug_logits": logits,
                "drug_probs": torch.sigmoid(logits),
                "runtime_timing": {"retrieval_time": 0.125},
            }

    dataset = [
        {
            "visit_mask": torch.tensor([True], dtype=torch.bool),
            "lab_values": torch.tensor([[0.1]], dtype=torch.float32),
            "vital_values": torch.tensor([[0.2]], dtype=torch.float32),
            "time_delta_hours": torch.tensor([0.0], dtype=torch.float32),
            "final_target_drugs": torch.tensor([1.0, 0.0], dtype=torch.float32),
        }
    ]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    def run_trainer(*, detailed_timing: bool) -> tuple[dict[str, float], TqdmCoreTrainer]:
        model = TimedCoreModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = TqdmCoreTrainer(
            model=model,
            loss_fn=MedicationRecommendationLoss(lambda_ddi=0.0),
            optimizer=optimizer,
            device=torch.device("cpu"),
            checkpoint_dir=tmp_path / ("checkpoints_detailed" if detailed_timing else "checkpoints_base"),
            log_dir=tmp_path / ("logs_detailed" if detailed_timing else "logs_base"),
            monitor_metric="val_total_loss",
            monitor_mode="min",
            detailed_timing=detailed_timing,
        )
        fit_result = trainer.fit(
            train_dataloader=dataloader,
            val_dataloader=dataloader,
            epochs=1,
        )
        return fit_result["history"][0], trainer

    history_without_detail, _ = run_trainer(detailed_timing=False)
    assert "train_retrieval_time" not in history_without_detail
    assert "checkpoint_write_time" not in history_without_detail
    assert "metrics_log_write_time" not in history_without_detail

    history_with_detail, trainer = run_trainer(detailed_timing=True)
    assert history_with_detail["train_retrieval_time"] == pytest.approx(0.125)
    assert history_with_detail["val_retrieval_time"] == pytest.approx(0.125)
    assert history_with_detail["checkpoint_write_time"] >= 0.0
    assert history_with_detail["metrics_log_write_time"] >= 0.0

    log_entry = json.loads(trainer.metrics_log_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert log_entry["run_context"]["runtime"]["detailed_timing"] is True
    assert "train_retrieval_time" in log_entry
    assert "checkpoint_write_time" in log_entry


def test_trainer_threshold_tuning_saves_effective_threshold_and_selection(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("pyarrow")

    project_root, configs = _prepare_runtime_project_with_active_manual_smoke_ddi(tmp_path)
    train_config = apply_profile_overrides(load_yaml_config(configs["train"]), profile_name="balanced")
    data_config = apply_profile_overrides(load_yaml_config(configs["data"]), profile_name="balanced")
    model_config = apply_profile_overrides(load_yaml_config(configs["model"]), profile_name="balanced")
    train_config["_resolved_paths"] = {"processed_root": str((project_root / "data" / "processed").resolve())}

    with tempfile.TemporaryDirectory(prefix="core_trainer_threshold_tuning_") as temp_dir_name:
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
        pos_weight, pos_weight_stats = build_positive_class_weight(
            dataset=train_loader.dataset,
            drug_vocab_size=4,
            mode="log_balanced",
            clip=12.0,
        )
        model, loss_fn = build_core_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            ddi_matrix_path=(project_root / "data" / "processed" / "ddi" / "drug_ddi.pt").resolve(),
            pos_weight=pos_weight,
        )

    optimizer = build_optimizer(model=model, train_config=train_config)
    trainer = TqdmCoreTrainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=torch.device("cpu"),
        checkpoint_dir=project_root / "outputs" / "checkpoints",
        log_dir=project_root / "outputs" / "logs",
        monitor_metric="val_f1_tuned",
        monitor_mode="max",
        decoder_top_k=int(train_config["runtime"]["train_decoder_top_k"]),
        amp=bool(train_config["runtime"]["amp"]),
        grad_accum_steps=int(train_config["runtime"]["grad_accum_steps"]),
        max_grad_norm=float(train_config["optimization"]["max_grad_norm"]),
        non_blocking_transfer=bool(train_config["runtime"]["non_blocking_transfer"]),
        log_interval=int(train_config["runtime"]["log_interval"]),
        profile_steps=train_config["runtime"]["profile_steps"],
        run_context={
            "threshold_tuning": {
                "enabled": True,
                "metric": "f1",
                "tie_breaker": "jaccard",
                "split": "val",
                "candidates": [0.10, 0.25, 0.50],
            },
            "effective_threshold": 0.5,
            "threshold_selection": {
                "source": "config.prediction.threshold",
                "split": "config",
                "metric": "f1",
                "tie_breaker": "jaccard",
                "candidates": [0.10, 0.25, 0.50],
                "best_threshold": 0.5,
            },
            "ddi_context": loss_fn.ddi_context,
            "configured_ddi_lambda": loss_fn.configured_lambda_ddi,
            "effective_ddi_lambda": loss_fn.effective_lambda_ddi,
            "pos_weight_stats": pos_weight_stats,
        },
    )
    fit_result = trainer.fit(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=1,
        extra_checkpoint_state={
            "train_config": {key: value for key, value in train_config.items() if not str(key).startswith("_")},
            "data_config": {key: value for key, value in data_config.items() if not str(key).startswith("_")},
            "model_config": {key: value for key, value in model_config.items() if not str(key).startswith("_")},
        },
    )

    assert fit_result["monitor_metric"] == "val_f1_tuned"
    history_entry = fit_result["history"][0]
    assert "val_f1_tuned" in history_entry
    assert "val_threshold_best" in history_entry

    checkpoint = torch.load(trainer.best_checkpoint_path, map_location="cpu", weights_only=False)
    assert checkpoint["effective_threshold"] == pytest.approx(checkpoint["threshold_selection"]["best_threshold"])
    assert checkpoint["threshold_selection"]["source"] == "validation_sweep"
    assert checkpoint["threshold_selection"]["split"] == "val"
    assert checkpoint["threshold_selection"]["metric"] == "f1"
    assert checkpoint["threshold_selection"]["tie_breaker"] == "jaccard"

    log_entry = json.loads(trainer.metrics_log_path.read_text(encoding="utf-8").strip().splitlines()[0])
    assert log_entry["run_context"]["effective_threshold"] == pytest.approx(
        checkpoint["threshold_selection"]["best_threshold"]
    )
    assert log_entry["run_context"]["threshold_selection"]["split"] == "val"


def test_trainer_threshold_tuning_reuses_cached_validation_predictions(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class CountingCoreModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.forward_calls = 0

        def forward(self, batch, *, mode, decoder_top_k, compute_ddi_metrics=False):
            _ = mode
            _ = decoder_top_k
            _ = compute_ddi_metrics
            self.forward_calls += 1
            batch_size = int(batch["visit_mask"].shape[0])
            logits = self.anchor.expand(batch_size, 2)
            return {
                "pooled_state": torch.zeros(batch_size, 2, dtype=torch.float32),
                "fused_repr": torch.zeros(batch_size, 2, dtype=torch.float32),
                "drug_logits": logits,
                "drug_probs": torch.sigmoid(logits),
                "final_target_drugs": batch["final_target_drugs"],
            }

    dataset = [
        {
            "visit_mask": torch.tensor([True], dtype=torch.bool),
            "lab_values": torch.tensor([[0.1]], dtype=torch.float32),
            "vital_values": torch.tensor([[0.2]], dtype=torch.float32),
            "time_delta_hours": torch.tensor([0.0], dtype=torch.float32),
            "final_target_drugs": torch.tensor([1.0, 0.0], dtype=torch.float32),
        },
        {
            "visit_mask": torch.tensor([True], dtype=torch.bool),
            "lab_values": torch.tensor([[0.3]], dtype=torch.float32),
            "vital_values": torch.tensor([[0.4]], dtype=torch.float32),
            "time_delta_hours": torch.tensor([0.0], dtype=torch.float32),
            "final_target_drugs": torch.tensor([0.0, 1.0], dtype=torch.float32),
        },
    ]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = CountingCoreModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = TqdmCoreTrainer(
        model=model,
        loss_fn=MedicationRecommendationLoss(lambda_ddi=0.0),
        optimizer=optimizer,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        monitor_metric="val_f1_tuned",
        monitor_mode="max",
        run_context={
            "threshold_tuning": {
                "enabled": True,
                "metric": "f1",
                "tie_breaker": "jaccard",
                "split": "val",
                "candidates": [0.1, 0.5],
            }
        },
    )

    trainer.validate_one_epoch(dataloader)
    forward_calls_after_validation = model.forward_calls
    tuning_metrics = trainer._run_threshold_tuning(dataloader)

    assert model.forward_calls == forward_calls_after_validation
    assert "val_threshold_best" in tuning_metrics
    assert tuning_metrics["val_threshold_best"] in {0.1, 0.5}


def test_extended_trainer_does_not_silent_fallback_to_core(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class ExtendedOnlyFailureModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def forward(self, batch, *, mode, memory_bank, records, decoder_top_k):
            _ = batch
            _ = memory_bank
            _ = records
            _ = decoder_top_k
            if mode == "extended":
                raise RuntimeError("retrieval failure should not fallback to core")
            batch_size = 1
            logits = self.anchor.expand(batch_size, 2)
            return {
                "pooled_state": torch.zeros(batch_size, 2, dtype=torch.float32),
                "fused_repr": torch.zeros(batch_size, 2, dtype=torch.float32),
                "drug_logits": logits,
                "drug_probs": torch.sigmoid(logits),
            }

    dataset = [
        {
            "visit_mask": torch.tensor([True], dtype=torch.bool),
            "lab_values": torch.tensor([[0.1]], dtype=torch.float32),
            "vital_values": torch.tensor([[0.2]], dtype=torch.float32),
            "time_delta_hours": torch.tensor([0.0], dtype=torch.float32),
            "target_drugs": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "records": [{}],
        }
    ]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = ExtendedOnlyFailureModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = ExtendedTrainer(
        model=model,
        loss_fn=torch.nn.Identity(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        use_retrieval=True,
    )

    with pytest.raises(RuntimeError, match="retrieval failure should not fallback to core"):
        trainer.train_one_epoch(dataloader)


def test_medication_recommendation_loss_supports_focal_and_fusion_regularization() -> None:
    torch = pytest.importorskip("torch")

    logits = torch.tensor([[0.2, -0.4], [1.0, -1.0]], dtype=torch.float32)
    targets = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    bce_loss = MedicationRecommendationLoss(objective="bce", lambda_ddi=0.0)
    focal_loss = MedicationRecommendationLoss(
        objective="focal_lite",
        focal_gamma=1.5,
        lambda_ddi=0.0,
        fusion_entropy_lambda=0.1,
        fusion_balance_lambda=0.2,
    )

    bce_outputs = bce_loss(drug_logits=logits, target_drugs=targets)
    focal_outputs = focal_loss(
        drug_logits=logits,
        target_drugs=targets,
        fusion_entropy_loss=torch.tensor([1.0, 3.0], dtype=torch.float32),
        fusion_balance_loss=torch.tensor([2.0, 4.0], dtype=torch.float32),
    )

    assert float(focal_outputs["prediction_loss"].item()) != pytest.approx(
        float(bce_outputs["prediction_loss"].item())
    )
    assert float(focal_outputs["weighted_fusion_entropy_loss"].item()) == pytest.approx(0.2)
    assert float(focal_outputs["weighted_fusion_balance_loss"].item()) == pytest.approx(0.6)
    assert float(focal_outputs["total_loss"].item()) == pytest.approx(
        float(focal_outputs["prediction_loss"].item())
        + float(focal_outputs["weighted_fusion_entropy_loss"].item())
        + float(focal_outputs["weighted_fusion_balance_loss"].item())
    )


def test_extended_trainer_train_only_policy_uses_train_bank_for_validation(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class ValidationBankRecorder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))
            self.last_memory_bank = None

        def forward(self, batch, *, mode, memory_bank, records, decoder_top_k, compute_ddi_metrics=False):
            _ = mode
            _ = records
            _ = decoder_top_k
            _ = compute_ddi_metrics
            self.last_memory_bank = memory_bank
            batch_size = int(batch["visit_mask"].shape[0])
            logits = self.anchor.expand(batch_size, 2)
            return {
                "pooled_state": torch.zeros(batch_size, 2, dtype=torch.float32),
                "fused_repr": torch.zeros(batch_size, 2, dtype=torch.float32),
                "drug_logits": logits,
                "drug_probs": torch.sigmoid(logits),
                "fusion_entropy_loss": torch.zeros(batch_size, dtype=torch.float32),
                "fusion_balance_loss": torch.zeros(batch_size, dtype=torch.float32),
            }

    dataset = [
        {
            "visit_mask": torch.tensor([True], dtype=torch.bool),
            "lab_values": torch.tensor([[0.1]], dtype=torch.float32),
            "vital_values": torch.tensor([[0.2]], dtype=torch.float32),
            "time_delta_hours": torch.tensor([0.0], dtype=torch.float32),
            "target_drugs": torch.tensor([[1.0, 0.0]], dtype=torch.float32),
            "records": [{}],
        }
    ]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = ValidationBankRecorder()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = ExtendedTrainer(
        model=model,
        loss_fn=MedicationRecommendationLoss(lambda_ddi=0.0),
        optimizer=optimizer,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        use_retrieval=True,
        validation_memory_bank_policy="train_only",
    )
    expected_bank = object()
    trainer.train_memory_bank = expected_bank
    trainer.val_memory_bank = None

    trainer.validate_one_epoch(dataloader)

    assert model.last_memory_bank is expected_bank


def test_build_extended_model_uses_core_runtime_copy_for_core_builder(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class DummyModel:
        def __init__(self) -> None:
            self.fusion_module = type(
                "FusionModuleStub",
                (),
                {"hidden_dim": 16, "strategy": "gated"},
            )()

    dummy_model = DummyModel()
    dummy_loss = mock.Mock(ddi_context={"active": False})
    train_config = {
        "runtime": {"mode": "extended", "profile": "safe"},
        "extended": {
            "mode": "extended",
            "use_retrieval": True,
            "use_group_encoder": False,
            "retrieval_top_k": 3,
            "temporal_decay_alpha": 0.1,
            "retrieval_backend": "bruteforce",
            "use_faiss_if_available": False,
            "allow_cross_split": False,
            "retrieval_scoring_mode": "temporal_relevance",
            "cross_split_policy": "train_bank_only",
            "validation_memory_bank_policy": "train_only",
        },
    }
    model_config = {
        "retrieval": {
            "top_k": 5,
            "temporal_decay_alpha": 0.05,
            "backend": "bruteforce",
            "use_faiss_if_available": True,
            "scoring_mode": "temporal_relevance",
            "cross_split_policy": "train_bank_only",
        }
    }

    def fake_build_core_model(**kwargs):
        captured["train_config"] = kwargs["train_config"]
        return dummy_model, dummy_loss

    with mock.patch("src.training.train_extended.build_core_model", side_effect=fake_build_core_model), mock.patch(
        "src.training.train_extended.build_optional_group_encoder",
        return_value=None,
    ):
        model, loss_fn = build_extended_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=tmp_path / "runtime_data.yaml",
            vocab_root=tmp_path / "vocab",
            ddi_matrix_path=tmp_path / "drug_ddi.pt",
        )

    forwarded_config = captured["train_config"]
    assert forwarded_config is not train_config
    assert isinstance(forwarded_config, dict)
    assert forwarded_config["runtime"] is not train_config["runtime"]
    assert forwarded_config["runtime"]["mode"] == "core"
    assert forwarded_config["runtime"]["profile"] == "safe"
    assert train_config["runtime"]["mode"] == "extended"
    assert model is dummy_model
    assert loss_fn is dummy_loss


def test_evaluate_extended_core_runs_and_reports_train_bank_truth(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pyarrow")

    project_root, configs = _prepare_runtime_project(tmp_path)
    train_config = load_yaml_config(configs["train"])
    data_config = load_yaml_config(configs["data"])
    model_config = load_yaml_config(configs["model"])
    train_config["_resolved_paths"] = {"processed_root": str((project_root / "data" / "processed").resolve())}

    with tempfile.TemporaryDirectory(prefix="eval_extended_runtime_smoke_") as temp_dir_name:
        runtime_data_config_path = build_runtime_data_config_file(
            data_config=data_config,
            processed_root=(project_root / "data" / "processed").resolve(),
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            temp_dir=Path(temp_dir_name),
        )
        model, loss_fn = build_extended_model(
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
            "monitor_metric": "val_prauc_tuned",
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
            "runtime_truth": dict(getattr(model, "runtime_truth", {})),
            "retrieval_active": True,
        },
        checkpoint_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.evaluate_extended_core",
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

    report = json.load((project_root / "outputs" / "reports" / "evaluate_extended_core_test.json").open())
    assert report["retrieval_active"] is True
    assert report["retrieval_bank_policy"] == "train_only"
    assert report["artifacts"]["attention_json"].endswith("_attention.json")
    assert Path(report["artifacts"]["attention_json"]).exists()


def test_evaluate_extended_safety_runs_with_train_bank_truth(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pyarrow")

    project_root, configs = _prepare_runtime_project(tmp_path)
    train_config = load_yaml_config(configs["train"])
    data_config = load_yaml_config(configs["data"])
    model_config = load_yaml_config(configs["model"])
    train_config["_resolved_paths"] = {"processed_root": str((project_root / "data" / "processed").resolve())}

    with tempfile.TemporaryDirectory(prefix="eval_extended_safety_runtime_") as temp_dir_name:
        runtime_data_config_path = build_runtime_data_config_file(
            data_config=data_config,
            processed_root=(project_root / "data" / "processed").resolve(),
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            temp_dir=Path(temp_dir_name),
        )
        model, loss_fn = build_extended_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            ddi_matrix_path=(project_root / "data" / "processed" / "ddi" / "drug_ddi.pt").resolve(),
        )

    checkpoint_path = project_root / "outputs" / "checkpoints" / "train_core_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(
        {
            "epoch": 1,
            "best_metric": 0.0,
            "monitor_metric": "val_prauc_tuned",
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
            "runtime_truth": dict(getattr(model, "runtime_truth", {})),
            "retrieval_active": True,
        },
        checkpoint_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.evaluate_extended_safety",
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

    report = json.load((project_root / "outputs" / "reports" / "evaluate_extended_safety_test.json").open())
    assert report["retrieval_active"] is True
    assert report["retrieval_bank_policy"] == "train_only"
    assert "patient_summary" in report


def test_evaluate_extended_subgroup_runs_with_train_bank_truth(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    pytest.importorskip("pyarrow")

    project_root, configs = _prepare_runtime_project(tmp_path)
    train_config = load_yaml_config(configs["train"])
    data_config = load_yaml_config(configs["data"])
    model_config = load_yaml_config(configs["model"])
    train_config["_resolved_paths"] = {"processed_root": str((project_root / "data" / "processed").resolve())}

    with tempfile.TemporaryDirectory(prefix="eval_extended_subgroup_runtime_") as temp_dir_name:
        runtime_data_config_path = build_runtime_data_config_file(
            data_config=data_config,
            processed_root=(project_root / "data" / "processed").resolve(),
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            temp_dir=Path(temp_dir_name),
        )
        model, loss_fn = build_extended_model(
            train_config=train_config,
            model_config=model_config,
            runtime_data_config_path=runtime_data_config_path,
            vocab_root=(project_root / "data" / "interim" / "vocab").resolve(),
            ddi_matrix_path=(project_root / "data" / "processed" / "ddi" / "drug_ddi.pt").resolve(),
        )

    checkpoint_path = project_root / "outputs" / "checkpoints" / "train_core_best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    import torch

    torch.save(
        {
            "epoch": 1,
            "best_metric": 0.0,
            "monitor_metric": "val_prauc_tuned",
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
            "runtime_truth": dict(getattr(model, "runtime_truth", {})),
            "retrieval_active": True,
        },
        checkpoint_path,
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "src.evaluation.evaluate_extended_subgroup",
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

    payload = json.load((project_root / "outputs" / "reports" / "evaluate_extended_subgroup_test.json").open())
    assert payload["report"]["retrieval_active"] is True
    assert payload["report"]["retrieval_bank_policy"] == "train_only"
    assert len(payload["subgroups"]) > 0


def test_resolve_precision_policy_cpu_uses_fp32() -> None:
    torch = pytest.importorskip("torch")

    policy = resolve_precision_policy(requested_amp=True, device=torch.device("cpu"))

    assert policy.requested_amp is True
    assert policy.resolved_precision == "fp32"
    assert policy.use_autocast is False
    assert policy.autocast_dtype is None
    assert policy.grad_scaler_enabled is False
    assert policy.warning_message is None


def test_resolve_precision_policy_cuda_prefers_bfloat16() -> None:
    torch = pytest.importorskip("torch")

    with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True, create=True):
        policy = resolve_precision_policy(requested_amp=True, device=torch.device("cuda"))

    assert policy.requested_amp is True
    assert policy.resolved_precision == "bf16"
    assert policy.use_autocast is True
    assert policy.autocast_dtype == torch.bfloat16
    assert policy.grad_scaler_enabled is False
    assert policy.warning_message is None


def test_resolve_precision_policy_cuda_without_bfloat16_falls_back_to_fp32() -> None:
    torch = pytest.importorskip("torch")

    with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=False, create=True):
        policy = resolve_precision_policy(requested_amp=True, device=torch.device("cuda"))

    assert policy.requested_amp is True
    assert policy.resolved_precision == "fp32"
    assert policy.use_autocast is False
    assert policy.autocast_dtype is None
    assert policy.grad_scaler_enabled is False
    assert policy.warning_message is not None
    assert "falling back to float32" in policy.warning_message


def test_trainer_raises_contextual_error_for_non_finite_forward_outputs(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class NaNForwardModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1, dtype=torch.float32))

        def forward(self, batch, *, mode, decoder_top_k, compute_ddi_metrics):
            _ = mode
            _ = decoder_top_k
            _ = compute_ddi_metrics
            batch_size = int(batch["visit_mask"].shape[0])
            fused = self.anchor.expand(batch_size, 2) * 0.0
            bad = torch.full_like(fused, float("nan"))
            return {
                "pooled_state": fused,
                "fused_repr": fused,
                "drug_logits": bad,
                "drug_probs": bad,
            }

    dataset = [
        {
            "visit_mask": torch.tensor([True], dtype=torch.bool),
            "lab_values": torch.tensor([[0.1]], dtype=torch.float32),
            "vital_values": torch.tensor([[0.2]], dtype=torch.float32),
            "time_delta_hours": torch.tensor([0.0], dtype=torch.float32),
            "final_target_drugs": torch.tensor([1.0, 0.0], dtype=torch.float32),
        }
    ]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = NaNForwardModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = Trainer(
        model=model,
        loss_fn=torch.nn.Identity(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
    )

    with pytest.raises(RuntimeError, match=r"val step 1 after forward: tensor `drug_logits`"):
        trainer.validate_one_epoch(dataloader)


def test_trainer_raises_when_optimizer_produces_non_finite_parameter(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    class StableModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.logit_bias = torch.nn.Parameter(torch.zeros(2, dtype=torch.float32))

        def forward(self, batch, *, mode, decoder_top_k, compute_ddi_metrics):
            _ = mode
            _ = decoder_top_k
            _ = compute_ddi_metrics
            batch_size = int(batch["visit_mask"].shape[0])
            logits = self.logit_bias.unsqueeze(0).expand(batch_size, -1)
            fused = logits * 0.0
            return {
                "pooled_state": fused,
                "fused_repr": fused,
                "drug_logits": logits,
                "drug_probs": torch.sigmoid(logits),
            }

    class SimpleLoss(torch.nn.Module):
        def forward(self, *, drug_logits, drug_probs, target_drugs, visit_mask):
            _ = drug_probs
            _ = visit_mask
            prediction_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                drug_logits,
                target_drugs,
            )
            zero = prediction_loss.new_zeros(())
            return {
                "total_loss": prediction_loss,
                "prediction_loss": prediction_loss,
                "ddi_loss": zero,
                "weighted_ddi_loss": zero,
            }

    class PoisonOptimizer(torch.optim.SGD):
        def __init__(self, params, *, poison_parameter, lr=0.1) -> None:
            super().__init__(params, lr=lr)
            self.poison_parameter = poison_parameter

        def step(self, closure=None):
            loss = super().step(closure)
            self.poison_parameter.data.fill_(float("nan"))
            return loss

    dataset = [
        {
            "visit_mask": torch.tensor([True], dtype=torch.bool),
            "lab_values": torch.tensor([[0.1]], dtype=torch.float32),
            "vital_values": torch.tensor([[0.2]], dtype=torch.float32),
            "time_delta_hours": torch.tensor([0.0], dtype=torch.float32),
            "final_target_drugs": torch.tensor([1.0, 0.0], dtype=torch.float32),
        }
    ]
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    model = StableModel()
    optimizer = PoisonOptimizer(model.parameters(), poison_parameter=model.logit_bias, lr=0.1)
    trainer = Trainer(
        model=model,
        loss_fn=SimpleLoss(),
        optimizer=optimizer,
        device=torch.device("cpu"),
        checkpoint_dir=tmp_path / "checkpoints",
        log_dir=tmp_path / "logs",
        max_grad_norm=1.0,
    )

    with pytest.raises(RuntimeError, match=r"train step 1 after optimizer step: tensor `parameter:logit_bias`"):
        trainer.train_one_epoch(dataloader)


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
