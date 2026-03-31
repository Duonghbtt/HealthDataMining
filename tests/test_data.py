from __future__ import annotations

import csv
import gzip
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from src.data.build_cohort import build_cohort
from src.data.build_ddi_matrix import build_ddi_matrix
from src.data.build_trajectories import build_trajectories
from src.data.build_vocab import build_vocab
from src.data.stage_filtered_tables import stage_filtered_tables
from src.utils.io import read_csv_gz, read_json, write_json, write_jsonl_gz


def _write_csv_gz(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_config(project_root: Path, *, spark_enabled: bool = True) -> Path:
    config_path = project_root / "configs" / "data.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "\n".join(
            [
                "seed: 7",
                "paths:",
                "  raw_root: data/raw",
                "  interim_root: data/interim",
                "  processed_root: data/processed",
                "  ddi_source_path: ''",
                "processed_format: parquet",
                "split:",
                "  train: 1.0",
                "  val: 0.0",
                "  test: 0.0",
                "cohort:",
                "  require_diagnosis: true",
                "  require_medication: true",
                "  min_los_hours: 0.0",
                "features:",
                "  time_bucket_hours: 24",
                "  top_k_labs: 4",
                "  top_k_vitals: 4",
                "  max_med_history: 8",
                "  normalization_eps: 1.0e-6",
                "spark:",
                f"  enabled: {'true' if spark_enabled else 'false'}",
                "  master: local[4]",
                "  driver_memory: 3g",
                "  default_parallelism: 8",
                "  sql_shuffle_partitions: 24",
                "  adaptive_enabled: true",
                "  adaptive_coalesce_enabled: true",
                "  files_max_partition_bytes: 64m",
                "  local_dir: /tmp/healthdm-spark-tests",
                "  stage_cache_dir: data/interim/spark_cache",
                "  cache_codec: snappy",
                "  trajectory_rows_per_file: 2",
                "  max_open_shards_per_dataset: 2",
            ]
        ),
        encoding="utf-8",
    )
    return config_path


def _build_mock_project(tmp_path: Path) -> Path:
    project_root = tmp_path / "mini_project"
    raw_hosp = project_root / "data" / "raw" / "hosp"
    raw_icu = project_root / "data" / "raw" / "icu"

    _write_csv_gz(
        raw_hosp / "patients.csv.gz",
        [
            {"subject_id": 1, "gender": "M", "anchor_age": 65, "anchor_year": 2020, "anchor_year_group": "2017 - 2019", "dod": ""},
            {"subject_id": 2, "gender": "F", "anchor_age": 72, "anchor_year": 2020, "anchor_year_group": "2017 - 2019", "dod": ""},
        ],
        ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group", "dod"],
    )
    _write_csv_gz(
        raw_hosp / "admissions.csv.gz",
        [
            {"subject_id": 1, "hadm_id": 11, "admittime": "2020-01-01 00:00:00", "dischtime": "2020-01-03 00:00:00", "deathtime": "", "admission_type": "URGENT", "admit_provider_id": "X", "admission_location": "ER", "discharge_location": "HOME", "insurance": "A", "language": "EN", "marital_status": "MARRIED", "race": "WHITE", "edregtime": "", "edouttime": "", "hospital_expire_flag": 0},
            {"subject_id": 2, "hadm_id": 22, "admittime": "2020-01-02 00:00:00", "dischtime": "2020-01-04 00:00:00", "deathtime": "", "admission_type": "URGENT", "admit_provider_id": "Y", "admission_location": "ER", "discharge_location": "HOME", "insurance": "B", "language": "EN", "marital_status": "WIDOWED", "race": "ASIAN", "edregtime": "", "edouttime": "", "hospital_expire_flag": 0},
        ],
        ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime", "admission_type", "admit_provider_id", "admission_location", "discharge_location", "insurance", "language", "marital_status", "race", "edregtime", "edouttime", "hospital_expire_flag"],
    )
    _write_csv_gz(
        raw_icu / "icustays.csv.gz",
        [
            {"subject_id": 1, "hadm_id": 11, "stay_id": 111, "first_careunit": "MICU", "last_careunit": "MICU", "intime": "2020-01-01 00:00:00", "outtime": "2020-01-02 12:00:00", "los": 1.5},
            {"subject_id": 1, "hadm_id": 11, "stay_id": 112, "first_careunit": "MICU", "last_careunit": "MICU", "intime": "2020-01-02 12:00:00", "outtime": "2020-01-03 00:00:00", "los": 0.5},
            {"subject_id": 2, "hadm_id": 22, "stay_id": 222, "first_careunit": "SICU", "last_careunit": "SICU", "intime": "2020-01-02 00:00:00", "outtime": "2020-01-03 12:00:00", "los": 1.5},
        ],
        ["subject_id", "hadm_id", "stay_id", "first_careunit", "last_careunit", "intime", "outtime", "los"],
    )
    _write_csv_gz(
        raw_hosp / "diagnoses_icd.csv.gz",
        [
            {"subject_id": 1, "hadm_id": 11, "seq_num": 1, "icd_code": "4019", "icd_version": 9},
            {"subject_id": 2, "hadm_id": 22, "seq_num": 1, "icd_code": "25000", "icd_version": 9},
        ],
        ["subject_id", "hadm_id", "seq_num", "icd_code", "icd_version"],
    )
    _write_csv_gz(
        raw_hosp / "procedures_icd.csv.gz",
        [
            {"subject_id": 1, "hadm_id": 11, "seq_num": 1, "chartdate": "2020-01-01", "icd_code": "3893", "icd_version": 9},
            {"subject_id": 2, "hadm_id": 22, "seq_num": 1, "chartdate": "2020-01-02", "icd_code": "8872", "icd_version": 9},
        ],
        ["subject_id", "hadm_id", "seq_num", "chartdate", "icd_code", "icd_version"],
    )
    _write_csv_gz(
        raw_hosp / "labevents.csv.gz",
        [
            {"labevent_id": 1, "subject_id": 1, "hadm_id": 11, "specimen_id": 1, "itemid": 50912, "order_provider_id": "A", "charttime": "2020-01-01 06:00:00", "storetime": "2020-01-01 06:30:00", "value": "100", "valuenum": 100, "valueuom": "mg/dL", "ref_range_lower": "", "ref_range_upper": "", "flag": "", "priority": "", "comments": ""},
            {"labevent_id": 2, "subject_id": 2, "hadm_id": 22, "specimen_id": 2, "itemid": 50912, "order_provider_id": "B", "charttime": "2020-01-02 06:00:00", "storetime": "2020-01-02 06:30:00", "value": "140", "valuenum": 140, "valueuom": "mg/dL", "ref_range_lower": "", "ref_range_upper": "", "flag": "", "priority": "", "comments": ""},
        ],
        ["labevent_id", "subject_id", "hadm_id", "specimen_id", "itemid", "order_provider_id", "charttime", "storetime", "value", "valuenum", "valueuom", "ref_range_lower", "ref_range_upper", "flag", "priority", "comments"],
    )
    _write_csv_gz(
        raw_hosp / "d_labitems.csv.gz",
        [{"itemid": 50912, "label": "Creatinine", "fluid": "Blood", "category": "Chemistry"}],
        ["itemid", "label", "fluid", "category"],
    )
    _write_csv_gz(
        raw_icu / "chartevents.csv.gz",
        [
            {"subject_id": 1, "hadm_id": 11, "stay_id": 111, "caregiver_id": 1, "charttime": "2020-01-01 05:00:00", "storetime": "2020-01-01 05:10:00", "itemid": 220045, "value": "80", "valuenum": 80, "valueuom": "bpm", "warning": 0},
            {"subject_id": 2, "hadm_id": 22, "stay_id": 222, "caregiver_id": 1, "charttime": "2020-01-02 07:00:00", "storetime": "2020-01-02 07:10:00", "itemid": 220045, "value": "88", "valuenum": 88, "valueuom": "bpm", "warning": 0},
        ],
        ["subject_id", "hadm_id", "stay_id", "caregiver_id", "charttime", "storetime", "itemid", "value", "valuenum", "valueuom", "warning"],
    )
    _write_csv_gz(
        raw_icu / "d_items.csv.gz",
        [{"itemid": 220045, "label": "Heart Rate", "abbreviation": "HR", "linksto": "chartevents", "category": "Routine Vital Signs", "unitname": "bpm", "param_type": "Numeric", "lownormalvalue": 60, "highnormalvalue": 100}],
        ["itemid", "label", "abbreviation", "linksto", "category", "unitname", "param_type", "lownormalvalue", "highnormalvalue"],
    )
    _write_csv_gz(
        raw_hosp / "prescriptions.csv.gz",
        [
            {"subject_id": 1, "hadm_id": 11, "pharmacy_id": 1, "poe_id": "p1", "poe_seq": 1, "order_provider_id": "A", "starttime": "2020-01-01 08:00:00", "stoptime": "2020-01-01 20:00:00", "drug_type": "MAIN", "drug": "Aspirin", "formulary_drug_cd": "ASP100", "gsn": "", "ndc": "", "prod_strength": "", "form_rx": "", "dose_val_rx": "", "dose_unit_rx": "", "form_val_disp": "", "form_unit_disp": "", "doses_per_24_hrs": "", "route": "PO"},
            {"subject_id": 2, "hadm_id": 22, "pharmacy_id": 2, "poe_id": "p2", "poe_seq": 1, "order_provider_id": "B", "starttime": "2020-01-02 09:00:00", "stoptime": "2020-01-02 20:00:00", "drug_type": "MAIN", "drug": "Heparin", "formulary_drug_cd": "HEP5000", "gsn": "", "ndc": "", "prod_strength": "", "form_rx": "", "dose_val_rx": "", "dose_unit_rx": "", "form_val_disp": "", "form_unit_disp": "", "doses_per_24_hrs": "", "route": "IV"},
        ],
        ["subject_id", "hadm_id", "pharmacy_id", "poe_id", "poe_seq", "order_provider_id", "starttime", "stoptime", "drug_type", "drug", "formulary_drug_cd", "gsn", "ndc", "prod_strength", "form_rx", "dose_val_rx", "dose_unit_rx", "form_val_disp", "form_unit_disp", "doses_per_24_hrs", "route"],
    )
    _write_csv_gz(
        raw_hosp / "emar.csv.gz",
        [{"subject_id": 1, "hadm_id": 11, "emar_id": "e1", "emar_seq": 1, "poe_id": "p1", "pharmacy_id": 1, "enter_provider_id": "", "charttime": "2020-01-01 08:00:00", "medication": "Aspirin", "event_txt": "Administered", "scheduletime": "2020-01-01 08:00:00", "storetime": "2020-01-01 08:05:00"}],
        ["subject_id", "hadm_id", "emar_id", "emar_seq", "poe_id", "pharmacy_id", "enter_provider_id", "charttime", "medication", "event_txt", "scheduletime", "storetime"],
    )
    _write_csv_gz(
        raw_hosp / "pharmacy.csv.gz",
        [{"subject_id": 2, "hadm_id": 22, "pharmacy_id": 2, "poe_id": "p2", "starttime": "2020-01-02 09:00:00", "stoptime": "2020-01-02 20:00:00", "medication": "Heparin", "proc_type": "Unit Dose", "status": "Active", "entertime": "2020-01-02 08:30:00", "verifiedtime": "2020-01-02 08:45:00", "route": "IV", "frequency": "BID", "disp_sched": "", "infusion_type": "", "sliding_scale": "", "lockout_interval": "", "basal_rate": "", "one_hr_max": "", "doses_per_24_hrs": "", "duration": "", "duration_interval": "", "expiration_value": "", "expiration_unit": "", "expirationdate": "", "dispensation": "", "fill_quantity": ""}],
        ["subject_id", "hadm_id", "pharmacy_id", "poe_id", "starttime", "stoptime", "medication", "proc_type", "status", "entertime", "verifiedtime", "route", "frequency", "disp_sched", "infusion_type", "sliding_scale", "lockout_interval", "basal_rate", "one_hr_max", "doses_per_24_hrs", "duration", "duration_interval", "expiration_value", "expiration_unit", "expirationdate", "dispensation", "fill_quantity"],
    )
    return project_root


def _write_minimal_vocab_bundle(project_root: Path) -> None:
    vocab_dir = project_root / "data" / "interim" / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    for name in ("diagnosis", "procedure", "drug", "lab", "vital"):
        write_json(
            vocab_dir / f"{name}_vocab.json",
            {
                "name": name,
                "size": 2,
                "pad_idx": 0,
                "unk_idx": 1,
                "idx_to_token": ["PAD", "UNK"],
                "token_to_idx": {"PAD": 0, "UNK": 1},
            },
        )


def test_build_vocab_requires_stage_cache_when_spark_enabled(tmp_path: Path) -> None:
    pytest.importorskip("pyspark")
    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=True)
    build_cohort(config_path)

    with pytest.raises(FileNotFoundError, match="stage_filtered_tables"):
        build_vocab(config_path)


def test_data_pipeline_builders_with_spark_cache(tmp_path: Path) -> None:
    pytest.importorskip("pyspark")
    pytest.importorskip("pyarrow")

    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=True)

    cohort_path = build_cohort(config_path)
    cohort_rows = read_csv_gz(cohort_path)
    assert len(cohort_rows) == 2
    assert {row["stay_id"] for row in cohort_rows} == {"111", "222"}
    assert (project_root / "data" / "interim" / "cohort" / "cohort_keys.parquet").exists()

    cache_dir = stage_filtered_tables(config_path)
    cache_manifest = read_json(cache_dir / "cache_manifest.json")
    assert set(cache_manifest["tables"]) == {"diagnoses_icd", "procedures_icd", "labevents", "chartevents", "medications"}

    build_vocab(config_path)
    diagnosis_vocab = read_json(project_root / "data" / "interim" / "vocab" / "diagnosis_vocab.json")
    assert diagnosis_vocab["idx_to_token"][:2] == ["PAD", "UNK"]
    assert "ICD9:4019" in diagnosis_vocab["token_to_idx"]

    ddi_path = build_ddi_matrix(config_path)
    assert ddi_path.exists()

    outputs = build_trajectories(config_path)
    assert outputs["train"].exists()

    manifest = read_json(project_root / "data" / "processed" / "trajectories" / "manifest.json")
    assert manifest["format"] == "parquet"
    assert manifest["counts_by_split"]["train"] == 2
    assert manifest["splits"]["train"]["shards"]

    metadata = read_json(project_root / "data" / "processed" / "trajectories" / "metadata.json")
    assert metadata["lab_feature_size"] >= 1
    assert metadata["vital_feature_size"] >= 1


def test_dataset_collate_when_torch_available(tmp_path: Path) -> None:
    pytest.importorskip("pyspark")
    pytest.importorskip("pyarrow")
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch.utils.data")

    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=True)
    build_cohort(config_path)
    stage_filtered_tables(config_path)
    build_vocab(config_path)
    build_ddi_matrix(config_path)
    build_trajectories(config_path)

    from src.data.dataset import MIMICTrajectoryDataset, collate_batch

    dataset = MIMICTrajectoryDataset("train", config_path)
    batch = collate_batch([dataset[0], dataset[1]])
    assert batch["diag_codes"].shape[0] == 2
    assert batch["visit_mask"].shape[0] == 2
    assert torch.all(batch["visit_mask"].sum(dim=1) >= 1)


def test_dataset_legacy_jsonl_fallback(tmp_path: Path) -> None:
    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=False)
    _write_minimal_vocab_bundle(project_root)
    record = {
        "subject_id": 1,
        "hadm_id": 11,
        "stay_id": 111,
        "split": "train",
        "intime": "2020-01-01 00:00:00",
        "outtime": "2020-01-02 00:00:00",
        "num_steps": 1,
        "drug_vocab_size": 2,
        "lab_feature_size": 0,
        "vital_feature_size": 0,
        "steps": [
            {
                "step_index": 0,
                "diagnosis_ids": [],
                "procedure_ids": [],
                "lab_values": [],
                "lab_mask": [],
                "vital_values": [],
                "vital_mask": [],
                "med_history_ids": [],
                "delta_hours": 0.0,
                "target_drugs": [],
            }
        ],
    }
    write_jsonl_gz(
        project_root / "data" / "processed" / "trajectories" / "train" / "trajectories.jsonl.gz",
        [record],
    )

    from src.data.dataset import MIMICTrajectoryDataset

    dataset = MIMICTrajectoryDataset("train", config_path)
    assert len(dataset) == 1
    assert dataset[0]["stay_id"] == 111


def test_dataset_raises_clear_error_when_outputs_missing(tmp_path: Path) -> None:
    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=True)
    _write_minimal_vocab_bundle(project_root)

    from src.data.dataset import MIMICTrajectoryDataset

    with pytest.raises(FileNotFoundError, match="Neither parquet manifest"):
        MIMICTrajectoryDataset("train", config_path)


def test_preprocess_script_smoke_if_pwsh_exists(tmp_path: Path) -> None:
    pytest.importorskip("pyspark")
    pytest.importorskip("pyarrow")

    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if not pwsh:
        pytest.skip("PowerShell is not available in this environment")

    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=True)
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "preprocess.ps1"
    result = subprocess.run(
        [
            pwsh,
            "-ExecutionPolicy",
            "Bypass",
            "-NoProfile",
            "-File",
            str(script_path),
            "-Config",
            str(config_path),
            "-Python",
            sys.executable,
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
