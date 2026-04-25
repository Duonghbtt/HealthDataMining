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
from src.data.build_drugbank_ddi_matrix import build_drugbank_ddi_matrix
from src.data.build_drugbank_metadata import build_drugbank_metadata
from src.data.build_trajectories import build_trajectories
from src.data.build_vocab import build_vocab
from src.data.stage_filtered_tables import stage_filtered_tables
from src.utils.io import iter_jsonl_gz, load_pt, read_csv_gz, read_json, write_json, write_jsonl_gz


def _stage_filtered_tables_or_skip(config_path: Path) -> Path:
    try:
        return stage_filtered_tables(config_path)
    except Exception as exc:  # pragma: no cover - depends on sandboxed Spark runtime
        message = str(exc)
        if (
            "JAVA_GATEWAY_EXITED" in message
            or "Failed to bind" in message
            or "Operation not permitted" in message
        ):
            pytest.skip(f"Spark gateway is unavailable in this environment: {message}")
        raise


def _skip_if_spark_wrapper_unavailable(result: subprocess.CompletedProcess[str]) -> None:
    message = f"{result.stderr}\n{result.stdout}"
    if (
        "JAVA_GATEWAY_EXITED" in message
        or "Failed to bind" in message
        or "Operation not permitted" in message
    ):
        pytest.skip(f"Spark gateway is unavailable in this environment: {message}")


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
                "ddi:",
                "  source_path: ''",
                "  source_format: twosides_csv",
                "  fallback_source_path: data/raw/ddi/drug_ddi_smoke.csv",
                "  fallback_source_format: manual_smoke_csv",
                "  canonical_pairs_path: data/processed/ddi/drug_ddi_pairs.csv.gz",
                "  min_support_a: 5",
                "  min_prr_ci_lower_bound: 1.0",
                "drugbank:",
                "  source_path: 'data/raw/drugbank/full database.xml'",
                "  summary_path: data/processed/drugbank/drugbank_summary.json",
                "  records_path: data/processed/drugbank/drugbank_drugs.jsonl.gz",
                "  vocab_metadata_path: data/interim/vocab/drugbank_drug_metadata.json",
                "  ddi_pairs_path: data/processed/ddi/drugbank_ddi_pairs.jsonl.gz",
                "  ddi_matrix_path: data/processed/ddi/drug_ddi_drugbank.pt",
                "  ddi_report_path: data/processed/ddi/drug_ddi_drugbank_report.json",
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


def _write_drugbank_test_vocab_bundle(project_root: Path) -> None:
    vocab_dir = project_root / "data" / "interim" / "vocab"
    vocab_dir.mkdir(parents=True, exist_ok=True)
    vocab_payloads = {
        "diagnosis": ["PAD", "UNK"],
        "procedure": ["PAD", "UNK"],
        "drug": ["PAD", "UNK", "NAME:ASPIRIN", "NAME:HEPARIN"],
        "lab": ["PAD", "UNK"],
        "vital": ["PAD", "UNK"],
    }
    for name, tokens in vocab_payloads.items():
        write_json(
            vocab_dir / f"{name}_vocab.json",
            {
                "name": name,
                "size": len(tokens),
                "pad_idx": 0,
                "unk_idx": 1,
                "idx_to_token": tokens,
                "token_to_idx": {token: index for index, token in enumerate(tokens)},
            },
        )
    write_json(vocab_dir / "lab_metadata.json", {})
    write_json(vocab_dir / "vital_metadata.json", {})
    write_json(
        vocab_dir / "vocab_summary.json",
        {
            "diagnosis_size": 2,
            "procedure_size": 2,
            "drug_size": 4,
            "lab_size": 2,
            "vital_size": 2,
            "built_from_split": "train",
        },
    )


def _write_drugbank_fixture_xml(project_root: Path) -> Path:
    drugbank_dir = project_root / "data" / "raw" / "drugbank"
    drugbank_dir.mkdir(parents=True, exist_ok=True)
    source_path = drugbank_dir / "full database.xml"
    source_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<drugbank xmlns="http://www.drugbank.ca">
  <drug type="small molecule">
    <drugbank-id primary="true">DB00001</drugbank-id>
    <name>Aspirin</name>
    <synonyms>
      <synonym language="english">Acetylsalicylic acid</synonym>
    </synonyms>
    <products>
      <product>
        <name>Bayer Aspirin</name>
      </product>
    </products>
    <drug-interactions>
      <drug-interaction>
        <drugbank-id>DB00002</drugbank-id>
        <name>Heparin sodium</name>
        <description>Interaction with Heparin sodium.</description>
      </drug-interaction>
      <drug-interaction>
        <drugbank-id>DB00003</drugbank-id>
        <name>Trial Agent</name>
        <description>Interaction with Trial Agent.</description>
      </drug-interaction>
      <drug-interaction>
        <drugbank-id>DB00004</drugbank-id>
        <name>Legacy Salicylate</name>
        <description>Interaction with Legacy Salicylate.</description>
      </drug-interaction>
    </drug-interactions>
  </drug>
  <drug type="small molecule">
    <drugbank-id primary="true">DB00002</drugbank-id>
    <drugbank-id>ALT00002</drugbank-id>
    <name>Heparin sodium</name>
    <synonyms>
      <synonym language="english">Heparin</synonym>
    </synonyms>
    <products>
      <product>
        <name>Heparin Flush</name>
      </product>
    </products>
    <drug-interactions>
      <drug-interaction>
        <drugbank-id>DB00001</drugbank-id>
        <name>Aspirin</name>
        <description>Interaction with Aspirin.</description>
      </drug-interaction>
    </drug-interactions>
  </drug>
  <drug type="small molecule">
    <drugbank-id primary="true">DB00003</drugbank-id>
    <name>Trial Agent</name>
    <synonyms>
      <synonym language="english">Aspirin</synonym>
      <synonym language="english">Heparin</synonym>
    </synonyms>
    <products/>
    <drug-interactions/>
  </drug>
  <drug type="small molecule">
    <drugbank-id primary="true">DB00004</drugbank-id>
    <name>Legacy Salicylate</name>
    <synonyms>
      <synonym language="english">Old Salicylate</synonym>
    </synonyms>
    <products>
      <product>
        <name>Aspirin</name>
      </product>
    </products>
    <drug-interactions/>
  </drug>
</drugbank>
""",
        encoding="utf-8",
    )
    return source_path


def _assert_optional_drugbank_outputs(project_root: Path) -> None:
    summary = read_json(project_root / "data" / "processed" / "drugbank" / "drugbank_summary.json")
    report = read_json(project_root / "data" / "processed" / "ddi" / "drug_ddi_drugbank_report.json")
    ddi_payload = load_pt(project_root / "data" / "processed" / "ddi" / "drug_ddi.pt")
    drugbank_ddi_payload = load_pt(project_root / "data" / "processed" / "ddi" / "drug_ddi_drugbank.pt")

    assert summary["active"] is False
    assert summary["reason"] == "missing_source_path"
    assert report["active"] is False
    assert report["reason"] == "missing_source_path"
    assert ddi_payload["matrix"] is not None
    assert drugbank_ddi_payload["matrix"] is not None


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

    cache_dir = _stage_filtered_tables_or_skip(config_path)
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
    _stage_filtered_tables_or_skip(config_path)
    build_vocab(config_path)
    build_ddi_matrix(config_path)
    build_trajectories(config_path)

    from src.data.dataset import MIMICTrajectoryDataset, collate_batch

    dataset = MIMICTrajectoryDataset("train", config_path)
    batch = collate_batch([dataset[0], dataset[1]])
    assert batch["diag_codes"].shape[0] == 2
    assert batch["visit_mask"].shape[0] == 2
    assert torch.all(batch["visit_mask"].sum(dim=1) >= 1)


def test_parquet_dataset_caches_arrow_tables_and_preserves_records(tmp_path: Path) -> None:
    pytest.importorskip("pyspark")
    pytest.importorskip("pyarrow")

    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=True)
    build_cohort(config_path)
    _stage_filtered_tables_or_skip(config_path)
    build_vocab(config_path)
    build_ddi_matrix(config_path)
    build_trajectories(config_path)

    from src.data.dataset import MIMICTrajectoryDataset

    dataset = MIMICTrajectoryDataset("train", config_path)
    first_record = dataset[0]
    second_record = dataset[0]

    assert first_record == second_record
    assert dataset._shard_cache
    cached_table = next(iter(dataset._shard_cache.values()))
    assert hasattr(cached_table, "num_rows")
    assert not isinstance(cached_table, list)


def test_collate_batch_can_emit_final_targets_without_full_target_tensor() -> None:
    torch = pytest.importorskip("torch")

    from src.data.dataset import collate_batch
    from src.training.losses import extract_last_valid_targets

    records = [
        {
            "subject_id": 1,
            "hadm_id": 10,
            "stay_id": 100,
            "num_steps": 3,
            "drug_vocab_size": 8,
            "lab_feature_size": 0,
            "vital_feature_size": 0,
            "steps": [
                {"diagnosis_ids": [1], "procedure_ids": [], "lab_values": [], "lab_mask": [], "vital_values": [], "vital_mask": [], "med_history_ids": [1, 2], "delta_hours": 0.0, "target_drugs": [1]},
                {"diagnosis_ids": [2], "procedure_ids": [], "lab_values": [], "lab_mask": [], "vital_values": [], "vital_mask": [], "med_history_ids": [2, 3], "delta_hours": 4.0, "target_drugs": [2, 4]},
                {"diagnosis_ids": [3], "procedure_ids": [], "lab_values": [], "lab_mask": [], "vital_values": [], "vital_mask": [], "med_history_ids": [4, 5], "delta_hours": 8.0, "target_drugs": [3, 6]},
            ],
        }
    ]

    full_batch = collate_batch(records, include_full_targets=True, include_final_target=True)
    fast_batch = collate_batch(records, include_full_targets=False, include_final_target=True)

    expected_final = extract_last_valid_targets(
        full_batch["target_drugs"],
        full_batch["visit_mask"],
    )
    assert "target_drugs" not in fast_batch
    assert torch.equal(fast_batch["final_target_drugs"], expected_final)


def test_collate_batch_applies_visit_and_history_truncation() -> None:
    torch = pytest.importorskip("torch")

    from src.data.dataset import collate_batch

    record = {
        "subject_id": 1,
        "hadm_id": 10,
        "stay_id": 100,
        "num_steps": 4,
        "drug_vocab_size": 12,
        "lab_feature_size": 0,
        "vital_feature_size": 0,
        "steps": [
            {"diagnosis_ids": [1], "procedure_ids": [], "lab_values": [], "lab_mask": [], "vital_values": [], "vital_mask": [], "med_history_ids": [1, 2, 3], "delta_hours": 0.0, "target_drugs": [1]},
            {"diagnosis_ids": [2], "procedure_ids": [], "lab_values": [], "lab_mask": [], "vital_values": [], "vital_mask": [], "med_history_ids": [4, 5, 6], "delta_hours": 2.0, "target_drugs": [2]},
            {"diagnosis_ids": [3], "procedure_ids": [], "lab_values": [], "lab_mask": [], "vital_values": [], "vital_mask": [], "med_history_ids": [7, 8, 9], "delta_hours": 4.0, "target_drugs": [3]},
            {"diagnosis_ids": [4], "procedure_ids": [], "lab_values": [], "lab_mask": [], "vital_values": [], "vital_mask": [], "med_history_ids": [10, 11, 12], "delta_hours": 6.0, "target_drugs": [4]},
        ],
    }

    batch = collate_batch([record], max_visits=2, max_history=2)

    assert int(batch["visit_lengths"][0].item()) == 2
    assert tuple(batch["visit_mask"].shape) == (1, 2)
    assert tuple(batch["med_history"].shape) == (1, 2, 2)
    assert torch.equal(batch["diag_codes"][0, :, 0], torch.tensor([3, 4], dtype=torch.long))
    assert torch.equal(batch["med_history"][0, 0], torch.tensor([8, 9], dtype=torch.long))
    assert torch.equal(batch["med_history"][0, 1], torch.tensor([11, 12], dtype=torch.long))


def test_shard_length_batch_sampler_keeps_batches_within_shards_and_covers_all_indices() -> None:
    from src.data.dataset import ShardLengthBatchSampler

    class DummyDataset:
        shard_row_indices = [[0, 1, 2], [3, 4]]
        row_num_steps = [5, 1, 3, 4, 2]

    sampler = ShardLengthBatchSampler(
        DummyDataset(),
        batch_size=2,
        length_bucket_window=2,
        shuffle=False,
        seed=0,
    )
    batches = list(sampler)
    flattened = [index for batch in batches for index in batch]

    assert flattened == [1, 0, 2, 4, 3]
    assert {tuple(batch) for batch in batches} == {(1, 0), (2,), (4, 3)}


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


def test_build_drugbank_metadata_parses_fixture_and_writes_vocab_metadata(tmp_path: Path) -> None:
    project_root = tmp_path / "drugbank_metadata_project"
    config_path = _write_config(project_root, spark_enabled=False)
    _write_drugbank_test_vocab_bundle(project_root)
    _write_drugbank_fixture_xml(project_root)

    records_path = build_drugbank_metadata(config_path)

    records = list(iter_jsonl_gz(records_path))
    summary = read_json(project_root / "data" / "processed" / "drugbank" / "drugbank_summary.json")
    vocab_metadata = read_json(project_root / "data" / "interim" / "vocab" / "drugbank_drug_metadata.json")

    assert len(records) == 4
    assert records[0]["primary_drugbank_id"] == "DB00001"
    assert records[1]["name"] == "Heparin sodium"
    assert summary["active"] is True
    assert summary["source_format"] == "drugbank_xml"
    assert summary["drugbank_drugs_parsed"] == 4
    assert summary["raw_interaction_edges"] == 4
    assert summary["matched_drugbank_records"] == 3
    assert summary["matched_vocab_drugs"] == 2
    assert summary["match_source_counts"] == {
        "primary_name": 1,
        "product_name": 1,
        "synonym": 1,
    }
    assert summary["collision_counts"]["ambiguous_record_matches"] == 1
    assert summary["collision_counts"]["vocab_tokens_with_multiple_drugbank_records"] == 1
    assert summary["auxiliary_only"] is True
    assert summary["research_grade"] is False
    assert summary["ambiguous_examples"][0]["primary_drugbank_id"] == "DB00003"
    assert vocab_metadata["NAME:ASPIRIN"]["collision"] is True
    assert vocab_metadata["NAME:ASPIRIN"]["matched_drugbank_ids"] == ["DB00001", "DB00004"]
    assert vocab_metadata["NAME:HEPARIN"]["match_sources"] == ["synonym"]


def test_build_drugbank_metadata_writes_inactive_summary_when_source_missing(tmp_path: Path) -> None:
    project_root = tmp_path / "drugbank_metadata_missing_project"
    config_path = _write_config(project_root, spark_enabled=False)
    _write_drugbank_test_vocab_bundle(project_root)

    records_path = build_drugbank_metadata(config_path)

    records = list(iter_jsonl_gz(records_path))
    summary = read_json(project_root / "data" / "processed" / "drugbank" / "drugbank_summary.json")
    vocab_metadata = read_json(project_root / "data" / "interim" / "vocab" / "drugbank_drug_metadata.json")

    assert records == []
    assert summary["active"] is False
    assert summary["reason"] == "missing_source_path"
    assert summary["matched_drugbank_records"] == 0
    assert summary["matched_vocab_drugs"] == 0
    assert vocab_metadata == {}


def test_build_drugbank_ddi_matrix_builds_separate_auxiliary_artifact(tmp_path: Path) -> None:
    project_root = tmp_path / "drugbank_ddi_project"
    config_path = _write_config(project_root, spark_enabled=False)
    _write_drugbank_test_vocab_bundle(project_root)
    _write_drugbank_fixture_xml(project_root)

    matrix_path = build_drugbank_ddi_matrix(config_path)

    payload = load_pt(matrix_path)
    report = read_json(project_root / "data" / "processed" / "ddi" / "drug_ddi_drugbank_report.json")
    pair_rows = list(iter_jsonl_gz(project_root / "data" / "processed" / "ddi" / "drugbank_ddi_pairs.jsonl.gz"))

    assert tuple((len(payload["matrix"]), len(payload["matrix"][0]))) == (4, 4)
    assert payload["matrix"][2][3] == 1
    assert payload["matrix"][3][2] == 1
    assert payload["matrix"][2][2] == 0
    assert report["active"] is True
    assert report["source_format"] == "drugbank_xml"
    assert report["ddi_type"] == "drugbank_knowledge_base_auxiliary"
    assert report["ddi_research_grade"] is False
    assert report["drugbank_drugs_parsed"] == 4
    assert report["mapped_drugbank_records"] == 3
    assert report["mapped_vocab_drugs"] == 2
    assert report["raw_interaction_edges"] == 4
    assert report["mapped_interaction_edges"] == 2
    assert report["matched_pairs"] == 1
    assert report["nonzero_pairs"] == 1
    assert report["mapped_pairs"] == 1
    assert report["dropped_interaction_edges"] == 2
    assert report["self_interaction_edges_skipped"] == 1
    assert len(pair_rows) == 4
    assert any(bool(row["kept"]) for row in pair_rows)


def test_build_drugbank_ddi_matrix_writes_inactive_artifact_when_source_missing(tmp_path: Path) -> None:
    project_root = tmp_path / "drugbank_ddi_missing_project"
    config_path = _write_config(project_root, spark_enabled=False)
    _write_drugbank_test_vocab_bundle(project_root)

    matrix_path = build_drugbank_ddi_matrix(config_path)

    payload = load_pt(matrix_path)
    report = read_json(project_root / "data" / "processed" / "ddi" / "drug_ddi_drugbank_report.json")
    pair_rows = list(iter_jsonl_gz(project_root / "data" / "processed" / "ddi" / "drugbank_ddi_pairs.jsonl.gz"))

    assert report["active"] is False
    assert report["reason"] == "missing_source_path"
    assert report["matched_pairs"] == 0
    assert payload["matrix"][2][3] == 0
    assert pair_rows == []


def test_preprocess_shell_script_smoke_with_missing_drugbank(tmp_path: Path) -> None:
    pytest.importorskip("pyspark")
    pytest.importorskip("pyarrow")

    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=True)
    script_path = Path(__file__).resolve().parents[1] / "scripts" / "preprocess.sh"
    result = subprocess.run(
        [
            "bash",
            str(script_path),
            "--config",
            str(config_path),
            "--python",
            sys.executable,
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    _skip_if_spark_wrapper_unavailable(result)
    assert result.returncode == 0, result.stderr
    _assert_optional_drugbank_outputs(project_root)


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
    _skip_if_spark_wrapper_unavailable(result)
    assert result.returncode == 0, result.stderr
    _assert_optional_drugbank_outputs(project_root)
