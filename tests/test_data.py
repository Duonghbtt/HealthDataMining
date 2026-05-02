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
from src.data.export_tensorized_trajectories import export_tensorized_trajectories
from src.data.build_trajectories import build_trajectories
from src.data.build_vocab import build_vocab
from src.data.dataset import validate_patient_level_splits
from src.utils.io import iter_jsonl_gz, read_csv_gz, read_json, save_pt, write_json, write_jsonl_gz


def _write_csv_gz(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
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
                "  cohort_root: data/interim/cohort",
                "  trajectory_interim_root: data/interim/trajectories",
                "  vocab_root: data/interim/vocab",
                "  processed_root: data/processed",
                "  ddi_root: data/processed/ddi",
                "  mimic_prescriptions_path: data/raw/hosp/prescriptions.csv.gz",
                "  rxnorm_root: data/processed/ddi/RxNorm_full_04062026",
                "  drugbank_vocab_path: data/processed/ddi/drugbank vocabulary.csv",
                "  ddinter_root: data/processed/ddi",
                "  ddinter_glob: ddinter_downloads_code_*.csv",
                "ddinter_include_unknown: false",
                "processed_format: parquet",
                "split:",
                "  train: 1.0",
                "  val: 0.0",
                "  test: 0.0",
                "cohort:",
                "  require_diagnosis: true",
                "  require_medication: true",
                "  min_los_hours: 0.0",
                "  min_patient_visits: 1",
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
                "  local_dir: data/interim/spark_tmp_tests",
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
    processed_ddi = project_root / "data" / "processed" / "ddi"

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
    prescription_rows = [
        {"subject_id": 1, "hadm_id": 11, "pharmacy_id": index, "poe_id": f"p_asp_{index}", "poe_seq": 1, "order_provider_id": "A", "starttime": "2020-01-01 08:00:00", "stoptime": "2020-01-01 20:00:00", "drug_type": "MAIN", "drug": "Aspirin", "formulary_drug_cd": "ASP100", "gsn": "", "ndc": "11111111111", "prod_strength": "", "form_rx": "", "dose_val_rx": "", "dose_unit_rx": "", "form_val_disp": "", "form_unit_disp": "", "doses_per_24_hrs": "", "route": "PO"}
        for index in range(1, 13)
    ]
    prescription_rows.extend(
        [
            {"subject_id": 2, "hadm_id": 22, "pharmacy_id": 200 + index, "poe_id": f"p_hep_{index}", "poe_seq": 1, "order_provider_id": "B", "starttime": "2020-01-02 09:00:00", "stoptime": "2020-01-02 20:00:00", "drug_type": "MAIN", "drug": "Heparin", "formulary_drug_cd": "HEP5000", "gsn": "", "ndc": "22222222222", "prod_strength": "", "form_rx": "", "dose_val_rx": "", "dose_unit_rx": "", "form_val_disp": "", "form_unit_disp": "", "doses_per_24_hrs": "", "route": "IV"}
            for index in range(1, 11)
        ]
    )
    prescription_rows.extend(
        [
            {"subject_id": 1, "hadm_id": 11, "pharmacy_id": 101, "poe_id": "p_art_1", "poe_seq": 1, "order_provider_id": "A", "starttime": "2020-01-01 10:00:00", "stoptime": "2020-01-01 10:30:00", "drug_type": "MAIN", "drug": "NS Flush", "formulary_drug_cd": "", "gsn": "", "ndc": "", "prod_strength": "", "form_rx": "", "dose_val_rx": "", "dose_unit_rx": "", "form_val_disp": "", "form_unit_disp": "", "doses_per_24_hrs": "", "route": "IV"},
            {"subject_id": 1, "hadm_id": 11, "pharmacy_id": 102, "poe_id": "p_art_2", "poe_seq": 1, "order_provider_id": "A", "starttime": "2020-01-01 11:00:00", "stoptime": "2020-01-01 11:30:00", "drug_type": "MAIN", "drug": "Sterile Water", "formulary_drug_cd": "", "gsn": "", "ndc": "", "prod_strength": "", "form_rx": "", "dose_val_rx": "", "dose_unit_rx": "", "form_val_disp": "", "form_unit_disp": "", "doses_per_24_hrs": "", "route": "IV"},
            {"subject_id": 1, "hadm_id": 11, "pharmacy_id": 103, "poe_id": "p_unknown", "poe_seq": 1, "order_provider_id": "A", "starttime": "2020-01-01 12:00:00", "stoptime": "2020-01-01 12:30:00", "drug_type": "MAIN", "drug": "MysteryDrug", "formulary_drug_cd": "", "gsn": "", "ndc": "", "prod_strength": "", "form_rx": "", "dose_val_rx": "", "dose_unit_rx": "", "form_val_disp": "", "form_unit_disp": "", "doses_per_24_hrs": "", "route": "PO"},
        ]
    )
    _write_csv_gz(
        raw_hosp / "prescriptions.csv.gz",
        prescription_rows,
        ["subject_id", "hadm_id", "pharmacy_id", "poe_id", "poe_seq", "order_provider_id", "starttime", "stoptime", "drug_type", "drug", "formulary_drug_cd", "gsn", "ndc", "prod_strength", "form_rx", "dose_val_rx", "dose_unit_rx", "form_val_disp", "form_unit_disp", "doses_per_24_hrs", "route"],
    )
    emar_rows = [
        {"subject_id": 1, "hadm_id": 11, "emar_id": f"e_asp_{index}", "emar_seq": index, "poe_id": f"p_asp_{index}", "pharmacy_id": index, "enter_provider_id": "", "charttime": "2020-01-01 08:00:00", "medication": "Aspirin", "event_txt": "Administered", "scheduletime": "2020-01-01 08:00:00", "storetime": "2020-01-01 08:05:00"}
        for index in range(1, 3)
    ]
    emar_rows.extend(
        [
            {"subject_id": 1, "hadm_id": 11, "emar_id": "e_art_1", "emar_seq": 100, "poe_id": "p_art_1", "pharmacy_id": 101, "enter_provider_id": "", "charttime": "2020-01-01 10:00:00", "medication": "NS Flush", "event_txt": "Administered", "scheduletime": "2020-01-01 10:00:00", "storetime": "2020-01-01 10:05:00"},
            {"subject_id": 1, "hadm_id": 11, "emar_id": "e_unknown", "emar_seq": 101, "poe_id": "p_unknown", "pharmacy_id": 103, "enter_provider_id": "", "charttime": "2020-01-01 12:00:00", "medication": "MysteryDrug", "event_txt": "Administered", "scheduletime": "2020-01-01 12:00:00", "storetime": "2020-01-01 12:05:00"},
        ]
    )
    _write_csv_gz(
        raw_hosp / "emar.csv.gz",
        emar_rows,
        ["subject_id", "hadm_id", "emar_id", "emar_seq", "poe_id", "pharmacy_id", "enter_provider_id", "charttime", "medication", "event_txt", "scheduletime", "storetime"],
    )
    pharmacy_rows = [
        {"subject_id": 2, "hadm_id": 22, "pharmacy_id": 201, "poe_id": "p_hep", "starttime": "2020-01-02 09:00:00", "stoptime": "2020-01-02 20:00:00", "medication": "Heparin", "proc_type": "Unit Dose", "status": "Active", "entertime": "2020-01-02 08:30:00", "verifiedtime": "2020-01-02 08:45:00", "route": "IV", "frequency": "BID", "disp_sched": "", "infusion_type": "", "sliding_scale": "", "lockout_interval": "", "basal_rate": "", "one_hr_max": "", "doses_per_24_hrs": "", "duration": "", "duration_interval": "", "expiration_value": "", "expiration_unit": "", "expirationdate": "", "dispensation": "", "fill_quantity": ""}
    ]
    _write_csv_gz(
        raw_hosp / "pharmacy.csv.gz",
        pharmacy_rows,
        ["subject_id", "hadm_id", "pharmacy_id", "poe_id", "starttime", "stoptime", "medication", "proc_type", "status", "entertime", "verifiedtime", "route", "frequency", "disp_sched", "infusion_type", "sliding_scale", "lockout_interval", "basal_rate", "one_hr_max", "doses_per_24_hrs", "duration", "duration_interval", "expiration_value", "expiration_unit", "expirationdate", "dispensation", "fill_quantity"],
    )
    rxnorm_rrf = processed_ddi / "RxNorm_full_04062026" / "rrf"
    rxnorm_rrf.mkdir(parents=True, exist_ok=True)
    (rxnorm_rrf / "RXNCONSO.RRF").write_text(
        "\n".join(
            [
                "111|ENG||||||123|123|111||RXNORM|SCD|111|Aspirin||N|4096|",
                "211|ENG||||||223|223|211||RXNORM|IN|211|Acetylsalicylic acid||N|4096|",
                "222|ENG||||||323|323|222||RXNORM|IN|222|Heparin||N|4096|",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (rxnorm_rrf / "RXNREL.RRF").write_text(
        "111|123|AUI|RO|211|223|AUI|has_ingredient|1||RXNORM||||N|\n",
        encoding="utf-8",
    )
    (rxnorm_rrf / "RXNSAT.RRF").write_text(
        "\n".join(
            [
                "111|||123|AUI|11111111111|||NDC|RXNORM|11111111111|N|4096|",
                "222|||323|AUI|22222222222|||NDC|RXNORM|22222222222|N|4096|",
                "111|||123|AUI|N02BA01|||ATC_LEVEL|ATC|5|N||",
                "222|||323|AUI|B01AB01|||ATC_LEVEL|ATC|5|N||",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_csv(
        processed_ddi / "drugbank vocabulary.csv",
        [
            {
                "DrugBank ID": "DB00945",
                "Accession Numbers": "",
                "Common name": "Acetylsalicylic acid",
                "CAS": "",
                "UNII": "",
                "Synonyms": "Aspirin | ASA",
                "Standard InChI Key": "",
            },
            {
                "DrugBank ID": "DB01109",
                "Accession Numbers": "",
                "Common name": "Heparin",
                "CAS": "",
                "UNII": "",
                "Synonyms": "Heparin",
                "Standard InChI Key": "",
            },
        ],
        ["DrugBank ID", "Accession Numbers", "Common name", "CAS", "UNII", "Synonyms", "Standard InChI Key"],
    )
    _write_csv(
        processed_ddi / "ddinter_downloads_code_A.csv",
        [
            {
                "DDInterID_A": "DDInter20",
                "Drug_A": "Acetylsalicylic acid",
                "DDInterID_B": "DDInter856",
                "Drug_B": "Heparin",
                "Level": "Moderate",
            }
        ],
        ["DDInterID_A", "Drug_A", "DDInterID_B", "Drug_B", "Level"],
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


def _build_patient_state_encoder_for_batch(
    *,
    batch,
    diagnosis_vocab_size: int,
    procedure_vocab_size: int,
    drug_vocab_size: int,
):
    from src.models.patient_state_encoder import PatientStateEncoder

    return PatientStateEncoder(
        diagnosis_vocab_size=diagnosis_vocab_size,
        procedure_vocab_size=procedure_vocab_size,
        drug_vocab_size=drug_vocab_size,
        num_lab_features=int(batch["lab_values"].shape[-1]),
        num_vital_features=int(batch["vital_values"].shape[-1]),
        code_embedding_dim=8,
        medication_embedding_dim=8,
        numeric_projection_dim=4,
        time_embedding_dim=4,
        visit_hidden_dim=16,
        hidden_dim=12,
        dropout=0.0,
        encoder_mode="modality_aware_gru",
        modality_hidden_dim=10,
        fusion_hidden_dim=16,
        modality_dropout=0.0,
        use_temporal_attention=True,
        temporal_attention_heads=1,
        temporal_attention_dropout=0.0,
    )


def test_build_vocab_ignores_spark_flag_and_uses_python_pipeline(tmp_path: Path) -> None:
    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=True)
    build_cohort(config_path)
    build_vocab(config_path)
    assert (project_root / "data" / "interim" / "vocab" / "diagnosis_vocab.json").exists()


def test_build_vocab_builds_canonical_med_vocab_main(tmp_path: Path) -> None:
    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=False)

    build_cohort(config_path)
    build_vocab(config_path)

    diag_vocab = read_json(project_root / "data" / "interim" / "vocab" / "diag_vocab.json")
    proc_vocab = read_json(project_root / "data" / "interim" / "vocab" / "proc_vocab.json")
    med_vocab_main = read_json(project_root / "data" / "interim" / "vocab" / "med_vocab_main.json")
    med_vocab_main_metadata = read_json(project_root / "data" / "interim" / "vocab" / "med_vocab_main_metadata.json")
    med_vocab_main_report = read_json(project_root / "data" / "interim" / "vocab" / "med_vocab_main_build_report.json")

    assert diag_vocab["idx_to_token"][:2] == ["PAD", "UNK"]
    assert "ICD9:4019" in diag_vocab["token_to_idx"]
    assert proc_vocab["idx_to_token"][:2] == ["PAD", "UNK"]
    assert "PROC9:3893" in proc_vocab["token_to_idx"]

    assert med_vocab_main["idx_to_token"] == ["PAD", "UNK", "MEDRX:211", "MEDRX:222"]
    assert med_vocab_main_metadata["MEDRX:211"]["canonical_name"] == "Acetylsalicylic acid"
    assert med_vocab_main_metadata["MEDRX:211"]["canonical_rxcui"] == "211"
    assert med_vocab_main_metadata["MEDRX:222"]["canonical_name"] == "Heparin"
    assert med_vocab_main_report["matched_rxcui_unique_token_count"] >= 2
    assert med_vocab_main_report["matched_canonical_token_unique_token_count"] >= 2
    assert any(
        example["raw_token"] == "NAME:MYSTERYDRUG"
        for example in med_vocab_main_report["unmatched_top_examples"]
    )
    assert not (project_root / "data" / "interim" / "vocab" / "drug_vocab.json").exists()


def test_validate_patient_level_splits_detects_disjoint_subject_manifests(tmp_path: Path) -> None:
    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=False)

    build_cohort(config_path)
    validation = validate_patient_level_splits(config_path)

    assert validation["validated"] is True
    assert validation["counts"]["train"] >= 1


def test_build_trajectories_maps_medication_targets_to_med_vocab_main_ids(tmp_path: Path) -> None:
    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=False)

    build_cohort(config_path)
    build_vocab(config_path)
    outputs = build_trajectories(config_path)

    med_vocab_main = read_json(project_root / "data" / "interim" / "vocab" / "med_vocab_main.json")
    aspirin_med_id = int(med_vocab_main["token_to_idx"]["MEDRX:211"])
    heparin_med_id = int(med_vocab_main["token_to_idx"]["MEDRX:222"])
    train_records = list(iter_jsonl_gz(outputs["train"]))
    assert train_records

    all_target_ids = [
        int(drug_id)
        for record in train_records
        for step in record["steps"]
        for drug_id in step["target_drugs"]
    ]
    all_history_ids = [
        int(drug_id)
        for record in train_records
        for step in record["steps"]
        for drug_id in step["med_history_ids"]
    ]

    assert aspirin_med_id in all_target_ids
    assert heparin_med_id in all_target_ids
    assert 1 not in all_target_ids
    assert all(drug_id != 1 for drug_id in all_history_ids)


def test_build_ddi_matrix_bridges_rxnorm_to_ddinter_via_drugbank(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")

    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=False)

    build_cohort(config_path)
    build_vocab(config_path)
    ddi_path = build_ddi_matrix(config_path)

    ddi_payload = torch.load(ddi_path, map_location="cpu", weights_only=False)
    ddi_report = read_json(project_root / "data" / "processed" / "ddi" / "drug_ddi_report.json")
    med_vocab_main = read_json(project_root / "data" / "interim" / "vocab" / "med_vocab_main.json")
    aspirin_id = int(med_vocab_main["token_to_idx"]["MEDRX:211"])
    heparin_id = int(med_vocab_main["token_to_idx"]["MEDRX:222"])

    matrix = ddi_payload["matrix"]
    severity_matrix = ddi_payload["severity_matrix"]
    assert int(matrix[aspirin_id, heparin_id]) == 1
    assert int(matrix[heparin_id, aspirin_id]) == 1
    assert int(severity_matrix[aspirin_id, heparin_id]) >= 1
    assert ddi_report["representation"] == "med_vocab_main"
    assert ddi_report["num_vocab_drugs_matched"] >= 2
    assert ddi_report["num_ddi_pairs_matched"] >= 1
    assert ddi_report["match_rate"] > 0.0


def test_data_pipeline_builders_and_tensorized_export(tmp_path: Path) -> None:
    pytest.importorskip("torch")
    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=False)

    cohort_path = build_cohort(config_path)
    cohort_rows = read_csv_gz(cohort_path)
    assert len(cohort_rows) == 2
    assert {row["stay_id"] for row in cohort_rows} == {"111", "222"}
    assert (project_root / "data" / "interim" / "cohort" / "cohort_keys.parquet").exists()

    build_vocab(config_path)
    diagnosis_vocab = read_json(project_root / "data" / "interim" / "vocab" / "diagnosis_vocab.json")
    assert diagnosis_vocab["idx_to_token"][:2] == ["PAD", "UNK"]
    assert "ICD9:4019" in diagnosis_vocab["token_to_idx"]

    ddi_path = build_ddi_matrix(config_path)
    assert ddi_path.exists()

    outputs = build_trajectories(config_path)
    assert outputs["train"].exists()
    assert (project_root / "data" / "interim" / "trajectories").exists()
    metadata = read_json(project_root / "data" / "interim" / "trajectories" / "metadata.json")
    summary = read_json(project_root / "data" / "interim" / "trajectories" / "trajectory_summary.json")
    assert metadata["drug_representation"] == "med_vocab_main"
    assert metadata["drug_vocab_size"] >= 4
    assert summary["num_trajectories"] >= 2
    assert summary["trajectory_length_max"] >= 1
    tensorized_root = export_tensorized_trajectories(config_path, overwrite=True, rows_per_pt_shard=2)
    assert (tensorized_root / "manifest.json").exists()
    tensorized_manifest = read_json(tensorized_root / "manifest.json")
    assert tensorized_manifest["drug_representation"] == "med_vocab_main"
    assert tensorized_manifest["splits"]["train"]


def test_dataset_collate_when_torch_available(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch.utils.data")

    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=False)
    build_cohort(config_path)
    build_vocab(config_path)
    build_ddi_matrix(config_path)
    build_trajectories(config_path)

    from src.data.dataset import MIMICTrajectoryDataset, collate_batch

    dataset = MIMICTrajectoryDataset("train", config_path)
    batch = collate_batch([dataset[0], dataset[1]])
    assert batch["diag_codes"].shape[0] == 2
    assert batch["visit_mask"].shape[0] == 2
    assert torch.all(batch["visit_mask"].sum(dim=1) >= 1)


def test_dataset_batch_runs_through_patient_state_encoder_with_new_benchmark(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch.utils.data")

    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=False)
    build_cohort(config_path)
    build_vocab(config_path)
    build_ddi_matrix(config_path)
    build_trajectories(config_path)

    from src.data.dataset import MIMICTrajectoryDataset, collate_batch

    dataset = MIMICTrajectoryDataset("train", config_path)
    batch = collate_batch([dataset[0], dataset[1]])
    diag_vocab = read_json(project_root / "data" / "interim" / "vocab" / "diag_vocab.json")
    proc_vocab = read_json(project_root / "data" / "interim" / "vocab" / "proc_vocab.json")
    med_vocab = read_json(project_root / "data" / "interim" / "vocab" / "med_vocab_main.json")
    encoder = _build_patient_state_encoder_for_batch(
        batch=batch,
        diagnosis_vocab_size=int(diag_vocab["size"]),
        procedure_vocab_size=int(proc_vocab["size"]),
        drug_vocab_size=int(med_vocab["size"]),
    )

    outputs = encoder(batch)

    assert batch["diag_codes"].ndim == 3
    assert batch["proc_codes"].ndim == 3
    assert batch["med_history"].ndim == 3
    assert batch["target_drugs"].ndim == 3
    assert batch["visit_mask"].ndim == 2
    assert batch["lab_values"].shape[-1] == 0
    assert batch["vital_values"].shape[-1] == 0
    assert batch["ddi_adj"].shape == (int(med_vocab["size"]), int(med_vocab["size"]))
    assert torch.isfinite(batch["target_drugs"]).all()
    assert torch.isfinite(batch["ddi_adj"]).all()
    assert outputs["visit_repr"].shape[:2] == batch["visit_mask"].shape
    assert torch.isfinite(outputs["visit_repr"]).all()
    assert torch.isfinite(outputs["current_state"]).all()


def test_tensorized_dataset_falls_back_for_missing_lab_vital_and_runs_encoder(tmp_path: Path) -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch.utils.data")

    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=False)
    build_cohort(config_path)
    build_vocab(config_path)
    build_ddi_matrix(config_path)
    build_trajectories(config_path)

    from src.data.dataset import MIMICTrajectoryDataset, collate_batch
    from src.data.tensorized_dataset import TensorizedTrajectoryDataset, tensorized_collate_batch

    source_dataset = MIMICTrajectoryDataset("train", config_path)
    source_batch = collate_batch([source_dataset[0], source_dataset[1]])
    tensorized_root = project_root / "data" / "artifacts" / "tensorized"
    train_dir = tensorized_root / "train"
    train_dir.mkdir(parents=True, exist_ok=True)
    shard_payload = {
        "diag_codes": source_batch["diag_codes"],
        "diag_mask": source_batch["diag_mask"],
        "proc_codes": source_batch["proc_codes"],
        "proc_mask": source_batch["proc_mask"],
        "med_history": source_batch["med_history"],
        "med_history_mask": source_batch["med_history_mask"],
        "time_delta_hours": source_batch["time_delta_hours"],
        "visit_mask": source_batch["visit_mask"],
        "target_drugs": source_batch["target_drugs"],
        "subject_ids": torch.as_tensor(source_batch["subject_ids"], dtype=torch.long),
        "hadm_ids": torch.as_tensor(source_batch["hadm_ids"], dtype=torch.long),
        "stay_ids": torch.as_tensor(source_batch["stay_ids"], dtype=torch.long),
    }
    save_pt(train_dir / "part-00000.pt", shard_payload)
    write_json(
        tensorized_root / "manifest.json",
        {
            "format": "tensorized_pt",
            "schema_version": 1,
            "drug_vocab_size": int(source_batch["target_drugs"].shape[-1]),
            "default_lab_feature_size": 0,
            "default_vital_feature_size": 0,
            "splits": {
                "train": [{"path": "train/part-00000.pt", "rows": 2}],
                "val": [],
                "test": [],
            },
        },
    )

    dataset = TensorizedTrajectoryDataset("train", config_path)
    batch = tensorized_collate_batch([dataset[0], dataset[1]])
    diag_vocab = read_json(project_root / "data" / "interim" / "vocab" / "diag_vocab.json")
    proc_vocab = read_json(project_root / "data" / "interim" / "vocab" / "proc_vocab.json")
    med_vocab = read_json(project_root / "data" / "interim" / "vocab" / "med_vocab_main.json")
    encoder = _build_patient_state_encoder_for_batch(
        batch=batch,
        diagnosis_vocab_size=int(diag_vocab["size"]),
        procedure_vocab_size=int(proc_vocab["size"]),
        drug_vocab_size=int(med_vocab["size"]),
    )

    outputs = encoder(batch)

    assert batch["lab_values"].shape[-1] == 0
    assert batch["vital_values"].shape[-1] == 0
    assert batch["ddi_adj"].shape == (int(med_vocab["size"]), int(med_vocab["size"]))
    assert torch.isfinite(batch["time_delta_hours"]).all()
    assert torch.isfinite(batch["target_drugs"]).all()
    assert outputs["visit_repr"].shape[:2] == batch["visit_mask"].shape
    assert torch.isfinite(outputs["visit_repr"]).all()


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
        project_root / "data" / "processed" / "train" / "trajectories.jsonl.gz",
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
    pytest.importorskip("torch")

    pwsh = shutil.which("pwsh") or shutil.which("powershell")
    if not pwsh:
        pytest.skip("PowerShell is not available in this environment")

    project_root = _build_mock_project(tmp_path)
    config_path = _write_config(project_root, spark_enabled=False)
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
