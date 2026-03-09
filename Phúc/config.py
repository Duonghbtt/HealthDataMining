# ============================================
# MODULE 3 - MIMIC-IV DATA PROCESSING
# Configuration File
# ============================================

import os
from pathlib import Path

# ============== PATH CONFIGURATION ==============
PROJECT_ROOT = Path("D:/ProjectBigData")
DATA_PATH = PROJECT_ROOT / "data" / "mimic-iv-3.1.zip"
OUTPUT_PATH = PROJECT_ROOT / "output" / "module3"

# Các thư mục dữ liệu chính
DATA_DIRS = {
    "core": OUTPUT_PATH / "01_core_data",
    "clinical": OUTPUT_PATH / "02_clinical_data",
    "icu": OUTPUT_PATH / "03_icu_data",
    "processed": OUTPUT_PATH / "04_processed_data",
    "features": OUTPUT_PATH / "05_features",
    "analysis": OUTPUT_PATH / "06_analysis"
}

# Tạo thư mục nếu chưa tồn tại
for dir_path in DATA_DIRS.values():
    dir_path.mkdir(parents=True, exist_ok=True)

# ============== FILE MAPPING ==============
# Ánh xạ các file dữ liệu từ MIMIC-IV
FILE_MAPPING = {
    # Core Data
    "admissions": "core/admissions.csv.gz",
    "patients": "core/patients.csv.gz",

    # Clinical Data
    "diagnoses_icd": "clinical/diagnoses_icd.csv.gz",
    "procedures_icd": "clinical/procedures_icd.csv.gz",
    "labevents": "clinical/labevents.csv.gz",
    "prescriptions": "clinical/prescriptions.csv.gz",
    "microbiologyevents": "clinical/microbiologyevents.csv.gz",

    # ICU Data
    "icustays": "icu/icustays.csv.gz",
    "chartevents": "icu/chartevents.csv.gz",
    "inputevents": "icu/inputevents.csv.gz",
    "outputevents": "icu/outputevents.csv.gz",
    "procedureevents": "icu/procedureevents.csv.gz",

    # Transfer Data
    "transfers": "clinical/transfers.csv.gz"
}

# ============== SPARK CONFIGURATION ==============
SPARK_CONFIG = {
    "appName": "MIMIC-IV-DataProcessing",
    "master": "local[*]",  # Sử dụng tất cả CPU cores
    "driver_memory": "4g",
    "executor_memory": "4g",
    "shuffle_partitions": "200",
    "broadcast_timeout": "600",
}

# ============== DATA SCHEMA DEFINITIONS ==============
SCHEMAS = {
    "admissions": {
        "columns": ["subject_id", "hadm_id", "admittime", "dischtime", "deathtime",
                    "admission_type", "admit_provider_id", "admission_location",
                    "discharge_location", "insurance", "language", "marital_status",
                    "ethnicity", "edregtime", "edouttime", "hospital_expire_flag"],
        "dtypes": {
            "subject_id": "integer",
            "hadm_id": "integer",
            "admittime": "timestamp",
            "hospital_expire_flag": "integer"
        }
    },
    "patients": {
        "columns": ["subject_id", "gender", "anchor_age", "anchor_year", "anchor_year_group"],
        "dtypes": {
            "subject_id": "integer",
            "anchor_age": "integer"
        }
    },
    "icustays": {
        "columns": ["stay_id", "subject_id", "hadm_id", "intime", "outtime",
                    "icu_level", "first_careunit", "last_careunit", "los"],
        "dtypes": {
            "stay_id": "integer",
            "subject_id": "integer",
            "hadm_id": "integer"
        }
    },
    "labevents": {
        "columns": ["labevent_id", "subject_id", "hadm_id", "specimen_id", "charttime",
                    "storetime", "test_id", "test_name", "value", "valuenum", "valueuom",
                    "ref_range_lower", "ref_range_upper", "flag", "priority"],
        "dtypes": {
            "subject_id": "integer",
            "hadm_id": "integer",
            "valuenum": "double"
        }
    },
    "prescriptions": {
        "columns": ["subject_id", "hadm_id", "pharmacy_id", "starttime", "stoptime",
                    "drug_type", "drug", "gsn", "ndc", "prod_strength", "form_rx",
                    "dose_val_rx", "dose_unit_rx", "form_val_disp", "form_unit_disp",
                    "doses_per_24hrs", "route"],
        "dtypes": {
            "subject_id": "integer",
            "hadm_id": "integer",
            "doses_per_24hrs": "double"
        }
    }
}

# ============== PROCESSING PARAMETERS ==============
PROCESSING_PARAMS = {
    "sample_ratio": 1.0,  # 1.0 = sử dụng toàn bộ dữ liệu
    "outlier_std": 3,  # số lần độ lệch chuẩn để phát hiện outliers
    "missing_threshold": 0.8,  # ngưỡng cột bị thiếu dữ liệu
    "date_range": {
        "start": "2008-01-01",
        "end": "2019-12-31"
    }
}

# ============== LOGGING CONFIGURATION ==============
LOG_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": OUTPUT_PATH / "module3.log"
}

# ============== QUALITY METRICS ==============
QUALITY_METRICS = {
    "data_completeness": 0.9,  # 90% completeness
    "duplicate_tolerance": 0.01,  # 1% duplicates allowed
    "outlier_tolerance": 0.05,  # 5% outliers allowed
}

# ============== FEATURE ENGINEERING PARAMS ==============
FEATURE_PARAMS = {
    "time_windows": [6, 12, 24, 48],  # hours
    "aggregation_funcs": ["mean", "std", "min", "max", "last"],
    "lab_tests_important": [
        "hematocrit", "hemoglobin", "platelets", "potassium",
        "sodium", "glucose", "creatinine", "bun"
    ]
}

# ============== ANALYSIS PARAMETERS ==============
ANALYSIS_PARAMS = {
    "mortality_threshold": 30,  # days
    "readmission_threshold": 30,  # days
    "los_percentile": [25, 50, 75, 90],
}

if __name__ == "__main__":
    print("✓ Module 3 Configuration Loaded")
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Output Path: {OUTPUT_PATH}")
    print(f"Spark Master: {SPARK_CONFIG['master']}")