# MODULE 3 - MIMIC-IV DATA PROCESSING WITH PYSPARK

## 📋 Overview

Module 3 là một hệ thống xử lý dữ liệu toàn diện cho MIMIC-IV sử dụng Apache PySpark. Nó cung cấp một pipeline hoàn chỉnh từ tải dữ liệu, làm sạch, xử lý, xây dựng features cho đến phân tích dữ liệu.

## 🏗️ Project Structure

```
module3/
├── module3_config.py              # Cấu hình chính
├── spark_session.py               # Quản lý Spark Session
├── data_loader.py                 # Đọc dữ liệu từ file ZIP
├── data_processor.py              # Làm sạch & biến đổi dữ liệu
├── feature_engineering.py         # Xây dựng features
├── data_analysis.py               # Phân tích dữ liệu
├── main.py                        # Script chính
├── requirements.txt               # Dependencies
├── README.md                      # Tài liệu này
└── tests/
    ├── test_data_loader.py
    ├── test_processor.py
    └── test_features.py
```

## 📦 Core Modules

### 1. **module3_config.py**
- Định cấu hình paths, file mapping
- Cấu hình Spark
- Định nghĩa schemas dữ liệu
- Tham số xử lý & phân tích

### 2. **spark_session.py**
- Quản lý Spark Session (Singleton pattern)
- Khởi tạo và cấu hình tối ưu
- Logging & monitoring

```python
from spark_session import SparkSessionManager

manager = SparkSessionManager()
spark = manager.get_or_create_spark_session()
```

### 3. **data_loader.py**
- Đọc file CSV.GZ từ file ZIP
- Xác thực dữ liệu
- Hỗ trợ sampling dữ liệu lớn

```python
from data_loader import MIMICDataLoader, DataValidator

loader = MIMICDataLoader(spark, "path/to/mimic-iv-3.1.zip")
df = loader.load_csv_from_zip("admissions")
```

### 4. **data_processor.py**
- Loại bỏ duplicates
- Xử lý missing values
- Loại bỏ outliers
- Chuẩn hóa dữ liệu
- Báo cáo chất lượng dữ liệu

```python
from data_processor import DataCleaner, DataTransformer

cleaner = DataCleaner()
df_clean = cleaner.remove_duplicates(df)
df_clean = cleaner.handle_missing_values(df_clean)
```

### 5. **feature_engineering.py**
- Xây dựng demographic features (tuổi, giới tính)
- Lab features (xét nghiệm)
- Diagnosis features (chẩn đoán)
- Vital signs features (dấu hiệu sinh tồn)
- Admission features (LOS, mortality)

```python
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer()
patient_features = engineer.build_patient_features(patients_df)
admission_features = engineer.build_admission_features(admissions_df)
```

### 6. **data_analysis.py**
- Phân tích nhân khẩu học
- Thống kê nhập viện
- Phân tích chẩn đoán
- Phân tích xét nghiệm
- Phân tích ICU
- Báo cáo toàn bộ

```python
from data_analysis import AnalysisReporter

reporter = AnalysisReporter()
report = reporter.generate_full_report(
    patients_df, admissions_df, diagnosis_df, lab_df, icustays_df
)
reporter.print_report(report)
```

## 🚀 Cách Sử Dụng

### 1. Cài đặt Dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình Paths

Sửa `module3_config.py`:
```python
DATA_PATH = Path("D:/ProjectBigData/data/mimic-iv-3.1.zip")
OUTPUT_PATH = Path("D:/ProjectBigData/output/module3")
```

### 3. Chạy Pipeline

```bash
python main.py
```

### 4. Chạy Modules Riêng Lẻ

```python
from spark_session import SparkSessionManager
from data_loader import MIMICDataLoader

# Setup
manager = SparkSessionManager()
spark = manager.get_or_create_spark_session()

# Load data
loader = MIMICDataLoader(spark, "path/to/data.zip")
df = loader.load_csv_from_zip("admissions")
df.show()
```

## 📊 Pipeline Stages

### Stage 1: Data Loading
- Đọc file CSV.GZ từ ZIP
- Validation cơ bản
- Sampling dữ liệu (nếu cần)

### Stage 2: Data Cleaning
- Loại bỏ duplicates
- Xử lý missing values
- Chuẩn hóa tên cột

### Stage 3: Core Data Processing
- Xử lý Patients DataFrame
- Xử lý Admissions DataFrame
- Xây dựng demographic features

### Stage 4: Clinical Data Processing
- Xử lý Lab events
- Xử lý Diagnoses
- Xử lý Prescriptions

### Stage 5: ICU Data Processing
- Xử lý ICU stays
- Xử lý Chart events
- Xây dựng vital signs features

### Stage 6: Data Analysis
- Phân tích nhân khẩu học
- Thống kê lâm sàng
- Tạo báo cáo

### Stage 7: Save Results
- Lưu DataFrames dưới định dạng Parquet
- Lưu báo cáo JSON

## 🔧 Configuration

### Spark Configuration
```python
SPARK_CONFIG = {
    "appName": "MIMIC-IV-DataProcessing",
    "master": "local[*]",           # Sử dụng tất cả CPU
    "driver_memory": "4g",
    "executor_memory": "4g",
    "shuffle_partitions": "200",
}
```

### Processing Parameters
```python
PROCESSING_PARAMS = {
    "sample_ratio": 1.0,            # 1.0 = toàn bộ dữ liệu
    "outlier_std": 3,               # Z-score threshold
    "missing_threshold": 0.8,       # 80% missing = drop
    "date_range": {
        "start": "2008-01-01",
        "end": "2019-12-31"
    }
}
```

### Feature Parameters
```python
FEATURE_PARAMS = {
    "time_windows": [6, 12, 24, 48],  # hours
    "aggregation_funcs": ["mean", "std", "min", "max", "last"],
    "lab_tests_important": [
        "hematocrit", "hemoglobin", "platelets", "potassium", 
        "sodium", "glucose", "creatinine", "bun"
    ]
}
```

## 📈 Output Files

```
output/module3/
├── 01_core_data/
│   ├── patients.parquet
│   └── admissions.parquet
├── 02_clinical_data/
│   ├── diagnoses_icd.parquet
│   ├── labevents.parquet
│   └── prescriptions.parquet
├── 03_icu_data/
│   ├── icustays.parquet
│   ├── chartevents.parquet
│   └── procedureevents.parquet
├── 04_processed_data/
│   ├── patients.parquet
│   └── admissions.parquet
├── 05_features/
│   ├── patient_features.parquet
│   ├── admission_features.parquet
│   └── lab_features.parquet
├── 06_analysis/
│   └── analysis_report.json
└── module3_execution.log
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_data_loader.py -v

# Run with coverage
pytest --cov=. tests/
```

## 📝 Example Usage

### Ví dụ 1: Tải và Làm Sạch Dữ liệu

```python
from spark_session import SparkSessionManager
from data_loader import MIMICDataLoader, DataValidator
from data_processor import DataCleaner

# Setup
manager = SparkSessionManager()
spark = manager.get_or_create_spark_session()

# Load
loader = MIMICDataLoader(spark, "path/to/data.zip")
df = loader.load_csv_from_zip("labevents")

# Validate
validator = DataValidator(spark)
validation = validator.validate_dataframe(df, "labevents")
validator.print_validation_report(validation)

# Clean
cleaner = DataCleaner()
df_clean = cleaner.remove_duplicates(df, subset=["labevent_id"])
df_clean = cleaner.handle_missing_values(df_clean, strategy="mean")

print(f"Original rows: {df.count():,}")
print(f"Cleaned rows: {df_clean.count():,}")
```

### Ví dụ 2: Xây Dựng Features

```python
from feature_engineering import FeatureEngineer

engineer = FeatureEngineer()

# Patient features
patients_features = engineer.build_patient_features(patients_df)
patients_features.show()

# Lab features
lab_summary = engineer.lab_features.create_lab_summary(
    lab_df,
    important_tests=["hemoglobin", "glucose", "creatinine"]
)
lab_summary.show()

# Diagnosis features
diag_features = engineer.diag_features.create_diagnosis_count_features(diagnosis_df)
diag_features.show()
```

### Ví dụ 3: Phân Tích Dữ Liệu

```python
from data_analysis import AnalysisReporter

reporter = AnalysisReporter()

# Patient demographics
demo = reporter.patient_analyzer.analyze_patient_demographics(patients_df)
print(f"Total patients: {demo['total_patients']:,}")
print(f"Mean age: {demo['age_statistics']['mean']}")

# Admission patterns
adm_stats = reporter.patient_analyzer.analyze_admission_patterns(admissions_df)
print(f"Total admissions: {adm_stats['total_admissions']:,}")
print(f"Mortality rate: {adm_stats['in_hospital_mortality_rate']}%")

# Top diagnoses
top_dx = reporter.diagnosis_analyzer.analyze_top_diagnoses(diagnosis_df, top_n=10)
for i, dx in enumerate(top_dx, 1):
    print(f"{i}. {dx['icd_code']}: {dx['frequency']:,} cases")
```

## ⚙️ Performance Optimization

### Partitioning
```python
# Repartition for better performance
df_partitioned = df.repartition(200, "subject_id")
```

### Caching
```python
# Cache frequently used DataFrames
df.cache()
df.count()  # Trigger caching
```

### Broadcasting
```python
from pyspark.sql.functions import broadcast

# Broadcast small DataFrames
df_large.join(broadcast(df_small), on="id")
```

## 🐛 Debugging

### Enable DEBUG logging
```python
import logging
logging.getLogger("pyspark").setLevel(logging.DEBUG)
```

### Check Spark UI
```
http://localhost:4040
```

### Monitor Memory
```python
spark.sparkContext.status.executorBlocks()
```

## 📚 Key Classes & Methods

| Class | Method | Purpose |
|-------|--------|---------|
| `SparkSessionManager` | `get_or_create_spark_session()` | Khởi tạo Spark |
| `MIMICDataLoader` | `load_csv_from_zip()` | Đọc dữ liệu từ ZIP |
| `DataValidator` | `validate_dataframe()` | Xác thực dữ liệu |
| `DataCleaner` | `remove_duplicates()` | Loại bỏ duplicates |
| `DataTransformer` | `aggregate_by_time_window()` | Tổng hợp theo thời gian |
| `FeatureEngineer` | `build_patient_features()` | Xây dựng features |
| `AnalysisReporter` | `generate_full_report()` | Tạo báo cáo phân tích |

## 🔗 Dependencies

- **Apache PySpark** >= 3.3.0: Xử lý dữ liệu phân tán
- **Pandas**: Xử lý dữ liệu nhỏ
- **NumPy**: Tính toán số học
- **PyTest**: Unit testing

## 📞 Support & Troubleshooting

### OutOfMemory Error
```python
# Tăng bộ nhớ Spark
spark_builder.config("spark.driver.memory", "8g")
spark_builder.config("spark.executor.memory", "8g")
```

### Slow Joins
```python
# Repartition trước khi join
df1 = df1.repartition(200, "key")
df2 = df2.repartition(200, "key")
df_joined = df1.join(df2, on="key")
```

### Large File Loading
```python
# Sử dụng sampling
df = loader.load_csv_from_zip("file", sample_ratio=0.1)
```

## 📄 License & Attribution

MIMIC-IV dataset: https://physionet.org/content/mimiciv/

## 📌 Version History

- **v1.0** (2024): Initial release
  - Data loading & validation
  - Data cleaning & processing
  - Feature engineering
  - Basic analysis

## 🎯 Next Steps

1. ✅ Implement advanced feature engineering
2. ✅ Add predictive modeling
3. ✅ Create visualization dashboard
4. ✅ Add distributed computation optimization
5. ✅ Implement data quality monitoring
