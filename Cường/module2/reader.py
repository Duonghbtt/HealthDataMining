"""
reader.py — Đọc dữ liệu MIMIC-IV bằng PySpark.

Tại sao dùng PySpark thay pandas chunks?
  - Pandas chunks: đọc tuần tự, 1 core, không tận dụng được CPU đa nhân
  - PySpark: đọc song song nhiều partition, tận dụng đa nhân, lazy evaluation
  - Với MIMIC-IV 10GB: PySpark nhanh hơn ~3-5x ở bước ETL (groupBy, join, pivot)

Lưu ý quan trọng:
  PySpark KHÔNG đọc được file bên trong .zip trực tiếp.
  → Hàm extract_zip() giải nén file .csv.gz ra thư mục tạm trước.
  → Spark đọc thẳng file .csv.gz đã giải nén (Spark hỗ trợ gzip natively).
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import List, Optional, Tuple

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, StringType


# 1. Khởi tạo SparkSession

def create_spark(
    driver_memory:   str = "8g",
    executor_memory: str = "4g",
    cores:           str = "4",
    app_name:        str = "Module2_Clustering",
) -> SparkSession:
    """
    Tạo SparkSession local (chạy trên 1 máy).

    Tham số:
        driver_memory  : RAM cho driver  (nơi chứa kết quả collect())
        executor_memory: RAM cho executor (nơi tính toán)
        cores          : Số core song song ("4" = dùng 4 core, "*" = dùng hết)

    Giải thích local[4]:
        "local[4]" nghĩa là chạy Spark local với 4 thread song song.
        Khác với pandas chunks chạy tuần tự 1 thread.
    """
    import sys
    import os

    os.environ["PYSPARK_PYTHON"]        = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    spark = (
        SparkSession.builder
        .appName(app_name)
        .master(f"local[{cores}]")
        .config("spark.driver.memory",          driver_memory)
        .config("spark.executor.memory",         executor_memory)
        .config("spark.sql.shuffle.partitions",  "8")     
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")  
        .config("spark.ui.showConsoleProgress",  "false")  
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")   
    print(f"SparkSession OK  — version={spark.version}  master={spark.sparkContext.master}")
    return spark


# 2. Giải nén zip → thư mục tạm

def extract_zip(
    zip_path: Path,
    extract_to: Path,
    files_needed: Optional[List[str]] = None,
) -> Path:
    """
    Giải nén các file cần thiết từ zip vào extract_to/.

    Tại sao cần bước này?
        PySpark đọc file trực tiếp từ filesystem (HDFS/local path).
        Nó không có API để đọc file bên trong .zip.
        Pandas có thể dùng zipfile+BytesIO nhưng Spark thì không.

    Args:
        zip_path     : đường dẫn file .zip
        extract_to   : thư mục đích
        files_needed : list tên file cần giải nén (None = giải nén hết)
                       VD: ["patients.csv.gz", "diagnoses_icd.csv.gz"]

    Returns:
        hosp_dir : Path đến thư mục hosp/ bên trong extract_to
    """
    extract_to = Path(extract_to)
    extract_to.mkdir(parents=True, exist_ok=True)

    zf = zipfile.ZipFile(zip_path, "r")
    all_names = zf.namelist()

    # Tìm prefix (vd: "mimic-iv-3.1/hosp/")
    candidates = [n for n in all_names if n.endswith("hosp/patients.csv.gz")]
    if not candidates:
        raise FileNotFoundError(f"Không tìm thấy hosp/patients.csv.gz trong {zip_path}")
    prefix = candidates[0].replace("patients.csv.gz", "")
    print(f"ZIP prefix: '{prefix}'")

    # Chọn file cần giải nén
    if files_needed is None:
        to_extract = [n for n in all_names if n.startswith(prefix) and n.endswith(".csv.gz")]
    else:
        to_extract = [prefix + f for f in files_needed if (prefix + f) in all_names]

    # Giải nén từng file (bỏ qua nếu đã tồn tại)
    for inner_path in to_extract:
        fname     = Path(inner_path).name
        dest_path = extract_to / fname

        if dest_path.exists():
            print(f"  Đã có: {fname} → bỏ qua")
            continue

        print(f"  Giải nén: {fname} ...", end=" ", flush=True)
        with zf.open(inner_path) as src, open(dest_path, "wb") as dst:
            dst.write(src.read())
        size_mb = dest_path.stat().st_size / 1024**2
        print(f"{size_mb:.0f} MB")

    zf.close()
    print(f"Giải nén xong → {extract_to}/")
    return extract_to

# 3. Đọc từng bảng bằng Spark

def load_patients_spark(
    spark: SparkSession,
    hosp_dir: Path,
    limit: Optional[int] = None,
) -> DataFrame:
    """
    Đọc patients.csv.gz bằng Spark.

    So sánh với pandas:
        pandas: pd.read_csv(path, usecols=[...])  → load vào RAM ngay
        Spark : spark.read.csv(path)              → lazy, chỉ tính khi action (.show, .collect)

    MIMIC-IV dùng anchor_age (không có BIRTHDATE).
    """
    path = str(hosp_dir / "patients.csv.gz")

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
        .select(
            F.col("subject_id").cast(IntegerType()),
            F.col("anchor_age").cast(IntegerType()).alias("age"),
            F.col("gender").cast(StringType()),
        )
        .withColumn("gender_num", (F.col("gender") == "M").cast(IntegerType()))
        .drop("gender")
    )

    if limit is not None:
        df = df.limit(limit)

    return df


def load_diagnoses_spark(
    spark: SparkSession,
    hosp_dir: Path,
) -> DataFrame:
    """
    Đọc diagnoses_icd.csv.gz bằng Spark.

    Spark tự động chia file thành nhiều partition và đọc song song.
    Với file 6M+ dòng, Spark nhanh hơn pandas ~3x ở bước groupBy / pivot.

    Tạo cột icd_token = "ICD9:4019" hoặc "ICD10:I50.9"
    """
    path = str(hosp_dir / "diagnoses_icd.csv.gz")

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
        .select(
            F.col("subject_id").cast(IntegerType()),
            F.col("icd_code").cast(StringType()),
            F.col("icd_version").cast(IntegerType()),
        )
        .dropna(subset=["icd_code"])
        .withColumn(
            "icd_token",
            F.concat(F.lit("ICD"), F.col("icd_version").cast("string"),
                     F.lit(":"), F.trim(F.col("icd_code")))
        )
        .select("subject_id", "icd_token")
    )
    return df


def load_prescriptions_spark(
    spark: SparkSession,
    hosp_dir: Path,
) -> Tuple[DataFrame, str]:
    """
    Đọc prescriptions.csv.gz bằng Spark.
    Tự động chọn cột drug: 'formulary_drug_cd' nếu có, fallback 'drug'.

    Prescriptions là file lớn nhất (~592MB, 20M+ dòng).
    Spark đọc song song nhanh hơn pandas chunking đáng kể ở đây.

    Returns:
        df       : DataFrame (subject_id, drug_token)
        drug_col : tên cột drug đang dùng
    """
    path = str(hosp_dir / "prescriptions.csv.gz")

    # Đọc header để detect cột
    header_df = spark.read.option("header", "true").csv(path).limit(0)
    cols = set(header_df.columns)

    if "formulary_drug_cd" in cols:
        drug_col   = "formulary_drug_cd"
        token_prefix = "DRUGCD:"
    elif "drug" in cols:
        drug_col   = "drug"
        token_prefix = "DRUG:"
    else:
        raise ValueError("prescriptions.csv.gz thiếu cả 'formulary_drug_cd' và 'drug'.")

    df = (
        spark.read
        .option("header", "true")
        .option("inferSchema", "true")
        .csv(path)
        .select(
            F.col("subject_id").cast(IntegerType()),
            F.col(drug_col).cast(StringType()),
        )
        .dropna(subset=[drug_col])
        .withColumn(
            "drug_token",
            F.concat(F.lit(token_prefix), F.upper(F.trim(F.col(drug_col))))
        )
        .select("subject_id", "drug_token")
    )
    return df, drug_col

# 4. Tính top-N tokens bằng Spark

def compute_top_tokens_spark(
    df: DataFrame,
    token_col: str,
    valid_pids_df: DataFrame,
    top_n: int,
) -> List[str]:
    """
    Tính top-N tokens phổ biến nhất bằng Spark groupBy + count.

    Tại sao Spark tốt hơn ở đây?
        pandas: Counter() chạy 1 thread, duyệt 20M dòng tuần tự
        Spark : groupBy chạy song song trên nhiều partition, nhanh hơn ~4x

    Args:
        df           : DataFrame có (subject_id, token_col)
        token_col    : tên cột token ("icd_token" hoặc "drug_token")
        valid_pids_df: DataFrame (subject_id) chỉ giữ bệnh nhân hợp lệ
        top_n        : số lượng token muốn lấy

    Returns:
        list top-N token strings
    """
    top_tokens = (
        df
        .join(valid_pids_df.select("subject_id"), on="subject_id", how="inner")
        .groupBy(token_col)
        .count()
        .orderBy(F.desc("count"))
        .limit(top_n)
        .select(token_col)
        .rdd.flatMap(lambda row: row)
        .collect()
    )
    return top_tokens
