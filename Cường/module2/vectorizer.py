"""
vectorizer.py — Feature engineering dùng SPARK .

Pipeline:
    patients_sdf + icd_sdf + drug_sdf   (Spark DataFrames)
        ↓  Spark: filter / pivot / join
        feature_sdf  (211k × ~102 cột)
        ↓  .toPandas()  (CHỈ 1 LẦN, sau aggregate)
        feature_df  (pandas, nhỏ gọn)
        ↓  StandardScalerScratch (numpy)
        X_scaled
        ↓  PCAScratch (numpy SVD)
        X_pca

"""
from __future__ import annotations

import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ─── Spark imports (lazy – chỉ fail khi dùng, không fail khi import) ───
try:
    from pyspark.sql import SparkSession, DataFrame as SDF
    import pyspark.sql.functions as F
    from pyspark.sql import types as T
    _SPARK_AVAILABLE = True
except ImportError:
    _SPARK_AVAILABLE = False


#  Scratch: StandardScaler
class StandardScalerScratch:
    """Z-score normalization: z = (x - mean) / std  (std clip ≥ 1e-8)."""

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_:  Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "StandardScalerScratch":
        self.mean_ = X.mean(axis=0)
        self.std_  = X.std(axis=0)
        self.std_  = np.where(self.std_ < 1e-8, 1.0, self.std_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) / self.std_).astype("float32")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        return (X * self.std_ + self.mean_).astype("float32")

#  Scratch: PCA (SVD-based)
class PCAScratch:
    """PCA bằng numpy SVD.  Tương đương sklearn.decomposition.PCA."""

    def __init__(self, n_components: int = 20, random_state: int = 42):
        self.n_components = n_components
        self.random_state = random_state
        self.components_: Optional[np.ndarray] = None
        self.explained_variance_ratio_: Optional[np.ndarray] = None
        self.mean_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> "PCAScratch":
        np.random.seed(self.random_state)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # Full SVD; với n << p thì dùng economy
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        self.components_ = Vt[: self.n_components]
        var_all = (s ** 2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = var_all[: self.n_components] / var_all.sum()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return ((X - self.mean_) @ self.components_.T).astype("float32")

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)


#  Helper: sanitize column name

def _safe_col(name: str) -> str:
    """Thay ký tự đặc biệt (. / khoảng trắng) → _ để dùng làm tên cột Spark."""
    return re.sub(r"[.\s/]", "_", name)


#  PatientVectorizer — SPARK 

class PatientVectorizer:
    """
    Toàn bộ feature engineering chạy trên Spark.

    Cách dùng:
        vec = PatientVectorizer(pca_dims=20)
        vec.fit_transform(
            spark          = spark,
            patients_sdf   = patients_sdf,   # subject_id, age, gender_num
            icd_sdf        = icd_sdf,        # subject_id, icd_token
            drug_sdf       = drug_sdf,       # subject_id, drug_token
            top_icd_codes  = top_icd_codes,
            top_drug_codes = top_drug_codes,
            limit_patients = None,
        )

    Sau fit:
        vec.feature_df   → pandas DataFrame (subject_id + tất cả features)
        vec.X_pca        → numpy array (dùng K-means / Euclidean)
        vec.X_scaled     → numpy array (dùng Cosine)
        vec.X_binary     → numpy array (dùng Jaccard)
        vec.patient_ids  → list subject_id tương ứng
        vec.icd_cols     → tên cột ICD
        vec.drug_cols    → tên cột Drug
    """

    def __init__(self, pca_dims: int = 20, random_state: int = 42):
        self.pca_dims     = pca_dims
        self.random_state = random_state

        self.scaler = StandardScalerScratch()
        self.pca    = PCAScratch(n_components=pca_dims, random_state=random_state)

        # Sẽ được set sau fit
        self.feature_df:  pd.DataFrame       = None
        self.X_scaled:    np.ndarray          = None
        self.X_pca:       np.ndarray          = None
        self.patient_ids: List[int]           = []
        self.icd_cols:    List[str]           = []
        self.drug_cols:   List[str]           = []
        self.binary_cols: List[str]           = []

    #  MAIN: fit_transform
    def fit_transform(
        self,
        spark,
        patients_sdf,
        icd_sdf,
        drug_sdf,
        top_icd_codes:  List[str],
        top_drug_codes: List[str],
        limit_patients: Optional[int] = None,
    ) -> "PatientVectorizer":
        """
        Chạy toàn bộ pipeline trên Spark rồi collect 1 lần về pandas.
        """

        #  Sanitize code names 
        safe_icd  = {c: _safe_col(c) for c in top_icd_codes}   
        safe_drug = {c: _safe_col(c) for c in top_drug_codes}

        # Tên cột cuối cùng (có prefix ICD_ / DRUG_)
        icd_col_names  = [f"ICD_{v}"  for v in safe_icd.values()]
        drug_col_names = [f"DRUG_{v}" for v in safe_drug.values()]

        print("=" * 55)
        print(f"[1/5] Chuẩn bị dữ liệu Spark...")
        print(f"      top ICD codes  : {len(top_icd_codes)}")
        print(f"      top Drug codes : {len(top_drug_codes)}")

        #  1. Lọc patients (optional limit) 
        base_sdf = patients_sdf.select("subject_id", "age", "gender_num")
        if limit_patients is not None:
            base_sdf = base_sdf.filter(F.col("subject_id") < limit_patients)
        base_sdf = base_sdf.cache()

        #  2. Pivot ICD 
        print("[2/5] Pivot ICD (Spark)...")
        icd_safe_sdf = (
            icd_sdf
            .join(base_sdf.select("subject_id"), on="subject_id", how="inner")
            .withColumn(
                "icd_safe",
                # sanitize ký tự đặc biệt trong giá trị token
                F.regexp_replace(F.col("icd_token"), r"[./\s]", "_")
            )
            .filter(F.col("icd_safe").isin(list(safe_icd.values())))
        )

        # groupBy + pivot = multi-hot
        icd_pivot = (
            icd_safe_sdf
            .groupBy("subject_id")
            .pivot("icd_safe", list(safe_icd.values()))
            .agg(F.lit(1))
        )
        # rename: code → ICD_code
        for v in safe_icd.values():
            if v in icd_pivot.columns:
                icd_pivot = icd_pivot.withColumnRenamed(v, f"ICD_{v}")

        #  3. Pivot Drug
        print("[3/5] Pivot Drug (Spark)...")
        drug_safe_sdf = (
            drug_sdf
            .join(base_sdf.select("subject_id"), on="subject_id", how="inner")
            .withColumn(
                "drug_safe",
                F.regexp_replace(F.col("drug_token"), r"[./\s]", "_")
            )
            .filter(F.col("drug_safe").isin(list(safe_drug.values())))
        )

        drug_pivot = (
            drug_safe_sdf
            .groupBy("subject_id")
            .pivot("drug_safe", list(safe_drug.values()))
            .agg(F.lit(1))
        )
        for v in safe_drug.values():
            if v in drug_pivot.columns:
                drug_pivot = drug_pivot.withColumnRenamed(v, f"DRUG_{v}")

        # 4. Join tất cả
        print("[4/5] Join base + ICD + Drug (Spark)...")
        feature_sdf = (
            base_sdf
            .join(icd_pivot,  on="subject_id", how="left")
            .join(drug_pivot, on="subject_id", how="left")
        )

        all_feature_cols = icd_col_names + drug_col_names
        existing_cols = set(feature_sdf.columns)
        fill_dict = {c: 0 for c in all_feature_cols if c in existing_cols}
        feature_sdf = feature_sdf.fillna(fill_dict)

        for c in all_feature_cols:
            if c not in existing_cols:
                feature_sdf = feature_sdf.withColumn(c, F.lit(0))

        # Cast tất cả feature cols về int
        for c in all_feature_cols:
            feature_sdf = feature_sdf.withColumn(c, F.col(f"`{c}`").cast(T.IntegerType()))

        # Lọc bệnh nhân có ít nhất 1 ICD hoặc 1 Drug (dùng backtick cho tên có :)
        icd_sum  = sum(F.col(f"`{c}`") for c in icd_col_names)
        drug_sum = sum(F.col(f"`{c}`") for c in drug_col_names)
        feature_sdf = feature_sdf.filter((icd_sum + drug_sum) > 0)

        # 5. Collect về pandas ( sau aggregate)
        print("[5/5] Collect về pandas (sau aggregate)...")
        ordered_cols = ["subject_id", "age", "gender_num"] + all_feature_cols
        self.feature_df = (
            feature_sdf
            .select(*[f"`{c}`" if c in all_feature_cols else c
                      for c in ordered_cols])
            .toPandas()
        )

        # 6. Lưu metadata
        self.icd_cols    = icd_col_names
        self.drug_cols   = drug_col_names
        self.binary_cols = icd_col_names + drug_col_names
        self.patient_ids = self.feature_df["subject_id"].tolist()

        print(f"\nFeature matrix  : {self.feature_df.shape}")
        print(f"  age + gender  : 2")
        print(f"  ICD features  : {len(self.icd_cols)}")
        print(f"  Drug features : {len(self.drug_cols)}")

        # 7. StandardScaler (scratch) 
        X = self.feature_df[["age", "gender_num"] + self.binary_cols].values.astype("float32")
        self.X_scaled = self.scaler.fit_transform(X)
        print(f"X_scaled shape  : {self.X_scaled.shape}")

        # 8. PCA (scratch) 
        actual_dims = min(self.pca_dims, X.shape[1], X.shape[0])
        self.pca = PCAScratch(n_components=actual_dims, random_state=self.random_state)
        self.X_pca = self.pca.fit_transform(self.X_scaled)
        print(f"X_pca shape     : {self.X_pca.shape}")
        print(f"PCA variance    : {self.pca.explained_variance_ratio_.sum()*100:.1f}%")
        print("=" * 55)

        base_sdf.unpersist()
        return self

    #  Transform bệnh nhân mới (không cần Spark)
    def transform_new_patient(
        self,
        age: int,
        gender: str,
        icd_tokens: List[str],
        drug_tokens: List[str],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Biến 1 bệnh nhân mới thành vector.
        Returns: (vec_scaled, vec_pca)
        """
        gender_num = 1 if gender.upper() == "M" else 0
        row = {"age": float(age), "gender_num": float(gender_num)}

        # Sanitize input tokens để khớp với tên cột đã safe
        icd_safe_set  = {_safe_col(t) for t in icd_tokens}
        drug_safe_set = {_safe_col(t) for t in drug_tokens}

        for col in self.icd_cols:
            code = col.replace("ICD_", "", 1)
            row[col] = 1.0 if code in icd_safe_set else 0.0
        for col in self.drug_cols:
            code = col.replace("DRUG_", "", 1)
            row[col] = 1.0 if code in drug_safe_set else 0.0

        all_cols = ["age", "gender_num"] + self.icd_cols + self.drug_cols
        vec = pd.DataFrame([row])[all_cols].fillna(0).values.astype("float32")
        vec_scaled = self.scaler.transform(vec).astype("float32")
        vec_pca    = self.pca.transform(vec_scaled).astype("float32")
        return vec_scaled, vec_pca

    #  Properties tiện lợi
    @property
    def X_binary(self) -> np.ndarray:
        """Binary features (ICD + drug) — dùng cho Jaccard."""
        return self.feature_df[self.binary_cols].values.astype("float32")