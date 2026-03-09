# ============================================
# MODULE 3 - FEATURE ENGINEERING
# ============================================

import logging
from typing import Dict, List, Optional
from pyspark.sql import DataFrame, Window
from pyspark.sql.functions import (
    col, when, count, sum, avg, min, max, stddev, 
    first, last, datediff, lag, row_number,
    collect_list, concat_ws, coalesce, isnull
)
from pyspark.sql.types import DoubleType

logger = logging.getLogger(__name__)


class DemographicFeatures:
    """
    Xây dựng các features liên quan đến nhân khẩu học
    """
    
    def __init__(self):
        self.logger = logger
    
    def create_age_features(self, df: DataFrame, anchor_age_col: str = "anchor_age") -> DataFrame:
        """
        Tạo age-based features
        
        Args:
            df (DataFrame): Patient DataFrame
            anchor_age_col (str): Cột chứa tuổi
        
        Returns:
            DataFrame: DataFrame với age features
        """
        try:
            df = df.withColumn(
                "age_group",
                when(col(anchor_age_col) < 18, "child")
                .when(col(anchor_age_col) < 35, "young_adult")
                .when(col(anchor_age_col) < 50, "middle_age")
                .when(col(anchor_age_col) < 65, "senior")
                .otherwise("elderly")
            ).withColumn(
                "is_elderly", 
                when(col(anchor_age_col) >= 65, 1).otherwise(0)
            ).withColumn(
                "age_squared", 
                col(anchor_age_col) ** 2
            )
            
            self.logger.info("✓ Created age features")
            return df
        
        except Exception as e:
            self.logger.error(f"✗ Error creating age features: {str(e)}")
            raise
    
    def create_gender_features(self, df: DataFrame, gender_col: str = "gender") -> DataFrame:
        """
        Tạo gender-based features
        
        Args:
            df (DataFrame): Patient DataFrame
            gender_col (str): Cột chứa giới tính
        
        Returns:
            DataFrame: DataFrame với gender features
        """
        try:
            df = df.withColumn(
                "gender_encoded",
                when(col(gender_col) == "M", 1).otherwise(0)
            )
            
            self.logger.info("✓ Created gender features")
            return df
        
        except Exception as e:
            self.logger.error(f"✗ Error creating gender features: {str(e)}")
            raise


class LabFeatures:
    """
    Xây dựng các features từ xét nghiệm (Lab Events)
    """
    
    def __init__(self):
        self.logger = logger
    
    def create_lab_summary(self, lab_df: DataFrame, 
                          important_tests: Optional[List[str]] = None) -> DataFrame:
        """
        Tạo summary features từ lab results
        
        Args:
            lab_df (DataFrame): Lab events DataFrame
            important_tests (List[str]): Danh sách các xét nghiệm quan trọng
        
        Returns:
            DataFrame: Lab summary features
        """
        try:
            if important_tests:
                lab_df = lab_df.filter(col("test_name").isin(important_tests))
            
            # Tổng hợp theo hadm_id
            lab_features = lab_df.groupBy("subject_id", "hadm_id").agg(
                count("labevent_id").alias("lab_test_count"),
                countDistinct("test_name").alias("unique_test_count"),
                avg("valuenum").alias("lab_valuenum_mean"),
                stddev("valuenum").alias("lab_valuenum_stddev"),
                min("valuenum").alias("lab_valuenum_min"),
                max("valuenum").alias("lab_valuenum_max")
            )
            
            self.logger.info("✓ Created lab summary features")
            return lab_features
        
        except Exception as e:
            self.logger.error(f"✗ Error creating lab features: {str(e)}")
            raise
    
    def create_lab_pivot_features(self, lab_df: DataFrame,
                                 important_tests: List[str]) -> DataFrame:
        """
        Tạo pivot features cho các xét nghiệm quan trọng
        
        Args:
            lab_df (DataFrame): Lab events DataFrame
            important_tests (List[str]): Danh sách các xét nghiệm quan trọng
        
        Returns:
            DataFrame: Pivoted lab features
        """
        try:
            # Lấy giá trị gần nhất của mỗi test
            window_spec = Window.partitionBy(
                "subject_id", "hadm_id", "test_name"
            ).orderBy(col("charttime").desc())
            
            lab_latest = lab_df.filter(
                col("test_name").isin(important_tests)
            ).withColumn(
                "row_num", row_number().over(window_spec)
            ).filter(
                col("row_num") == 1
            ).drop("row_num")
            
            # Pivot
            lab_pivot = lab_latest.groupBy(
                "subject_id", "hadm_id"
            ).pivot("test_name").agg(
                first("valuenum")
            )
            
            self.logger.info("✓ Created pivoted lab features")
            return lab_pivot
        
        except Exception as e:
            self.logger.error(f"✗ Error creating pivoted lab features: {str(e)}")
            raise


class DiagnosisFeatures:
    """
    Xây dựng các features từ chẩn đoán
    """
    
    def __init__(self):
        self.logger = logger
    
    def create_diagnosis_count_features(self, diagnosis_df: DataFrame) -> DataFrame:
        """
        Tạo count features từ chẩn đoán
        
        Args:
            diagnosis_df (DataFrame): Diagnoses ICD DataFrame
        
        Returns:
            DataFrame: Diagnosis count features
        """
        try:
            diag_features = diagnosis_df.groupBy("subject_id", "hadm_id").agg(
                count("icd_code").alias("diagnosis_count"),
                countDistinct("icd_code").alias("unique_diagnosis_count"),
                max("seq_num").alias("max_diagnosis_seq")
            )
            
            self.logger.info("✓ Created diagnosis count features")
            return diag_features
        
        except Exception as e:
            self.logger.error(f"✗ Error creating diagnosis features: {str(e)}")
            raise
    
    def create_diagnosis_categories(self, diagnosis_df: DataFrame) -> DataFrame:
        """
        Tạo categorical features từ chẩn đoán (presence of specific diagnoses)
        
        Args:
            diagnosis_df (DataFrame): Diagnoses ICD DataFrame
        
        Returns:
            DataFrame: Diagnosis category features
        """
        try:
            # Định nghĩa các chẩn đoán quan trọng
            important_diagnoses = {
                "sepsis": "^A40|^A41|^R65",
                "acute_kidney_injury": "^N17",
                "pneumonia": "^J1[0-9]|^J2[0-9]",
                "ami": "^I2[1-3]",
                "ards": "^J80"
            }
            
            diag_cat = diagnosis_df.groupBy("subject_id", "hadm_id").agg(
                countDistinct("icd_code").alias("num_diagnoses")
            )
            
            # Thêm features cho từng chẩn đoán quan trọng
            for diag_name, icd_pattern in important_diagnoses.items():
                # Tạo feature binary
                diag_df_filtered = diagnosis_df.filter(
                    col("icd_code").rlike(icd_pattern)
                ).select("subject_id", "hadm_id").distinct()
                
                diag_df_filtered = diag_df_filtered.withColumn(
                    f"has_{diag_name}", col("hadm_id").isNotNull().cast("integer")
                )
                
                diag_cat = diag_cat.join(
                    diag_df_filtered.select("subject_id", "hadm_id", f"has_{diag_name}"),
                    on=["subject_id", "hadm_id"],
                    how="left"
                ).fillna(0, subset=[f"has_{diag_name}"])
            
            self.logger.info("✓ Created diagnosis category features")
            return diag_cat
        
        except Exception as e:
            self.logger.error(f"✗ Error creating diagnosis categories: {str(e)}")
            raise


class VitalSignsFeatures:
    """
    Xây dựng các features từ dấu hiệu sinh tồn (Chart Events)
    """
    
    def __init__(self):
        self.logger = logger
    
    def create_vital_signs_summary(self, chartevents_df: DataFrame,
                                   vital_items: Optional[List[int]] = None) -> DataFrame:
        """
        Tạo summary features từ dấu hiệu sinh tồn
        
        Args:
            chartevents_df (DataFrame): Chart events DataFrame
            vital_items (List[int]): Danh sách itemid của các dấu hiệu quan trọng
        
        Returns:
            DataFrame: Vital signs summary features
        """
        try:
            if vital_items:
                chartevents_df = chartevents_df.filter(
                    col("itemid").isin(vital_items)
                )
            
            vital_features = chartevents_df.groupBy(
                "subject_id", "hadm_id", "stay_id"
            ).agg(
                avg("valuenum").alias("vital_valuenum_mean"),
                stddev("valuenum").alias("vital_valuenum_stddev"),
                min("valuenum").alias("vital_valuenum_min"),
                max("valuenum").alias("vital_valuenum_max"),
                count("value").alias("vital_measurements_count")
            )
            
            self.logger.info("✓ Created vital signs summary features")
            return vital_features
        
        except Exception as e:
            self.logger.error(f"✗ Error creating vital signs features: {str(e)}")
            raise


class AdmissionFeatures:
    """
    Xây dựng các features từ thông tin nhập viện
    """
    
    def __init__(self):
        self.logger = logger
    
    def create_los_features(self, admissions_df: DataFrame) -> DataFrame:
        """
        Tạo features từ Length of Stay (LOS)
        
        Args:
            admissions_df (DataFrame): Admissions DataFrame
        
        Returns:
            DataFrame: LOS features
        """
        try:
            los_features = admissions_df.withColumn(
                "los_days",
                datediff(col("dischtime"), col("admittime"))
            ).withColumn(
                "los_hours",
                (datediff(col("dischtime"), col("admittime")) * 24).cast(DoubleType())
            ).withColumn(
                "is_long_stay",
                when(col("los_days") > 7, 1).otherwise(0)
            ).withColumn(
                "is_short_stay",
                when(col("los_days") < 1, 1).otherwise(0)
            )
            
            self.logger.info("✓ Created LOS features")
            return los_features
        
        except Exception as e:
            self.logger.error(f"✗ Error creating LOS features: {str(e)}")
            raise
    
    def create_admission_type_features(self, admissions_df: DataFrame) -> DataFrame:
        """
        Tạo features từ loại nhập viện
        
        Args:
            admissions_df (DataFrame): Admissions DataFrame
        
        Returns:
            DataFrame: Admission type features
        """
        try:
            # One-hot encoding for admission type
            admission_types = admissions_df.select("admission_type").distinct().rdd.flatMap(
                lambda x: x
            ).collect()
            
            result_df = admissions_df
            for adm_type in admission_types:
                col_name = f"admission_type_{adm_type.lower().replace(' ', '_')}"
                result_df = result_df.withColumn(
                    col_name,
                    when(col("admission_type") == adm_type, 1).otherwise(0)
                )
            
            self.logger.info("✓ Created admission type features")
            return result_df
        
        except Exception as e:
            self.logger.error(f"✗ Error creating admission type features: {str(e)}")
            raise
    
    def create_mortality_features(self, admissions_df: DataFrame) -> DataFrame:
        """
        Tạo mortality-related features
        
        Args:
            admissions_df (DataFrame): Admissions DataFrame
        
        Returns:
            DataFrame: Mortality features
        """
        try:
            mortality_features = admissions_df.withColumn(
                "in_hospital_mortality",
                when(col("hospital_expire_flag") == 1, 1).otherwise(0)
            ).withColumn(
                "deceased",
                when(col("deathtime").isNotNull(), 1).otherwise(0)
            )
            
            self.logger.info("✓ Created mortality features")
            return mortality_features
        
        except Exception as e:
            self.logger.error(f"✗ Error creating mortality features: {str(e)}")
            raise


class FeatureEngineer:
    """
    Master class để xây dựng toàn bộ features
    """
    
    def __init__(self):
        self.logger = logger
        self.demo_features = DemographicFeatures()
        self.lab_features = LabFeatures()
        self.diag_features = DiagnosisFeatures()
        self.vital_features = VitalSignsFeatures()
        self.adm_features = AdmissionFeatures()
    
    def build_patient_features(self, patients_df: DataFrame) -> DataFrame:
        """
        Xây dựng patient-level features
        
        Args:
            patients_df (DataFrame): Patients DataFrame
        
        Returns:
            DataFrame: Patient features
        """
        try:
            df = self.demo_features.create_age_features(patients_df)
            df = self.demo_features.create_gender_features(df)
            
            self.logger.info("✓ Built patient features")
            return df
        
        except Exception as e:
            self.logger.error(f"✗ Error building patient features: {str(e)}")
            raise
    
    def build_admission_features(self, admissions_df: DataFrame) -> DataFrame:
        """
        Xây dựng admission-level features
        
        Args:
            admissions_df (DataFrame): Admissions DataFrame
        
        Returns:
            DataFrame: Admission features
        """
        try:
            df = self.adm_features.create_los_features(admissions_df)
            df = self.adm_features.create_admission_type_features(df)
            df = self.adm_features.create_mortality_features(df)
            
            self.logger.info("✓ Built admission features")
            return df
        
        except Exception as e:
            self.logger.error(f"✗ Error building admission features: {str(e)}")
            raise


if __name__ == "__main__":
    print("Feature engineering modules loaded successfully")
