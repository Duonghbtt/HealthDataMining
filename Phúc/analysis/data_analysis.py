# ============================================
# MODULE 3 - DATA ANALYSIS & REPORTING
# ============================================

import logging
from typing import Dict, List, Tuple
from pyspark.sql import DataFrame
from pyspark.sql.functions import (
    col, count, sum, avg, stddev, min, max, 
    percentile_approx, when, datediff, year, month,
    row_number, dense_rank
)
from pyspark.sql import Window
from datetime import datetime

logger = logging.getLogger(__name__)


class PatientAnalyzer:
    """
    Phân tích dữ liệu bệnh nhân
    """
    
    def __init__(self):
        self.logger = logger
    
    def analyze_patient_demographics(self, patients_df: DataFrame) -> Dict:
        """
        Phân tích nhân khẩu học bệnh nhân
        
        Args:
            patients_df (DataFrame): Patients DataFrame
        
        Returns:
            Dict: Thống kê nhân khẩu học
        """
        try:
            total_patients = patients_df.count()
            
            # Phân tích theo giới tính
            gender_dist = patients_df.groupBy("gender").agg(
                count("subject_id").alias("count")
            ).collect()
            
            # Phân tích theo độ tuổi
            age_stats = patients_df.agg(
                avg("anchor_age").alias("avg_age"),
                min("anchor_age").alias("min_age"),
                max("anchor_age").alias("max_age"),
                percentile_approx("anchor_age", 0.5).alias("median_age")
            ).collect()[0]
            
            results = {
                "total_patients": total_patients,
                "gender_distribution": {row.gender: row['count'] for row in gender_dist},
                "age_statistics": {
                    "mean": round(age_stats.avg_age, 2),
                    "min": age_stats.min_age,
                    "max": age_stats.max_age,
                    "median": age_stats.median_age
                }
            }
            
            self.logger.info(f"✓ Analyzed demographics for {total_patients:,} patients")
            return results
        
        except Exception as e:
            self.logger.error(f"✗ Error analyzing demographics: {str(e)}")
            raise
    
    def analyze_admission_patterns(self, admissions_df: DataFrame) -> Dict:
        """
        Phân tích mẫu nhập viện
        
        Args:
            admissions_df (DataFrame): Admissions DataFrame
        
        Returns:
            Dict: Thống kê nhập viện
        """
        try:
            total_admissions = admissions_df.count()
            
            # Phân tích theo loại nhập viện
            admission_type_dist = admissions_df.groupBy("admission_type").agg(
                count("hadm_id").alias("count")
            ).collect()
            
            # Phân tích LOS
            los_stats = admissions_df.withColumn(
                "los_days",
                datediff(col("dischtime"), col("admittime"))
            ).agg(
                avg("los_days").alias("avg_los"),
                percentile_approx("los_days", 0.5).alias("median_los"),
                min("los_days").alias("min_los"),
                max("los_days").alias("max_los")
            ).collect()[0]
            
            # Tỷ lệ tử vong
            mortality_rate = admissions_df.agg(
                sum("hospital_expire_flag").alias("deaths")
            ).collect()[0]["deaths"] / total_admissions * 100
            
            # Phân tích theo năm
            admissions_by_year = admissions_df.withColumn(
                "year", year(col("admittime"))
            ).groupBy("year").agg(
                count("hadm_id").alias("count")
            ).orderBy("year").collect()
            
            results = {
                "total_admissions": total_admissions,
                "admission_type_distribution": {
                    row.admission_type: row['count'] for row in admission_type_dist
                },
                "length_of_stay": {
                    "mean_days": round(los_stats.avg_los, 2),
                    "median_days": los_stats.median_los,
                    "min_days": los_stats.min_los,
                    "max_days": los_stats.max_los
                },
                "in_hospital_mortality_rate": round(mortality_rate, 2),
                "admissions_by_year": {
                    row.year: row['count'] for row in admissions_by_year
                }
            }
            
            self.logger.info(f"✓ Analyzed {total_admissions:,} admissions")
            return results
        
        except Exception as e:
            self.logger.error(f"✗ Error analyzing admissions: {str(e)}")
            raise


class DiagnosisAnalyzer:
    """
    Phân tích chẩn đoán
    """
    
    def __init__(self):
        self.logger = logger
    
    def analyze_top_diagnoses(self, diagnosis_df: DataFrame, top_n: int = 20) -> List[Dict]:
        """
        Phân tích top N chẩn đoán phổ biến nhất
        
        Args:
            diagnosis_df (DataFrame): Diagnoses ICD DataFrame
            top_n (int): Số chẩn đoán top
        
        Returns:
            List[Dict]: Danh sách top diagnoses
        """
        try:
            top_diagnoses = diagnosis_df.groupBy("icd_code", "icd_version").agg(
                count("hadm_id").alias("frequency")
            ).orderBy(col("frequency").desc()).limit(top_n).collect()
            
            results = [
                {
                    "icd_code": row.icd_code,
                    "icd_version": row.icd_version,
                    "frequency": row.frequency
                }
                for row in top_diagnoses
            ]
            
            self.logger.info(f"✓ Analyzed top {len(results)} diagnoses")
            return results
        
        except Exception as e:
            self.logger.error(f"✗ Error analyzing diagnoses: {str(e)}")
            raise
    
    def analyze_comorbidities(self, diagnosis_df: DataFrame) -> Dict:
        """
        Phân tích các bệnh kèm theo (comorbidities)
        
        Args:
            diagnosis_df (DataFrame): Diagnoses ICD DataFrame
        
        Returns:
            Dict: Thống kê comorbidities
        """
        try:
            # Số bệnh trung bình trên mỗi admission
            comorbidity_stats = diagnosis_df.groupBy("hadm_id").agg(
                count("icd_code").alias("diagnosis_count")
            ).agg(
                avg("diagnosis_count").alias("avg_diagnoses"),
                percentile_approx("diagnosis_count", 0.5).alias("median_diagnoses"),
                min("diagnosis_count").alias("min_diagnoses"),
                max("diagnosis_count").alias("max_diagnoses")
            ).collect()[0]
            
            results = {
                "average_diagnoses_per_admission": round(comorbidity_stats.avg_diagnoses, 2),
                "median_diagnoses_per_admission": comorbidity_stats.median_diagnoses,
                "min_diagnoses": comorbidity_stats.min_diagnoses,
                "max_diagnoses": comorbidity_stats.max_diagnoses
            }
            
            self.logger.info("✓ Analyzed comorbidities")
            return results
        
        except Exception as e:
            self.logger.error(f"✗ Error analyzing comorbidities: {str(e)}")
            raise


class LabAnalyzer:
    """
    Phân tích xét nghiệm
    """
    
    def __init__(self):
        self.logger = logger
    
    def analyze_top_lab_tests(self, lab_df: DataFrame, top_n: int = 20) -> List[Dict]:
        """
        Phân tích top N xét nghiệm phổ biến nhất
        
        Args:
            lab_df (DataFrame): Lab events DataFrame
            top_n (int): Số xét nghiệm top
        
        Returns:
            List[Dict]: Danh sách top lab tests
        """
        try:
            top_tests = lab_df.groupBy("test_name").agg(
                count("labevent_id").alias("frequency"),
                sum(when(col("flag").isNotNull(), 1).otherwise(0)).alias("abnormal_count")
            ).orderBy(col("frequency").desc()).limit(top_n).collect()
            
            results = [
                {
                    "test_name": row.test_name,
                    "frequency": row.frequency,
                    "abnormal_count": row.abnormal_count,
                    "abnormal_percentage": round(
                        (row.abnormal_count / row.frequency * 100) if row.frequency > 0 else 0, 2
                    )
                }
                for row in top_tests
            ]
            
            self.logger.info(f"✓ Analyzed top {len(results)} lab tests")
            return results
        
        except Exception as e:
            self.logger.error(f"✗ Error analyzing lab tests: {str(e)}")
            raise
    
    def analyze_lab_value_statistics(self, lab_df: DataFrame, 
                                    test_names: List[str]) -> Dict:
        """
        Phân tích thống kê giá trị xét nghiệm
        
        Args:
            lab_df (DataFrame): Lab events DataFrame
            test_names (List[str]): Danh sách các xét nghiệm cần phân tích
        
        Returns:
            Dict: Thống kê giá trị
        """
        try:
            results = {}
            
            for test_name in test_names:
                test_data = lab_df.filter(col("test_name") == test_name)
                
                stats = test_data.agg(
                    count("labevent_id").alias("count"),
                    avg("valuenum").alias("mean"),
                    stddev("valuenum").alias("stddev"),
                    min("valuenum").alias("min"),
                    max("valuenum").alias("max"),
                    percentile_approx("valuenum", 0.25).alias("q1"),
                    percentile_approx("valuenum", 0.5).alias("median"),
                    percentile_approx("valuenum", 0.75).alias("q3")
                ).collect()[0]
                
                results[test_name] = {
                    "count": stats['count'],
                    "mean": round(stats['mean'], 4) if stats['mean'] else None,
                    "stddev": round(stats['stddev'], 4) if stats['stddev'] else None,
                    "min": round(stats['min'], 4) if stats['min'] else None,
                    "max": round(stats['max'], 4) if stats['max'] else None,
                    "median": round(stats['median'], 4) if stats['median'] else None,
                    "q1": round(stats['q1'], 4) if stats['q1'] else None,
                    "q3": round(stats['q3'], 4) if stats['q3'] else None
                }
            
            self.logger.info(f"✓ Analyzed statistics for {len(results)} lab tests")
            return results
        
        except Exception as e:
            self.logger.error(f"✗ Error analyzing lab values: {str(e)}")
            raise


class ICUAnalyzer:
    """
    Phân tích dữ liệu ICU
    """
    
    def __init__(self):
        self.logger = logger
    
    def analyze_icu_stays(self, icustays_df: DataFrame) -> Dict:
        """
        Phân tích ICU stays
        
        Args:
            icustays_df (DataFrame): ICU stays DataFrame
        
        Returns:
            Dict: Thống kê ICU
        """
        try:
            total_stays = icustays_df.count()
            
            # Phân tích theo ICU level
            icu_level_dist = icustays_df.groupBy("icu_level").agg(
                count("stay_id").alias("count")
            ).collect()
            
            # Phân tích LOS trong ICU
            los_stats = icustays_df.agg(
                avg("los").alias("avg_los"),
                percentile_approx("los", 0.5).alias("median_los"),
                min("los").alias("min_los"),
                max("los").alias("max_los")
            ).collect()[0]
            
            results = {
                "total_stays": total_stays,
                "icu_level_distribution": {
                    row.icu_level: row['count'] for row in icu_level_dist
                },
                "length_of_icu_stay": {
                    "mean_days": round(los_stats.avg_los, 2),
                    "median_days": round(los_stats.median_los, 2),
                    "min_days": round(los_stats.min_los, 2),
                    "max_days": round(los_stats.max_los, 2)
                }
            }
            
            self.logger.info(f"✓ Analyzed {total_stays:,} ICU stays")
            return results
        
        except Exception as e:
            self.logger.error(f"✗ Error analyzing ICU stays: {str(e)}")
            raise


class AnalysisReporter:
    """
    Tạo báo cáo phân tích
    """
    
    def __init__(self):
        self.logger = logger
        self.patient_analyzer = PatientAnalyzer()
        self.diagnosis_analyzer = DiagnosisAnalyzer()
        self.lab_analyzer = LabAnalyzer()
        self.icu_analyzer = ICUAnalyzer()
    
    def generate_full_report(self, 
                            patients_df: DataFrame,
                            admissions_df: DataFrame,
                            diagnosis_df: DataFrame,
                            lab_df: DataFrame,
                            icustays_df: DataFrame) -> Dict:
        """
        Tạo báo cáo toàn bộ
        
        Args:
            patients_df: Patients DataFrame
            admissions_df: Admissions DataFrame
            diagnosis_df: Diagnoses DataFrame
            lab_df: Lab events DataFrame
            icustays_df: ICU stays DataFrame
        
        Returns:
            Dict: Báo cáo hoàn chỉnh
        """
        try:
            report = {
                "timestamp": datetime.now().isoformat(),
                "patient_demographics": self.patient_analyzer.analyze_patient_demographics(
                    patients_df
                ),
                "admission_statistics": self.patient_analyzer.analyze_admission_patterns(
                    admissions_df
                ),
                "diagnosis_statistics": self.diagnosis_analyzer.analyze_comorbidities(
                    diagnosis_df
                ),
                "top_diagnoses": self.diagnosis_analyzer.analyze_top_diagnoses(
                    diagnosis_df, top_n=10
                ),
                "top_lab_tests": self.lab_analyzer.analyze_top_lab_tests(
                    lab_df, top_n=10
                ),
                "icu_statistics": self.icu_analyzer.analyze_icu_stays(icustays_df)
            }
            
            self.logger.info("✓ Generated full analysis report")
            return report
        
        except Exception as e:
            self.logger.error(f"✗ Error generating report: {str(e)}")
            raise
    
    def print_report(self, report: Dict):
        """In báo cáo"""
        print("\n" + "="*100)
        print("COMPREHENSIVE DATA ANALYSIS REPORT")
        print("="*100)
        print(f"\nReport Generated: {report['timestamp']}")
        
        # Patient Demographics
        print("\n" + "-"*100)
        print("PATIENT DEMOGRAPHICS")
        print("-"*100)
        demo = report['patient_demographics']
        print(f"Total Patients: {demo['total_patients']:,}")
        print("Gender Distribution:")
        for gender, count in demo['gender_distribution'].items():
            print(f"  {gender}: {count:,}")
        print("Age Statistics:")
        for stat, value in demo['age_statistics'].items():
            print(f"  {stat.capitalize()}: {value}")
        
        # Admission Statistics
        print("\n" + "-"*100)
        print("ADMISSION STATISTICS")
        print("-"*100)
        adm = report['admission_statistics']
        print(f"Total Admissions: {adm['total_admissions']:,}")
        print(f"In-Hospital Mortality Rate: {adm['in_hospital_mortality_rate']}%")
        print("Length of Stay Statistics (days):")
        for stat, value in adm['length_of_stay'].items():
            print(f"  {stat}: {value}")
        
        # Diagnosis Statistics
        print("\n" + "-"*100)
        print("DIAGNOSIS STATISTICS")
        print("-"*100)
        diag = report['diagnosis_statistics']
        print(f"Average Diagnoses per Admission: {diag['average_diagnoses_per_admission']}")
        print(f"Median Diagnoses per Admission: {diag['median_diagnoses_per_admission']}")
        
        # Top Diagnoses
        print("\nTop 10 Diagnoses:")
        for i, dx in enumerate(report['top_diagnoses'], 1):
            print(f"  {i}. {dx['icd_code']} ({dx['icd_version']}): {dx['frequency']:,} cases")
        
        # Top Lab Tests
        print("\n" + "-"*100)
        print("TOP LAB TESTS")
        print("-"*100)
        for i, test in enumerate(report['top_lab_tests'][:5], 1):
            print(f"  {i}. {test['test_name']}: {test['frequency']:,} tests "
                  f"({test['abnormal_percentage']}% abnormal)")
        
        # ICU Statistics
        print("\n" + "-"*100)
        print("ICU STATISTICS")
        print("-"*100)
        icu = report['icu_statistics']
        print(f"Total ICU Stays: {icu['total_stays']:,}")
        print("Length of ICU Stay (days):")
        for stat, value in icu['length_of_icu_stay'].items():
            print(f"  {stat}: {value}")
        
        print("\n" + "="*100 + "\n")


if __name__ == "__main__":
    print("Analysis and reporting modules loaded successfully")
