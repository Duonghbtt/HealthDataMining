#!/usr/bin/env python3
# ============================================
# MODULE 3 - MAIN EXECUTION SCRIPT
# MIMIC-IV Data Processing Pipeline
# ============================================

import sys
import logging
from pathlib import Path
from datetime import datetime

# Import các modules
from src.spark_session import SparkSessionManager, get_spark
from config import (
    DATA_PATH, OUTPUT_PATH, FILE_MAPPING,
    SPARK_CONFIG, PROCESSING_PARAMS, FEATURE_PARAMS
)
from data.data_loader import MIMICDataLoader, DataValidator
from data.data_processor import DataCleaner, DataTransformer, DataQualityReport
from features.feature_engineering import FeatureEngineer
from analysis.data_analysis import AnalysisReporter

# ============== LOGGING SETUP ==============
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(OUTPUT_PATH / "module3_execution.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


class Module3Pipeline:
    """
    Pipeline chính để xử lý dữ liệu MIMIC-IV
    """

    def __init__(self):
        """Khởi tạo pipeline"""
        self.spark = None
        self.loader = None
        self.validator = None
        self.cleaner = DataCleaner()
        self.transformer = DataTransformer()
        self.quality_report = DataQualityReport()
        self.feature_engineer = FeatureEngineer()
        self.analyzer = AnalysisReporter()

        self.data_store = {}  # Lưu trữ DataFrames

        logger.info("[OK] Module 3 Pipeline initialized")

    def setup_spark(self):
        """Khởi tạo Spark Session"""
        try:
            spark_manager = SparkSessionManager()
            self.spark = spark_manager.get_or_create_spark_session(
                app_name="MIMIC-IV-Module3",
                config_dict={
                    "spark.sql.shuffle.partitions": "200",
                    "spark.driver.memory": "4g",
                    "spark.executor.memory": "4g"
                }
            )

            logger.info(f"✓ Spark Session Setup Complete")
            logger.info(f"  Master: {self.spark.sparkContext.master()}")
            logger.info(f"  App ID: {self.spark.sparkContext.applicationId}")

        except Exception as e:
            logger.error(f"✗ Failed to setup Spark: {str(e)}")
            raise

    def load_data(self, file_names: list):
        """
        Đọc dữ liệu từ file ZIP

        Args:
            file_names (list): Danh sách file cần đọc
        """
        try:
            logger.info("=" * 80)
            logger.info("STAGE 1: LOADING DATA")
            logger.info("=" * 80)

            if not DATA_PATH.exists():
                raise FileNotFoundError(f"Data path not found: {DATA_PATH}")

            self.loader = MIMICDataLoader(self.spark, DATA_PATH)
            self.validator = DataValidator(self.spark)

            # Đọc các file
            self.data_store = self.loader.load_multiple_files(
                file_names,
                sample_ratio=PROCESSING_PARAMS["sample_ratio"]
            )

            logger.info(f"✓ Loaded {len(self.data_store)} files successfully")

            # Validation
            for file_name, df in self.data_store.items():
                validation = self.validator.validate_dataframe(df, file_name)
                self.validator.print_validation_report(validation)

        except Exception as e:
            logger.error(f"✗ Data loading failed: {str(e)}")
            raise

    def clean_data(self):
        """
        Làm sạch dữ liệu
        """
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 2: DATA CLEANING")
            logger.info("=" * 80)

            for file_name, df in self.data_store.items():
                logger.info(f"\nCleaning '{file_name}'...")

                # Loại bỏ duplicates
                df = self.cleaner.remove_duplicates(df)

                # Xử lý missing values
                df = self.cleaner.handle_missing_values(
                    df,
                    strategy="drop",
                    threshold=PROCESSING_PARAMS["missing_threshold"]
                )

                # Chuẩn hóa tên cột
                df = self.cleaner.standardize_column_names(df)

                # Lưu lại
                self.data_store[file_name] = df

            logger.info(f"✓ Data cleaning completed")

        except Exception as e:
            logger.error(f"✗ Data cleaning failed: {str(e)}")
            raise

    def process_core_data(self):
        """
        Xử lý core data (patients, admissions)
        """
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 3: PROCESSING CORE DATA")
            logger.info("=" * 80)

            # Patients
            if "patients" in self.data_store:
                patients_df = self.data_store["patients"]
                logger.info(f"Patients: {patients_df.count():,} records")

                # Xây dựng patient features
                patients_features = self.feature_engineer.build_patient_features(patients_df)
                self.data_store["patients_features"] = patients_features

            # Admissions
            if "admissions" in self.data_store:
                admissions_df = self.data_store["admissions"]
                logger.info(f"Admissions: {admissions_df.count():,} records")

                # Xây dựng admission features
                admission_features = self.feature_engineer.build_admission_features(
                    admissions_df
                )
                self.data_store["admission_features"] = admission_features

            logger.info("✓ Core data processing completed")

        except Exception as e:
            logger.error(f"✗ Core data processing failed: {str(e)}")
            raise

    def process_clinical_data(self):
        """
        Xử lý clinical data (lab, diagnoses, prescriptions)
        """
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 4: PROCESSING CLINICAL DATA")
            logger.info("=" * 80)

            # Lab events
            if "labevents" in self.data_store:
                lab_df = self.data_store["labevents"]
                logger.info(f"Lab Events: {lab_df.count():,} records")

                # Xây dựng lab features
                lab_features = self.feature_engineer.lab_features.create_lab_summary(
                    lab_df,
                    important_tests=FEATURE_PARAMS["lab_tests_important"]
                )
                self.data_store["lab_features"] = lab_features

            # Diagnoses
            if "diagnoses_icd" in self.data_store:
                diag_df = self.data_store["diagnoses_icd"]
                logger.info(f"Diagnoses: {diag_df.count():,} records")

                # Xây dựng diagnosis features
                diag_features = self.feature_engineer.diag_features.create_diagnosis_count_features(
                    diag_df
                )
                self.data_store["diagnosis_features"] = diag_features

            logger.info("✓ Clinical data processing completed")

        except Exception as e:
            logger.error(f"✗ Clinical data processing failed: {str(e)}")
            raise

    def process_icu_data(self):
        """
        Xử lý ICU data
        """
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 5: PROCESSING ICU DATA")
            logger.info("=" * 80)

            # ICU stays
            if "icustays" in self.data_store:
                icu_df = self.data_store["icustays"]
                logger.info(f"ICU Stays: {icu_df.count():,} records")

            # Chart events
            if "chartevents" in self.data_store:
                chart_df = self.data_store["chartevents"]
                logger.info(f"Chart Events: {chart_df.count():,} records")

                # Xây dựng vital signs features (nếu có itemid)
                try:
                    vital_features = self.feature_engineer.vital_features.create_vital_signs_summary(
                        chart_df
                    )
                    self.data_store["vital_features"] = vital_features
                except Exception as e:
                    logger.warning(f"⚠ Could not create vital signs features: {str(e)}")

            logger.info("✓ ICU data processing completed")

        except Exception as e:
            logger.error(f"✗ ICU data processing failed: {str(e)}")
            raise

    def perform_analysis(self):
        """
        Thực hiện phân tích dữ liệu
        """
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 6: DATA ANALYSIS")
            logger.info("=" * 80)

            # Đảm bảo có đủ dữ liệu để phân tích
            required_files = ["patients", "admissions", "diagnoses_icd",
                              "labevents", "icustays"]
            available_files = [f for f in required_files if f in self.data_store]

            if len(available_files) >= 3:
                # Generate report
                try:
                    report = self.analyzer.generate_full_report(
                        patients_df=self.data_store.get("patients"),
                        admissions_df=self.data_store.get("admissions"),
                        diagnosis_df=self.data_store.get("diagnoses_icd"),
                        lab_df=self.data_store.get("labevents"),
                        icustays_df=self.data_store.get("icustays")
                    )

                    self.analyzer.print_report(report)

                    # Lưu report
                    import json
                    report_file = OUTPUT_PATH / "analysis_report.json"
                    with open(report_file, 'w') as f:
                        json.dump(report, f, indent=2, default=str)

                    logger.info(f"✓ Analysis completed, report saved to {report_file}")

                except Exception as e:
                    logger.warning(f"⚠ Analysis generation failed: {str(e)}")
            else:
                logger.warning(f"⚠ Not enough data for full analysis. Available: {available_files}")

        except Exception as e:
            logger.error(f"✗ Analysis failed: {str(e)}")
            raise

    def save_results(self):
        """
        Lưu kết quả xử lý
        """
        try:
            logger.info("\n" + "=" * 80)
            logger.info("STAGE 7: SAVING RESULTS")
            logger.info("=" * 80)

            for name, df in self.data_store.items():
                if df is not None:
                    output_file = OUTPUT_PATH / f"{name}.parquet"
                    df.write.mode("overwrite").parquet(str(output_file))
                    logger.info(f"✓ Saved {name} to {output_file}")

            logger.info("✓ All results saved successfully")

        except Exception as e:
            logger.error(f"✗ Failed to save results: {str(e)}")
            raise

    def run(self, file_names: list):
        """
        Chạy pipeline hoàn chỉnh

        Args:
            file_names (list): Danh sách file cần xử lý
        """
        try:
            start_time = datetime.now()
            logger.info("\n" + "#" * 80)
            logger.info("# MODULE 3 - MIMIC-IV DATA PROCESSING PIPELINE")
            logger.info(f"# Started: {start_time}")
            logger.info("#" * 80)

            # Chạy các stage
            self.setup_spark()
            self.load_data(file_names)
            self.clean_data()
            self.process_core_data()
            self.process_clinical_data()
            self.process_icu_data()
            self.perform_analysis()
            self.save_results()

            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()

            logger.info("\n" + "#" * 80)
            logger.info("# PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"# Ended: {end_time}")
            logger.info(f"# Total Duration: {duration:.2f} seconds")
            logger.info("#" * 80 + "\n")

        except Exception as e:
            logger.error(f"\n✗ PIPELINE FAILED: {str(e)}\n")
            raise

        finally:
            # Cleanup
            try:
                spark_manager = SparkSessionManager()
                spark_manager.stop_spark_session()
            except:
                pass


def main():
    """
    Entry point
    """
    try:
        # Files cần xử lý
        files_to_process = [
            "admissions",
            "patients",
            "diagnoses_icd",
            "procedures_icd",
            "labevents",
            "prescriptions",
            "microbiologyevents",
            "transfers",
            "icustays",
            "chartevents",
            "inputevents",
            "outputevents",
            "procedureevents"
        ]

        # Khởi tạo và chạy pipeline
        pipeline = Module3Pipeline()
        pipeline.run(files_to_process)

    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()