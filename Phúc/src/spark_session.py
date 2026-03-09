# ============================================
# MODULE 3 - SPARK SESSION INITIALIZATION
# ============================================

import logging
from pathlib import Path
from pyspark.sql import SparkSession
from pyspark.sql.types import *
import sys


class SparkSessionManager:
    """
    Quản lý Spark Session cho xử lý dữ liệu MIMIC-IV
    """

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SparkSessionManager, cls).__new__(cls)
        return cls._instance

    def __init__(self):
        if not hasattr(self, 'spark'):
            self.spark = None
            self.logger = None
            self._initialize_logger()

    def _initialize_logger(self):
        """Khởi tạo logger"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

    def get_or_create_spark_session(self, app_name="MIMIC-IV", config_dict=None):
        """
        Lấy hoặc tạo Spark Session

        Args:
            app_name (str): Tên ứng dụng Spark
            config_dict (dict): Cấu hình Spark tùy chỉnh

        Returns:
            SparkSession: Spark Session instance
        """
        if self.spark is None:
            try:
                spark_builder = SparkSession.builder.appName(app_name)

                # Cấu hình mặc định
                default_config = {
                    "spark.sql.adaptive.enabled": "true",
                    "spark.sql.adaptive.coalescePartitions.enabled": "true",
                    "spark.sql.shuffle.partitions": "200",
                    "spark.broadcast.blockManager.maxRetries": "3",
                    "spark.driver.maxResultSize": "4g",
                    "spark.sql.parquet.compression.codec": "snappy",
                    "spark.sql.inMemoryColumnarStorage.compressed": "true",
                    "spark.memory.storageFraction": "0.3",
                }

                # Ghi đè cấu hình nếu có
                if config_dict:
                    default_config.update(config_dict)

                # Áp dụng cấu hình
                for key, value in default_config.items():
                    spark_builder = spark_builder.config(key, value)

                self.spark = spark_builder.getOrCreate()

                self.logger.info(f"✓ Spark Session created: {app_name}")
                self.logger.info(f"  - Master: {self.spark.sparkContext.master()}")
                self.logger.info(f"  - App ID: {self.spark.sparkContext.applicationId}")

                return self.spark

            except Exception as e:
                self.logger.error(f"✗ Error creating Spark Session: {str(e)}")
                raise

        return self.spark

    def get_spark_session(self):
        """Lấy Spark Session hiện tại"""
        if self.spark is None:
            return self.get_or_create_spark_session()
        return self.spark

    def stop_spark_session(self):
        """Dừng Spark Session"""
        if self.spark is not None:
            try:
                self.spark.stop()
                self.spark = None
                self.logger.info("✓ Spark Session stopped")
            except Exception as e:
                self.logger.error(f"✗ Error stopping Spark Session: {str(e)}")

    def get_spark_context(self):
        """Lấy Spark Context"""
        return self.get_spark_session().sparkContext

    def get_sql_context(self):
        """Lấy SQL Context"""
        return self.get_spark_session().sql

    def log_info(self, message):
        """Ghi log thông tin"""
        self.logger.info(message)

    def log_warning(self, message):
        """Ghi log cảnh báo"""
        self.logger.warning(message)

    def log_error(self, message):
        """Ghi log lỗi"""
        self.logger.error(message)

    def get_spark_version(self):
        """Lấy phiên bản Spark"""
        return self.get_spark_session().version

    def print_spark_config(self):
        """In ra cấu hình Spark hiện tại"""
        spark = self.get_spark_session()
        print("\n" + "=" * 60)
        print("SPARK CONFIGURATION")
        print("=" * 60)

        config = spark.sparkContext.getConf().getAll()
        for key, value in sorted(config):
            print(f"{key:<45} {value}")

        print("=" * 60 + "\n")


# Singleton instance
spark_manager = SparkSessionManager()


def get_spark():
    """Hàm tiện ích lấy Spark Session"""
    return spark_manager.get_spark_session()


def get_logger():
    """Hàm tiện ích lấy logger"""
    return spark_manager.logger


if __name__ == "__main__":
    # Test
    manager = SparkSessionManager()
    spark = manager.get_or_create_spark_session()

    print(f"\n✓ Spark Version: {manager.get_spark_version()}")
    manager.print_spark_config()

    # Test tạo DataFrame
    data = [("Alice", 25), ("Bob", 30), ("Charlie", 35)]
    df = spark.createDataFrame(data, ["name", "age"])
    df.show()

    # Cleanup
    # manager.stop_spark_session()