# ============================================
# MODULE 3 - UTILITIES & HELPER FUNCTIONS
# ============================================

import logging
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, to_timestamp, from_unixtime
import pandas as pd


logger = logging.getLogger(__name__)


class FileManager:
    """
    Quản lý file và directories
    """
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
    
    def create_directory(self, dir_name: str) -> Path:
        """
        Tạo thư mục nếu chưa tồn tại
        
        Args:
            dir_name (str): Tên thư mục
        
        Returns:
            Path: Đường dẫn thư mục
        """
        dir_path = self.base_path / dir_name
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"✓ Directory created: {dir_path}")
        return dir_path
    
    def save_dataframe_parquet(self, df: DataFrame, file_name: str) -> Path:
        """
        Lưu DataFrame dưới định dạng Parquet
        
        Args:
            df (DataFrame): Spark DataFrame
            file_name (str): Tên file
        
        Returns:
            Path: Đường dẫn file
        """
        output_path = self.base_path / f"{file_name}.parquet"
        df.write.mode("overwrite").parquet(str(output_path))
        logger.info(f"✓ Saved Parquet: {output_path}")
        return output_path
    
    def save_dataframe_csv(self, df: DataFrame, file_name: str, 
                          single_file: bool = True) -> Path:
        """
        Lưu DataFrame dưới định dạng CSV
        
        Args:
            df (DataFrame): Spark DataFrame
            file_name (str): Tên file
            single_file (bool): Lưu thành một file hay nhiều partitions
        
        Returns:
            Path: Đường dẫn file/thư mục
        """
        output_path = self.base_path / f"{file_name}.csv"
        
        if single_file:
            df = df.coalesce(1)
        
        df.write.mode("overwrite").csv(str(output_path), header=True)
        logger.info(f"✓ Saved CSV: {output_path}")
        return output_path
    
    def save_json(self, data: Dict, file_name: str) -> Path:
        """
        Lưu dictionary thành JSON file
        
        Args:
            data (Dict): Dữ liệu cần lưu
            file_name (str): Tên file
        
        Returns:
            Path: Đường dẫn file
        """
        output_path = self.base_path / f"{file_name}.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"✓ Saved JSON: {output_path}")
        return output_path
    
    def load_json(self, file_name: str) -> Dict:
        """
        Đọc JSON file
        
        Args:
            file_name (str): Tên file
        
        Returns:
            Dict: Dữ liệu đã đọc
        """
        file_path = self.base_path / f"{file_name}.json"
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"✓ Loaded JSON: {file_path}")
        return data


class DataFrameUtils:
    """
    Utility functions cho Spark DataFrame
    """
    
    @staticmethod
    def print_schema(df: DataFrame, max_cols: int = 10):
        """
        In schema của DataFrame
        
        Args:
            df (DataFrame): Spark DataFrame
            max_cols (int): Số cột tối đa hiển thị
        """
        print("\n" + "="*80)
        print("DATAFRAME SCHEMA")
        print("="*80)
        print(f"Total Columns: {len(df.columns)}")
        print("-"*80)
        
        for i, field in enumerate(df.schema.fields[:max_cols]):
            print(f"{i+1:3d}. {field.name:<30s} {str(field.dataType):<20s} "
                  f"nullable={field.nullable}")
        
        if len(df.columns) > max_cols:
            print(f"... and {len(df.columns) - max_cols} more columns")
        
        print("="*80 + "\n")
    
    @staticmethod
    def get_memory_usage(df: DataFrame) -> Dict:
        """
        Ước tính sử dụng bộ nhớ của DataFrame
        
        Args:
            df (DataFrame): Spark DataFrame
        
        Returns:
            Dict: Thông tin sử dụng bộ nhớ
        """
        # Ước tính thô từ schema
        row_count = df.count()
        
        # Tính kích thước từng cột
        col_sizes = {}
        total_size = 0
        
        for field in df.schema.fields:
            col_name = field.name
            data_type = str(field.dataType)
            
            # Ước tính kích thước theo type
            if "string" in data_type:
                estimated_size = 50  # bytes
            elif "int" in data_type:
                estimated_size = 4  # bytes
            elif "double" in data_type:
                estimated_size = 8  # bytes
            elif "timestamp" in data_type:
                estimated_size = 8  # bytes
            else:
                estimated_size = 10  # bytes
            
            col_bytes = row_count * estimated_size
            col_sizes[col_name] = col_bytes
            total_size += col_bytes
        
        return {
            "row_count": row_count,
            "column_count": len(df.columns),
            "estimated_total_bytes": total_size,
            "estimated_total_mb": total_size / (1024 * 1024),
            "estimated_total_gb": total_size / (1024 * 1024 * 1024),
            "column_sizes_bytes": col_sizes
        }
    
    @staticmethod
    def sample_rows(df: DataFrame, n_rows: int = 5) -> pd.DataFrame:
        """
        Lấy sample rows và convert thành Pandas
        
        Args:
            df (DataFrame): Spark DataFrame
            n_rows (int): Số rows lấy
        
        Returns:
            pd.DataFrame: Pandas DataFrame
        """
        return df.limit(n_rows).toPandas()
    
    @staticmethod
    def column_statistics(df: DataFrame) -> Dict:
        """
        Tính thống kê cho các cột numeric
        
        Args:
            df (DataFrame): Spark DataFrame
        
        Returns:
            Dict: Thống kê cột
        """
        from pyspark.sql.functions import avg, stddev, min, max
        
        numeric_cols = [f.name for f in df.schema.fields 
                       if f.dataType.simpleString() in 
                       ['long', 'integer', 'double', 'float']]
        
        if not numeric_cols:
            return {}
        
        agg_dict = {}
        for col_name in numeric_cols:
            agg_dict.update({
                f"{col_name}_mean": avg(col_name),
                f"{col_name}_stddev": stddev(col_name),
                f"{col_name}_min": min(col_name),
                f"{col_name}_max": max(col_name)
            })
        
        result = df.agg(agg_dict).collect()[0]
        
        return {k: v for k, v in result.asDict().items()}


class TimeSeriesUtils:
    """
    Utility functions cho time series data
    """
    
    @staticmethod
    def convert_to_timestamp(df: DataFrame, col_name: str, 
                            input_format: str = "unix") -> DataFrame:
        """
        Chuyển đổi cột thành timestamp
        
        Args:
            df (DataFrame): Spark DataFrame
            col_name (str): Tên cột
            input_format (str): "unix" hoặc format string
        
        Returns:
            DataFrame: DataFrame với timestamp converted
        """
        if input_format == "unix":
            df = df.withColumn(col_name, from_unixtime(col(col_name)))
        else:
            df = df.withColumn(col_name, to_timestamp(col(col_name), input_format))
        
        return df
    
    @staticmethod
    def extract_date_features(df: DataFrame, col_name: str) -> DataFrame:
        """
        Extract date features từ timestamp column
        
        Args:
            df (DataFrame): Spark DataFrame
            col_name (str): Tên cột timestamp
        
        Returns:
            DataFrame: DataFrame với date features
        """
        from pyspark.sql.functions import year, month, dayofmonth, hour, dayofweek
        
        df = df.withColumn(
            f"{col_name}_year", year(col(col_name))
        ).withColumn(
            f"{col_name}_month", month(col(col_name))
        ).withColumn(
            f"{col_name}_day", dayofmonth(col(col_name))
        ).withColumn(
            f"{col_name}_hour", hour(col(col_name))
        ).withColumn(
            f"{col_name}_dayofweek", dayofweek(col(col_name))
        )
        
        return df
    
    @staticmethod
    def create_time_bins(df: DataFrame, col_name: str, 
                        bin_hours: int) -> DataFrame:
        """
        Tạo time bins từ timestamp
        
        Args:
            df (DataFrame): Spark DataFrame
            col_name (str): Tên cột timestamp
            bin_hours (int): Kích thước bin (giờ)
        
        Returns:
            DataFrame: DataFrame với time_bin column
        """
        from pyspark.sql.functions import date_trunc
        
        df = df.withColumn(
            f"{col_name}_bin",
            date_trunc(f"{bin_hours} hour", col(col_name))
        )
        
        return df


class PerformanceMonitor:
    """
    Monitor performance của Spark jobs
    """
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
        self.logger = logger
    
    def get_execution_time(self, func, *args, **kwargs) -> tuple:
        """
        Đo thời gian thực thi
        
        Args:
            func: Hàm cần đo
            *args: Positional arguments
            **kwargs: Keyword arguments
        
        Returns:
            tuple: (result, execution_time_seconds)
        """
        import time
        
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        execution_time = end_time - start_time
        self.logger.info(f"✓ Execution time: {execution_time:.2f} seconds")
        
        return result, execution_time
    
    def get_spark_config(self) -> Dict:
        """
        Lấy Spark configuration
        
        Returns:
            Dict: Spark config
        """
        config = self.spark.sparkContext.getConf().getAll()
        return dict(config)
    
    def get_executor_info(self) -> Dict:
        """
        Lấy thông tin executor
        
        Returns:
            Dict: Executor info
        """
        return {
            "executor_cores": self.spark.sparkContext.defaultParallelism,
            "executor_memory": self.spark.conf.get("spark.executor.memory"),
            "driver_memory": self.spark.conf.get("spark.driver.memory")
        }
    
    def optimize_for_large_dataset(self):
        """
        Tối ưu Spark configuration cho dataset lớn
        """
        self.spark.conf.set("spark.sql.shuffle.partitions", "500")
        self.spark.conf.set("spark.sql.adaptive.enabled", "true")
        self.spark.conf.set("spark.sql.adaptive.coalescePartitions.enabled", "true")
        
        self.logger.info("✓ Optimized for large dataset")


class ValidationUtils:
    """
    Validation utilities
    """
    
    @staticmethod
    def validate_required_columns(df: DataFrame, required_cols: List[str]) -> bool:
        """
        Kiểm tra DataFrame có chứa các cột bắt buộc
        
        Args:
            df (DataFrame): Spark DataFrame
            required_cols (List[str]): Danh sách cột bắt buộc
        
        Returns:
            bool: True nếu hợp lệ
        """
        missing_cols = set(required_cols) - set(df.columns)
        
        if missing_cols:
            logger.error(f"✗ Missing required columns: {missing_cols}")
            return False
        
        logger.info(f"✓ All required columns present")
        return True
    
    @staticmethod
    def validate_null_percentage(df: DataFrame, col_name: str, 
                                max_null_pct: float = 0.5) -> bool:
        """
        Kiểm tra percentage NULL values trong cột
        
        Args:
            df (DataFrame): Spark DataFrame
            col_name (str): Tên cột
            max_null_pct (float): Phần trăm NULL tối đa cho phép
        
        Returns:
            bool: True nếu hợp lệ
        """
        from pyspark.sql.functions import count, sum as spark_sum, when, isnull
        
        total_rows = df.count()
        null_count = df.filter(isnull(col(col_name))).count()
        null_pct = null_count / total_rows if total_rows > 0 else 0
        
        if null_pct > max_null_pct:
            logger.warning(f"⚠ Column '{col_name}' has {null_pct*100:.2f}% NULL values")
            return False
        
        logger.info(f"✓ Column '{col_name}' has {null_pct*100:.2f}% NULL values")
        return True
    
    @staticmethod
    def validate_data_types(df: DataFrame, expected_types: Dict[str, str]) -> bool:
        """
        Kiểm tra kiểu dữ liệu của các cột
        
        Args:
            df (DataFrame): Spark DataFrame
            expected_types (Dict[str, str]): {col_name: expected_type}
        
        Returns:
            bool: True nếu hợp lệ
        """
        current_types = {f.name: str(f.dataType) for f in df.schema.fields}
        
        mismatches = []
        for col_name, expected_type in expected_types.items():
            if col_name in current_types:
                if expected_type not in current_types[col_name]:
                    mismatches.append(f"{col_name}: expected {expected_type}, "
                                    f"got {current_types[col_name]}")
        
        if mismatches:
            logger.warning(f"⚠ Data type mismatches: {mismatches}")
            return False
        
        logger.info(f"✓ All data types match expected types")
        return True


if __name__ == "__main__":
    print("Utilities module loaded successfully")
