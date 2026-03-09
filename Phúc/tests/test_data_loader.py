# ============================================
# MODULE 3 - TEST DATA LOADER
# ============================================

import pytest
from pathlib import Path
from pyspark.sql import SparkSession
from data_loader import MIMICDataLoader, DataValidator


class TestMIMICDataLoader:
    """Test cases for MIMICDataLoader"""
    
    @pytest.fixture(scope="class")
    def spark(self):
        """Create Spark Session for tests"""
        spark = SparkSession.builder \
            .appName("test") \
            .master("local[1]") \
            .getOrCreate()
        yield spark
        spark.stop()
    
    @pytest.fixture
    def loader(self, spark):
        """Create MIMICDataLoader instance"""
        zip_path = "D:/ProjectBigData/data/mimic-iv-3.1.zip"
        return MIMICDataLoader(spark, zip_path)
    
    def test_loader_initialization(self, loader):
        """Test loader initialization"""
        assert loader is not None
        assert loader.zip_path.exists()
    
    def test_list_files_in_zip(self, loader):
        """Test listing files in ZIP"""
        files = loader.list_files_in_zip()
        assert len(files) > 0
        assert "admissions" in files
        assert "patients" in files
    
    def test_load_csv_from_zip(self, loader):
        """Test loading single CSV file"""
        df = loader.load_csv_from_zip("admissions", sample_ratio=0.01)
        assert df is not None
        assert df.count() > 0
        assert len(df.columns) > 0
    
    def test_load_multiple_files(self, loader):
        """Test loading multiple files"""
        files = ["patients", "admissions"]
        dfs = loader.load_multiple_files(files, sample_ratio=0.01)
        assert len(dfs) == 2
        assert all(df.count() > 0 for df in dfs.values())


class TestDataValidator:
    """Test cases for DataValidator"""
    
    @pytest.fixture(scope="class")
    def spark(self):
        """Create Spark Session"""
        spark = SparkSession.builder \
            .appName("test") \
            .master("local[1]") \
            .getOrCreate()
        yield spark
        spark.stop()
    
    @pytest.fixture
    def validator(self, spark):
        """Create DataValidator instance"""
        return DataValidator(spark)
    
    @pytest.fixture
    def sample_df(self, spark):
        """Create sample DataFrame"""
        data = [
            (1, "Alice", 25, None),
            (2, "Bob", 30, "M"),
            (3, "Charlie", None, "M")
        ]
        return spark.createDataFrame(data, ["id", "name", "age", "gender"])
    
    def test_validate_dataframe(self, validator, sample_df):
        """Test DataFrame validation"""
        result = validator.validate_dataframe(sample_df, "test")
        assert result["file_name"] == "test"
        assert result["total_rows"] == 3
        assert result["total_columns"] == 4
    
    def test_null_detection(self, validator, sample_df):
        """Test NULL value detection"""
        result = validator.validate_dataframe(sample_df, "test")
        null_info = result["null_counts"]
        assert "age" in null_info
        assert null_info["age"]["count"] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
