from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    
    # Đường dẫn hệ thống
    project_root: Path = Path(__file__).resolve().parents[3]
    data_raw_dir: Path = project_root / "data" / "raw"
    artifacts_dir: Path = project_root / "artifacts"

    # Tham số module1 (MinHash + LSH)
    num_perm: int = 128 # số hash function dùng trong Minhash
    lsh_threshold: float = 0.4 # ngưỡng similarity của LSH
    top_k_default: int = 10 # số lượng bệnh nhân trả về mặc định

    # Giới hạn feature mỗi bệnh nhân
    max_unique_drugs_per_patient: int = 200
    max_unique_codes_per_patient: int = 300

settings = Settings()