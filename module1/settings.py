from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class Settings:
    # Đường dẫn root MIMIC-IV v3.1 (chứa thư mục hosp/)
    mimic_root: Path

    # Đọc .csv.gz theo chunks để không nổ RAM
    chunksize: int = 300_000

    # Bật/tắt nguồn feature
    use_diagnoses: bool = True
    use_prescriptions: bool = True
    use_labs: bool = True

    # LAB: chỉ lấy top-K itemid phổ biến (để chạy ổn)
    top_k_labs: int = 200

    # Cap unique tokens mỗi bệnh nhân cho từng nhóm
    max_unique_icd_per_patient: int = 200
    max_unique_drug_per_patient: int = 200
    max_unique_lab_per_patient: int = 200

    # MinHash/LSH
    num_perm: int = 128

    # Manual LSH banding (nếu không dùng auto_threshold)
    bands: int = 32
    rows: int = 4   # bands * rows phải == num_perm

    # Auto chọn b,r theo S-curve: set None để tắt
    auto_threshold: float | None = None  # ví dụ 0.5

    # Query
    top_k: int = 20
    shortlist_factor: int = 50  # candidates -> shortlist -> exact rerank

    # Demo nhanh: chỉ lấy subject_id < limit_patients
    limit_patients: int | None = None

    # Explain: in tối đa bao nhiêu token overlap mỗi nhóm
    top_n_explain_tokens: int = 15