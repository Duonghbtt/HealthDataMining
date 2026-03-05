from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterator, Optional, Set, Tuple

import pandas as pd


def _s(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip()


def _hosp_path(mimic_root: Path, filename: str) -> Path:
    return Path(mimic_root) / "hosp" / filename


def sniff_columns(csv_gz_path: Path) -> list[str]:
    """
    Đọc header để biết cột nào tồn tại.
    Mục tiêu: prescriptions có thể khác phiên bản (formulary_drug_cd/drug).
    """
    df0 = pd.read_csv(csv_gz_path, compression="gzip", nrows=0)
    return [c.strip() for c in df0.columns.tolist()]


def iter_diagnoses(mimic_root: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    """
    hosp/diagnoses_icd.csv.gz
    Lấy 3 cột đủ để token hoá ICD.
    """
    path = _hosp_path(mimic_root, "diagnoses_icd.csv.gz")
    usecols = ["subject_id", "icd_code", "icd_version"]
    dtypes = {"subject_id": "int32", "icd_code": "string", "icd_version": "int8"}

    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=usecols,
        dtype=dtypes,
        chunksize=chunksize,
        low_memory=False,
    ):
        yield chunk


def iter_labevents(mimic_root: Path, chunksize: int) -> Iterator[pd.DataFrame]:
    """
    hosp/labevents.csv.gz
    Lấy (subject_id, itemid) để làm token LAB.
    """
    path = _hosp_path(mimic_root, "labevents.csv.gz")
    usecols = ["subject_id", "itemid"]
    dtypes = {"subject_id": "int32", "itemid": "int32"}

    for chunk in pd.read_csv(
        path,
        compression="gzip",
        usecols=usecols,
        dtype=dtypes,
        chunksize=chunksize,
        low_memory=False,
    ):
        yield chunk


def iter_prescriptions(mimic_root: Path, chunksize: int) -> Tuple[Iterator[pd.DataFrame], str]:
    """
    hosp/prescriptions.csv.gz
    Ưu tiên formulary_drug_cd nếu có, fallback drug.

    Return:
      - iterator chunks
      - mode: "formulary_drug_cd" hoặc "drug"
    """
    path = _hosp_path(mimic_root, "prescriptions.csv.gz")
    cols = set(sniff_columns(path))

    if "subject_id" not in cols:
        raise ValueError("prescriptions.csv.gz không có cột 'subject_id'")

    has_fcd = "formulary_drug_cd" in cols
    has_drug = "drug" in cols

    if has_fcd:
        usecols = ["subject_id", "formulary_drug_cd"]
        dtypes = {"subject_id": "int32", "formulary_drug_cd": "string"}
        mode = "formulary_drug_cd"
    elif has_drug:
        usecols = ["subject_id", "drug"]
        dtypes = {"subject_id": "int32", "drug": "string"}
        mode = "drug"
    else:
        raise ValueError("prescriptions.csv.gz thiếu cả 'formulary_drug_cd' và 'drug'.")

    def _iter() -> Iterator[pd.DataFrame]:
        for chunk in pd.read_csv(
            path,
            compression="gzip",
            usecols=usecols,
            dtype=dtypes,
            chunksize=chunksize,
            low_memory=False,
        ):
            yield chunk

    return _iter(), mode


def compute_top_lab_itemids(mimic_root: Path, chunksize: int, top_k: int) -> Set[int]:
    """
    Tính top-K lab itemid phổ biến nhất bằng streaming.
    """
    counter = Counter()
    for chunk in iter_labevents(mimic_root, chunksize):
        counter.update(chunk["itemid"].tolist())
    return set(int(x) for x, _ in counter.most_common(top_k))


def build_patient_tokens(
    mimic_root: Path,
    chunksize: int,
    *,
    use_diagnoses: bool,
    use_prescriptions: bool,
    use_labs: bool,
    top_lab_itemids: Optional[Set[int]],
    max_icd: int,
    max_drug: int,
    max_lab: int,
    limit_patients: int | None = None
) -> Dict[int, Set[str]]:
    """
    Xây token set cho mỗi patient (subject_id):

    ICD token:
      ICD{icd_version}:{icd_code}

    DRUG token:
      DRUGCD:{formulary_drug_cd} hoặc DRUG:{drug} (lower)

    LAB token:
      LAB:{itemid} (chỉ top_lab_itemids)

    Có cap theo từng nhóm để tránh feature_set quá lớn.
    """
    feats: DefaultDict[int, Set[str]] = defaultdict(set)
    cnt_icd: DefaultDict[int, int] = defaultdict(int)
    cnt_drug: DefaultDict[int, int] = defaultdict(int)
    cnt_lab: DefaultDict[int, int] = defaultdict(int)

    # ===== ICD =====
    if use_diagnoses:
        for chunk in iter_diagnoses(mimic_root, chunksize):
            for sid, code, ver in zip(chunk["subject_id"], chunk["icd_code"], chunk["icd_version"]):
                sid = int(sid)
                if limit_patients is not None and sid >= limit_patients:
                    continue
                if cnt_icd[sid] >= max_icd:
                    continue
                code = _s(code)
                if not code:
                    continue

                tok = f"ICD{int(ver)}:{code}"
                if tok not in feats[sid]:
                    feats[sid].add(tok)
                    cnt_icd[sid] += 1

    # ===== DRUG =====
    if use_prescriptions:
        it, mode = iter_prescriptions(mimic_root, chunksize)

        if mode == "formulary_drug_cd":
            for chunk in it:
                for sid, fcd in zip(chunk["subject_id"], chunk["formulary_drug_cd"]):
                    sid = int(sid)
                    if limit_patients is not None and sid >= limit_patients:
                        continue
                    if cnt_drug[sid] >= max_drug:
                        continue

                    fcd = _s(fcd)
                    if not fcd:
                        continue

                    tok = f"DRUGCD:{fcd}"
                    if tok not in feats[sid]:
                        feats[sid].add(tok)
                        cnt_drug[sid] += 1
        else:
            for chunk in it:
                for sid, drug in zip(chunk["subject_id"], chunk["drug"]):
                    sid = int(sid)
                    if limit_patients is not None and sid >= limit_patients:
                        continue
                    if cnt_drug[sid] >= max_drug:
                        continue

                    drug = _s(drug)
                    if not drug:
                        continue

                    tok = f"DRUG:{drug.lower()}"
                    if tok not in feats[sid]:
                        feats[sid].add(tok)
                        cnt_drug[sid] += 1

    # ===== LAB =====
    if use_labs:
        if top_lab_itemids is None:
            raise ValueError("use_labs=True thì cần top_lab_itemids")

        for chunk in iter_labevents(mimic_root, chunksize):
            for sid, itemid in zip(chunk["subject_id"], chunk["itemid"]):
                sid = int(sid)
                if limit_patients is not None and sid >= limit_patients:
                    continue
                if cnt_lab[sid] >= max_lab:
                    continue

                itemid = int(itemid)
                if itemid not in top_lab_itemids:
                    continue

                tok = f"LAB:{itemid}"
                if tok not in feats[sid]:
                    feats[sid].add(tok)
                    cnt_lab[sid] += 1

    return dict(feats)