from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Iterable

from src.data.load_mimic import MIMICDataPaths, iter_table, read_lookup
from src.features.medication_history import extract_medication_token
from src.utils.io import (
    ensure_dir,
    load_yaml_config,
    parse_float,
    parse_int,
    read_csv_gz,
    read_json,
    resolve_path,
    write_json,
)


def vocab_dir_from_config(config: dict) -> Path:
    interim_root = resolve_path(config["_project_root"], config["paths"]["interim_root"])
    return ensure_dir(Path(interim_root) / "vocab")


def cohort_path_from_config(config: dict) -> Path:
    interim_root = resolve_path(config["_project_root"], config["paths"]["interim_root"])
    return Path(interim_root) / "cohort" / "cohort.csv.gz"


def _sorted_counter_items(counter: Counter[str], top_k: int | None = None) -> list[str]:
    items = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    tokens = [token for token, _ in items]
    if top_k is not None and top_k > 0:
        return tokens[:top_k]
    return tokens


def _build_vocab_payload(tokens: Iterable[str], vocab_name: str) -> dict[str, object]:
    idx_to_token = ["PAD", "UNK", *list(tokens)]
    token_to_idx = {token: index for index, token in enumerate(idx_to_token)}
    return {
        "name": vocab_name,
        "size": len(idx_to_token),
        "pad_idx": 0,
        "unk_idx": 1,
        "idx_to_token": idx_to_token,
        "token_to_idx": token_to_idx,
    }


def load_vocab_bundle(config_path: str | Path | dict) -> dict[str, dict]:
    config = config_path if isinstance(config_path, dict) else load_yaml_config(config_path)
    vocab_dir = vocab_dir_from_config(config)
    bundle: dict[str, dict] = {}
    for name in ("diagnosis", "procedure", "drug", "lab", "vital"):
        bundle[name] = read_json(vocab_dir / f"{name}_vocab.json")
    return bundle


def build_vocab(config_path: str | Path) -> Path:
    config = load_yaml_config(config_path)
    paths = MIMICDataPaths.from_config(config)
    vocab_dir = vocab_dir_from_config(config)
    cohort_rows = read_csv_gz(cohort_path_from_config(config))

    hadm_ids = {parse_int(row["hadm_id"]) for row in cohort_rows if row.get("hadm_id")}
    stay_ids = {parse_int(row["stay_id"]) for row in cohort_rows if row.get("stay_id")}
    hadm_ids.discard(None)
    stay_ids.discard(None)

    feature_cfg = config.get("features", {})
    top_k_labs = int(feature_cfg.get("top_k_labs", 64))
    top_k_vitals = int(feature_cfg.get("top_k_vitals", 64))

    diagnosis_counter: Counter[str] = Counter()
    procedure_counter: Counter[str] = Counter()
    drug_counter: Counter[str] = Counter()
    lab_counter: Counter[str] = Counter()
    vital_counter: Counter[str] = Counter()

    for row in iter_table(paths, "diagnoses_icd", fields=["hadm_id", "icd_code", "icd_version"]):
        hadm_id = parse_int(row.get("hadm_id"))
        code = str(row.get("icd_code", "")).strip()
        version = str(row.get("icd_version", "")).strip()
        if hadm_id in hadm_ids and code and version:
            diagnosis_counter[f"ICD{version}:{code}"] += 1

    for row in iter_table(paths, "procedures_icd", fields=["hadm_id", "icd_code", "icd_version"]):
        hadm_id = parse_int(row.get("hadm_id"))
        code = str(row.get("icd_code", "")).strip()
        version = str(row.get("icd_version", "")).strip()
        if hadm_id in hadm_ids and code and version:
            procedure_counter[f"PROC{version}:{code}"] += 1

    for table_name, fields in (
        ("emar", ["hadm_id", "medication"]),
        ("prescriptions", ["hadm_id", "drug", "formulary_drug_cd"]),
        ("pharmacy", ["hadm_id", "medication"]),
    ):
        for row in iter_table(paths, table_name, fields=fields):
            hadm_id = parse_int(row.get("hadm_id"))
            if hadm_id not in hadm_ids:
                continue
            token = extract_medication_token(row)
            if token:
                drug_counter[token] += 1

    for row in iter_table(paths, "labevents", fields=["hadm_id", "itemid", "valuenum"]):
        hadm_id = parse_int(row.get("hadm_id"))
        itemid = str(row.get("itemid", "")).strip()
        value = parse_float(row.get("valuenum"))
        if hadm_id in hadm_ids and itemid and value is not None:
            lab_counter[f"LAB:{itemid}"] += 1

    for row in iter_table(paths, "chartevents", fields=["stay_id", "itemid", "valuenum"]):
        stay_id = parse_int(row.get("stay_id"))
        itemid = str(row.get("itemid", "")).strip()
        value = parse_float(row.get("valuenum"))
        if stay_id in stay_ids and itemid and value is not None:
            vital_counter[f"VITAL:{itemid}"] += 1

    diagnosis_vocab = _build_vocab_payload(_sorted_counter_items(diagnosis_counter), "diagnosis")
    procedure_vocab = _build_vocab_payload(_sorted_counter_items(procedure_counter), "procedure")
    drug_vocab = _build_vocab_payload(_sorted_counter_items(drug_counter), "drug")
    lab_vocab = _build_vocab_payload(_sorted_counter_items(lab_counter, top_k=top_k_labs), "lab")
    vital_vocab = _build_vocab_payload(_sorted_counter_items(vital_counter, top_k=top_k_vitals), "vital")

    lab_lookup = read_lookup(paths, "d_labitems", "itemid", ["label", "category", "fluid"])
    vital_lookup = read_lookup(paths, "d_items", "itemid", ["label", "category", "unitname"])

    lab_metadata = {
        token: {
            "index": index,
            "label": lab_lookup.get(token.split(":", 1)[1], {}).get("label", ""),
            "category": lab_lookup.get(token.split(":", 1)[1], {}).get("category", ""),
            "fluid": lab_lookup.get(token.split(":", 1)[1], {}).get("fluid", ""),
        }
        for token, index in lab_vocab["token_to_idx"].items()
        if token not in {"PAD", "UNK"}
    }
    vital_metadata = {
        token: {
            "index": index,
            "label": vital_lookup.get(token.split(":", 1)[1], {}).get("label", ""),
            "category": vital_lookup.get(token.split(":", 1)[1], {}).get("category", ""),
            "unitname": vital_lookup.get(token.split(":", 1)[1], {}).get("unitname", ""),
        }
        for token, index in vital_vocab["token_to_idx"].items()
        if token not in {"PAD", "UNK"}
    }

    write_json(vocab_dir / "diagnosis_vocab.json", diagnosis_vocab)
    write_json(vocab_dir / "procedure_vocab.json", procedure_vocab)
    write_json(vocab_dir / "drug_vocab.json", drug_vocab)
    write_json(vocab_dir / "lab_vocab.json", lab_vocab)
    write_json(vocab_dir / "vital_vocab.json", vital_vocab)
    write_json(vocab_dir / "lab_metadata.json", lab_metadata)
    write_json(vocab_dir / "vital_metadata.json", vital_metadata)
    write_json(
        vocab_dir / "vocab_summary.json",
        {
            "diagnosis_size": diagnosis_vocab["size"],
            "procedure_size": procedure_vocab["size"],
            "drug_size": drug_vocab["size"],
            "lab_size": lab_vocab["size"],
            "vital_size": vital_vocab["size"],
        },
    )
    return vocab_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build vocabularies from cohort-filtered MIMIC tables.")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    args = parser.parse_args()
    build_vocab(args.config)


if __name__ == "__main__":
    main()
