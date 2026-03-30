from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from src.data.build_vocab import load_vocab_bundle
from src.utils.io import iter_jsonl_gz, load_yaml_config, resolve_path


def _trajectory_file(config: dict, split: str) -> Path:
    processed_root = resolve_path(config["_project_root"], config["paths"]["processed_root"])
    return Path(processed_root) / "trajectories" / split / "trajectories.jsonl.gz"


class MIMICTrajectoryDataset(Dataset):
    def __init__(self, split: str, config_path: str | Path) -> None:
        self.config = load_yaml_config(config_path)
        self.split = split
        self.vocab_bundle = load_vocab_bundle(self.config)
        self.records = list(iter_jsonl_gz(_trajectory_file(self.config, split)))
        self.drug_vocab_size = len(self.vocab_bundle["drug"]["idx_to_token"])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = dict(self.records[index])
        record["drug_vocab_size"] = self.drug_vocab_size
        return record


def _max_length(records: list[dict[str, Any]], field_name: str) -> int:
    return max(
        (
            len(step.get(field_name, []))
            for record in records
            for step in record.get("steps", [])
        ),
        default=0,
    )


def collate_batch(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("collate_batch requires at least one record")

    batch_size = len(records)
    max_steps = max(int(record["num_steps"]) for record in records)
    max_diag_codes = _max_length(records, "diagnosis_ids")
    max_proc_codes = _max_length(records, "procedure_ids")
    max_history = _max_length(records, "med_history_ids")
    drug_vocab_size = max(int(record.get("drug_vocab_size", 0)) for record in records)
    lab_feature_size = max(int(record.get("lab_feature_size", 0)) for record in records)
    vital_feature_size = max(int(record.get("vital_feature_size", 0)) for record in records)

    diag_codes = torch.zeros(batch_size, max_steps, max_diag_codes, dtype=torch.long)
    diag_mask = torch.zeros(batch_size, max_steps, max_diag_codes, dtype=torch.bool)
    proc_codes = torch.zeros(batch_size, max_steps, max_proc_codes, dtype=torch.long)
    proc_mask = torch.zeros(batch_size, max_steps, max_proc_codes, dtype=torch.bool)
    med_history = torch.zeros(batch_size, max_steps, max_history, dtype=torch.long)
    med_history_mask = torch.zeros(batch_size, max_steps, max_history, dtype=torch.bool)
    lab_values = torch.zeros(batch_size, max_steps, lab_feature_size, dtype=torch.float32)
    lab_mask = torch.zeros(batch_size, max_steps, lab_feature_size, dtype=torch.bool)
    vital_values = torch.zeros(batch_size, max_steps, vital_feature_size, dtype=torch.float32)
    vital_mask = torch.zeros(batch_size, max_steps, vital_feature_size, dtype=torch.bool)
    time_delta_hours = torch.zeros(batch_size, max_steps, dtype=torch.float32)
    visit_mask = torch.zeros(batch_size, max_steps, dtype=torch.bool)
    target_drugs = torch.zeros(batch_size, max_steps, drug_vocab_size, dtype=torch.float32)

    subject_ids: list[int] = []
    hadm_ids: list[int] = []
    stay_ids: list[int] = []

    for batch_index, record in enumerate(records):
        subject_ids.append(int(record["subject_id"]))
        hadm_ids.append(int(record["hadm_id"]))
        stay_ids.append(int(record["stay_id"]))
        for step_index, step in enumerate(record["steps"]):
            visit_mask[batch_index, step_index] = True
            diagnosis_ids = list(step.get("diagnosis_ids", []))
            procedure_ids = list(step.get("procedure_ids", []))
            history_ids = list(step.get("med_history_ids", []))
            target_ids = list(step.get("target_drugs", []))

            if diagnosis_ids:
                diag_codes[batch_index, step_index, : len(diagnosis_ids)] = torch.tensor(diagnosis_ids, dtype=torch.long)
                diag_mask[batch_index, step_index, : len(diagnosis_ids)] = True
            if procedure_ids:
                proc_codes[batch_index, step_index, : len(procedure_ids)] = torch.tensor(procedure_ids, dtype=torch.long)
                proc_mask[batch_index, step_index, : len(procedure_ids)] = True
            if history_ids:
                med_history[batch_index, step_index, : len(history_ids)] = torch.tensor(history_ids, dtype=torch.long)
                med_history_mask[batch_index, step_index, : len(history_ids)] = True

            if lab_feature_size:
                lab_values[batch_index, step_index] = torch.tensor(step.get("lab_values", []), dtype=torch.float32)
                lab_mask[batch_index, step_index] = torch.tensor(step.get("lab_mask", []), dtype=torch.bool)
            if vital_feature_size:
                vital_values[batch_index, step_index] = torch.tensor(step.get("vital_values", []), dtype=torch.float32)
                vital_mask[batch_index, step_index] = torch.tensor(step.get("vital_mask", []), dtype=torch.bool)

            for drug_id in target_ids:
                if 0 <= int(drug_id) < drug_vocab_size:
                    target_drugs[batch_index, step_index, int(drug_id)] = 1.0
            time_delta_hours[batch_index, step_index] = float(step.get("delta_hours", 0.0))

    return {
        "diag_codes": diag_codes,
        "diag_mask": diag_mask,
        "proc_codes": proc_codes,
        "proc_mask": proc_mask,
        "lab_values": lab_values,
        "lab_mask": lab_mask,
        "vital_values": vital_values,
        "vital_mask": vital_mask,
        "med_history": med_history,
        "med_history_mask": med_history_mask,
        "time_delta_hours": time_delta_hours,
        "visit_mask": visit_mask,
        "target_drugs": target_drugs,
        "subject_ids": subject_ids,
        "hadm_ids": hadm_ids,
        "stay_ids": stay_ids,
    }
