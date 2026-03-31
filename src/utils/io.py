from __future__ import annotations

import csv
import gzip
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

import yaml


DATETIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d",
)


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def infer_project_root(config_path: str | Path) -> Path:
    path = Path(config_path).resolve()
    if path.parent.name == "configs":
        return path.parent.parent
    return path.parent


def ensure_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def resolve_path(base_dir: str | Path, target: str | Path) -> Path:
    candidate = Path(target)
    if candidate.is_absolute():
        return candidate
    return Path(base_dir) / candidate


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    config_file = Path(config_path).resolve()
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config["_config_path"] = str(config_file)
    config["_project_root"] = str(infer_project_root(config_file))
    return config


def stable_fraction(key: str | int, seed: int) -> float:
    payload = f"{seed}:{key}".encode("utf-8", errors="ignore")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) / float(1 << 64)


def assign_split(subject_id: int, split_config: Mapping[str, float], seed: int) -> str:
    train_ratio = float(split_config.get("train", 0.7))
    val_ratio = float(split_config.get("val", 0.15))
    test_ratio = float(split_config.get("test", 0.15))
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")

    value = stable_fraction(subject_id, seed)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def parse_datetime(value: str | None) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    for fmt in DATETIME_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported datetime value: {value}")


def parse_int(value: str | None, default: int | None = None) -> int | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return default
    return int(float(text))


def parse_float(value: str | None, default: float | None = None) -> float | None:
    text = "" if value is None else str(value).strip()
    if not text:
        return default
    return float(text)


def write_json(path: str | Path, payload: Any, *, indent: int = 2) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=indent, ensure_ascii=True, sort_keys=True)
    return destination


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def stable_json_dumps(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))


def fingerprint_payload(payload: Any, *, digest_size: int = 16) -> str:
    return hashlib.blake2b(
        stable_json_dumps(payload).encode("utf-8"),
        digest_size=digest_size,
    ).hexdigest()


def fingerprint_path(path: str | Path) -> str:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Cannot fingerprint missing path: {source}")
    stat = source.stat()
    payload = {
        "path": str(source.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    return fingerprint_payload(payload)


def write_jsonl_gz(path: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    with gzip.open(destination, "wt", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=True, sort_keys=True))
            handle.write("\n")
    return destination


def iter_jsonl_gz(path: str | Path) -> Iterator[dict[str, Any]]:
    with gzip.open(Path(path), "rt", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if text:
                yield json.loads(text)


def write_csv_gz(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    with gzip.open(destination, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name, "") for name in fieldnames})
    return destination


def read_csv_gz(path: str | Path) -> list[dict[str, str]]:
    with gzip.open(Path(path), "rt", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def save_pt(path: str | Path, payload: Any) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    try:
        import torch

        torch.save(payload, destination)
    except Exception:
        with destination.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return destination


def load_pt(path: str | Path) -> Any:
    source = Path(path)
    try:
        import torch

        try:
            return torch.load(source, map_location="cpu", weights_only=False)
        except Exception:
            pass
    except Exception:
        pass

    with source.open("rb") as handle:
        return pickle.load(handle)


def hours_between(start: datetime | None, end: datetime | None) -> float:
    if start is None or end is None:
        return 0.0
    return max((end - start).total_seconds() / 3600.0, 0.0)


def write_parquet_pylist(path: str | Path, records: Sequence[Mapping[str, Any]]) -> Path:
    destination = Path(path)
    ensure_dir(destination.parent)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required for parquet output. Install dependencies from requirements.txt first."
        ) from exc

    table = pa.Table.from_pylist(list(records))
    pq.write_table(table, destination, compression="snappy")
    return destination
