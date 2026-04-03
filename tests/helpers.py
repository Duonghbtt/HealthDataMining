from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HANDOVER_ROOT = PROJECT_ROOT / "handover_data"


def load_handover_records(*, split: str = "train", limit: int = 4) -> list[dict]:
    split_dir = HANDOVER_ROOT / "processed" / split
    first_shard = sorted(split_dir.glob("part-*.parquet"))[0]
    rows = pq.read_table(first_shard).slice(0, limit).to_pylist()
    return [dict(row) for row in rows]


def load_vocab_size(name: str) -> int:
    with (HANDOVER_ROOT / "vocab" / f"{name}_vocab.json").open("r", encoding="utf-8") as handle:
        return int(json.load(handle)["size"])
