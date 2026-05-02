from __future__ import annotations

from pathlib import Path

from src.data.ddinter_utils import (
    build_ddinter_dataset,
    load_ddinter_raw_rows,
    map_vocab_items_to_ddinter_ids,
    match_names_to_ddinter_ids,
    normalize_ddinter_text,
    project_ddinter_pairs_to_vocab,
)
from src.data.drugbank_vocab_utils import DrugBankVocabularyIndex


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_mock_ddinter_root(tmp_path: Path) -> Path:
    root = tmp_path / "ddi"
    _write_text(
        root / "ddinter_downloads_code_A.csv",
        "\n".join(
            [
                "DDInterID_A,Drug_A,DDInterID_B,Drug_B,Level",
                "DDInter1,Aspirin,DDInter2,Warfarin,Major",
                "DDInter2,Warfarin,DDInter1,Aspirin oral solution,Moderate",
                "DDInter1,Aspirin tablet,DDInter3,Ibuprofen,Unknown",
                "DDInter4,,DDInter5,Acetaminophen,Minor",
                "DDInter6,Self drug,DDInter6,Self drug,Moderate",
            ]
        )
        + "\n",
    )
    _write_text(
        root / "ddinter_downloads_code_B.csv",
        "\n".join(
            [
                "DDInterID_A,Drug_A,DDInterID_B,Drug_B,Level",
                "DDInter1,Aspirin,DDInter2,Warfarin sodium,Major",
                "DDInter7,Albuterol sulfate inhalation,DDInter8,Ipratropium bromide aerosol,Minor",
            ]
        )
        + "\n",
    )
    return root


def _build_mock_drugbank_index() -> DrugBankVocabularyIndex:
    return DrugBankVocabularyIndex(
        normalized_name_to_drugbank_id={
            "acetylsalicylic acid": "DB0001",
        },
        synonym_to_drugbank_id={
            "asa": "DB0001",
            "coumadin": "DB0002",
        },
        drugbank_id_to_canonical_name={
            "DB0001": "Aspirin",
            "DB0002": "Warfarin",
        },
        drugbank_id_to_all_normalized_names={
            "DB0001": ("acetylsalicylic acid", "asa", "aspirin"),
            "DB0002": ("coumadin", "warfarin"),
        },
        ambiguous_normalized_names={},
        ambiguous_synonyms={},
        meta={},
    )


def test_normalize_ddinter_text_collapses_punctuation_and_simple_dosage_forms() -> None:
    assert normalize_ddinter_text(" Aspirin, oral solution ") == "aspirin"
    assert normalize_ddinter_text("Ipratropium-bromide aerosol") == "ipratropium bromide"


def test_load_ddinter_raw_rows_and_build_dataset_filters_unknown_and_self_pairs(tmp_path: Path) -> None:
    ddinter_root = _build_mock_ddinter_root(tmp_path)

    raw_rows, raw_meta = load_ddinter_raw_rows(ddinter_root)
    dataset = build_ddinter_dataset(raw_rows, include_unknown=False)

    assert raw_meta["file_count"] == 2
    assert raw_meta["rows_loaded"] == 7
    assert raw_meta["rows_with_missing_left_name"] == 1

    assert dataset.entities["DDInter1"].canonical_name == "Aspirin"
    assert dataset.entities["DDInter1"].normalized_name == "aspirin"
    assert dataset.normalized_name_to_ids["aspirin"] == ("DDInter1",)

    pair_meta = dataset.meta["pair_index"]
    assert pair_meta["rows_missing_required_fields"] == 1
    assert pair_meta["rows_self_pairs"] == 1
    assert pair_meta["rows_unknown_filtered"] == 1
    assert pair_meta["unique_pair_count"] == 2

    pair_record = dataset.pair_key_to_record[("DDInter1", "DDInter2")]
    assert pair_record.row_count == 3
    assert pair_record.dominant_level == "Major"
    assert pair_record.max_severity_level == "Major"
    assert pair_record.left_name == "Aspirin"
    assert pair_record.right_name == "Warfarin"


def test_match_and_project_ddinter_pairs_to_vocab_with_drugbank_bridge(tmp_path: Path) -> None:
    ddinter_root = _build_mock_ddinter_root(tmp_path)
    raw_rows, _ = load_ddinter_raw_rows(ddinter_root)
    dataset = build_ddinter_dataset(raw_rows, include_unknown=False)
    drugbank_index = _build_mock_drugbank_index()

    matched_ids, match_report = match_names_to_ddinter_ids(
        ["ASA"],
        ddinter_dataset=dataset,
        drugbank_index=drugbank_index,
    )

    assert matched_ids == {"DDInter1"}
    assert match_report["drugbank_bridge_matches"]["ASA"] == ["DDInter1"]

    vocab_item_to_names = {
        0: ["ASA"],
        1: ["Warfarin"],
        2: ["Not a DDInter drug"],
    }
    vocab_to_ddinter_ids, vocab_meta = map_vocab_items_to_ddinter_ids(
        vocab_item_to_names,
        ddinter_dataset=dataset,
        drugbank_index=drugbank_index,
    )
    projected_pairs, projection_meta = project_ddinter_pairs_to_vocab(
        vocab_to_ddinter_ids,
        ddinter_dataset=dataset,
    )

    assert vocab_to_ddinter_ids[0] == ("DDInter1",)
    assert vocab_to_ddinter_ids[1] == ("DDInter2",)
    assert vocab_meta["num_vocab_items_matched"] == 2
    assert vocab_meta["num_vocab_items_unmatched"] == 1

    assert len(projected_pairs) == 1
    assert projected_pairs[0].left_item_id == 0
    assert projected_pairs[0].right_item_id == 1
    assert projected_pairs[0].max_severity_level == "Major"
    assert projection_meta["num_ddinter_pair_records_with_vocab_match"] == 1
    assert projection_meta["num_ddinter_pair_records_without_vocab_match"] == 1
