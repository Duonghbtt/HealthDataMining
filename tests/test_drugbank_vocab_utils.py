from __future__ import annotations

from pathlib import Path

from src.data.drugbank_vocab_utils import (
    load_drugbank_vocabulary_index,
    normalize_drugbank_text,
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_normalize_drugbank_text_collapses_spacing_and_simple_dosage_forms() -> None:
    assert normalize_drugbank_text(" Aspirin, oral solution ") == "aspirin"
    assert normalize_drugbank_text("Acetylsalicylic-acid") == "acetylsalicylic acid"


def test_load_drugbank_vocabulary_index_builds_canonical_and_synonym_indexes(tmp_path: Path) -> None:
    csv_path = tmp_path / "drugbank vocabulary.csv"
    _write_text(
        csv_path,
        "\n".join(
            [
                "DrugBank ID,Common name,Synonyms",
                'DB0001,Aspirin,"Acetylsalicylic acid|ASA|Aspirin tablet"',
                'DB0002,Heparin,"Heparin sodium injection|Heparin"',
            ]
        )
        + "\n",
    )

    index = load_drugbank_vocabulary_index(tmp_path)

    assert index.normalized_name_to_drugbank_id["aspirin"] == "DB0001"
    assert index.synonym_to_drugbank_id["asa"] == "DB0001"
    assert index.synonym_to_drugbank_id["heparin sodium"] == "DB0002"
    assert index.drugbank_id_to_canonical_name["DB0002"] == "Heparin"
    assert "aspirin" in index.drugbank_id_to_all_normalized_names["DB0001"]
