from __future__ import annotations

from pathlib import Path

from src.data.rxnorm_utils import (
    build_minimal_rxnorm_index,
    build_ndc_to_rxcui_map,
    build_rxcui_ingredient_index,
    build_rxcui_name_index,
    load_rxnorm_tables,
)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _build_mock_rxnorm_root(tmp_path: Path) -> Path:
    root = tmp_path / "RxNorm_full_04062026" / "rrf"
    _write_text(
        root / "RXNCONSO.RRF",
        "\n".join(
            [
                "111|ENG||||||AUI111|CODE111|111||RXNORM|SCD|111|Aspirin 81 MG Oral Tablet||N|4096|",
                "111|ENG||||||AUI112|CODE112|111||RXNORM|BN|111|Aspirin||N|4096|",
                "111|ENG||||||AUI113|CODE113|111||MTHSPL|SY|111|ASA||N|4096|",
                "211|ENG||||||AUI211|CODE211|211||RXNORM|IN|211|Acetylsalicylic acid||N|4096|",
                "222|ENG||||||AUI222|CODE222|222||RXNORM|IN|222|Heparin||N|4096|",
                "333|SPA||||||AUI333|CODE333|333||RXNORM|IN|333|Nombre Espanol||N|4096|",
                "BAD|ROW|",
            ]
        )
        + "\n",
    )
    _write_text(
        root / "RXNREL.RRF",
        "\n".join(
            [
                "111|AUI111|SRC|RO|211|AUI211|SRC|has_ingredient|1||RXNORM||||N|",
                "BAD|REL|",
            ]
        )
        + "\n",
    )
    _write_text(
        root / "RXNSAT.RRF",
        "\n".join(
            [
                "111|||AUI111|AUI|11111111111|||NDC|RXNORM|11111111111|N|4096|",
                "999|||AUI999|AUI|11111111111|||NDC|NDDF|11111111111|N|4096|",
                "ABC|||AUIABC|AUI|33333333333|||NDC|RXNORM|33333333333|N|4096|",
                "SHORT|ROW|",
            ]
        )
        + "\n",
    )
    return root.parent


def test_load_rxnorm_tables_resolves_rrf_layout(tmp_path: Path) -> None:
    rxnorm_root = _build_mock_rxnorm_root(tmp_path)

    tables = load_rxnorm_tables(rxnorm_root)

    assert tables.root == rxnorm_root
    assert tables.conso_path.name == "RXNCONSO.RRF"
    assert tables.rel_path.name == "RXNREL.RRF"
    assert tables.sat_path.name == "RXNSAT.RRF"


def test_build_ndc_to_rxcui_map_prefers_high_priority_source_and_reports_malformed_rows(tmp_path: Path) -> None:
    rxnorm_root = _build_mock_rxnorm_root(tmp_path)
    tables = load_rxnorm_tables(rxnorm_root)

    ndc_index = build_ndc_to_rxcui_map(tables)

    assert ndc_index.ndc_to_rxcui["11111111111"] == "111"
    assert "33333333333" not in ndc_index.ndc_to_rxcui
    assert ndc_index.meta["ambiguous_ndc_count"] == 1
    assert ndc_index.meta["rows_malformed"] == 1
    assert ndc_index.meta["rows_with_invalid_rxcui"] == 1


def test_build_rxcui_name_index_collects_preferred_names_and_synonyms(tmp_path: Path) -> None:
    rxnorm_root = _build_mock_rxnorm_root(tmp_path)
    tables = load_rxnorm_tables(rxnorm_root)

    name_index = build_rxcui_name_index(tables)

    aspirin = name_index.rxcui_to_names["111"]
    assert "Aspirin" in aspirin.synonyms
    assert aspirin.normalized_preferred_name == "ASPIRIN_81_MG_ORAL_TABLET"
    assert "ASA" in aspirin.normalized_synonyms
    assert name_index.normalized_name_to_rxcuis["ASPIRIN"] == ("111",)
    assert name_index.normalized_name_to_rxcuis["ASA"] == ("111",)
    assert name_index.meta["rows_non_english"] == 1
    assert name_index.meta["rows_malformed"] == 1


def test_build_rxcui_ingredient_index_maps_products_to_ingredients_and_self_normalizes_ingredient_terms(tmp_path: Path) -> None:
    rxnorm_root = _build_mock_rxnorm_root(tmp_path)
    tables = load_rxnorm_tables(rxnorm_root)
    name_index = build_rxcui_name_index(tables)

    ingredient_index = build_rxcui_ingredient_index(tables, name_index=name_index)

    assert ingredient_index.rxcui_to_ingredient_rxcuis["111"] == ("211",)
    assert ingredient_index.rxcui_to_ingredient_names["111"] == ("Acetylsalicylic acid",)
    assert ingredient_index.rxcui_to_ingredient_rxcuis["222"] == ("222",)
    assert ingredient_index.ingredient_rxcui_to_name["222"] == "Heparin"
    assert ingredient_index.meta["rel_rows_malformed"] == 1


def test_build_minimal_rxnorm_index_aggregates_all_minimal_indexes(tmp_path: Path) -> None:
    rxnorm_root = _build_mock_rxnorm_root(tmp_path)

    minimal_index = build_minimal_rxnorm_index(rxnorm_root)

    assert minimal_index.meta["ndc_entries"] == 1
    assert minimal_index.meta["named_rxcui_entries"] >= 3
    assert minimal_index.meta["ingredient_entries"] >= 2
