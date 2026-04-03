from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Mapping

from src.utils.io import ensure_dir, read_json, resolve_path, write_csv_gz, write_json


ABLATION_ORDER = (
    "Base",
    "TempSim",
    "SelfSel",
    "NbrSel",
    "NoAttrGate",
    "NoGroupAware",
    "NoFusionReg",
    "Full Core",
    "Extended",
)
PRIMARY_METRIC_CANDIDATES = ("jaccard", "prauc", "f1", "roc_auc")
PERSON3_VARIANT_ALIASES = {
    "attribute_gate_on": ("AttrGate", "AttributeGate", "Full Core", "Extended"),
    "attribute_gate_off": ("NoAttrGate", "WithoutAttrGate", "No Attribute Gate"),
    "group_aware_on": ("GroupAware", "GroupAwareReweight", "Full Core", "Extended"),
    "group_aware_off": ("NoGroupAware", "WithoutGroupAware", "No Group Aware"),
    "fusion_reg_on": ("FusionReg", "FusionRegularized", "Full Core", "Extended"),
    "fusion_reg_off": ("NoFusionReg", "WithoutFusionReg", "FusionUnregularized"),
}


def _normalize_name(name: str) -> str:
    return name.strip().lower().replace("_", "").replace("-", "").replace(" ", "")


def _display_name(name: str) -> str:
    lookup = {_normalize_name(value): value for value in ABLATION_ORDER}
    return lookup.get(_normalize_name(name), name)


def _primary_metric(metrics_by_variant: Mapping[str, Mapping[str, float]]) -> str:
    available = {
        metric_name
        for payload in metrics_by_variant.values()
        for metric_name in payload.keys()
        if isinstance(payload[metric_name], (int, float))
    }
    for metric_name in PRIMARY_METRIC_CANDIDATES:
        if metric_name in available:
            return metric_name
    if not available:
        raise ValueError("No numeric metrics found for ablation report")
    return sorted(available)[0]


def _metric_summary(value: Any) -> tuple[float, float, int] | None:
    if isinstance(value, (int, float)):
        return float(value), 0.0, 1
    if isinstance(value, list) and value and all(isinstance(item, (int, float)) for item in value):
        numbers = [float(item) for item in value]
        mean_value = sum(numbers) / float(len(numbers))
        variance = sum((item - mean_value) ** 2 for item in numbers) / float(len(numbers))
        return mean_value, math.sqrt(variance), len(numbers)
    return None


def _metric_series(value: Any) -> list[float] | None:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, list) and value and all(isinstance(item, (int, float)) for item in value):
        return [float(item) for item in value]
    return None


def _normalize_metrics(payload: Mapping[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, value in payload.items():
        summary = _metric_summary(value)
        if summary is None:
            continue
        mean_value, std_value, num_seeds = summary
        normalized[key] = mean_value
        if num_seeds > 1:
            normalized[f"{key}_std"] = std_value
            normalized[f"{key}_num_seeds"] = float(num_seeds)
    return normalized


def _extract_metric_series(payload: Mapping[str, Any]) -> dict[str, list[float]]:
    series: dict[str, list[float]] = {}
    for key, value in payload.items():
        metric_values = _metric_series(value)
        if metric_values is not None:
            series[key] = metric_values
    return series


def _comparison_summary(
    left_name: str,
    right_name: str,
    metric_name: str,
    series_by_variant: Mapping[str, Mapping[str, list[float]]],
) -> dict[str, Any]:
    left_series = series_by_variant.get(left_name, {}).get(metric_name, [])
    right_series = series_by_variant.get(right_name, {}).get(metric_name, [])
    if not left_series or not right_series:
        return {
            "left": left_name,
            "right": right_name,
            "metric": metric_name,
            "mean_delta": None,
            "win_rate": None,
            "effect_size": None,
        }
    paired_count = min(len(left_series), len(right_series))
    paired_left = left_series[:paired_count]
    paired_right = right_series[:paired_count]
    deltas = [left - right for left, right in zip(paired_left, paired_right)]
    mean_delta = sum(deltas) / float(len(deltas))
    win_rate = sum(1 for delta in deltas if delta > 0.0) / float(len(deltas))
    pooled_values = paired_left + paired_right
    pooled_mean = sum(pooled_values) / float(len(pooled_values))
    pooled_variance = sum((value - pooled_mean) ** 2 for value in pooled_values) / float(len(pooled_values))
    pooled_std = math.sqrt(pooled_variance)
    return {
        "left": left_name,
        "right": right_name,
        "metric": metric_name,
        "mean_delta": mean_delta,
        "win_rate": win_rate,
        "effect_size": 0.0 if pooled_std == 0.0 else mean_delta / pooled_std,
    }


def _first_metric_value(
    metrics_by_variant: Mapping[str, Mapping[str, float]],
    *,
    metric_name: str,
    variant_names: tuple[str, ...],
) -> float | None:
    normalized_lookup = {_normalize_name(name): payload for name, payload in metrics_by_variant.items()}
    for variant_name in variant_names:
        payload = normalized_lookup.get(_normalize_name(variant_name))
        if payload is None:
            continue
        value = payload.get(metric_name)
        if isinstance(value, (int, float)):
            return float(value)
    return None


def _delta_if_present(
    metrics_by_variant: Mapping[str, Mapping[str, float]],
    *,
    metric_name: str,
    left_aliases: tuple[str, ...],
    right_aliases: tuple[str, ...],
) -> float | None:
    left_value = _first_metric_value(metrics_by_variant, metric_name=metric_name, variant_names=left_aliases)
    right_value = _first_metric_value(metrics_by_variant, metric_name=metric_name, variant_names=right_aliases)
    if left_value is None or right_value is None:
        return None
    return left_value - right_value


def _positive_if_present(
    metrics_by_variant: Mapping[str, Mapping[str, float]],
    *,
    metric_name: str,
    left_aliases: tuple[str, ...],
    right_aliases: tuple[str, ...],
) -> bool | None:
    delta = _delta_if_present(
        metrics_by_variant,
        metric_name=metric_name,
        left_aliases=left_aliases,
        right_aliases=right_aliases,
    )
    if delta is None:
        return None
    return delta > 0.0


def build_ablation_summary(metrics_by_variant: Mapping[str, Mapping[str, float]]) -> dict[str, Any]:
    resolved = {_display_name(name): _normalize_metrics(payload) for name, payload in metrics_by_variant.items()}
    series_by_variant = {_display_name(name): _extract_metric_series(payload) for name, payload in metrics_by_variant.items()}
    metric_name = _primary_metric(resolved)
    ordered_variants = [name for name in ABLATION_ORDER if name in resolved]
    extra_variants = sorted(name for name in resolved.keys() if name not in ABLATION_ORDER)
    variant_names = [*ordered_variants, *extra_variants]
    rows = []
    base_value = float(resolved.get("Base", {}).get(metric_name, 0.0))
    full_core_value = float(resolved.get("Full Core", {}).get(metric_name, 0.0))

    for variant_name in variant_names:
        payload = resolved.get(variant_name)
        if payload is None:
            continue
        numeric_metrics = {
            key: float(value)
            for key, value in payload.items()
            if isinstance(value, (int, float))
        }
        primary_value = float(numeric_metrics.get(metric_name, 0.0))
        rows.append(
            {
                "variant": variant_name,
                "primary_metric": metric_name,
                "primary_value": primary_value,
                "primary_std": float(numeric_metrics.get(f"{metric_name}_std", 0.0)),
                "num_seeds": int(numeric_metrics.get(f"{metric_name}_num_seeds", 1.0)),
                "delta_vs_base": primary_value - base_value,
                "delta_vs_full_core": primary_value - full_core_value,
                **numeric_metrics,
            }
        )

    questions = {
        "selection_beats_tempsim": (
            float(resolved.get("SelfSel", {}).get(metric_name, float("-inf"))) > float(resolved.get("TempSim", {}).get(metric_name, float("-inf")))
            or float(resolved.get("NbrSel", {}).get(metric_name, float("-inf"))) > float(resolved.get("TempSim", {}).get(metric_name, float("-inf")))
            or float(resolved.get("Full Core", {}).get(metric_name, float("-inf"))) > float(resolved.get("TempSim", {}).get(metric_name, float("-inf")))
        ),
        "hypergraph_beats_full_core": float(resolved.get("Extended", {}).get(metric_name, float("-inf"))) > float(
            resolved.get("Full Core", {}).get(metric_name, float("-inf"))
        ),
        "attribute_gate_helps": _positive_if_present(
            resolved,
            metric_name=metric_name,
            left_aliases=PERSON3_VARIANT_ALIASES["attribute_gate_on"],
            right_aliases=PERSON3_VARIANT_ALIASES["attribute_gate_off"],
        ),
        "group_aware_helps": _positive_if_present(
            resolved,
            metric_name=metric_name,
            left_aliases=PERSON3_VARIANT_ALIASES["group_aware_on"],
            right_aliases=PERSON3_VARIANT_ALIASES["group_aware_off"],
        ),
        "fusion_reg_helps": _positive_if_present(
            resolved,
            metric_name=metric_name,
            left_aliases=PERSON3_VARIANT_ALIASES["fusion_reg_on"],
            right_aliases=PERSON3_VARIANT_ALIASES["fusion_reg_off"],
        ),
    }
    diagnostics = {
        "history_selection": {
            "self_only_vs_neighbor_only": float(resolved.get("SelfSel", {}).get(metric_name, float("-inf")))
            - float(resolved.get("NbrSel", {}).get(metric_name, float("-inf"))),
            "full_core_vs_tempsim": float(resolved.get("Full Core", {}).get(metric_name, float("-inf")))
            - float(resolved.get("TempSim", {}).get(metric_name, float("-inf"))),
        },
        "fusion": {
            "gated_vs_concat": float(resolved.get("FusionGated", {}).get(metric_name, float("-inf")))
            - float(resolved.get("FusionConcat", {}).get(metric_name, float("-inf"))),
            "gated_vs_mean": float(resolved.get("FusionGated", {}).get(metric_name, float("-inf")))
            - float(resolved.get("FusionMean", {}).get(metric_name, float("-inf"))),
        },
        "hypergraph": {
            "weighted_vs_unweighted": float(resolved.get("HypergraphWeighted", {}).get(metric_name, float("-inf")))
            - float(resolved.get("HypergraphUnweighted", {}).get(metric_name, float("-inf"))),
            "extended_vs_full_core": float(resolved.get("Extended", {}).get(metric_name, float("-inf")))
            - float(resolved.get("Full Core", {}).get(metric_name, float("-inf"))),
        },
        "person3": {
            "attribute_gate_vs_off": _delta_if_present(
                resolved,
                metric_name=metric_name,
                left_aliases=PERSON3_VARIANT_ALIASES["attribute_gate_on"],
                right_aliases=PERSON3_VARIANT_ALIASES["attribute_gate_off"],
            ),
            "group_aware_vs_off": _delta_if_present(
                resolved,
                metric_name=metric_name,
                left_aliases=PERSON3_VARIANT_ALIASES["group_aware_on"],
                right_aliases=PERSON3_VARIANT_ALIASES["group_aware_off"],
            ),
            "fusion_reg_vs_off": _delta_if_present(
                resolved,
                metric_name=metric_name,
                left_aliases=PERSON3_VARIANT_ALIASES["fusion_reg_on"],
                right_aliases=PERSON3_VARIANT_ALIASES["fusion_reg_off"],
            ),
        },
    }
    comparisons = {
        "selection_vs_tempsim": _comparison_summary("Full Core", "TempSim", metric_name, series_by_variant),
        "hypergraph_vs_full_core": _comparison_summary("Extended", "Full Core", metric_name, series_by_variant),
        "self_vs_neighbor": _comparison_summary("SelfSel", "NbrSel", metric_name, series_by_variant),
        "fusion_gated_vs_concat": _comparison_summary("FusionGated", "FusionConcat", metric_name, series_by_variant),
        "fusion_gated_vs_mean": _comparison_summary("FusionGated", "FusionMean", metric_name, series_by_variant),
        "hypergraph_weighted_vs_unweighted": _comparison_summary(
            "HypergraphWeighted",
            "HypergraphUnweighted",
            metric_name,
            series_by_variant,
        ),
        "attribute_gate_vs_off": _comparison_summary("Full Core", "NoAttrGate", metric_name, series_by_variant),
        "group_aware_vs_off": _comparison_summary("Full Core", "NoGroupAware", metric_name, series_by_variant),
        "fusion_reg_vs_off": _comparison_summary("Full Core", "NoFusionReg", metric_name, series_by_variant),
    }
    return {
        "primary_metric": metric_name,
        "rows": rows,
        "questions": questions,
        "diagnostics": diagnostics,
        "comparisons": comparisons,
    }


def report_dir(project_root: str | Path) -> Path:
    return ensure_dir(resolve_path(project_root, "outputs/reports"))


def save_ablation_report(
    project_root: str | Path,
    metrics_by_variant: Mapping[str, Mapping[str, float]],
    *,
    report_name: str = "ablation",
) -> dict[str, Path]:
    summary = build_ablation_summary(metrics_by_variant)
    output_dir = report_dir(project_root)
    json_path = write_json(output_dir / f"{report_name}.json", summary)
    csv_path = write_csv_gz(
        output_dir / f"{report_name}.csv.gz",
        summary["rows"],
        fieldnames=sorted({key for row in summary["rows"] for key in row.keys()}),
    )
    markdown_lines = [
        f"# Ablation Summary ({summary['primary_metric']})",
        "",
        f"- selection_beats_tempsim: {summary['questions']['selection_beats_tempsim']}",
        f"- hypergraph_beats_full_core: {summary['questions']['hypergraph_beats_full_core']}",
        "",
        "| Variant | Primary | Std | Seeds | Delta vs Base | Delta vs Full Core |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary["rows"]:
        markdown_lines.append(
            f"| {row['variant']} | {row['primary_value']:.6f} | {row['primary_std']:.6f} | {row['num_seeds']} | {row['delta_vs_base']:.6f} | {row['delta_vs_full_core']:.6f} |"
        )
    markdown_lines.extend(
        [
            "",
            "## Diagnostics",
            "",
            f"- history.self_only_vs_neighbor_only: {summary['diagnostics']['history_selection']['self_only_vs_neighbor_only']:.6f}",
            f"- history.full_core_vs_tempsim: {summary['diagnostics']['history_selection']['full_core_vs_tempsim']:.6f}",
            f"- fusion.gated_vs_concat: {summary['diagnostics']['fusion']['gated_vs_concat']:.6f}",
            f"- fusion.gated_vs_mean: {summary['diagnostics']['fusion']['gated_vs_mean']:.6f}",
            f"- hypergraph.weighted_vs_unweighted: {summary['diagnostics']['hypergraph']['weighted_vs_unweighted']:.6f}",
            f"- hypergraph.extended_vs_full_core: {summary['diagnostics']['hypergraph']['extended_vs_full_core']:.6f}",
            f"- person3.attribute_gate_vs_off: {summary['diagnostics']['person3']['attribute_gate_vs_off']}",
            f"- person3.group_aware_vs_off: {summary['diagnostics']['person3']['group_aware_vs_off']}",
            f"- person3.fusion_reg_vs_off: {summary['diagnostics']['person3']['fusion_reg_vs_off']}",
            "",
            "## Seed Comparisons",
            "",
        ]
    )
    for comparison_name, payload in summary["comparisons"].items():
        markdown_lines.append(
            f"- {comparison_name}: mean_delta={payload['mean_delta']} win_rate={payload['win_rate']} effect_size={payload['effect_size']}"
        )
    markdown_path = output_dir / f"{report_name}.md"
    markdown_path.write_text("\n".join(markdown_lines), encoding="utf-8")
    return {"json": json_path, "csv": csv_path, "markdown": markdown_path}


def _load_metrics_from_dir(input_dir: Path) -> dict[str, dict[str, float]]:
    payload = {}
    for path in sorted(input_dir.glob("*.json")):
        if path.name.startswith("ablation"):
            continue
        data = read_json(path)
        if isinstance(data, dict):
            payload[_display_name(path.stem)] = {
                key: value
                for key, value in data.items()
                if isinstance(value, (int, float))
            }
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate ablation metrics into JSON/CSV/Markdown artifacts.")
    parser.add_argument("--project-root", default=".", help="Project root containing outputs/reports.")
    parser.add_argument(
        "--input-dir",
        default="outputs/reports/ablation_inputs",
        help="Directory of JSON metric payloads such as Base.json or Full Core.json.",
    )
    parser.add_argument("--report-name", default="ablation", help="Stem for the output report files.")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    input_dir = resolve_path(project_root, args.input_dir)
    metrics_by_variant = _load_metrics_from_dir(input_dir)
    if not metrics_by_variant:
        raise FileNotFoundError(f"No ablation metric JSON files found in {input_dir}")
    save_ablation_report(project_root, metrics_by_variant, report_name=args.report_name)


if __name__ == "__main__":
    main()
