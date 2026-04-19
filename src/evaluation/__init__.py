from __future__ import annotations

from importlib import import_module
from typing import Any

from src.evaluation.metrics import (
    binarize_predictions,
    compute_core_metrics,
    compute_ddi_flags,
    compute_ddi_rate,
    compute_samplewise_f1,
    compute_samplewise_jaccard,
    multilabel_f1,
    multilabel_jaccard,
    multilabel_prauc,
)


_OPTIONAL_EXPORTS = {
    "ABLATION_ORDER": ("src.evaluation.evaluate_ablation", "ABLATION_ORDER"),
    "build_ablation_summary": ("src.evaluation.evaluate_ablation", "build_ablation_summary"),
    "save_ablation_report": ("src.evaluation.evaluate_ablation", "save_ablation_report"),
    "build_demo_pipeline_inputs": ("src.evaluation.evaluate_person3", "build_demo_pipeline_inputs"),
    "run_person3_analysis": ("src.evaluation.evaluate_person3", "run_person3_analysis"),
    "load_baseline_results": ("src.evaluation.baseline_results", "load_baseline_results"),
    "normalize_baseline_payload": ("src.evaluation.baseline_results", "normalize_baseline_payload"),
}


def __getattr__(name: str) -> Any:
    if name not in _OPTIONAL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _OPTIONAL_EXPORTS[name]
    module = import_module(module_name)
    return getattr(module, attr_name)


__all__ = [
    "binarize_predictions",
    "compute_core_metrics",
    "compute_ddi_flags",
    "compute_ddi_rate",
    "compute_samplewise_f1",
    "compute_samplewise_jaccard",
    "multilabel_f1",
    "multilabel_jaccard",
    "multilabel_prauc",
    *sorted(_OPTIONAL_EXPORTS.keys()),
]
