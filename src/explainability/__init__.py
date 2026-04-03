from src.explainability.attention_export import (
    build_attention_payload,
    build_attention_rows,
    save_attention_artifacts,
)
from src.explainability.counterfactual import run_counterfactual_analysis
from src.explainability.nl_explainer import build_nl_explanation

try:
    from src.explainability.person3_analysis import (
        build_case_study_markdown,
        build_case_study_payload,
        build_faithfulness_analysis,
        build_fusion_analysis,
        build_hypergraph_analysis,
        build_plot_payloads,
        build_selection_analysis,
    )
except ImportError:
    build_case_study_markdown = None
    build_case_study_payload = None
    build_faithfulness_analysis = None
    build_fusion_analysis = None
    build_hypergraph_analysis = None
    build_plot_payloads = None
    build_selection_analysis = None

__all__ = [
    "build_attention_payload",
    "build_attention_rows",
    "build_nl_explanation",
    "run_counterfactual_analysis",
    "save_attention_artifacts",
]

if build_case_study_markdown is not None:
    __all__.extend(
        [
            "build_case_study_markdown",
            "build_case_study_payload",
            "build_faithfulness_analysis",
            "build_fusion_analysis",
            "build_hypergraph_analysis",
            "build_plot_payloads",
            "build_selection_analysis",
        ]
    )
