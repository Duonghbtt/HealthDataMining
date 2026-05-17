from __future__ import annotations

from src.features.lab_processor import NumericFeatureEncoder, NumericFeatureProcessor


class VitalProcessor(NumericFeatureProcessor):
    """Kept for optional lab/vital pipeline support."""


class VitalFeatureEncoder(NumericFeatureEncoder):
    """Numeric vital-sign branch that returns visit-level vital embeddings."""
