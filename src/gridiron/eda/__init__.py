"""Exploratory analysis: charts and correlation review."""

from gridiron.eda.charts import (
    ChartError,
    ats_class_balance,
    build_all_charts,
    correlation_heatmap,
    defensive_epa_by_team_season,
    epa_differential_vs_cover,
    feature_distributions,
    missing_value_heatmap,
    offensive_epa_by_team_season,
    pace_distribution,
    rest_advantage_vs_cover,
    spread_bins_vs_cover,
)
from gridiron.eda.correlation import (
    CORRELATION_THRESHOLD,
    DEFENSIVE_EPA_DECISION,
    CorrelationDecision,
    CorrelationError,
    correlated_pairs,
    correlation_decisions,
    correlation_matrix,
    numeric_fields,
)
from gridiron.eda.theme import apply_theme

__all__ = [
    "CORRELATION_THRESHOLD",
    "DEFENSIVE_EPA_DECISION",
    "ChartError",
    "CorrelationDecision",
    "CorrelationError",
    "apply_theme",
    "ats_class_balance",
    "build_all_charts",
    "correlated_pairs",
    "correlation_decisions",
    "correlation_heatmap",
    "correlation_matrix",
    "defensive_epa_by_team_season",
    "epa_differential_vs_cover",
    "feature_distributions",
    "missing_value_heatmap",
    "numeric_fields",
    "offensive_epa_by_team_season",
    "pace_distribution",
    "rest_advantage_vs_cover",
    "spread_bins_vs_cover",
]
