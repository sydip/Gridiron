"""Feature and target construction utilities."""

from gridiron.features.epa import (
    MIN_DEFENSIVE_PLAYS,
    MIN_OFFENSIVE_PLAYS,
    OffensiveEpaError,
    aggregate_defensive_epa,
    aggregate_offensive_epa,
    eligible_plays,
    epa_reconciliation_failures,
    filter_offensive_plays,
    low_volume_team_games,
    reconcile_offense_and_defense,
)
from gridiron.features.pace import (
    MAX_SNAP_INTERVAL_SECONDS,
    MIN_NEUTRAL_INTERVALS,
    MIN_PACE_PLAYS,
    PaceError,
    aggregate_pace,
    flag_pace_outliers,
    pace_distribution_report,
    timing_fields_available,
)
from gridiron.features.target import (
    add_ats_target,
    calculate_adjusted_home_margin,
    classify_ats_result,
)

__all__ = [
    "MAX_SNAP_INTERVAL_SECONDS",
    "MIN_DEFENSIVE_PLAYS",
    "MIN_NEUTRAL_INTERVALS",
    "MIN_OFFENSIVE_PLAYS",
    "MIN_PACE_PLAYS",
    "OffensiveEpaError",
    "PaceError",
    "add_ats_target",
    "aggregate_defensive_epa",
    "aggregate_offensive_epa",
    "aggregate_pace",
    "calculate_adjusted_home_margin",
    "classify_ats_result",
    "eligible_plays",
    "epa_reconciliation_failures",
    "filter_offensive_plays",
    "flag_pace_outliers",
    "low_volume_team_games",
    "pace_distribution_report",
    "reconcile_offense_and_defense",
    "timing_fields_available",
]
