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
from gridiron.features.target import (
    add_ats_target,
    calculate_adjusted_home_margin,
    classify_ats_result,
)

__all__ = [
    "MIN_DEFENSIVE_PLAYS",
    "MIN_OFFENSIVE_PLAYS",
    "OffensiveEpaError",
    "add_ats_target",
    "aggregate_defensive_epa",
    "aggregate_offensive_epa",
    "calculate_adjusted_home_margin",
    "classify_ats_result",
    "eligible_plays",
    "epa_reconciliation_failures",
    "filter_offensive_plays",
    "low_volume_team_games",
    "reconcile_offense_and_defense",
]
