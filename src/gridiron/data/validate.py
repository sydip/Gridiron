"""Validation contract for canonical NFL schedules."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from gridiron.data.team_names import CANONICAL_TEAMS

LOGGER = logging.getLogger(__name__)

CANONICAL_SCHEDULE_COLUMNS = (
    "game_id",
    "season",
    "week",
    "gameday",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "spread_line",
    "home_margin",
    "is_completed",
)


class ScheduleValidationError(ValueError):
    """Raised when a schedule violates the canonical data contract."""


@dataclass(frozen=True)
class ScheduleValidationReport:
    """Auditable schedule quality metrics."""

    regular_season_games: dict[int, int]
    missing_spreads: dict[int, int]
    missing_spread_percentage: dict[int, float]
    duplicate_game_ids: tuple[str, ...]
    unknown_teams: tuple[str, ...]
    completed_games_missing_scores: int
    implausible_historical_counts: dict[int, int]

    @property
    def is_valid(self) -> bool:
        return not (
            self.duplicate_game_ids
            or self.unknown_teams
            or self.completed_games_missing_scores
            or self.implausible_historical_counts
        )


def _integer_dict(series: pd.Series) -> dict[int, int]:
    return {int(key): int(value) for key, value in series.items()}


def validate_schedule(
    schedule: pd.DataFrame,
    historical_seasons: list[int] | None = None,
    plausible_game_range: tuple[int, int] = (250, 285),
) -> ScheduleValidationReport:
    """Validate a canonical schedule and return its quality report.

    Historical count validation is enabled when ``historical_seasons`` is
    supplied, allowing small schedule slices to be structurally validated.
    """
    missing_columns = sorted(set(CANONICAL_SCHEDULE_COLUMNS).difference(schedule))
    if missing_columns:
        raise ScheduleValidationError(
            "Canonical schedule is missing required column(s): "
            + ", ".join(missing_columns)
        )

    duplicates = tuple(
        sorted(
            schedule.loc[schedule["game_id"].duplicated(keep=False), "game_id"].astype(
                str
            )
        )
    )
    observed_teams = set(schedule["home_team"].dropna()).union(
        schedule["away_team"].dropna()
    )
    unknown_teams = tuple(sorted(observed_teams.difference(CANONICAL_TEAMS)))
    completed_without_scores = int(
        (
            schedule["is_completed"]
            & (schedule["home_score"].isna() | schedule["away_score"].isna())
        ).sum()
    )

    counts = _integer_dict(schedule.groupby("season", sort=True).size())
    missing_spreads = _integer_dict(
        schedule["spread_line"].isna().groupby(schedule["season"]).sum()
    )
    spread_percentages = {
        int(season): float(percentage)
        for season, percentage in (
            schedule["spread_line"]
            .isna()
            .groupby(schedule["season"])
            .mean()
            .mul(100)
            .items()
        )
    }

    minimum, maximum = plausible_game_range
    historical_seasons = historical_seasons or []
    implausible_counts = {
        season: counts.get(season, 0)
        for season in historical_seasons
        if not minimum <= counts.get(season, 0) <= maximum
    }
    report = ScheduleValidationReport(
        regular_season_games=counts,
        missing_spreads=missing_spreads,
        missing_spread_percentage=spread_percentages,
        duplicate_game_ids=duplicates,
        unknown_teams=unknown_teams,
        completed_games_missing_scores=completed_without_scores,
        implausible_historical_counts=implausible_counts,
    )

    for season, count in missing_spreads.items():
        if count:
            LOGGER.warning(
                "Season %d has %d games without spread lines (%.2f%%); "
                "the games were preserved.",
                season,
                count,
                spread_percentages[season],
            )

    violations = []
    if duplicates:
        violations.append(f"duplicate game IDs: {list(duplicates)}")
    if unknown_teams:
        violations.append(f"unknown team abbreviations: {list(unknown_teams)}")
    if completed_without_scores:
        violations.append(
            f"{completed_without_scores} completed games lack one or both scores"
        )
    if implausible_counts:
        violations.append(
            "implausible historical regular-season counts: "
            f"{implausible_counts} (expected {minimum}–{maximum})"
        )
    if violations:
        raise ScheduleValidationError("; ".join(violations))
    return report
