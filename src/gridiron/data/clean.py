"""Cleaning functions for a canonical, join-safe NFL schedule."""

from __future__ import annotations

import logging

import pandas as pd

from gridiron.data.team_names import needed_team_mappings, normalize_team_names
from gridiron.data.validate import CANONICAL_SCHEDULE_COLUMNS, validate_schedule

LOGGER = logging.getLogger(__name__)

RAW_SCHEDULE_COLUMNS = frozenset(
    {
        "game_id",
        "season",
        "week",
        "gameday",
        "game_type",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "spread_line",
    }
)


class ScheduleCleaningError(ValueError):
    """Raised when raw schedule values cannot be cleaned safely."""


def _require_raw_columns(schedule: pd.DataFrame) -> None:
    missing = sorted(RAW_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise ScheduleCleaningError(
            "Raw schedule is missing required column(s): " + ", ".join(missing)
        )


def _to_numeric(series: pd.Series, column: str) -> pd.Series:
    converted = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & converted.isna()
    if invalid.any():
        examples = sorted(set(series.loc[invalid].astype(str)))[:5]
        raise ScheduleCleaningError(
            f"Column {column!r} contains non-numeric value(s): {examples}"
        )
    return converted


def clean_schedule(schedule: pd.DataFrame) -> pd.DataFrame:
    """Return one canonical row per regular-season game.

    In addition to the core contract, ``is_prediction_ready`` explicitly marks
    unplayed games that currently have a spread line.
    """
    _require_raw_columns(schedule)
    cleaned = schedule.loc[schedule["game_type"].eq("REG")].copy()

    source_dates = cleaned["gameday"]
    cleaned["gameday"] = pd.to_datetime(source_dates, errors="coerce")
    invalid_dates = source_dates.notna() & cleaned["gameday"].isna()
    if invalid_dates.any():
        examples = sorted(set(source_dates.loc[invalid_dates].astype(str)))[:5]
        raise ScheduleCleaningError(f"Invalid gameday value(s): {examples}")

    for column in ("season", "week", "home_score", "away_score", "spread_line"):
        cleaned[column] = _to_numeric(cleaned[column], column)

    observed_teams = pd.concat([cleaned["home_team"], cleaned["away_team"]])
    applied_mappings = needed_team_mappings(observed_teams)
    if applied_mappings:
        LOGGER.info("Applying historical team mappings: %s", applied_mappings)
    cleaned["home_team"] = normalize_team_names(cleaned["home_team"])
    cleaned["away_team"] = normalize_team_names(cleaned["away_team"])

    # Prefer the most complete record when the upstream source repeats a game.
    cleaned["_source_order"] = range(len(cleaned))
    cleaned["_record_completeness"] = (
        cleaned[["home_score", "away_score", "spread_line"]].notna().sum(axis=1)
    )
    cleaned = cleaned.sort_values(
        ["game_id", "_record_completeness", "_source_order"], kind="stable"
    ).drop_duplicates("game_id", keep="last")

    cleaned["home_margin"] = cleaned["home_score"] - cleaned["away_score"]
    cleaned["is_completed"] = (
        cleaned["home_score"].notna() & cleaned["away_score"].notna()
    )
    cleaned["is_prediction_ready"] = (
        ~cleaned["is_completed"] & cleaned["spread_line"].notna()
    )

    cleaned = cleaned.sort_values(
        ["season", "week", "gameday", "game_id"], kind="stable"
    ).reset_index(drop=True)
    result = cleaned[[*CANONICAL_SCHEDULE_COLUMNS, "is_prediction_ready"]].copy()
    validate_schedule(result)
    return result


def clean_schedules(schedule: pd.DataFrame) -> pd.DataFrame:
    """Plural alias for :func:`clean_schedule`."""
    return clean_schedule(schedule)
