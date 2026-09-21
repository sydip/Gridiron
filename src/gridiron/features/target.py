"""Against-the-spread target construction."""

from __future__ import annotations

import math

import pandas as pd

HOME_COVER = "HOME_COVER"
AWAY_COVER = "AWAY_COVER"
PUSH = "PUSH"
UNRESOLVED = "UNRESOLVED"

TARGET_SOURCE_COLUMNS = frozenset({"home_score", "away_score", "spread_line"})


class AtsTargetError(ValueError):
    """Raised when a schedule cannot safely produce ATS labels."""


def calculate_adjusted_home_margin(
    home_score: float,
    away_score: float,
    spread_line: float,
) -> float:
    """Return the home margin after applying nflverse's home spread.

    nflverse represents a home favorite with a positive spread and a home
    underdog with a negative spread, so the line is subtracted from the raw
    home margin.
    """
    if any(pd.isna(value) for value in (home_score, away_score, spread_line)):
        return float("nan")
    return float(home_score) - float(away_score) - float(spread_line)


def classify_ats_result(adjusted_home_margin: float) -> str:
    """Classify an adjusted margin as home cover, away cover, push, or unknown."""
    if pd.isna(adjusted_home_margin):
        return UNRESOLVED
    margin = float(adjusted_home_margin)
    if math.isclose(margin, 0.0, abs_tol=1e-9):
        return PUSH
    return HOME_COVER if margin > 0 else AWAY_COVER


def _numeric_column(schedules: pd.DataFrame, column: str) -> pd.Series:
    converted = pd.to_numeric(schedules[column], errors="coerce")
    invalid = schedules[column].notna() & converted.isna()
    if invalid.any():
        examples = sorted(set(schedules.loc[invalid, column].astype(str)))[:5]
        raise AtsTargetError(
            f"Cannot build ATS target: {column!r} contains non-numeric "
            f"value(s): {examples}"
        )
    return converted


def add_ats_target(schedules: pd.DataFrame) -> pd.DataFrame:
    """Add nullable against-the-spread outcomes to a schedule.

    ``home_cover`` uses pandas' nullable integer dtype: 1 for a home cover, 0
    for an away cover, and null for both pushes and unresolved games.
    """
    missing = sorted(TARGET_SOURCE_COLUMNS.difference(schedules.columns))
    if missing:
        raise AtsTargetError(
            "Cannot build ATS target; missing required column(s): " + ", ".join(missing)
        )

    result = schedules.copy()
    home_score = _numeric_column(result, "home_score")
    away_score = _numeric_column(result, "away_score")
    spread_line = _numeric_column(result, "spread_line")

    result["home_margin"] = home_score - away_score
    result["adjusted_home_margin"] = result["home_margin"] - spread_line
    result["ats_result"] = result["adjusted_home_margin"].map(classify_ats_result)
    result["home_cover"] = (
        result["ats_result"].map({HOME_COVER: 1, AWAY_COVER: 0}).astype("Int8")
    )
    result["is_push"] = result["ats_result"].eq(PUSH)
    return result
