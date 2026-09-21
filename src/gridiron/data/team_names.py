"""Canonical NFL team abbreviations."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

TEAM_MAP = {
    "JAC": "JAX",
    "LA": "LAR",
    "STL": "LAR",
    "SD": "LAC",
    "OAK": "LV",
}

CANONICAL_TEAMS = frozenset(
    {
        "ARI",
        "ATL",
        "BAL",
        "BUF",
        "CAR",
        "CHI",
        "CIN",
        "CLE",
        "DAL",
        "DEN",
        "DET",
        "GB",
        "HOU",
        "IND",
        "JAX",
        "KC",
        "LAC",
        "LAR",
        "LV",
        "MIA",
        "MIN",
        "NE",
        "NO",
        "NYG",
        "NYJ",
        "PHI",
        "PIT",
        "SEA",
        "SF",
        "TB",
        "TEN",
        "WAS",
    }
)


def needed_team_mappings(values: Iterable[object]) -> dict[str, str]:
    """Return only historical mappings represented in ``values``."""
    observed = {str(value).strip().upper() for value in values if pd.notna(value)}
    return {old: new for old, new in TEAM_MAP.items() if old in observed}


def normalize_team_name(team: object) -> object:
    """Normalize one team abbreviation while preserving missing values."""
    if pd.isna(team):
        return team
    abbreviation = str(team).strip().upper()
    return TEAM_MAP.get(abbreviation, abbreviation)


def normalize_team_names(teams: pd.Series) -> pd.Series:
    """Normalize a series using only mappings needed by its observed values."""
    normalized = teams.astype("string").str.strip().str.upper()
    mappings = needed_team_mappings(normalized.dropna())
    return normalized.replace(mappings)
