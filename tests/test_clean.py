"""Tests for canonical schedule cleaning and validation."""

from __future__ import annotations

import pandas as pd
import pytest

from gridiron.data.clean import ScheduleCleaningError, clean_schedule
from gridiron.data.team_names import TEAM_MAP, needed_team_mappings
from gridiron.data.validate import ScheduleValidationError, validate_schedule


def _raw_schedule() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": ["g2", "g1", "g1", "post", "g3"],
            "season": [2026, 2016, 2016, 2016, 2026],
            "week": [2, 1, 1, 19, 1],
            "gameday": [
                "2026-09-10",
                "2016-09-01",
                "2016-09-01",
                "2017-01-01",
                "2026-09-03",
            ],
            "game_type": ["REG", "REG", "REG", "WC", "REG"],
            "home_team": ["LA", "OAK", "OAK", "NE", "KC"],
            "away_team": ["SD", "JAC", "JAC", "MIA", "LV"],
            "home_score": [None, 20, 20, 30, None],
            "away_score": [None, 17, 17, 20, None],
            "spread_line": [None, None, 2.5, 3.0, -1.5],
        }
    )


def test_clean_schedule_preserves_future_games_and_normalizes_teams():
    cleaned = clean_schedule(_raw_schedule())

    assert list(cleaned["game_id"]) == ["g1", "g3", "g2"]
    assert len(cleaned) == 3
    assert cleaned.loc[cleaned["game_id"].eq("g1"), "home_team"].item() == "LV"
    assert cleaned.loc[cleaned["game_id"].eq("g1"), "away_team"].item() == "JAX"
    assert cleaned.loc[cleaned["game_id"].eq("g2"), "home_team"].item() == "LAR"
    assert cleaned.loc[cleaned["game_id"].eq("g2"), "away_team"].item() == "LAC"
    assert cleaned.loc[cleaned["game_id"].eq("g1"), "home_margin"].item() == 3
    assert cleaned.loc[cleaned["game_id"].eq("g1"), "is_completed"].item()
    assert not cleaned.loc[cleaned["game_id"].eq("g2"), "is_prediction_ready"].item()
    assert cleaned.loc[cleaned["game_id"].eq("g3"), "is_prediction_ready"].item()


def test_only_observed_team_mappings_are_selected():
    assert needed_team_mappings(["LA", "SD", "KC"]) == {
        "LA": TEAM_MAP["LA"],
        "SD": TEAM_MAP["SD"],
    }


def test_invalid_numeric_value_fails_clearly():
    schedule = _raw_schedule()
    schedule["spread_line"] = schedule["spread_line"].astype(object)
    schedule.loc[0, "spread_line"] = "not-a-number"

    with pytest.raises(ScheduleCleaningError, match="spread_line.*not-a-number"):
        clean_schedule(schedule)


def test_unknown_team_fails_validation():
    schedule = _raw_schedule()
    schedule.loc[schedule["game_id"].eq("g2"), "home_team"] = "XYZ"

    with pytest.raises(ScheduleValidationError, match="XYZ"):
        clean_schedule(schedule)


def test_historical_game_counts_can_be_enforced():
    cleaned = clean_schedule(_raw_schedule())

    with pytest.raises(ScheduleValidationError, match="implausible historical"):
        validate_schedule(cleaned, historical_seasons=[2016])
