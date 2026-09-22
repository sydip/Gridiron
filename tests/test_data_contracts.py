"""Contract tests for the raw data, and for the messages that report breaches.

Every test here breaks one contract on purpose and checks two things: that the
breach is caught, and that the error says which contract broke. A validator
that raises ``ValueError: invalid input`` is only marginally better than one
that does not raise at all -- whoever hits it still has to go and find out
what happened.

The fixtures are deliberately tiny and fixed. A five-row frame that fails on
exactly one thing localises a bug; a thousand random rows do not.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridiron.data.clean import ScheduleCleaningError, clean_schedule
from gridiron.data.team_names import CANONICAL_TEAMS, normalize_team_names
from gridiron.data.validate import (
    CANONICAL_SCHEDULE_COLUMNS,
    ScheduleValidationError,
    validate_schedule,
)

RAW_COLUMNS = [
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
]


def _raw(**overrides) -> pd.DataFrame:
    """Four valid regular-season games, with overrides applied by column."""
    frame = pd.DataFrame(
        {
            "game_id": [
                "2023_01_BBB_AAA",
                "2023_01_DDD_CCC",
                "2023_02_AAA_BBB",
                "2023_02_CCC_DDD",
            ],
            "season": [2023, 2023, 2023, 2023],
            "week": [1, 1, 2, 2],
            "gameday": ["2023-09-10", "2023-09-10", "2023-09-17", "2023-09-17"],
            "game_type": ["REG", "REG", "REG", "REG"],
            "home_team": ["AAA", "CCC", "BBB", "DDD"],
            "away_team": ["BBB", "DDD", "AAA", "CCC"],
            "home_score": [24.0, 17.0, 20.0, 31.0],
            "away_score": [17.0, 21.0, 27.0, 28.0],
            "spread_line": [3.0, -2.5, 1.0, 7.0],
        }
    )
    for column, values in overrides.items():
        frame[column] = values
    return frame


def _canonical(**overrides) -> pd.DataFrame:
    """A raw frame using real team codes, so validation can run on it.

    Team codes are defaults here, so a caller overriding one replaces it
    rather than colliding with it.
    """
    settings: dict = {
        "home_team": ["KC", "BUF", "MIA", "NYJ"],
        "away_team": ["MIA", "NYJ", "KC", "BUF"],
    }
    settings.update(overrides)
    return _raw(**settings)


# --- required columns ------------------------------------------------------


@pytest.mark.parametrize("column", RAW_COLUMNS)
def test_a_missing_raw_column_is_named_in_the_error(column):
    """The message must say which column, not merely that one is missing."""
    frame = _canonical().drop(columns=column)

    with pytest.raises(ScheduleCleaningError) as error:
        clean_schedule(frame)

    assert column in str(error.value)
    assert "missing required column" in str(error.value).lower()


def test_several_missing_columns_are_all_named():
    frame = _canonical().drop(columns=["spread_line", "gameday"])

    with pytest.raises(ScheduleCleaningError) as error:
        clean_schedule(frame)

    message = str(error.value)
    assert "spread_line" in message
    assert "gameday" in message


def test_the_cleaned_frame_carries_the_agreed_columns():
    cleaned = clean_schedule(_canonical())

    for column in CANONICAL_SCHEDULE_COLUMNS:
        assert column in cleaned.columns


def test_validation_names_a_column_the_contract_requires():
    cleaned = clean_schedule(_canonical()).drop(columns="spread_line")

    with pytest.raises(ScheduleValidationError, match="spread_line"):
        validate_schedule(cleaned)


# --- game identifiers ------------------------------------------------------


def test_game_ids_are_unique_after_cleaning():
    cleaned = clean_schedule(_canonical())

    assert not cleaned["game_id"].duplicated().any()
    assert len(cleaned) == cleaned["game_id"].nunique()


def test_a_repeated_game_keeps_only_the_most_complete_record():
    """Duplicates are collapsed, not carried, and the fuller row survives."""
    frame = pd.concat([_canonical(), _canonical()], ignore_index=True)
    # Make the first copy of one game incomplete.
    frame.loc[0, ["home_score", "away_score", "spread_line"]] = np.nan

    cleaned = clean_schedule(frame)

    assert not cleaned["game_id"].duplicated().any()
    kept = cleaned.loc[cleaned["game_id"].eq("2023_01_BBB_AAA")].iloc[0]
    assert kept["home_score"] == 24.0


def test_duplicate_ids_are_reported_by_the_validator():
    cleaned = clean_schedule(_canonical())
    doubled = pd.concat([cleaned, cleaned.head(1)], ignore_index=True)

    with pytest.raises(ScheduleValidationError, match="duplicate game IDs"):
        validate_schedule(doubled)


# --- team names ------------------------------------------------------------


def test_every_cleaned_team_code_is_canonical():
    cleaned = clean_schedule(_canonical())

    observed = set(cleaned["home_team"]) | set(cleaned["away_team"])
    assert observed.issubset(CANONICAL_TEAMS)


def test_historical_codes_are_mapped_to_current_ones():
    frame = _canonical(
        home_team=["OAK", "SD", "LA", "JAC"],
        away_team=["KC", "BUF", "MIA", "NYJ"],
    )

    cleaned = clean_schedule(frame)

    assert set(cleaned["home_team"]) == {"LV", "LAC", "LAR", "JAX"}


def test_an_unknown_team_code_is_named_in_the_error():
    frame = _canonical(home_team=["ZZZ", "BUF", "MIA", "NYJ"])

    with pytest.raises(ScheduleValidationError) as error:
        clean_schedule(frame)

    assert "ZZZ" in str(error.value)
    assert "unknown team" in str(error.value).lower()


def test_normalisation_leaves_missing_values_alone():
    values = normalize_team_names(pd.Series(["KC", None, "OAK"]))

    assert values.iloc[0] == "KC"
    assert pd.isna(values.iloc[1])
    assert values.iloc[2] == "LV"


def test_normalisation_handles_whitespace_and_case():
    values = normalize_team_names(pd.Series([" kc ", "Buf"]))

    assert list(values) == ["KC", "BUF"]


# --- dates -----------------------------------------------------------------


def test_game_dates_parse_to_timestamps():
    cleaned = clean_schedule(_canonical())

    assert pd.api.types.is_datetime64_any_dtype(cleaned["gameday"])
    assert cleaned["gameday"].notna().all()
    assert cleaned["gameday"].min() == pd.Timestamp("2023-09-10")


def test_an_unparseable_date_is_named_in_the_error():
    frame = _canonical(gameday=["2023-09-10", "not a date", "2023-09-17", "2023-09-17"])

    with pytest.raises(ScheduleCleaningError) as error:
        clean_schedule(frame)

    assert "not a date" in str(error.value)
    assert "gameday" in str(error.value).lower()


def test_an_impossible_date_is_rejected():
    """A day that does not exist must not be coerced into one that does."""
    frame = _canonical(gameday=["2023-09-31", "2023-09-10", "2023-09-17", "2023-09-17"])

    with pytest.raises(ScheduleCleaningError, match="2023-09-31"):
        clean_schedule(frame)


def test_a_missing_date_is_allowed_through():
    """An unscheduled game has no date yet; that is not a data error."""
    frame = _canonical(gameday=["2023-09-10", None, "2023-09-17", "2023-09-17"])

    cleaned = clean_schedule(frame)

    assert cleaned["gameday"].isna().sum() == 1


# --- numeric fields --------------------------------------------------------


@pytest.mark.parametrize(
    "column", ["season", "week", "home_score", "away_score", "spread_line"]
)
def test_a_non_numeric_value_is_named_with_its_column(column):
    values = list(_canonical()[column])
    values[1] = "not a number"
    frame = _canonical(**{column: values})

    with pytest.raises(ScheduleCleaningError) as error:
        clean_schedule(frame)

    message = str(error.value)
    assert column in message
    assert "not a number" in message
    assert "non-numeric" in message.lower()


def test_scores_and_spreads_are_numeric_after_cleaning():
    cleaned = clean_schedule(_canonical())

    for column in ("home_score", "away_score", "spread_line", "home_margin"):
        assert pd.api.types.is_numeric_dtype(cleaned[column])


def test_an_unplayed_game_keeps_null_scores_rather_than_zeros():
    """A game with no result must not be read as nil-nil."""
    frame = _canonical(
        home_score=[24.0, np.nan, 20.0, 31.0],
        away_score=[17.0, np.nan, 27.0, 28.0],
    )

    cleaned = clean_schedule(frame)
    unplayed = cleaned.loc[~cleaned["is_completed"]]

    assert len(unplayed) == 1
    assert unplayed["home_score"].isna().all()
    assert unplayed["away_score"].isna().all()


def test_a_completed_game_missing_a_score_is_reported():
    cleaned = clean_schedule(_canonical())
    cleaned.loc[0, "home_score"] = np.nan
    cleaned.loc[0, "is_completed"] = True

    with pytest.raises(ScheduleValidationError, match="lack one or both scores"):
        validate_schedule(cleaned)


# --- scope -----------------------------------------------------------------


def test_only_regular_season_games_survive_cleaning():
    frame = _canonical(game_type=["REG", "WC", "REG", "DIV"])

    cleaned = clean_schedule(frame)

    assert len(cleaned) == 2
    assert set(cleaned["game_id"]) == {"2023_01_BBB_AAA", "2023_02_AAA_BBB"}
