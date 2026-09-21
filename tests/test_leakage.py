"""Leakage tests for rolling features.

The procedure for each test is the one the feature contract promises:

1. build features from all the data,
2. remove the current game's own values and every later game,
3. rebuild,
4. confirm the selected game's features did not move.

A feature that passes cannot have been reading its own outcome or anything that
happened afterwards. :func:`test_the_leak_detector_actually_detects_a_leak`
runs the same comparison against a deliberately leaky implementation to show the
check has teeth.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gridiron.features.rolling import ROLLING_FEATURES, add_rolling_features

PBP_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "pbp_2016_2025.parquet"
)
SCHEDULE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "schedules_2016_2026.parquet"
)

# Every per-game measurement a rolling feature reads. Blanking these on the
# target row is what "remove the current game" means in practice: the row must
# survive so its features can be read, but it must contribute nothing.
SOURCE_COLUMNS = (
    "offensive_epa_per_play",
    "defensive_epa_strength_per_play",
    "pace_plays_per_game",
    "point_margin",
    "team_spread",
    "covered",
    "win_value",
    "ats_margin",
    "covered_value",
)

TARGET_WEEK = 8


def _synthetic_history(teams: tuple[str, ...] = ("AAA", "BBB")) -> pd.DataFrame:
    """Two teams, two seasons, twelve games each, with distinct values."""
    rows = []
    for team_index, team in enumerate(teams):
        for season in (2022, 2023):
            for week in range(1, 13):
                value = float(week + 10 * team_index + 100 * (season - 2022))
                rows.append(
                    {
                        "team": team,
                        "season": season,
                        "week": week,
                        "game_id": f"{season}_{week:02d}_ZZZ_{team}",
                        "gameday": pd.Timestamp(f"{season}-09-07")
                        + pd.Timedelta(days=7 * (week - 1)),
                        "offensive_epa_per_play": value / 100.0,
                        "defensive_epa_strength_per_play": -value / 100.0,
                        "pace_plays_per_game": 60.0 + value / 10.0,
                        "point_margin": value - 6.0,
                        "team_spread": 1.5,
                        "covered": 1 if week % 2 else 0,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["covered"] = frame["covered"].astype("Int8")
    return frame


def _target_mask(frame: pd.DataFrame, team: str, season: int, week: int) -> pd.Series:
    return frame["team"].eq(team) & frame["season"].eq(season) & frame["week"].eq(week)


def _features_for(frame: pd.DataFrame, mask_args: tuple[str, int, int]) -> pd.Series:
    built = add_rolling_features(frame)
    mask = _target_mask(built, *mask_args)
    # rename(None) drops the row label, which differs between a full frame and
    # a truncated one and is not part of what the comparison is testing.
    return built.loc[mask, list(ROLLING_FEATURES)].iloc[0].rename(None)


def _blank_current_and_drop_future(
    frame: pd.DataFrame, team: str, season: int, week: int
) -> pd.DataFrame:
    """Remove the target's own values and everything that happened after it."""
    target = frame.loc[_target_mask(frame, team, season, week)].iloc[0]
    kept = frame.loc[frame["gameday"].le(target["gameday"])].copy()

    mask = _target_mask(kept, team, season, week)
    for column in SOURCE_COLUMNS:
        if column in kept.columns:
            kept.loc[mask, column] = pd.NA
    return kept


def test_removing_current_and_future_games_leaves_features_unchanged():
    frame = _synthetic_history()
    target = ("AAA", 2023, TARGET_WEEK)

    full = _features_for(frame, target)
    truncated = _features_for(_blank_current_and_drop_future(frame, *target), target)

    pd.testing.assert_series_equal(full, truncated)


def test_future_games_do_not_change_historical_features():
    """Truncating the season after the target must not move its features."""
    frame = _synthetic_history()
    target = ("AAA", 2023, TARGET_WEEK)
    target_date = frame.loc[_target_mask(frame, *target), "gameday"].iloc[0]

    full = _features_for(frame, target)
    truncated = _features_for(frame.loc[frame["gameday"].le(target_date)], target)

    pd.testing.assert_series_equal(full, truncated)


def test_changing_a_future_result_does_not_change_the_past():
    frame = _synthetic_history()
    target = ("AAA", 2023, TARGET_WEEK)

    full = _features_for(frame, target)
    tampered = frame.copy()
    later = (
        tampered["team"].eq("AAA")
        & tampered["season"].eq(2023)
        & tampered["week"].gt(TARGET_WEEK)
    )
    tampered.loc[later, "offensive_epa_per_play"] = 999.0
    tampered.loc[later, "point_margin"] = 999.0

    pd.testing.assert_series_equal(full, _features_for(tampered, target))


def test_changing_the_current_game_does_not_change_its_own_features():
    """The row's own outcome is invisible to the features attached to it."""
    frame = _synthetic_history()
    target = ("AAA", 2023, TARGET_WEEK)

    full = _features_for(frame, target)
    tampered = frame.copy()
    mask = _target_mask(tampered, *target)
    tampered.loc[mask, "offensive_epa_per_play"] = 999.0
    tampered.loc[mask, "pace_plays_per_game"] = 999.0
    tampered.loc[mask, "point_margin"] = 999.0
    tampered.loc[mask, "covered"] = 0

    pd.testing.assert_series_equal(full, _features_for(tampered, target))


def test_previous_season_does_not_reach_into_this_one():
    frame = _synthetic_history()
    target = ("AAA", 2023, 3)

    full = _features_for(frame, target)
    without_history = _features_for(frame.loc[frame["season"].eq(2023)], target)

    pd.testing.assert_series_equal(full, without_history)


def test_the_leak_detector_actually_detects_a_leak():
    """The same comparison must fail for a window that omits the shift.

    Without this, every leakage test above could be passing vacuously. The
    deliberately wrong implementation below is the one the phase brief warns
    against, and the comparison catches it.
    """

    def _build(frame: pd.DataFrame) -> pd.Series:
        ordered = frame.sort_values(
            ["team", "season", "gameday"], kind="stable"
        ).reset_index(drop=True)
        grouped = ordered.groupby(["team", "season"], sort=False)
        # Deliberately wrong: no shift, so the current game is in its own window.
        ordered["off_epa_last_3"] = grouped["offensive_epa_per_play"].transform(
            lambda series: series.rolling(3, min_periods=1).mean()
        )
        mask = _target_mask(ordered, "AAA", 2023, TARGET_WEEK)
        return ordered.loc[mask, ["off_epa_last_3"]].iloc[0]

    frame = _synthetic_history()
    full = _build(frame)
    blanked = _build(_blank_current_and_drop_future(frame, "AAA", 2023, TARGET_WEEK))

    assert full["off_epa_last_3"] != pytest.approx(blanked["off_epa_last_3"])


def test_every_rolling_feature_is_covered_by_the_comparison():
    """Guards against a new feature being added without a leakage check."""
    frame = _synthetic_history()
    built = add_rolling_features(frame)

    assert set(ROLLING_FEATURES).issubset(built.columns)
    assert len(ROLLING_FEATURES) == 14


@pytest.fixture(scope="module")
def real_features() -> pd.DataFrame:
    if not (PBP_PATH.exists() and SCHEDULE_PATH.exists()):
        pytest.skip("raw parquet data is not available")

    from gridiron.data.clean import clean_schedules
    from gridiron.features.epa import (
        aggregate_defensive_epa,
        aggregate_offensive_epa,
    )
    from gridiron.features.pace import aggregate_pace
    from gridiron.features.rolling import add_derived_sources, assemble_team_game_inputs
    from gridiron.features.team_games import team_game_history

    columns = [
        "season",
        "week",
        "game_id",
        "posteam",
        "defteam",
        "play_type",
        "epa",
        "success",
        "pass",
        "rush",
        "qb_kneel",
        "qb_spike",
        "game_seconds_remaining",
        "half_seconds_remaining",
        "no_huddle",
        "fixed_drive",
        "play_id",
        "score_differential",
    ]
    pbp = pd.read_parquet(PBP_PATH, columns=columns)
    history = team_game_history(clean_schedules(pd.read_parquet(SCHEDULE_PATH)))
    assembled = assemble_team_game_inputs(
        history,
        aggregate_offensive_epa(pbp),
        aggregate_defensive_epa(pbp),
        aggregate_pace(pbp),
    )
    return add_derived_sources(assembled)


def _nth_game(features: pd.DataFrame, team: str, season: int, n: int) -> tuple:
    """The team's nth game of a season, by date.

    Picking by position rather than by week number avoids depending on a team
    having played in any particular week -- a bye makes that assumption false.
    """
    played = (
        features.loc[features["team"].eq(team) & features["season"].eq(season)]
        .sort_values(["gameday", "game_id"])
        .reset_index(drop=True)
    )
    assert len(played) > n, f"{team} {season} has no game {n}"
    return (team, season, int(played.loc[n, "week"]))


def test_real_target_game_survives_truncation(real_features):
    target = _nth_game(real_features, "KC", 2023, 9)

    full = _features_for(real_features, target)
    truncated = _features_for(
        _blank_current_and_drop_future(real_features, *target), target
    )

    pd.testing.assert_series_equal(full, truncated)


def test_real_features_are_stable_across_several_teams(real_features):
    for team, season, index in (
        ("BUF", 2022, 11),
        ("SF", 2021, 8),
        ("LAR", 2018, 5),
        ("NYJ", 2024, 13),
    ):
        target = _nth_game(real_features, team, season, index)
        full = _features_for(real_features, target)
        truncated = _features_for(
            _blank_current_and_drop_future(real_features, *target), target
        )
        pd.testing.assert_series_equal(full, truncated)


def test_real_week_one_features_are_empty(real_features):
    built = add_rolling_features(real_features)
    openers = built.loc[built["prior_games_this_season"].eq(0)]

    assert len(openers) > 300
    assert built.loc[openers.index, list(ROLLING_FEATURES)].isna().all().all()


def test_real_season_to_date_never_crosses_a_season(real_features):
    """Rebuilding one season alone must reproduce that season's features."""
    built = add_rolling_features(real_features)
    one_season = add_rolling_features(
        real_features.loc[real_features["season"].eq(2023)]
    )

    columns = ["team", "season", "week", "game_id", *ROLLING_FEATURES]
    left = (
        built.loc[built["season"].eq(2023), columns]
        .sort_values(["team", "week", "game_id"])
        .reset_index(drop=True)
    )
    right = (
        one_season[columns]
        .sort_values(["team", "week", "game_id"])
        .reset_index(drop=True)
    )
    pd.testing.assert_frame_equal(left, right)
