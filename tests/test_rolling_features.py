"""Tests for rolling team form features."""

from __future__ import annotations

import pandas as pd
import pytest

from gridiron.features.rolling import (
    EARLY_SEASON_WEEKS,
    MIN_PERIODS,
    ROLLING_FEATURES,
    RollingFeatureError,
    RollingSpec,
    add_derived_sources,
    add_rolling_features,
    assemble_team_game_inputs,
    rolling_feature_coverage,
)

REQUIRED_FEATURES = (
    "off_epa_last_3",
    "off_epa_last_5",
    "off_epa_season",
    "def_epa_strength_last_3",
    "def_epa_strength_last_5",
    "def_epa_strength_season",
    "pace_last_3",
    "pace_last_5",
    "pace_season",
    "point_margin_last_3",
    "point_margin_last_5",
    "win_pct_last_5",
    "ats_margin_last_5",
    "cover_rate_last_5",
)


def _team_games(
    team: str = "AAA",
    season: int = 2023,
    *,
    weeks: list[int] | None = None,
    offensive_epa: list[float] | None = None,
    pace: list[float] | None = None,
    point_margin: list[float] | None = None,
    team_spread: list[float] | None = None,
) -> pd.DataFrame:
    """A run of games for one team, with controllable per-game values.

    The number of games follows whichever value list is supplied, so a caller
    only has to state the series it cares about.
    """
    supplied = [
        values
        for values in (offensive_epa, pace, point_margin, team_spread)
        if values is not None
    ]
    if weeks is None:
        weeks = list(range(1, len(supplied[0]) + 1)) if supplied else [1, 2, 3, 4, 5, 6]
    count = len(weeks)
    for values in supplied:
        assert len(values) == count, "value lists must match the number of weeks"

    default = [float(index) for index in range(count)]
    values = offensive_epa if offensive_epa is not None else default
    return pd.DataFrame(
        {
            "team": team,
            "season": season,
            "week": weeks,
            "game_id": [f"{season}_{week:02d}_ZZZ_{team}" for week in weeks],
            "gameday": [
                pd.Timestamp(f"{season}-09-07") + pd.Timedelta(days=7 * index)
                for index in range(count)
            ],
            "offensive_epa_per_play": [float(value) for value in values],
            "defensive_epa_strength_per_play": [float(value) for value in values],
            "pace_plays_per_game": [
                float(value) for value in (pace if pace is not None else values)
            ],
            "point_margin": [
                float(value)
                for value in (point_margin if point_margin is not None else values)
            ],
            "team_spread": [
                float(value)
                for value in (team_spread if team_spread is not None else [0.0] * count)
            ],
            "covered": pd.array([pd.NA] * count, dtype="Int8"),
        }
    )


def _feature(frame: pd.DataFrame, week: int, column: str) -> float:
    return frame.loc[frame["week"].eq(week), column].iloc[0]


def test_all_required_features_are_produced():
    features = add_rolling_features(_team_games())

    assert set(REQUIRED_FEATURES).issubset(features.columns)
    assert set(REQUIRED_FEATURES) == set(ROLLING_FEATURES)


def test_current_game_is_excluded_from_its_own_feature():
    """Week 2's value must be week 1's number alone, not an average with itself."""
    features = add_rolling_features(
        _team_games(offensive_epa=[10.0, 99.0, 0.0, 0.0, 0.0, 0.0])
    )

    assert _feature(features, 2, "off_epa_last_3") == 10.0
    assert _feature(features, 2, "off_epa_season") == 10.0


def test_week_one_has_no_rolling_value():
    features = add_rolling_features(_team_games())

    for column in REQUIRED_FEATURES:
        assert pd.isna(_feature(features, 1, column))


def test_min_periods_allows_a_value_from_a_single_prior_game():
    features = add_rolling_features(_team_games(offensive_epa=[4.0, 0.0, 0.0]))

    assert MIN_PERIODS == 1
    assert _feature(features, 2, "off_epa_last_5") == 4.0


def test_short_window_forgets_games_beyond_its_length():
    """last_3 at week 5 averages weeks 2-4 and must not see week 1."""
    features = add_rolling_features(
        _team_games(offensive_epa=[100.0, 1.0, 2.0, 3.0, 0.0])
    )

    assert _feature(features, 5, "off_epa_last_3") == pytest.approx(2.0)


def test_long_window_averages_the_previous_five_games():
    features = add_rolling_features(
        _team_games(
            weeks=[1, 2, 3, 4, 5, 6, 7],
            offensive_epa=[100.0, 1.0, 2.0, 3.0, 4.0, 5.0, 0.0],
        )
    )

    assert _feature(features, 7, "off_epa_last_5") == pytest.approx(3.0)


def test_season_to_date_expands_over_every_prior_game():
    features = add_rolling_features(
        _team_games(offensive_epa=[1.0, 2.0, 3.0, 4.0, 0.0])
    )

    assert _feature(features, 5, "off_epa_season") == pytest.approx(2.5)


def test_season_to_date_resets_at_the_start_of_each_season():
    """A new season starts empty rather than inheriting last season's form."""
    first = _team_games(season=2022, offensive_epa=[50.0, 50.0, 50.0])
    second = _team_games(season=2023, offensive_epa=[1.0, 2.0, 3.0])
    features = add_rolling_features(pd.concat([first, second], ignore_index=True))
    new_season = features.loc[features["season"].eq(2023)]

    assert pd.isna(_feature(new_season, 1, "off_epa_season"))
    assert _feature(new_season, 2, "off_epa_season") == 1.0
    assert _feature(new_season, 3, "off_epa_season") == pytest.approx(1.5)


def test_fixed_windows_also_reset_across_seasons():
    first = _team_games(season=2022, offensive_epa=[50.0, 50.0, 50.0])
    second = _team_games(season=2023, offensive_epa=[1.0, 2.0, 3.0])
    features = add_rolling_features(pd.concat([first, second], ignore_index=True))
    new_season = features.loc[features["season"].eq(2023)]

    assert pd.isna(_feature(new_season, 1, "off_epa_last_5"))
    assert _feature(new_season, 2, "off_epa_last_5") == 1.0


def test_teams_do_not_borrow_from_each_other():
    frame = pd.concat(
        [
            _team_games(team="AAA", offensive_epa=[1.0, 1.0, 1.0]),
            _team_games(team="BBB", offensive_epa=[9.0, 9.0, 9.0]),
        ],
        ignore_index=True,
    )
    features = add_rolling_features(frame)

    away = features.loc[features["team"].eq("BBB")]
    assert _feature(away, 2, "off_epa_season") == 9.0


def test_rows_are_sorted_before_shifting():
    """Input order cannot change the result; dates decide what came first."""
    ordered = _team_games(offensive_epa=[1.0, 2.0, 3.0, 4.0])
    shuffled = ordered.iloc[::-1].reset_index(drop=True)

    pd.testing.assert_series_equal(
        add_rolling_features(shuffled)["off_epa_season"],
        add_rolling_features(ordered)["off_epa_season"],
    )


def test_win_value_scores_a_tie_as_half_a_win():
    derived = add_derived_sources(
        _team_games(point_margin=[7.0, -3.0, 0.0, 1.0, 1.0, 1.0])
    )

    assert derived["win_value"].tolist()[:3] == [1.0, 0.0, 0.5]


def test_win_pct_averages_prior_results_only():
    features = add_rolling_features(
        _team_games(point_margin=[7.0, 7.0, -7.0, -7.0, 0.0])
    )

    assert _feature(features, 5, "win_pct_last_5") == pytest.approx(0.5)


def test_ats_margin_is_relative_to_the_team_spread():
    derived = add_derived_sources(
        _team_games(
            point_margin=[7.0, 7.0, 7.0], team_spread=[3.0, -3.0, 7.0], weeks=[1, 2, 3]
        )
    )

    assert derived["ats_margin"].tolist() == [4.0, 10.0, 0.0]


def test_cover_rate_ignores_pushes():
    """A push is neither a cover nor a failure, so it leaves the rate alone."""
    frame = _team_games(weeks=[1, 2, 3, 4])
    frame["covered"] = pd.array([1, pd.NA, 0, pd.NA], dtype="Int8")
    features = add_rolling_features(frame)

    assert _feature(features, 3, "cover_rate_last_5") == 1.0
    assert _feature(features, 4, "cover_rate_last_5") == pytest.approx(0.5)


def test_context_columns_describe_the_available_history():
    features = add_rolling_features(_team_games(weeks=[1, 2, 3, 4, 5, 6]))

    assert _feature(features, 1, "prior_games_this_season") == 0
    assert _feature(features, 4, "prior_games_this_season") == 3
    assert bool(_feature(features, EARLY_SEASON_WEEKS, "is_early_season"))
    assert not bool(_feature(features, EARLY_SEASON_WEEKS + 1, "is_early_season"))
    assert _feature(features, 3, "week_number") == 3


def test_unplayed_games_carry_prior_form_without_contributing():
    """A future game has no result yet but still gets its team's prior form."""
    frame = _team_games(weeks=[1, 2, 3], offensive_epa=[2.0, 4.0, 0.0])
    frame.loc[frame["week"].eq(3), "offensive_epa_per_play"] = float("nan")
    features = add_rolling_features(frame)

    assert _feature(features, 3, "off_epa_season") == pytest.approx(3.0)


def test_custom_spec_list_is_honoured():
    spec = RollingSpec("custom_last_2", "offensive_epa_per_play", 2)
    features = add_rolling_features(
        _team_games(offensive_epa=[1.0, 3.0, 0.0]), specs=(spec,)
    )

    assert _feature(features, 3, "custom_last_2") == pytest.approx(2.0)
    assert "off_epa_last_3" not in features.columns


def test_missing_required_column_raises():
    frame = _team_games().drop(columns="gameday")

    with pytest.raises(RollingFeatureError, match="gameday"):
        add_rolling_features(frame)


def test_missing_source_column_raises():
    spec = RollingSpec("nope", "not_a_column", 3)

    with pytest.raises(RollingFeatureError, match="not_a_column"):
        add_rolling_features(_team_games(), specs=(spec,))


def test_derived_sources_require_margin_and_spread():
    with pytest.raises(RollingFeatureError, match="point_margin"):
        add_derived_sources(pd.DataFrame({"team": ["AAA"]}))


def test_assemble_normalizes_legacy_team_codes():
    """Play-by-play calls the Rams LA while the schedule says LAR."""
    history = pd.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "game_id": ["2023_01_ZZZ_LAR"],
            "team": ["LAR"],
            "opponent": ["ZZZ"],
            "gameday": pd.to_datetime(["2023-09-10"]),
        }
    )
    measured = pd.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "game_id": ["2023_01_ZZZ_LAR"],
            "team": ["LA"],
            "offensive_epa_per_play": [0.25],
            "defensive_epa_strength_per_play": [0.1],
            "pace_plays_per_game": [64.0],
        }
    )

    assembled = assemble_team_game_inputs(history, measured, measured, measured)

    assert assembled["offensive_epa_per_play"].notna().all()
    assert assembled.loc[0, "pace_plays_per_game"] == 64.0


def test_assemble_requires_the_join_columns():
    history = pd.DataFrame(
        {
            "season": [2023],
            "week": [1],
            "game_id": ["g"],
            "team": ["AAA"],
            "gameday": pd.to_datetime(["2023-09-10"]),
        }
    )

    with pytest.raises(RollingFeatureError, match="offensive_epa_per_play"):
        assemble_team_game_inputs(
            history,
            pd.DataFrame(columns=["season", "week", "game_id", "team"]),
            history,
            history,
        )


def test_coverage_report_separates_week_one_nulls():
    features = add_rolling_features(_team_games(weeks=[1, 2, 3]))
    coverage = rolling_feature_coverage(features)

    row = coverage.loc[coverage["feature"].eq("off_epa_last_3")].iloc[0]
    assert row["missing_week_1"] == 1
    assert row["missing_after_week_1"] == 0


def test_coverage_report_requires_a_week_column():
    with pytest.raises(RollingFeatureError, match="week"):
        rolling_feature_coverage(pd.DataFrame({"off_epa_last_3": [1.0]}))
