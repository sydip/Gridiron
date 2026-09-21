"""Tests for team-game history and rest days."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gridiron.features.team_games import (
    SEASON_OPENER_REST_DAYS,
    TeamGameError,
    add_rest_days,
    build_team_games,
    rest_distribution,
    team_game_history,
    team_game_reconciliation,
    team_game_reconciliation_failures,
)

SCHEDULE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "schedules_2016_2026.parquet"
)


def _game(**overrides) -> dict:
    """Return one ordinary completed game, with overrides applied."""
    game = {
        "game_id": "2023_01_BBB_AAA",
        "season": 2023,
        "week": 1,
        "gameday": "2023-09-10",
        "home_team": "AAA",
        "away_team": "BBB",
        "home_score": 24.0,
        "away_score": 17.0,
        "spread_line": 3.0,
    }
    game.update(overrides)
    return game


def _schedule(*games: dict) -> pd.DataFrame:
    return pd.DataFrame([_game(**game) for game in games])


def _season(team: str, dates: list[str], weeks: list[int]) -> pd.DataFrame:
    """A run of games for one team, as the home side each week."""
    return _schedule(
        *[
            _game(
                game_id=f"2023_{week:02d}_ZZZ_{team}",
                week=week,
                gameday=date,
                home_team=team,
                away_team="ZZZ",
            )
            for date, week in zip(dates, weeks, strict=True)
        ]
    )


def test_each_game_becomes_exactly_two_rows():
    team_games = build_team_games(_schedule(_game()))

    assert len(team_games) == 2
    assert set(team_games["team"]) == {"AAA", "BBB"}
    assert list(team_games["is_home"]) == sorted([True, False], reverse=True)


def test_home_and_away_margins_sum_to_zero():
    team_games = build_team_games(_schedule(_game(home_score=24.0, away_score=17.0)))

    assert team_games["point_margin"].sum() == 0
    home = team_games.loc[team_games["is_home"]].iloc[0]
    away = team_games.loc[~team_games["is_home"]].iloc[0]
    assert home["point_margin"] == 7
    assert away["point_margin"] == -7


def test_team_spread_signs_reconcile_across_rows():
    """A positive spread means this team is favoured, so the pair cancels."""
    team_games = build_team_games(_schedule(_game(spread_line=3.0)))

    home = team_games.loc[team_games["is_home"]].iloc[0]
    away = team_games.loc[~team_games["is_home"]].iloc[0]
    assert home["team_spread"] == 3.0
    assert away["team_spread"] == -3.0
    assert team_games["team_spread"].sum() == 0


def test_points_for_and_against_mirror_each_other():
    team_games = build_team_games(_schedule(_game(home_score=24.0, away_score=17.0)))

    home = team_games.loc[team_games["is_home"]].iloc[0]
    away = team_games.loc[~team_games["is_home"]].iloc[0]
    assert (home["points_for"], home["points_against"]) == (24.0, 17.0)
    assert (away["points_for"], away["points_against"]) == (17.0, 24.0)
    assert home["opponent"] == "BBB"
    assert away["opponent"] == "AAA"


def test_covered_is_exclusive_between_the_two_rows():
    """The home team wins by 7 against a 3 point spread, so it covers alone."""
    team_games = build_team_games(
        _schedule(_game(home_score=24.0, away_score=17.0, spread_line=3.0))
    )

    home = team_games.loc[team_games["is_home"]].iloc[0]
    away = team_games.loc[~team_games["is_home"]].iloc[0]
    assert home["covered"] == 1
    assert away["covered"] == 0


def test_push_is_null_for_both_rows_not_a_loss():
    team_games = build_team_games(
        _schedule(_game(home_score=24.0, away_score=17.0, spread_line=7.0))
    )

    assert team_games["covered"].isna().all()


def test_unplayed_game_is_kept_with_null_scores():
    team_games = build_team_games(_schedule(_game(home_score=None, away_score=None)))

    assert len(team_games) == 2
    assert not team_games["is_completed"].any()
    assert team_games["points_for"].isna().all()
    assert team_games["covered"].isna().all()


def test_missing_schedule_column_raises():
    schedule = _schedule(_game()).drop(columns="spread_line")

    with pytest.raises(TeamGameError, match="spread_line"):
        build_team_games(schedule)


def test_rest_days_measure_the_gap_to_the_previous_game():
    history = team_game_history(
        _season("AAA", ["2023-09-10", "2023-09-17", "2023-09-21"], [1, 2, 3])
    )
    team = history.loc[history["team"].eq("AAA")].sort_values("week")

    assert list(team["rest_days"]) == [SEASON_OPENER_REST_DAYS, 7, 4]


def test_season_opener_uses_asserted_rest_and_is_flagged():
    history = team_game_history(_season("AAA", ["2023-09-10", "2023-09-17"], [1, 2]))
    team = history.loc[history["team"].eq("AAA")].sort_values("week")

    assert list(team["week_1_flag"]) == [1, 0]
    assert team.iloc[0]["rest_days"] == SEASON_OPENER_REST_DAYS
    assert pd.isna(team.iloc[0]["previous_game_date"])


def test_rest_never_crosses_a_season_boundary():
    """A new season restarts the clock instead of reaching back months."""
    schedule = pd.concat(
        [
            _schedule(
                _game(
                    game_id="2022_18_ZZZ_AAA",
                    season=2022,
                    week=18,
                    gameday="2023-01-07",
                    home_team="AAA",
                )
            ),
            _schedule(
                _game(
                    game_id="2023_01_ZZZ_AAA",
                    season=2023,
                    week=1,
                    gameday="2023-09-10",
                    home_team="AAA",
                )
            ),
        ],
        ignore_index=True,
    )

    history = team_game_history(schedule)
    opener = history.loc[history["team"].eq("AAA") & history["season"].eq(2023)].iloc[0]

    assert opener["week_1_flag"] == 1
    assert opener["rest_days"] == SEASON_OPENER_REST_DAYS
    assert pd.isna(opener["previous_game_date"])


def test_rest_days_never_use_a_future_date():
    history = team_game_history(
        _season("AAA", ["2023-09-10", "2023-09-17", "2023-09-24"], [1, 2, 3])
    )
    measured = history.dropna(subset=["previous_game_date"])

    assert (measured["previous_game_date"] < measured["gameday"]).all()


def test_out_of_order_input_still_measures_rest_backwards():
    """Rows are sorted by date before shifting, so input order cannot matter."""
    ordered = _season("AAA", ["2023-09-10", "2023-09-17", "2023-09-24"], [1, 2, 3])
    shuffled = ordered.iloc[::-1].reset_index(drop=True)

    history = team_game_history(shuffled)
    team = history.loc[history["team"].eq("AAA")].sort_values("week")

    assert list(team["rest_days"]) == [SEASON_OPENER_REST_DAYS, 7, 7]
    measured = team.dropna(subset=["previous_game_date"])
    assert (measured["previous_game_date"] < measured["gameday"]).all()


def test_short_week_flags_a_thursday_turnaround():
    history = team_game_history(_season("AAA", ["2023-09-10", "2023-09-14"], [1, 2]))
    team = history.loc[history["team"].eq("AAA")].sort_values("week")

    assert team.iloc[1]["rest_days"] == 4
    assert bool(team.iloc[1]["short_week"])
    assert not bool(team.iloc[1]["extra_rest"])


def test_extra_rest_flags_a_longer_than_normal_gap():
    history = team_game_history(_season("AAA", ["2023-09-11", "2023-09-21"], [1, 2]))
    team = history.loc[history["team"].eq("AAA")].sort_values("week")

    assert team.iloc[1]["rest_days"] == 10
    assert bool(team.iloc[1]["extra_rest"])
    assert not bool(team.iloc[1]["short_week"])


def test_bye_week_is_detected_from_the_skipped_week():
    history = team_game_history(_season("AAA", ["2023-09-10", "2023-09-24"], [1, 3]))
    team = history.loc[history["team"].eq("AAA")].sort_values("week")

    assert team.iloc[1]["rest_days"] == 14
    assert bool(team.iloc[1]["bye_week_rest"])
    assert bool(team.iloc[1]["extra_rest"])


def test_thursday_to_thursday_is_not_a_short_week():
    """Thanksgiving into the next Thursday is a normal seven day turnaround."""
    history = team_game_history(_season("AAA", ["2023-11-23", "2023-11-30"], [12, 13]))
    team = history.loc[history["team"].eq("AAA")].sort_values("week")

    assert team.iloc[1]["rest_days"] == 7
    assert not bool(team.iloc[1]["short_week"])


def test_mini_bye_is_extra_rest_but_not_a_bye_week():
    """A Thursday game then the next Sunday is ten days without a missed week."""
    history = team_game_history(_season("AAA", ["2023-09-14", "2023-09-24"], [2, 3]))
    team = history.loc[history["team"].eq("AAA")].sort_values("week")

    assert team.iloc[1]["rest_days"] == 10
    assert bool(team.iloc[1]["extra_rest"])
    assert not bool(team.iloc[1]["bye_week_rest"])


def test_rest_advantage_placeholder_is_reserved_and_empty():
    history = team_game_history(_schedule(_game()))

    assert "rest_advantage_placeholder" in history.columns
    assert history["rest_advantage_placeholder"].isna().all()


def test_add_rest_days_requires_a_gameday():
    with pytest.raises(TeamGameError, match="gameday"):
        add_rest_days(pd.DataFrame({"team": ["AAA"]}))


def test_reconciliation_passes_for_a_well_formed_game():
    report = team_game_reconciliation(build_team_games(_schedule(_game())))

    assert len(report) == 1
    assert bool(report.iloc[0]["reconciled"])
    assert team_game_reconciliation_failures(report).empty


def test_reconciliation_catches_a_dropped_row():
    team_games = build_team_games(_schedule(_game())).iloc[:1]
    report = team_game_reconciliation(team_games)

    assert not bool(report.iloc[0]["reconciled"])
    assert len(team_game_reconciliation_failures(report)) == 1


def test_reconciliation_requires_the_expected_columns():
    with pytest.raises(TeamGameError, match="missing column"):
        team_game_reconciliation(pd.DataFrame({"game_id": ["x"]}))


def test_rest_distribution_reports_flag_shares():
    history = team_game_history(_season("AAA", ["2023-09-10", "2023-09-14"], [1, 2]))
    summary = rest_distribution(history)

    assert "rest_days" in set(summary["metric"])
    assert "short_week" in set(summary["metric"])


def test_rest_distribution_requires_rest_days():
    with pytest.raises(TeamGameError, match="rest days"):
        rest_distribution(pd.DataFrame({"team": ["AAA"]}))


@pytest.fixture(scope="module")
def real_history() -> pd.DataFrame:
    if not SCHEDULE_PATH.exists():
        pytest.skip("raw schedule parquet is not available")
    from gridiron.data.clean import clean_schedules

    return team_game_history(clean_schedules(pd.read_parquet(SCHEDULE_PATH)))


def test_real_completed_games_produce_exactly_two_rows(real_history):
    completed = real_history.loc[real_history["is_completed"]]
    report = team_game_reconciliation(completed)

    assert report["two_rows"].all()
    assert len(completed) == 2 * completed["game_id"].nunique()


def test_real_margins_and_spreads_reconcile(real_history):
    report = team_game_reconciliation(real_history)

    assert team_game_reconciliation_failures(report).empty


def test_real_rest_never_uses_a_future_date(real_history):
    measured = real_history.dropna(subset=["previous_game_date"])

    assert (measured["previous_game_date"] < measured["gameday"]).all()
    assert measured["rest_days"].min() > 0


def test_real_openers_are_consistent(real_history):
    openers = real_history.loc[real_history["week_1_flag"].eq(1)]

    assert (openers["rest_days"] == SEASON_OPENER_REST_DAYS).all()
    assert openers["previous_game_date"].isna().all()
    # Exactly one opener per team per season, including the two 2017 teams
    # whose week 1 game was cancelled and who therefore open in week 2.
    per_team_season = openers.groupby(["team", "season"]).size()
    assert (per_team_season == 1).all()


def test_real_thursday_games_are_usually_short_weeks(real_history):
    thursday = real_history.loc[
        real_history["gameday"].dt.day_name().eq("Thursday")
        & real_history["week_1_flag"].eq(0)
    ]

    assert len(thursday) > 200
    assert thursday["short_week"].mean() > 0.9


def test_real_bye_weeks_always_show_extra_rest(real_history):
    bye = real_history.loc[real_history["bye_week_rest"]]

    assert len(bye) > 300
    assert bye["extra_rest"].all()
    assert bye["rest_days"].median() >= 13


def test_real_covered_is_exclusive_within_each_game(real_history):
    played = real_history.loc[
        real_history["is_completed"] & real_history["team_spread"].notna()
    ]
    outcomes = played.groupby("game_id")["covered"].apply(
        lambda values: set(values.dropna())
    )

    assert all(outcome in ({0, 1}, set()) for outcome in outcomes)
