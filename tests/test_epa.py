"""Tests for game-level offensive and defensive EPA aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gridiron.features.epa import (
    MIN_DEFENSIVE_PLAYS,
    MIN_OFFENSIVE_PLAYS,
    OffensiveEpaError,
    aggregate_defensive_epa,
    aggregate_offensive_epa,
    epa_reconciliation_failures,
    filter_offensive_plays,
    low_volume_team_games,
    reconcile_offense_and_defense,
)

GAME_ID = "2023_01_AAA_BBB"
PBP_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "pbp_2016_2025.parquet"
)


def _play(**overrides) -> dict:
    """Return one ordinary completed pass play, with overrides applied."""
    play = {
        "season": 2023,
        "week": 1,
        "game_id": GAME_ID,
        "posteam": "AAA",
        "defteam": "BBB",
        "play_type": "pass",
        "epa": 0.0,
        "success": 0.0,
        "pass": 1.0,
        "rush": 0.0,
        "qb_kneel": 0.0,
        "qb_spike": 0.0,
    }
    play.update(overrides)
    return play


def _pbp(*plays: dict) -> pd.DataFrame:
    return pd.DataFrame([_play(**play) for play in plays])


def _run(**overrides) -> dict:
    return _play(play_type="run", **{"pass": 0.0, "rush": 1.0}, **overrides)


def _flip(**overrides) -> dict:
    """A play for the other offense in the same game."""
    return _play(posteam="BBB", defteam="AAA", **overrides)


def test_completed_game_produces_two_offensive_rows():
    offense = aggregate_offensive_epa(
        _pbp(_play(epa=1.0), _flip(epa=-1.0)), min_plays=1
    )

    assert len(offense) == 2
    assert list(offense["team"]) == ["AAA", "BBB"]
    assert list(offense["opponent"]) == ["BBB", "AAA"]
    assert set(offense["game_id"]) == {GAME_ID}


def test_no_game_team_combination_appears_twice():
    offense = aggregate_offensive_epa(
        _pbp(
            _play(epa=1.0),
            _play(epa=2.0),
            _run(epa=0.5),
            _flip(epa=-1.0),
            _flip(epa=-2.0),
        ),
        min_plays=1,
    )

    assert not offense.duplicated(["season", "week", "game_id", "team"]).any()
    assert len(offense) == 2


def test_totals_and_rates_match_manual_calculation():
    offense = aggregate_offensive_epa(
        _pbp(
            _play(epa=2.0, success=1.0),
            _play(epa=-1.0, success=0.0),
            _run(epa=0.5, success=1.0),
            _run(epa=-0.5, success=0.0),
        ),
        min_plays=1,
    )
    row = offense.loc[offense["team"].eq("AAA")].iloc[0]

    # 2.0 - 1.0 + 0.5 - 0.5 = 1.0 over four plays; two of them successful.
    assert row["offensive_plays"] == 4
    assert row["offensive_epa"] == pytest.approx(1.0)
    assert row["offensive_epa_per_play"] == pytest.approx(0.25)
    assert row["offensive_success_rate"] == pytest.approx(0.5)
    assert row["pass_plays"] == 2
    assert row["pass_epa"] == pytest.approx(1.0)
    assert row["pass_epa_per_play"] == pytest.approx(0.5)
    assert row["rush_plays"] == 2
    assert row["rush_epa"] == pytest.approx(0.0)
    assert row["rush_epa_per_play"] == pytest.approx(0.0)


def test_epa_columns_are_numeric():
    offense = aggregate_offensive_epa(
        _pbp(_play(epa="1.5"), _run(epa="-0.5"), _flip(epa="0.25")), min_plays=1
    )

    for column in (
        "offensive_epa",
        "offensive_epa_per_play",
        "offensive_success_rate",
        "pass_epa",
        "rush_epa",
    ):
        assert pd.api.types.is_numeric_dtype(offense[column])
        assert offense[column].notna().all()
    assert pd.api.types.is_integer_dtype(offense["offensive_plays"])


@pytest.mark.parametrize(
    "play_type",
    ["no_play", "kickoff", "punt", "field_goal", "extra_point", None],
    ids=["penalty", "kickoff", "punt", "field-goal", "extra-point", "untyped"],
)
def test_administrative_and_special_teams_rows_are_excluded(play_type):
    offense = aggregate_offensive_epa(
        _pbp(_play(epa=1.0), _play(play_type=play_type, epa=99.0)), min_plays=1
    )

    assert offense.loc[0, "offensive_plays"] == 1
    assert offense.loc[0, "offensive_epa"] == pytest.approx(1.0)


@pytest.mark.parametrize("indicator", ["qb_kneel", "qb_spike"])
def test_kneels_and_spikes_are_excluded_by_indicator_fields(indicator):
    # play_type still claims a normal scrimmage play, so only the indicator
    # column can catch these rows.
    offense = aggregate_offensive_epa(
        _pbp(_play(epa=1.0), _play(epa=-9.0, **{indicator: 1.0})), min_plays=1
    )

    assert offense.loc[0, "offensive_plays"] == 1
    assert offense.loc[0, "offensive_epa"] == pytest.approx(1.0)


@pytest.mark.parametrize("indicator", ["qb_kneel", "qb_spike"])
def test_unreliable_indicator_column_is_ignored(indicator):
    """An all-null indicator must not silently discard every play."""
    offense = aggregate_offensive_epa(
        _pbp(_play(epa=1.0), _play(epa=3.0)).assign(**{indicator: None}),
        min_plays=1,
    )

    assert offense.loc[0, "offensive_plays"] == 2
    assert offense.loc[0, "offensive_epa"] == pytest.approx(4.0)


def test_missing_indicator_columns_fall_back_to_play_type():
    pbp = _pbp(
        _play(epa=1.0),
        _run(epa=2.0),
        _play(play_type="qb_kneel", epa=-1.0),
        _play(play_type="qb_spike", epa=-1.0),
    ).drop(columns=["qb_kneel", "qb_spike", "pass", "rush", "success"])

    offense = aggregate_offensive_epa(pbp, min_plays=1)
    row = offense.iloc[0]

    assert row["offensive_plays"] == 2
    assert row["pass_plays"] == 1
    assert row["rush_plays"] == 1
    assert row["pass_epa"] == pytest.approx(1.0)
    assert row["rush_epa"] == pytest.approx(2.0)
    # success falls back to nflverse's definition: a play with positive EPA.
    assert row["offensive_success_rate"] == pytest.approx(1.0)


def test_plays_without_epa_are_removed():
    offense = aggregate_offensive_epa(
        _pbp(_play(epa=1.0), _play(epa=None), _play(epa=float("nan"))), min_plays=1
    )

    assert offense.loc[0, "offensive_plays"] == 1
    assert offense.loc[0, "offensive_epa"] == pytest.approx(1.0)


@pytest.mark.parametrize("blank", [None, "", "   "], ids=["null", "empty", "spaces"])
@pytest.mark.parametrize("column", ["posteam", "defteam"])
def test_plays_without_a_known_offense_or_defense_are_removed(column, blank):
    offense = aggregate_offensive_epa(
        _pbp(_play(epa=1.0), _play(epa=5.0, **{column: blank})), min_plays=1
    )

    assert len(offense) == 1
    assert offense.loc[0, "offensive_plays"] == 1
    assert offense.loc[0, "offensive_epa"] == pytest.approx(1.0)


def test_low_volume_team_games_are_flagged_not_dropped():
    pbp = _pbp(
        *[_play(epa=0.1) for _ in range(2)],
        *[_flip(epa=0.1) for _ in range(5)],
    )

    offense = aggregate_offensive_epa(pbp, min_plays=5)

    assert len(offense) == 2
    pairs = zip(offense["team"], offense["valid_game"], strict=True)
    assert dict(pairs) == {
        "AAA": False,
        "BBB": True,
    }

    flagged = low_volume_team_games(offense)
    assert list(flagged["team"]) == ["AAA"]
    assert list(flagged["offensive_plays"]) == [2]


def test_default_minimum_play_threshold_is_thirty():
    assert MIN_OFFENSIVE_PLAYS == 30

    offense = aggregate_offensive_epa(_pbp(*[_play(epa=0.1) for _ in range(29)]))

    assert not offense.loc[0, "valid_game"]
    assert offense.loc[0, "offensive_plays"] == 29


def test_teams_are_grouped_across_seasons_and_weeks():
    pbp = pd.DataFrame(
        [
            _play(season=2022, week=1, game_id="2022_01_AAA_BBB", epa=1.0),
            _play(season=2023, week=1, game_id="2023_01_AAA_BBB", epa=2.0),
            _play(season=2023, week=2, game_id="2023_02_AAA_BBB", epa=3.0),
        ]
    )

    offense = aggregate_offensive_epa(pbp, min_plays=1)

    assert len(offense) == 3
    assert list(offense["offensive_epa"]) == [1.0, 2.0, 3.0]
    assert list(offense["season"]) == [2022, 2023, 2023]
    assert list(offense["week"]) == [1, 1, 2]


def test_missing_required_columns_raise():
    with pytest.raises(OffensiveEpaError, match="defteam"):
        aggregate_offensive_epa(_pbp(_play()).drop(columns=["defteam"]))


def test_input_is_not_modified():
    pbp = _pbp(_play(epa=1.0), _flip(epa=-1.0))
    before = pbp.copy()

    aggregate_offensive_epa(pbp, min_plays=1)

    pd.testing.assert_frame_equal(pbp, before)


def test_filter_keeps_only_eligible_scrimmage_plays():
    pbp = _pbp(
        _play(epa=1.0),
        _run(epa=2.0),
        _play(play_type="punt", epa=3.0),
        _play(epa=None),
        _play(posteam=None, epa=4.0),
        _play(epa=5.0, qb_kneel=1.0),
    )

    plays = filter_offensive_plays(pbp)

    assert list(plays["epa"]) == [1.0, 2.0]


@pytest.mark.skipif(
    not PBP_PATH.exists(), reason="raw play-by-play parquet is not available"
)
def test_real_play_by_play_aggregates_plausibly():
    pbp = pd.read_parquet(PBP_PATH)
    offense = aggregate_offensive_epa(pbp)

    assert not offense.duplicated(["season", "week", "game_id", "team"]).any()
    assert (offense.groupby("game_id").size() == 2).all()
    assert offense["offensive_epa"].notna().all()
    assert offense["offensive_plays"].between(20, 110).all()
    assert 55 < offense["offensive_plays"].mean() < 70
    assert 0.35 < offense["offensive_success_rate"].mean() < 0.55

    # Each team's opponent is the other team's offense in the same game.
    paired = offense.merge(
        offense, left_on=["game_id", "team"], right_on=["game_id", "opponent"]
    )
    assert len(paired) == len(offense)


@pytest.mark.skipif(
    not PBP_PATH.exists(), reason="raw play-by-play parquet is not available"
)
def test_sampled_real_game_matches_manual_calculation():
    pbp = pd.read_parquet(PBP_PATH)
    offense = aggregate_offensive_epa(pbp)

    sampled = pbp.loc[pbp["game_id"].eq("2019_01_PIT_NE")]
    eligible = sampled.loc[
        sampled["play_type"].isin(["pass", "run"])
        & sampled["epa"].notna()
        & sampled["posteam"].notna()
        & sampled["defteam"].notna()
        & sampled["qb_kneel"].ne(1)
        & sampled["qb_spike"].ne(1)
    ]

    for team, plays in eligible.groupby("posteam"):
        row = offense.loc[
            offense["game_id"].eq("2019_01_PIT_NE") & offense["team"].eq(team)
        ].iloc[0]
        assert row["offensive_plays"] == len(plays)
        assert row["offensive_epa"] == pytest.approx(plays["epa"].sum())
        assert row["offensive_epa_per_play"] == pytest.approx(plays["epa"].mean())
        assert row["offensive_success_rate"] == pytest.approx(plays["success"].mean())


# --- Phase 6: defensive EPA -------------------------------------------------


def test_completed_game_produces_two_defensive_rows():
    defense = aggregate_defensive_epa(
        _pbp(_play(epa=1.0), _flip(epa=-1.0)), min_plays=1
    )

    assert len(defense) == 2
    assert list(defense["team"]) == ["AAA", "BBB"]
    assert list(defense["opponent"]) == ["BBB", "AAA"]
    assert not defense.duplicated(["season", "week", "game_id", "team"]).any()


def test_defensive_epa_strength_reverses_the_sign():
    defense = aggregate_defensive_epa(
        _pbp(_play(epa=2.0), _play(epa=-0.5), _flip(epa=-3.0)), min_plays=1
    )

    # BBB's defense faced AAA's two plays and allowed 1.5 EPA, so its strength
    # is -1.5; AAA's defense allowed -3.0, a good game, so strength is +3.0.
    strength = dict(
        zip(defense["team"], defense["defensive_epa_strength"], strict=True)
    )
    allowed = dict(zip(defense["team"], defense["defensive_epa_allowed"], strict=True))

    assert allowed["BBB"] == pytest.approx(1.5)
    assert strength["BBB"] == pytest.approx(-1.5)
    assert allowed["AAA"] == pytest.approx(-3.0)
    assert strength["AAA"] == pytest.approx(3.0)
    assert (
        defense["defensive_epa_strength"] == -defense["defensive_epa_allowed"]
    ).all()
    assert (
        defense["defensive_epa_strength_per_play"]
        == -defense["defensive_epa_allowed_per_play"]
    ).all()


def test_defense_allowed_equals_the_opposing_offense():
    pbp = _pbp(
        _play(epa=2.0, success=1.0),
        _run(epa=-0.5, success=0.0),
        _flip(epa=1.0, success=1.0),
    )

    offense = aggregate_offensive_epa(pbp, min_plays=1)
    defense = aggregate_defensive_epa(pbp, min_plays=1)

    offense_aaa = offense.loc[offense["team"].eq("AAA")].iloc[0]
    defense_bbb = defense.loc[defense["team"].eq("BBB")].iloc[0]

    assert defense_bbb["defensive_plays"] == offense_aaa["offensive_plays"]
    assert defense_bbb["defensive_epa_allowed"] == pytest.approx(
        offense_aaa["offensive_epa"]
    )
    assert defense_bbb["defensive_success_rate_allowed"] == pytest.approx(
        offense_aaa["offensive_success_rate"]
    )
    assert defense_bbb["pass_epa_allowed"] == pytest.approx(offense_aaa["pass_epa"])
    assert defense_bbb["rush_epa_allowed"] == pytest.approx(offense_aaa["rush_epa"])


def test_defense_uses_the_same_play_filter_as_offense():
    pbp = _pbp(
        _play(epa=1.0),
        _play(play_type="no_play", epa=99.0),
        _play(play_type="punt", epa=99.0),
        _play(epa=99.0, qb_kneel=1.0),
        _play(epa=99.0, qb_spike=1.0),
        _play(epa=None),
        _play(defteam=None, epa=99.0),
    )

    defense = aggregate_defensive_epa(pbp, min_plays=1)

    assert len(defense) == 1
    assert defense.loc[0, "team"] == "BBB"
    assert defense.loc[0, "defensive_plays"] == 1
    assert defense.loc[0, "defensive_epa_allowed"] == pytest.approx(1.0)


def test_defensive_low_volume_games_are_flagged_not_dropped():
    pbp = _pbp(
        *[_play(epa=0.1) for _ in range(2)],
        *[_flip(epa=0.1) for _ in range(5)],
    )

    defense = aggregate_defensive_epa(pbp, min_plays=5)

    # AAA's defense faced BBB's five plays; BBB's defense faced only two.
    assert dict(zip(defense["team"], defense["valid_game"], strict=True)) == {
        "AAA": True,
        "BBB": False,
    }

    flagged = low_volume_team_games(defense)
    assert list(flagged["team"]) == ["BBB"]
    assert list(flagged["defensive_plays"]) == [2]


def test_default_defensive_threshold_matches_the_offensive_one():
    assert MIN_DEFENSIVE_PLAYS == MIN_OFFENSIVE_PLAYS == 30

    defense = aggregate_defensive_epa(_pbp(*[_play(epa=0.1) for _ in range(29)]))

    assert not defense.loc[0, "valid_game"]
    assert defense.loc[0, "defensive_plays"] == 29


def test_aggregations_reconcile_by_game():
    pbp = _pbp(
        _play(epa=2.0, success=1.0),
        _run(epa=-0.5, success=0.0),
        _flip(epa=1.0, success=1.0),
        _flip(epa=-1.5, success=0.0),
    )

    report = reconcile_offense_and_defense(
        aggregate_offensive_epa(pbp, min_plays=1),
        aggregate_defensive_epa(pbp, min_plays=1),
    )

    assert len(report) == 2
    assert report["paired"].all()
    assert report["reconciled"].all()
    assert (report["plays_difference"] == 0).all()
    assert report["epa_difference"].abs().max() == pytest.approx(0.0)
    assert report["success_rate_difference"].abs().max() == pytest.approx(0.0)
    assert epa_reconciliation_failures(report).empty


def test_reconciliation_detects_a_filtering_difference():
    """A defense built from a narrower play set must not silently pass."""
    pbp = _pbp(
        _play(epa=2.0),
        _play(epa=1.0),
        _play(epa=0.5),
        _play(epa=0.5),
        _flip(epa=-1.0),
    )
    # The defensive side loses two of AAA's plays, as a stricter filter would.
    narrowed = pbp.drop(index=[2, 3])

    report = reconcile_offense_and_defense(
        aggregate_offensive_epa(pbp, min_plays=1),
        aggregate_defensive_epa(narrowed, min_plays=1),
    )
    failures = epa_reconciliation_failures(report)

    assert len(failures) == 1
    failure = failures.iloc[0]
    assert failure["offense"] == "AAA"
    assert failure["defense"] == "BBB"
    assert failure["paired"]
    assert not failure["reconciled"]
    assert failure["plays_difference"] == 2
    assert failure["epa_difference"] == pytest.approx(1.0)


def test_reconciliation_detects_a_missing_counterpart():
    pbp = _pbp(_play(epa=1.0), _flip(epa=-1.0))
    offense = aggregate_offensive_epa(pbp, min_plays=1)
    defense = aggregate_defensive_epa(pbp, min_plays=1)

    report = reconcile_offense_and_defense(
        offense, defense.loc[defense["team"].ne("BBB")]
    )
    failures = epa_reconciliation_failures(report)

    assert len(failures) == 1
    assert not failures.loc[0, "paired"]
    assert not failures.loc[0, "reconciled"]
    assert pd.isna(failures.loc[0, "defensive_plays"])


def test_aggregations_ignore_current_game_target_information():
    """Scores, spreads and results must not reach the EPA features."""
    pbp = _pbp(_play(epa=1.0), _run(epa=-0.5), _flip(epa=2.0))
    leaky = pbp.assign(
        home_score=99,
        away_score=0,
        result=99,
        spread_line=-14.0,
        total_line=77.0,
        home_cover=1,
    )

    pd.testing.assert_frame_equal(
        aggregate_offensive_epa(pbp, min_plays=1),
        aggregate_offensive_epa(leaky, min_plays=1),
    )
    pd.testing.assert_frame_equal(
        aggregate_defensive_epa(pbp, min_plays=1),
        aggregate_defensive_epa(leaky, min_plays=1),
    )

    forbidden = {
        "home_score",
        "away_score",
        "result",
        "spread_line",
        "total_line",
        "home_cover",
    }
    assert forbidden.isdisjoint(aggregate_offensive_epa(leaky, min_plays=1).columns)
    assert forbidden.isdisjoint(aggregate_defensive_epa(leaky, min_plays=1).columns)


def test_defensive_input_is_not_modified():
    pbp = _pbp(_play(epa=1.0), _flip(epa=-1.0))
    before = pbp.copy()

    aggregate_defensive_epa(pbp, min_plays=1)

    pd.testing.assert_frame_equal(pbp, before)


@pytest.mark.skipif(
    not PBP_PATH.exists(), reason="raw play-by-play parquet is not available"
)
def test_real_defensive_aggregation_reconciles_with_offense():
    pbp = pd.read_parquet(PBP_PATH)
    offense = aggregate_offensive_epa(pbp)
    defense = aggregate_defensive_epa(pbp)

    assert len(defense) == len(offense)
    assert (defense.groupby("game_id").size() == 2).all()
    assert not defense.duplicated(["season", "week", "game_id", "team"]).any()
    assert defense["defensive_epa_allowed"].notna().all()
    assert (
        defense["defensive_epa_strength"] == -defense["defensive_epa_allowed"]
    ).all()

    report = reconcile_offense_and_defense(offense, defense)
    assert len(report) == len(offense)
    assert report["paired"].all()
    assert report["reconciled"].all()
    assert epa_reconciliation_failures(report).empty


@pytest.mark.skipif(
    not PBP_PATH.exists(), reason="raw play-by-play parquet is not available"
)
def test_real_home_offense_matches_away_defense_allowed():
    pbp = pd.read_parquet(PBP_PATH)
    offense = aggregate_offensive_epa(pbp)
    defense = aggregate_defensive_epa(pbp)

    sides = (
        pbp[["game_id", "home_team", "away_team"]]
        .dropna()
        .drop_duplicates(subset="game_id")
    )
    home = offense.merge(
        sides, left_on=["game_id", "team"], right_on=["game_id", "home_team"]
    ).merge(defense, left_on=["game_id", "away_team"], right_on=["game_id", "team"])
    away = offense.merge(
        sides, left_on=["game_id", "team"], right_on=["game_id", "away_team"]
    ).merge(defense, left_on=["game_id", "home_team"], right_on=["game_id", "team"])

    assert len(home) == len(away) == sides["game_id"].nunique()
    for paired in (home, away):
        assert (paired["offensive_plays"] == paired["defensive_plays"]).all()
        assert (
            paired["offensive_epa"] - paired["defensive_epa_allowed"]
        ).abs().max() == pytest.approx(0.0)
        assert (paired["defensive_epa_strength"] == -paired["offensive_epa"]).all()
