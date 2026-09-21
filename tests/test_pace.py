"""Tests for team-game offensive pace aggregation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gridiron.features.epa import aggregate_offensive_epa
from gridiron.features.pace import (
    MIN_NEUTRAL_INTERVALS,
    MIN_PACE_PLAYS,
    PaceError,
    aggregate_pace,
    flag_pace_outliers,
    pace_distribution_report,
    timing_fields_available,
)

GAME_ID = "2023_01_AAA_BBB"
PBP_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "pbp_2016_2025.parquet"
)
PACE_PBP_COLUMNS = [
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

_PLAY_COUNTER = {"play_id": 0}


def _play(**overrides) -> dict:
    """Return one ordinary completed pass play, with overrides applied."""
    _PLAY_COUNTER["play_id"] += 1
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
        "game_seconds_remaining": 3600.0,
        "half_seconds_remaining": 1800.0,
        "no_huddle": 0.0,
        "fixed_drive": 1,
        "play_id": float(_PLAY_COUNTER["play_id"]),
        "score_differential": 0.0,
    }
    play.update(overrides)
    return play


def _pbp(*plays: dict) -> pd.DataFrame:
    return pd.DataFrame([_play(**play) for play in plays])


def _drive(*, clock: list[float], **overrides) -> list[dict]:
    """A single drive whose snaps sit at the given game-clock readings."""
    return [_play(game_seconds_remaining=seconds, **overrides) for seconds in clock]


def _flip(**overrides) -> dict:
    """A play for the other offense in the same game."""
    return _play(posteam="BBB", defteam="AAA", **overrides)


def test_completed_game_produces_two_pace_rows():
    pace = aggregate_pace(_pbp(_play(), _flip()), min_plays=1)

    assert len(pace) == 2
    assert list(pace["team"]) == ["AAA", "BBB"]
    assert list(pace["opponent"]) == ["BBB", "AAA"]


def test_pace_equals_eligible_offensive_play_count():
    pace = aggregate_pace(
        _pbp(_play(), _play(), _play(play_type="run", **{"pass": 0.0, "rush": 1.0})),
        min_plays=1,
    )

    row = pace.loc[pace["team"].eq("AAA")].iloc[0]
    assert row["offensive_play_count"] == 3
    assert row["pace_plays_per_game"] == 3


def test_pace_ignores_non_offensive_plays():
    """Punts, kickoffs, and kneels are not offensive plays and must not count."""
    pace = aggregate_pace(
        _pbp(
            _play(),
            _play(play_type="punt"),
            _play(play_type="kickoff"),
            _play(play_type="field_goal"),
            _play(play_type="no_play"),
            _play(qb_kneel=1.0),
            _play(qb_spike=1.0),
        ),
        min_plays=1,
    )

    assert pace.loc[pace["team"].eq("AAA"), "pace_plays_per_game"].iloc[0] == 1


def test_pace_play_count_matches_epa_play_count():
    """Pace and EPA must never disagree about how many plays a team ran."""
    pbp = _pbp(
        _play(epa=1.0),
        _play(epa=-0.5),
        _play(play_type="punt"),
        _flip(epa=0.25),
    )

    pace = aggregate_pace(pbp, min_plays=1)
    epa = aggregate_offensive_epa(pbp, min_plays=1)
    merged = pace.merge(epa, on=["season", "week", "game_id", "team"])

    assert len(merged) == 2
    assert (merged["pace_plays_per_game"] == merged["offensive_plays"]).all()


def test_seconds_per_play_uses_within_drive_intervals():
    """Three snaps 30s apart give a 30s mean over two intervals."""
    pace = aggregate_pace(
        pd.DataFrame(_drive(clock=[3600.0, 3570.0, 3540.0])), min_plays=1
    )

    row = pace.iloc[0]
    assert row["pace_intervals"] == 2
    assert row["seconds_per_play"] == pytest.approx(30.0)


def test_first_snap_of_each_drive_has_no_interval():
    pace = aggregate_pace(
        pd.DataFrame(
            _drive(clock=[3600.0, 3570.0], fixed_drive=1)
            + _drive(clock=[3000.0, 2980.0], fixed_drive=3)
        ),
        min_plays=1,
    )

    row = pace.iloc[0]
    assert row["offensive_play_count"] == 4
    assert row["pace_intervals"] == 2
    assert row["seconds_per_play"] == pytest.approx(25.0)


def test_clock_stoppage_intervals_are_excluded():
    """A gap longer than the play clock allows measures a stoppage, not tempo."""
    pace = aggregate_pace(
        pd.DataFrame(_drive(clock=[3600.0, 3570.0, 3400.0])), min_plays=1
    )

    row = pace.iloc[0]
    assert row["pace_intervals"] == 1
    assert row["seconds_per_play"] == pytest.approx(30.0)


def test_non_positive_intervals_are_excluded():
    pace = aggregate_pace(
        pd.DataFrame(_drive(clock=[3600.0, 3600.0, 3570.0])), min_plays=1
    )

    assert pace.iloc[0]["pace_intervals"] == 1


def test_neutral_pace_excludes_blowouts_and_end_of_half():
    """Only one-possession, non-endgame snaps feed the neutral mean."""
    plays = (
        _drive(clock=[3600.0, 3570.0], score_differential=0.0)
        + _drive(clock=[3000.0, 2980.0], score_differential=28.0, fixed_drive=3)
        + _drive(
            clock=[1900.0, 1880.0],
            half_seconds_remaining=60.0,
            fixed_drive=5,
        )
    )
    pace = aggregate_pace(pd.DataFrame(plays), min_plays=1)

    row = pace.iloc[0]
    assert row["pace_intervals"] == 3
    assert row["neutral_pace_intervals"] == 1
    assert row["neutral_seconds_per_play"] == pytest.approx(30.0)


def test_no_huddle_rate_is_share_of_eligible_plays():
    pace = aggregate_pace(
        _pbp(_play(no_huddle=1.0), _play(no_huddle=0.0), _play(no_huddle=0.0)),
        min_plays=1,
    )

    assert pace.iloc[0]["no_huddle_rate"] == pytest.approx(1 / 3)


def test_missing_tempo_is_null_not_fabricated():
    """A lone snap has no interval, so tempo is null rather than invented."""
    pace = aggregate_pace(_pbp(_play()), min_plays=1)

    row = pace.iloc[0]
    assert row["pace_plays_per_game"] == 1
    assert row["pace_intervals"] == 0
    assert pd.isna(row["seconds_per_play"])
    assert not row["valid_neutral_pace"]


def test_low_volume_team_game_is_flagged_not_dropped():
    pace = aggregate_pace(_pbp(_play(), _flip()), min_plays=MIN_PACE_PLAYS)

    assert len(pace) == 2
    assert not pace["valid_pace"].any()


def test_timing_columns_omitted_when_clock_is_unusable():
    pbp = _pbp(_play(), _play())
    pbp["game_seconds_remaining"] = pd.NA

    pace = aggregate_pace(pbp, min_plays=1)

    assert not timing_fields_available(pbp)
    assert "seconds_per_play" not in pace.columns
    assert pace.iloc[0]["pace_plays_per_game"] == 2


def test_include_timing_false_produces_baseline_only():
    pace = aggregate_pace(_pbp(_play(), _play()), min_plays=1, include_timing=False)

    assert "seconds_per_play" not in pace.columns
    assert "pace_plays_per_game" in pace.columns


def test_missing_required_column_raises_pace_error():
    pbp = _pbp(_play()).drop(columns="epa")

    with pytest.raises(PaceError, match="epa"):
        aggregate_pace(pbp)


def test_team_facing_two_opponents_raises():
    pbp = _pbp(_play(defteam="BBB"), _play(defteam="CCC"))

    with pytest.raises(PaceError, match="more than one opponent"):
        aggregate_pace(pbp, min_plays=1)


def test_distribution_report_has_the_required_statistics():
    pace = aggregate_pace(
        pd.DataFrame(_drive(clock=[3600.0, 3570.0, 3540.0])), min_plays=1
    )
    report = pace_distribution_report(pace)

    assert list(report.columns) == [
        "statistic",
        "count",
        "missing",
        "mean",
        "median",
        "std",
        "min",
        "p01",
        "p99",
        "max",
    ]
    assert "pace_plays_per_game" in set(report["statistic"])


def test_distribution_report_counts_missing_values():
    pace = aggregate_pace(_pbp(_play(), _flip()), min_plays=1)
    report = pace_distribution_report(pace)
    seconds = report.loc[report["statistic"].eq("seconds_per_play")].iloc[0]

    assert seconds["count"] == 0
    assert seconds["missing"] == 2


def test_distribution_report_rejects_a_frame_without_pace_columns():
    with pytest.raises(PaceError, match="No numeric pace columns"):
        pace_distribution_report(pd.DataFrame({"team": ["AAA"]}))


def test_outlier_report_flags_thin_neutral_samples_with_a_reason():
    pace = aggregate_pace(
        pd.DataFrame(_drive(clock=[3600.0, 3570.0, 3540.0])), min_plays=1
    )
    flagged = flag_pace_outliers(pace)

    assert len(flagged) == 1
    assert str(MIN_NEUTRAL_INTERVALS) in flagged.iloc[0]["flag_reasons"]


def test_outlier_report_requires_a_pace_aggregation():
    with pytest.raises(PaceError, match="pace_plays_per_game"):
        flag_pace_outliers(pd.DataFrame({"team": ["AAA"]}))


@pytest.mark.skipif(
    not PBP_PATH.exists(), reason="raw play-by-play parquet is not available"
)
def test_real_pace_covers_both_teams_in_every_completed_game():
    pbp = pd.read_parquet(PBP_PATH, columns=PACE_PBP_COLUMNS)
    pace = aggregate_pace(pbp)

    teams_per_game = pace.groupby("game_id")["team"].nunique()
    assert teams_per_game.eq(2).all()
    assert not pace.duplicated(["game_id", "team"]).any()


@pytest.mark.skipif(
    not PBP_PATH.exists(), reason="raw play-by-play parquet is not available"
)
def test_real_pace_distribution_is_plausible():
    pbp = pd.read_parquet(PBP_PATH, columns=PACE_PBP_COLUMNS)
    pace = aggregate_pace(pbp)

    plays = pace["pace_plays_per_game"]
    assert 55 <= plays.mean() <= 70
    assert plays.min() >= 20
    assert plays.max() <= 110

    seconds = pace["seconds_per_play"].dropna()
    assert 25 <= seconds.mean() <= 35
    assert seconds.between(15.0, 50.0).all()


@pytest.mark.skipif(
    not PBP_PATH.exists(), reason="raw play-by-play parquet is not available"
)
def test_real_pace_agrees_with_real_epa_play_counts():
    pbp = pd.read_parquet(PBP_PATH, columns=PACE_PBP_COLUMNS)

    pace = aggregate_pace(pbp)
    epa = aggregate_offensive_epa(pbp)
    merged = pace.merge(epa, on=["season", "week", "game_id", "team"], how="outer")

    assert len(merged) == len(pace) == len(epa)
    assert (merged["pace_plays_per_game"] == merged["offensive_plays"]).all()
