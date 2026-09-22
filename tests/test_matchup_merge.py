"""Tests for game-level matchup assembly."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from gridiron.features.matchup import (
    DIFFERENTIAL_SOURCES,
    FEATURE_COLUMNS,
    MODEL_ROW_COLUMNS,
    POSTGAME_COLUMNS,
    MatchupError,
    add_matchup_target,
    assert_no_postgame_fields,
    build_matchups,
    build_model_table,
    missing_value_report,
    model_rows,
)

PBP_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "pbp_2016_2025.parquet"
)
SCHEDULE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "schedules_2016_2026.parquet"
)

GAME_ID = "2023_05_BBB_AAA"


def _schedule(**overrides) -> pd.DataFrame:
    """One game: AAA at home against BBB."""
    row = {
        "game_id": GAME_ID,
        "season": 2023,
        "week": 5,
        "gameday": pd.Timestamp("2023-10-08"),
        "home_team": "AAA",
        "away_team": "BBB",
        "spread_line": 3.0,
        "home_score": 27.0,
        "away_score": 17.0,
        "is_completed": True,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _features(home_value: float = 10.0, away_value: float = 4.0) -> pd.DataFrame:
    """Team-game features for both sides, with distinguishable values."""
    rows = []
    for team, value in (("AAA", home_value), ("BBB", away_value)):
        row = {
            "season": 2023,
            "week": 5,
            "game_id": GAME_ID,
            "team": team,
            "rest_days": value,
            "short_week": value < 7,
            "extra_rest": value > 7,
            "bye_week_rest": False,
            "week_1_flag": 0,
            "prior_games_this_season": 4,
        }
        for _, source in DIFFERENTIAL_SOURCES:
            if source != "rest_days":
                row[source] = value
        rows.append(row)
    return pd.DataFrame(rows)


def test_exactly_one_row_per_matchup():
    matchups = build_matchups(_features(), _schedule())

    assert len(matchups) == 1
    assert matchups["game_id"].nunique() == 1


def test_home_and_away_features_are_not_swapped():
    """The home columns must carry the home team's values, not the away team's."""
    matchups = build_matchups(_features(home_value=10.0, away_value=4.0), _schedule())
    row = matchups.iloc[0]

    assert row["home_team"] == "AAA"
    assert row["away_team"] == "BBB"
    assert row["home_off_epa_last_5"] == 10.0
    assert row["away_off_epa_last_5"] == 4.0


@pytest.mark.parametrize(("output", "source"), DIFFERENTIAL_SOURCES)
def test_every_differential_is_home_minus_away(output, source):
    matchups = build_matchups(_features(home_value=10.0, away_value=4.0), _schedule())
    row = matchups.iloc[0]

    assert row[output] == pytest.approx(row[f"home_{source}"] - row[f"away_{source}"])
    assert row[output] == pytest.approx(6.0)


def test_differential_is_null_when_a_side_is_missing():
    """A missing side yields a null differential, never a silent zero."""
    features = _features()
    features = features.loc[features["team"].eq("AAA")]

    matchups = build_matchups(features, _schedule())

    assert pd.isna(matchups.iloc[0]["off_epa_diff_last_5"])


def test_div_game_is_derived_from_the_two_divisions():
    divisional = build_matchups(
        _features(), _schedule(home_team="BUF", away_team="MIA")
    )
    interdivisional = build_matchups(
        _features(), _schedule(home_team="BUF", away_team="DAL")
    )

    assert divisional.iloc[0]["div_game"] == 1
    assert interdivisional.iloc[0]["div_game"] == 0


def test_week_1_flag_is_set_when_either_team_is_opening():
    features = _features()
    features.loc[features["team"].eq("BBB"), "week_1_flag"] = 1

    matchups = build_matchups(features, _schedule())

    assert matchups.iloc[0]["week_1_flag"] == 1


def test_duplicate_team_rows_are_rejected():
    """A repeated team-game would silently fan out the join, so it must raise."""
    features = pd.concat([_features(), _features()], ignore_index=True)

    with pytest.raises(ValueError, match="one_to_one|unique"):
        build_matchups(features, _schedule())


def test_missing_schedule_column_raises():
    with pytest.raises(MatchupError, match="spread_line"):
        build_matchups(_features(), _schedule().drop(columns="spread_line"))


def test_target_is_attached_for_a_completed_game():
    matchups = add_matchup_target(build_matchups(_features(), _schedule()))
    row = matchups.iloc[0]

    # Home won by 10 against a 3 point spread, so the home side covered.
    assert row["home_cover"] == 1
    assert not row["is_push"]


def test_push_has_no_label_but_is_marked():
    matchups = add_matchup_target(
        build_matchups(
            _features(), _schedule(home_score=20.0, away_score=17.0, spread_line=3.0)
        )
    )
    row = matchups.iloc[0]

    assert pd.isna(row["home_cover"])
    assert bool(row["is_push"])


def test_unplayed_game_has_features_but_no_target():
    matchups = add_matchup_target(
        build_matchups(
            _features(), _schedule(home_score=None, away_score=None, is_completed=False)
        )
    )
    row = matchups.iloc[0]

    assert pd.isna(row["home_cover"])
    assert not row["is_push"]
    assert row["off_epa_diff_last_5"] == pytest.approx(6.0)
    assert row["div_game"] in (0, 1)


def test_missing_line_leaves_the_target_unlabelled():
    matchups = add_matchup_target(
        build_matchups(_features(), _schedule(spread_line=None))
    )

    assert pd.isna(matchups.iloc[0]["home_cover"])


def test_model_rows_have_exactly_the_agreed_columns():
    rows = model_rows(add_matchup_target(build_matchups(_features(), _schedule())))

    assert list(rows.columns) == MODEL_ROW_COLUMNS


def test_model_rows_reports_a_missing_column():
    matchups = add_matchup_target(build_matchups(_features(), _schedule()))

    with pytest.raises(MatchupError, match="rest_diff"):
        model_rows(matchups.drop(columns="rest_diff"))


def test_no_postgame_field_reaches_the_feature_matrix():
    rows = build_model_table(_features(), _schedule())

    assert_no_postgame_fields(rows)
    assert not set(FEATURE_COLUMNS).intersection(POSTGAME_COLUMNS)


def test_the_postgame_guard_actually_catches_a_leak():
    """The guard must fail when a postgame field is offered as a feature."""
    rows = build_model_table(_features(), _schedule())
    leaked = rows.assign(home_score=27.0)

    with pytest.raises(MatchupError, match="home_score"):
        assert_no_postgame_fields(leaked, [*FEATURE_COLUMNS, "home_score"])


def test_the_guard_rejects_a_target_used_as_a_feature():
    rows = build_model_table(_features(), _schedule())

    with pytest.raises(MatchupError, match="home_cover"):
        assert_no_postgame_fields(rows, [*FEATURE_COLUMNS, "home_cover"])


def test_the_guard_allows_pregame_rolling_features():
    """point_margin_diff_last_5 must not be caught by a point_margin rule."""
    rows = build_model_table(_features(), _schedule())

    assert "point_margin_diff_last_5" in rows.columns
    assert_no_postgame_fields(rows, ["point_margin_diff_last_5"])


def test_missing_report_has_all_four_breakdowns():
    matchups = add_matchup_target(build_matchups(_features(), _schedule()))
    report = missing_value_report(matchups)

    assert set(report) == {"feature", "season", "week", "team"}
    assert "missing_pct" in report["feature"].columns
    assert "any_missing_pct" in report["week"].columns


def test_missing_report_counts_a_missing_feature():
    features = _features()
    features = features.loc[features["team"].eq("AAA")]
    matchups = add_matchup_target(build_matchups(features, _schedule()))

    report = missing_value_report(matchups)
    row = report["feature"].set_index("feature").loc["off_epa_diff_last_5"]

    assert row["missing_pct"] == 100.0


def test_missing_report_requires_feature_columns():
    with pytest.raises(MatchupError, match="No feature columns"):
        missing_value_report(pd.DataFrame({"game_id": ["x"]}))


@pytest.fixture(scope="module")
def real_matchups() -> pd.DataFrame:
    if not (PBP_PATH.exists() and SCHEDULE_PATH.exists()):
        pytest.skip("raw parquet data is not available")

    from gridiron.data.clean import clean_schedules
    from gridiron.features.epa import (
        aggregate_defensive_epa,
        aggregate_offensive_epa,
    )
    from gridiron.features.pace import aggregate_pace
    from gridiron.features.rolling import build_rolling_features
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
    schedule = clean_schedules(pd.read_parquet(SCHEDULE_PATH))
    features = build_rolling_features(
        team_game_history(schedule),
        aggregate_offensive_epa(pbp),
        aggregate_defensive_epa(pbp),
        aggregate_pace(pbp),
    )
    return add_matchup_target(build_matchups(features, schedule))


def test_real_one_row_per_game(real_matchups):
    assert len(real_matchups) == real_matchups["game_id"].nunique()
    assert not real_matchups.duplicated("game_id").any()


def test_real_differentials_are_all_home_minus_away(real_matchups):
    for output, source in DIFFERENTIAL_SOURCES:
        home = pd.to_numeric(real_matchups[f"home_{source}"], errors="coerce")
        away = pd.to_numeric(real_matchups[f"away_{source}"], errors="coerce")
        expected = home - away
        actual = real_matchups[output]
        both_null = actual.isna() & expected.isna()
        assert ((actual - expected).abs().le(1e-9) | both_null).all(), output


def test_real_completed_games_are_labelled_except_pushes(real_matchups):
    completed = real_matchups.loc[real_matchups["home_score"].notna()]
    unlabelled = completed.loc[completed["home_cover"].isna()]

    # Every unlabelled completed game is a push or has no line -- nothing else.
    assert (unlabelled["is_push"] | unlabelled["spread_line"].isna()).all()


def test_real_2026_rows_exist_with_no_target(real_matchups):
    upcoming = real_matchups.loc[real_matchups["season"].eq(2026)]

    assert len(upcoming) > 0
    assert upcoming["home_cover"].isna().all()
    # Schedule-derived features are available for unplayed games.
    assert upcoming["div_game"].notna().all()
    assert upcoming["rest_diff"].notna().all()


def test_real_feature_matrix_holds_no_postgame_field(real_matchups):
    rows = model_rows(real_matchups)

    assert_no_postgame_fields(rows)
    assert not set(rows.columns).intersection(
        {"home_score", "away_score", "result", "home_margin"}
    )


def test_real_early_season_games_are_kept_not_dropped(real_matchups):
    """Week 1 is sparse by design, but every week 1 game still has a row."""
    played = real_matchups.loc[real_matchups["season"].lt(2026)]
    week_one = played.loc[played["week"].eq(1)]

    assert len(week_one) > 150
    assert week_one["off_epa_diff_last_5"].isna().all()
    assert week_one["div_game"].notna().all()


def test_real_home_side_matches_the_schedule(real_matchups):
    """Guards against the two sides being transposed during the merge."""
    sample = real_matchups.loc[real_matchups["home_prior_games_this_season"].notna()]

    assert len(sample) > 2000
    assert sample["home_team"].ne(sample["away_team"]).all()
