"""Tests for weekly prediction generation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridiron.modeling.pipeline import build_pipeline, get_feature_columns
from gridiron.prediction.weekly import (
    PREDICTION_COLUMNS,
    SPREAD_SOURCE,
    STATUS_PENDING_NO_SPREAD,
    STATUS_PREDICTED,
    RefreshRecord,
    WeeklyPredictionError,
    format_report,
    predict_week,
    save_predictions,
    validate_schema,
)

FEATURES = get_feature_columns()
UPDATED_AT = "2026-09-22T20:04:54+00:00"


def _schema(names: list[str] | None = None) -> dict[str, object]:
    names = names or FEATURES
    return {"feature_names": names, "n_features": len(names), "target": "home_cover"}


def _fitted_pipeline():
    rng = np.random.default_rng(3)
    features = pd.DataFrame(rng.normal(0, 1, (300, len(FEATURES))), columns=FEATURES)
    labels = pd.Series((features["spread_line"] > 0).astype(int))
    return build_pipeline().fit(features, labels)


def _matchups(
    *,
    week: int = 3,
    games: int = 4,
    completed: int = 0,
    without_spread: int = 0,
    season: int = 2026,
) -> pd.DataFrame:
    rng = np.random.default_rng(9)
    rows = []
    for index in range(games):
        rows.append(
            {
                "game_id": f"{season}_{week:02d}_A{index}_H{index}",
                "season": season,
                "week": week,
                "gameday": pd.Timestamp(f"{season}-09-27")
                + pd.Timedelta(days=index % 2),
                "home_team": f"H{index}",
                "away_team": f"A{index}",
                "spread_line": float(index - 1),
                "home_score": np.nan,
                "away_score": np.nan,
                "home_prior_games_this_season": 2,
            }
        )
    frame = pd.DataFrame(rows)

    for name in FEATURES:
        if name not in frame.columns:
            frame[name] = rng.normal(0, 1, len(frame))
    frame["spread_line"] = [float(index - 1) for index in range(games)]

    if completed:
        frame.loc[frame.index[:completed], "home_score"] = 24.0
        frame.loc[frame.index[:completed], "away_score"] = 17.0
    if without_spread:
        frame.loc[frame.index[-without_spread:], "spread_line"] = np.nan
    return frame


def _record(**overrides) -> RefreshRecord:
    settings = {
        "refreshed": True,
        "data_updated_at": UPDATED_AT,
        "schedule_path": "data/raw/schedules.parquet",
        "schedule_games": 272,
        "completed_games": 32,
    }
    settings.update(overrides)
    return RefreshRecord(**settings)


# --- the prediction table --------------------------------------------------


def test_the_table_has_every_required_column():
    table = predict_week(
        _matchups(), 2026, 3, _fitted_pipeline(), _schema(), UPDATED_AT
    )

    assert list(table.columns) == PREDICTION_COLUMNS
    for required in (
        "season",
        "week",
        "game_id",
        "gameday",
        "away_team",
        "home_team",
        "spread_line",
        "away_cover_probability",
        "home_cover_probability",
        "predicted_side",
        "confidence",
        "recommendation_tier",
        "data_updated_at",
    ):
        assert required in table.columns


def test_probabilities_sum_to_one():
    table = predict_week(
        _matchups(), 2026, 3, _fitted_pipeline(), _schema(), UPDATED_AT
    )
    total = table["home_cover_probability"] + table["away_cover_probability"]

    assert total.to_numpy() == pytest.approx(np.ones(len(table)))


def test_completed_games_are_excluded():
    """A played game has a result, not a recommendation."""
    table = predict_week(
        _matchups(games=4, completed=2),
        2026,
        3,
        _fitted_pipeline(),
        _schema(),
        UPDATED_AT,
    )

    assert len(table) == 2
    assert not table["game_id"].str.endswith(("A0_H0", "A1_H1")).any()


def test_a_fully_completed_week_is_refused():
    with pytest.raises(WeeklyPredictionError, match="already been played"):
        predict_week(
            _matchups(games=3, completed=3),
            2026,
            3,
            _fitted_pipeline(),
            _schema(),
            UPDATED_AT,
        )


def test_games_without_a_spread_are_marked_pending():
    table = predict_week(
        _matchups(games=4, without_spread=2),
        2026,
        3,
        _fitted_pipeline(),
        _schema(),
        UPDATED_AT,
    )

    pending = table.loc[table["status"].eq(STATUS_PENDING_NO_SPREAD)]
    assert len(pending) == 2
    assert pending["home_cover_probability"].isna().all()
    assert (pending["predicted_side"] == "").all()
    assert (pending["recommendation_tier"] == "").all()


def test_a_pending_game_is_still_listed():
    """A missing line means the target is undefined, not the game absent."""
    table = predict_week(
        _matchups(games=4, without_spread=4),
        2026,
        3,
        _fitted_pipeline(),
        _schema(),
        UPDATED_AT,
    )

    assert len(table) == 4
    assert table["status"].eq(STATUS_PENDING_NO_SPREAD).all()


def test_the_predicted_side_follows_the_probability():
    table = predict_week(
        _matchups(games=6), 2026, 3, _fitted_pipeline(), _schema(), UPDATED_AT
    )
    priced = table.loc[table["status"].eq(STATUS_PREDICTED)]

    for row in priced.itertuples():
        expected = row.home_team if row.home_cover_probability >= 0.5 else row.away_team
        assert row.predicted_side == expected


def test_confidence_is_the_distance_from_a_coin_flip():
    table = predict_week(
        _matchups(games=6), 2026, 3, _fitted_pipeline(), _schema(), UPDATED_AT
    )
    priced = table.loc[table["status"].eq(STATUS_PREDICTED)]

    expected = (priced["home_cover_probability"] - 0.5).abs()
    assert priced["confidence"].to_numpy() == pytest.approx(expected.to_numpy())
    assert (priced["confidence"] <= 0.5).all()


def test_every_priced_game_gets_a_recommendation_tier():
    table = predict_week(
        _matchups(games=6), 2026, 3, _fitted_pipeline(), _schema(), UPDATED_AT
    )
    priced = table.loc[table["status"].eq(STATUS_PREDICTED)]

    assert (priced["recommendation_tier"].str.len() > 0).all()


def test_an_unknown_week_is_refused():
    with pytest.raises(WeeklyPredictionError, match="No games found"):
        predict_week(
            _matchups(week=3), 2026, 14, _fitted_pipeline(), _schema(), UPDATED_AT
        )


# --- spread provenance -----------------------------------------------------


def test_the_spread_used_is_recorded_with_its_source():
    table = predict_week(
        _matchups(), 2026, 3, _fitted_pipeline(), _schema(), UPDATED_AT
    )

    assert (table["spread_line_used"] == table["spread_line"]).all()
    assert (table["spread_source"] == SPREAD_SOURCE).all()
    assert (table["spread_timestamp"] == UPDATED_AT).all()


def test_the_spread_is_not_described_as_a_live_quote():
    """The label must not imply a real-time sportsbook price."""
    assert "not a live sportsbook quote" in SPREAD_SOURCE
    assert "schedule line" in SPREAD_SOURCE


def test_the_refresh_timestamp_is_carried_on_every_row():
    table = predict_week(
        _matchups(), 2026, 3, _fitted_pipeline(), _schema(), UPDATED_AT
    )

    assert (table["data_updated_at"] == UPDATED_AT).all()


# --- schema validation -----------------------------------------------------


def test_the_generated_features_match_the_training_schema():
    validate_schema(_matchups(), _schema())


def test_a_missing_feature_is_rejected():
    frame = _matchups().drop(columns="rest_diff")

    with pytest.raises(WeeklyPredictionError, match="rest_diff"):
        validate_schema(frame, _schema())


def test_a_reordered_schema_is_rejected():
    """Order matters: the model is scored against positions, not names."""
    reversed_schema = _schema(list(reversed(FEATURES)))

    with pytest.raises(WeeklyPredictionError, match="order does not match"):
        validate_schema(_matchups(), reversed_schema)


def test_an_empty_schema_is_rejected():
    with pytest.raises(WeeklyPredictionError, match="no feature names"):
        validate_schema(_matchups(), {"feature_names": []})


def test_prediction_validates_the_schema_before_scoring():
    with pytest.raises(WeeklyPredictionError, match="order does not match"):
        predict_week(
            _matchups(),
            2026,
            3,
            _fitted_pipeline(),
            _schema(list(reversed(FEATURES))),
            UPDATED_AT,
        )


# --- the report ------------------------------------------------------------


def test_the_report_carries_the_refresh_timestamp():
    table = predict_week(
        _matchups(), 2026, 3, _fitted_pipeline(), _schema(), UPDATED_AT
    )

    report = format_report(table, _record())

    assert UPDATED_AT in report
    assert "Data refreshed at" in report


def test_the_report_states_what_the_spread_is():
    table = predict_week(
        _matchups(), 2026, 3, _fitted_pipeline(), _schema(), UPDATED_AT
    )

    report = format_report(table, _record())

    assert SPREAD_SOURCE in report


def test_the_report_carries_the_no_edge_warning():
    """No weekly output may read as betting advice."""
    table = predict_week(
        _matchups(), 2026, 3, _fitted_pipeline(), _schema(), UPDATED_AT
    )

    report = format_report(table, _record())

    assert "no demonstrated edge" in report
    assert "not betting advice" in report


def test_the_report_lists_pending_games_as_pending():
    table = predict_week(
        _matchups(games=4, without_spread=2),
        2026,
        3,
        _fitted_pipeline(),
        _schema(),
        UPDATED_AT,
    )

    report = format_report(table, _record())

    assert "PENDING" in report


def test_the_report_repeats_any_refresh_notes():
    table = predict_week(
        _matchups(), 2026, 3, _fitted_pipeline(), _schema(), UPDATED_AT
    )

    report = format_report(table, _record(notes=["No play-by-play published yet."]))

    assert "No play-by-play published yet." in report


def test_the_report_includes_the_validation_summary_when_given():
    class _Metadata:
        validation_summary = {"roc_auc": 0.4817, "roi": -0.034}

    table = predict_week(
        _matchups(), 2026, 3, _fitted_pipeline(), _schema(), UPDATED_AT
    )

    report = format_report(table, _record(), _Metadata())

    assert "roc_auc" in report
    assert "0.48170" in report


# --- saving ----------------------------------------------------------------


def test_predictions_and_report_are_written(tmp_path):
    table = predict_week(
        _matchups(), 2026, 3, _fitted_pipeline(), _schema(), UPDATED_AT
    )
    report = format_report(table, _record())

    paths = save_predictions(table, report, 2026, 3, tmp_path)

    assert paths["csv"].exists()
    assert paths["report"].exists()
    assert paths["csv"].name == "week_03_predictions.csv"
    assert pd.read_csv(paths["csv"]).shape[0] == len(table)


def test_the_saved_report_matches_what_was_generated(tmp_path):
    table = predict_week(
        _matchups(), 2026, 3, _fitted_pipeline(), _schema(), UPDATED_AT
    )
    report = format_report(table, _record())

    paths = save_predictions(table, report, 2026, 3, tmp_path)

    assert paths["report"].read_text(encoding="utf-8") == report


# --- rolling metrics update --------------------------------------------------


def test_rebuilding_after_new_results_changes_the_features():
    """The acceptance criterion: new games must move the rolling numbers.

    Rather than re-download, this exercises the property the weekly rebuild
    relies on -- that a team's rolling feature for a week is built from the
    games completed before it, so adding a result changes the next week.
    """
    from gridiron.features.rolling import add_rolling_features

    base = pd.DataFrame(
        {
            "team": ["AAA"] * 3,
            "season": [2026] * 3,
            "week": [1, 2, 3],
            "game_id": ["g1", "g2", "g3"],
            "gameday": pd.to_datetime(["2026-09-10", "2026-09-17", "2026-09-24"]),
            "offensive_epa_per_play": [0.10, 0.30, np.nan],
            "defensive_epa_strength_per_play": [0.0, 0.0, np.nan],
            "pace_plays_per_game": [60.0, 60.0, np.nan],
            "point_margin": [7.0, -3.0, np.nan],
            "team_spread": [1.0, 1.0, 1.0],
            "covered": pd.array([1, 0, pd.NA], dtype="Int8"),
        }
    )

    before = add_rolling_features(base.iloc[[0, 2]].copy())
    after = add_rolling_features(base.copy())

    week_three_before = before.loc[before["week"].eq(3), "off_epa_season"].iloc[0]
    week_three_after = after.loc[after["week"].eq(3), "off_epa_season"].iloc[0]

    assert week_three_before == pytest.approx(0.10)
    assert week_three_after == pytest.approx(0.20)
    assert week_three_before != pytest.approx(week_three_after)
