"""Tests for season-based walk-forward validation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridiron.modeling.evaluate import (
    AGGREGATE_ROWS,
    CONFIDENCE_TIERS,
    PREDICTION_COLUMNS,
    UNINFORMATIVE_BRIER,
    UNINFORMATIVE_LOG_LOSS,
    EvaluationError,
    aggregate_metrics,
    probability_quality,
    profit_by_season,
    run_walk_forward,
    season_metrics,
)
from gridiron.modeling.splits import (
    MIN_TRAIN_ROWS,
    SplitError,
    describe_splits,
    season_walk_forward_splits,
)


def _frame(
    seasons: tuple[int, ...] = (2016, 2017, 2018, 2019, 2020),
    per_season: int = 260,
    *,
    seed: int = 5,
) -> pd.DataFrame:
    """A matchup frame with several complete, chronologically ordered seasons."""
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for index in range(per_season):
            week = index % 17 + 1
            rows.append(
                {
                    "game_id": f"{season}_{index:03d}",
                    "season": season,
                    "week": week,
                    # September of the season year, running into December.
                    "gameday": pd.Timestamp(f"{season}-09-05")
                    + pd.Timedelta(days=7 * (week - 1)),
                }
            )
    frame = pd.DataFrame(rows)
    count = len(frame)

    frame["spread_line"] = rng.normal(0, 5, count)
    frame["off_epa_diff_last_5"] = rng.normal(0, 0.2, count)
    frame["def_epa_strength_diff_last_5"] = rng.normal(0, 0.2, count)
    frame["pace_diff_last_5"] = rng.normal(0, 5, count)
    frame["rest_diff"] = rng.integers(-7, 8, count).astype(float)
    frame["point_margin_diff_last_5"] = rng.normal(0, 8, count)
    frame["win_pct_diff_last_5"] = rng.normal(0, 0.3, count)
    frame["home_short_week"] = rng.random(count) < 0.15
    frame["away_short_week"] = rng.random(count) < 0.15
    frame["div_game"] = rng.integers(0, 2, count).astype("int8")
    frame["week_1_flag"] = (frame["week"] == 1).astype("int8")

    signal = frame["off_epa_diff_last_5"] * 2.5 + rng.normal(0, 0.6, count)
    frame["home_cover"] = (signal > 0).astype(float)
    frame["is_push"] = False
    return frame


# --- split generation ------------------------------------------------------


def test_training_data_expands_forward_through_time():
    frame = _frame()
    splits = list(season_walk_forward_splits(frame, 2017, 2020))

    sizes = [split.train_size for split in splits]
    counts = [len(split.train_seasons) for split in splits]

    assert sizes == sorted(sizes)
    assert counts == [1, 2, 3, 4]
    assert [split.validation_season for split in splits] == [2017, 2018, 2019, 2020]


def test_no_season_appears_in_both_halves_of_a_fold():
    frame = _frame()

    for split in season_walk_forward_splits(frame, 2017, 2020):
        assert split.validation_season not in split.train_seasons
        train_seasons = set(frame.loc[split.train_indices, "season"])
        validation_seasons = set(frame.loc[split.validation_indices, "season"])
        assert not train_seasons & validation_seasons
        assert validation_seasons == {split.validation_season}


def test_every_validation_game_is_after_every_training_game():
    frame = _frame()

    for split in season_walk_forward_splits(frame, 2017, 2020):
        latest_train = frame.loc[split.train_indices, "gameday"].max()
        earliest_validation = frame.loc[split.validation_indices, "gameday"].min()
        assert latest_train < earliest_validation


def test_training_seasons_are_all_earlier_than_the_validation_season():
    frame = _frame()

    for split in season_walk_forward_splits(frame, 2017, 2020):
        assert all(season < split.validation_season for season in split.train_seasons)


def test_split_carries_the_four_required_fields():
    frame = _frame()
    split = next(iter(season_walk_forward_splits(frame, 2018, 2018)))

    assert isinstance(split.train_indices, pd.Index)
    assert isinstance(split.validation_indices, pd.Index)
    assert isinstance(split.train_seasons, tuple)
    assert isinstance(split.validation_season, int)


def test_out_of_order_input_still_splits_by_season():
    frame = _frame().sample(frac=1.0, random_state=1)

    for split in season_walk_forward_splits(frame, 2018, 2020):
        assert set(frame.loc[split.validation_indices, "season"]) == {
            split.validation_season
        }
        latest_train = frame.loc[split.train_indices, "gameday"].max()
        assert latest_train < frame.loc[split.validation_indices, "gameday"].min()


def test_chronology_violation_is_caught():
    """A training game moved past the validation season must raise."""
    frame = _frame()
    frame.loc[frame.index[0], "gameday"] = pd.Timestamp("2030-01-01")

    with pytest.raises(SplitError, match="at or after"):
        list(season_walk_forward_splits(frame, 2018, 2018))


def test_seasons_absent_from_the_frame_are_skipped():
    frame = _frame(seasons=(2016, 2017, 2018))

    splits = list(season_walk_forward_splits(frame, 2017, 2025))

    assert [split.validation_season for split in splits] == [2017, 2018]


def test_the_first_season_is_never_validated():
    """There is nothing earlier to train on."""
    frame = _frame()

    splits = list(season_walk_forward_splits(frame, 2016, 2020))

    assert 2016 not in [split.validation_season for split in splits]


def test_a_fold_with_too_little_history_is_skipped():
    frame = _frame(seasons=(2016, 2017), per_season=MIN_TRAIN_ROWS - 50)

    with pytest.raises(SplitError, match="No usable folds"):
        list(season_walk_forward_splits(frame, 2017, 2017))


def test_reversed_season_bounds_are_rejected():
    with pytest.raises(SplitError, match="is after"):
        list(season_walk_forward_splits(_frame(), 2020, 2018))


def test_missing_season_column_is_rejected():
    with pytest.raises(SplitError, match="season"):
        list(season_walk_forward_splits(pd.DataFrame({"game_id": ["a"]})))


def test_describe_splits_shows_the_expanding_window():
    frame = _frame()
    table = describe_splits(list(season_walk_forward_splits(frame, 2017, 2020)))

    assert list(table["train_season_count"]) == [1, 2, 3, 4]
    assert table["train_games"].is_monotonic_increasing


# --- running the backtest --------------------------------------------------


@pytest.fixture(scope="module")
def backtest() -> tuple[pd.DataFrame, pd.DataFrame]:
    return run_walk_forward(_frame(), 2018, 2020)


def test_predictions_have_every_required_column(backtest):
    predictions, _ = backtest

    assert list(predictions.columns) == PREDICTION_COLUMNS


def test_folds_are_concatenated_into_one_out_of_sample_table(backtest):
    predictions, folds = backtest

    assert len(predictions) == folds["validation_games"].sum()
    assert set(predictions["season"]) == {2018, 2019, 2020}


def test_every_game_is_predicted_exactly_once(backtest):
    predictions, _ = backtest

    assert not predictions["game_id"].duplicated().any()


def test_predicted_side_agrees_with_the_prediction(backtest):
    predictions, _ = backtest

    home = predictions["predicted_home_cover"].eq(1)
    assert (predictions.loc[home, "predicted_side"] == "HOME").all()
    assert (predictions.loc[~home, "predicted_side"] == "AWAY").all()


def test_probability_and_prediction_agree_at_the_threshold(backtest):
    predictions, _ = backtest

    assert (
        predictions["predicted_home_cover"]
        == (predictions["home_cover_probability"] >= 0.5).astype(int)
    ).all()


def test_confidence_is_the_edge_over_a_coin_flip(backtest):
    """The specification defines confidence as abs(p - 0.50)."""
    predictions, _ = backtest
    probability = predictions["home_cover_probability"]

    expected = (probability - 0.5).abs()
    assert predictions["confidence"].to_numpy() == pytest.approx(expected.to_numpy())
    assert (predictions["confidence"] >= 0.0).all()
    assert (predictions["confidence"] <= 0.5).all()


def test_away_probability_is_the_complement(backtest):
    predictions, _ = backtest

    total = (
        predictions["home_cover_probability"] + predictions["away_cover_probability"]
    )
    assert total.to_numpy() == pytest.approx(np.ones(len(predictions)))


def test_every_prediction_carries_a_confidence_tier(backtest):
    predictions, _ = backtest

    assert predictions["confidence_tier"].notna().all()
    assert set(predictions["confidence_tier"]).issubset(
        {label for _, label in CONFIDENCE_TIERS}
    )


def test_fold_summary_records_the_expanding_window(backtest):
    _, folds = backtest

    assert folds["train_games"].is_monotonic_increasing
    assert list(folds["validation_season"]) == [2018, 2019, 2020]


def test_a_later_fold_trains_on_more_data_than_an_earlier_one(backtest):
    _, folds = backtest

    assert folds.iloc[-1]["train_games"] > folds.iloc[0]["train_games"]


def test_backtest_is_reproducible():
    first, _ = run_walk_forward(_frame(), 2018, 2019)
    second, _ = run_walk_forward(_frame(), 2018, 2019)

    pd.testing.assert_frame_equal(first, second)


def test_backtest_excludes_unsettled_games():
    frame = _frame()
    frame.loc[frame["season"].eq(2019), "home_cover"] = np.nan

    predictions, _ = run_walk_forward(frame, 2018, 2020)

    assert 2019 not in set(predictions["season"])


# --- metrics ---------------------------------------------------------------


def test_season_metrics_cover_every_validation_season(backtest):
    predictions, _ = backtest
    seasons = season_metrics(predictions)

    assert list(seasons["season"]) == [2018, 2019, 2020]
    assert (seasons["wins"] + seasons["losses"] == seasons["games"]).all()


def test_aggregate_reports_every_required_metric(backtest):
    predictions, _ = backtest
    table = aggregate_metrics(predictions)

    assert list(table["metric"]) == AGGREGATE_ROWS
    for name in (
        "mean_season_accuracy",
        "median_season_accuracy",
        "worst_season_accuracy",
        "best_season_accuracy",
        "accuracy_std",
        "log_loss",
        "brier_score",
        "roi",
    ):
        assert name in set(table["metric"])


def test_worst_and_best_bracket_the_mean(backtest):
    predictions, _ = backtest
    values = aggregate_metrics(predictions).set_index("metric")["value"]

    assert values["worst_season_accuracy"] <= values["mean_season_accuracy"]
    assert values["mean_season_accuracy"] <= values["best_season_accuracy"]


def test_aggregate_names_the_worst_and_best_seasons(backtest):
    predictions, _ = backtest
    table = aggregate_metrics(predictions)
    seasons = season_metrics(predictions)

    assert table.attrs["worst_season"] == int(
        seasons.loc[seasons["accuracy"].idxmin(), "season"]
    )
    assert table.attrs["best_season"] == int(
        seasons.loc[seasons["accuracy"].idxmax(), "season"]
    )


def test_a_perfect_model_scores_as_expected():
    """Sanity check on the metric code itself, not on any real model."""
    predictions = pd.DataFrame(
        {
            "game_id": ["a", "b", "c", "d"],
            "season": [2019, 2019, 2020, 2020],
            "week": [1, 2, 1, 2],
            "actual_home_cover": [1, 0, 1, 0],
            "predicted_home_cover": [1, 0, 1, 0],
            "home_cover_probability": [0.99, 0.01, 0.99, 0.01],
            "away_cover_probability": [0.01, 0.99, 0.01, 0.99],
            "predicted_side": ["HOME", "AWAY", "HOME", "AWAY"],
            "confidence": [0.49, 0.49, 0.49, 0.49],
            "confidence_tier": ["high model confidence"] * 4,
        }
    )
    values = aggregate_metrics(predictions).set_index("metric")["value"]

    assert values["pooled_accuracy"] == 1.0
    assert values["accuracy_std"] == 0.0
    assert values["brier_score"] < 0.001
    assert values["roi"] > 0


def test_uninformative_reference_values_are_what_they_claim():
    assert pytest.approx(0.6931, abs=1e-4) == UNINFORMATIVE_LOG_LOSS
    assert UNINFORMATIVE_BRIER == 0.25


def test_probability_quality_compares_against_knowing_nothing():
    """A flat 0.5 predictor must score exactly at the reference values."""
    predictions = pd.DataFrame(
        {
            "game_id": ["a", "b", "c", "d"],
            "season": [2019, 2019, 2020, 2020],
            "week": [1, 2, 1, 2],
            "actual_home_cover": [1, 0, 1, 0],
            "predicted_home_cover": [1, 1, 1, 1],
            "home_cover_probability": [0.5, 0.5, 0.5, 0.5],
            "away_cover_probability": [0.5, 0.5, 0.5, 0.5],
            "predicted_side": ["HOME"] * 4,
            "confidence": [0.0] * 4,
            "confidence_tier": ["lean only"] * 4,
        }
    )
    quality = probability_quality(predictions)

    assert quality.loc[0, "model"] == pytest.approx(UNINFORMATIVE_LOG_LOSS)
    assert quality.loc[1, "model"] == pytest.approx(UNINFORMATIVE_BRIER)
    assert not quality["model_is_better"].any()


def test_profit_curve_covers_every_out_of_sample_game(backtest):
    predictions, _ = backtest
    curve = profit_by_season(predictions)

    assert len(curve) == len(predictions)


def test_metrics_reject_a_frame_without_predictions():
    with pytest.raises(EvaluationError, match="missing column"):
        aggregate_metrics(pd.DataFrame({"game_id": ["a"]}))


def test_metrics_reject_an_empty_frame():
    empty = pd.DataFrame(columns=PREDICTION_COLUMNS)

    with pytest.raises(EvaluationError, match="No predictions"):
        aggregate_metrics(empty)
