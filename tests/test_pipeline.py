"""Tests for the model pipeline and training."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from gridiron.modeling.pipeline import (
    IMPUTE_STEP,
    MODEL_STEP,
    SCALE_STEP,
    PipelineError,
    build_pipeline,
    feature_matrix,
    fitted_feature_names,
    get_feature_columns,
    imputation_medians,
    model_coefficients,
    scaling_statistics,
    target_vector,
)
from gridiron.modeling.train import (
    METADATA_FILENAME,
    TrainingError,
    chronological_split,
    load_model,
    save_model,
    train_from_matchups,
    train_model,
    training_metadata,
)

FEATURES = get_feature_columns()


def _frame(rows: int = 400, *, seed: int = 3, missing: int = 0) -> pd.DataFrame:
    """A synthetic matchup frame with a mild, learnable signal."""
    rng = np.random.default_rng(seed)
    seasons = np.repeat([2020, 2021, 2022, 2023], rows // 4)[:rows]
    frame = pd.DataFrame(
        {
            "game_id": [f"g{index:04d}" for index in range(rows)],
            "season": seasons,
            "week": np.tile(np.arange(1, 18), rows)[:rows],
            "gameday": [
                pd.Timestamp("2020-09-10") + pd.Timedelta(days=index)
                for index in range(rows)
            ],
            "spread_line": rng.normal(0, 5, rows),
            "off_epa_diff_last_5": rng.normal(0, 0.2, rows),
            "def_epa_strength_diff_last_5": rng.normal(0, 0.2, rows),
            "pace_diff_last_5": rng.normal(0, 5, rows),
            "rest_diff": rng.integers(-7, 8, rows).astype(float),
            "point_margin_diff_last_5": rng.normal(0, 8, rows),
            "win_pct_diff_last_5": rng.normal(0, 0.3, rows),
            "home_short_week": rng.random(rows) < 0.15,
            "away_short_week": rng.random(rows) < 0.15,
            "div_game": rng.integers(0, 2, rows).astype("int8"),
            "week_1_flag": (np.tile(np.arange(1, 18), rows)[:rows] == 1).astype("int8"),
        }
    )
    signal = frame["off_epa_diff_last_5"] * 3 + rng.normal(0, 0.5, rows)
    frame["home_cover"] = (signal > 0).astype(float)
    frame["is_push"] = False

    if missing:
        frame.loc[frame.index[:missing], "off_epa_diff_last_5"] = np.nan
    return frame


# --- feature columns and pipeline shape ------------------------------------


def test_feature_columns_are_stable_and_ordered():
    first = get_feature_columns()
    second = get_feature_columns()

    assert first == second
    assert len(first) == len(set(first))
    assert first[0] == "spread_line"


def test_mutating_the_returned_list_does_not_affect_the_next_call():
    columns = get_feature_columns()
    columns.append("injected")

    assert "injected" not in get_feature_columns()


def test_pipeline_steps_are_in_the_required_order():
    """Impute, then standardise, then fit. The order is not interchangeable."""
    pipeline = build_pipeline()

    assert [name for name, _ in pipeline.steps] == [
        IMPUTE_STEP,
        SCALE_STEP,
        MODEL_STEP,
    ]
    assert isinstance(pipeline.named_steps[IMPUTE_STEP], SimpleImputer)
    assert isinstance(pipeline.named_steps[SCALE_STEP], StandardScaler)
    assert isinstance(pipeline.named_steps[MODEL_STEP], LogisticRegression)


def test_imputation_uses_the_median():
    assert build_pipeline().named_steps[IMPUTE_STEP].strategy == "median"


def test_build_pipeline_passes_through_its_arguments():
    pipeline = build_pipeline(c=0.25, class_weight="balanced")
    classifier = pipeline.named_steps[MODEL_STEP]

    assert classifier.C == 0.25
    assert classifier.class_weight == "balanced"


def test_build_pipeline_rejects_a_nonpositive_c():
    with pytest.raises(PipelineError, match="positive"):
        build_pipeline(c=0.0)


def test_pipeline_is_unfitted_until_fit_is_called():
    pipeline = build_pipeline()

    assert not hasattr(pipeline.named_steps[IMPUTE_STEP], "statistics_")
    with pytest.raises(PipelineError, match="not been fitted"):
        fitted_feature_names(pipeline)


# --- the feature matrix ----------------------------------------------------


def test_pipeline_accepts_the_raw_feature_columns():
    """No manual imputation or scaling before handing data to the pipeline."""
    frame = _frame(missing=20)
    matrix = feature_matrix(frame)

    assert matrix.isna().any().any()

    pipeline = build_pipeline().fit(matrix, target_vector(frame))

    assert pipeline.predict(matrix).shape == (len(frame),)


def test_feature_matrix_keeps_the_agreed_order():
    frame = _frame()
    shuffled = frame[list(reversed(frame.columns))]

    assert list(feature_matrix(shuffled).columns) == FEATURES


def test_feature_matrix_casts_booleans_to_float():
    matrix = feature_matrix(_frame())

    assert matrix["home_short_week"].dtype == float


def test_feature_matrix_names_every_missing_column():
    frame = _frame().drop(columns=["rest_diff", "div_game"])

    with pytest.raises(PipelineError, match="div_game") as error:
        feature_matrix(frame)
    assert "rest_diff" in str(error.value)


def test_feature_matrix_refuses_a_postgame_field():
    frame = _frame().assign(home_score=21.0)

    with pytest.raises(PipelineError, match="home_score"):
        feature_matrix(frame, [*FEATURES, "home_score"])


def test_feature_matrix_refuses_the_target_as_a_feature():
    with pytest.raises(PipelineError, match="home_cover"):
        feature_matrix(_frame(), [*FEATURES, "home_cover"])


def test_target_rejects_unsettled_rows():
    frame = _frame()
    frame.loc[frame.index[0], "home_cover"] = np.nan

    with pytest.raises(PipelineError, match="no settled result"):
        target_vector(frame)


# --- leakage: the test fold must not reach the fitted statistics -----------


def test_imputation_medians_come_only_from_the_training_rows():
    """The held-out rows must not move the median the imputer learned."""
    frame = _frame(rows=400, missing=40)
    train_frame, test_frame = chronological_split(frame, test_fraction=0.25)

    pipeline = train_model(feature_matrix(train_frame), target_vector(train_frame))
    learned = imputation_medians(pipeline)

    expected = feature_matrix(train_frame).median()
    pd.testing.assert_series_equal(
        learned, expected, check_names=False, check_dtype=False
    )

    full_median = feature_matrix(frame)["off_epa_diff_last_5"].median()
    assert learned["off_epa_diff_last_5"] != pytest.approx(full_median)


def test_scaling_statistics_come_only_from_the_training_rows():
    frame = _frame(rows=400)
    train_frame, _ = chronological_split(frame, test_fraction=0.25)

    pipeline = train_model(feature_matrix(train_frame), target_vector(train_frame))
    learned = scaling_statistics(pipeline)

    train_mean = feature_matrix(train_frame).mean()
    full_mean = feature_matrix(frame).mean()

    assert learned["mean"].to_numpy() == pytest.approx(train_mean.to_numpy())
    assert learned["mean"]["spread_line"] != pytest.approx(full_mean["spread_line"])


def test_fitting_on_more_rows_changes_the_statistics():
    """Confirms the previous two tests could have failed."""
    frame = _frame(rows=400, missing=40)
    train_frame, _ = chronological_split(frame, test_fraction=0.25)

    on_train = train_model(feature_matrix(train_frame), target_vector(train_frame))
    on_all = train_model(feature_matrix(frame), target_vector(frame))

    assert not np.allclose(
        imputation_medians(on_train).to_numpy(),
        imputation_medians(on_all).to_numpy(),
    )


def test_predicting_does_not_refit_the_statistics():
    frame = _frame(rows=400, missing=40)
    train_frame, test_frame = chronological_split(frame, test_fraction=0.25)
    pipeline = train_model(feature_matrix(train_frame), target_vector(train_frame))

    before = imputation_medians(pipeline).copy()
    pipeline.predict(feature_matrix(test_frame))

    pd.testing.assert_series_equal(before, imputation_medians(pipeline))


def test_chronological_split_never_puts_later_games_in_training():
    frame = _frame(rows=400)
    train_frame, test_frame = chronological_split(frame, test_fraction=0.25)

    assert train_frame["gameday"].max() <= test_frame["gameday"].min()
    assert len(train_frame) + len(test_frame) == len(frame)


def test_season_holdout_keeps_seasons_whole():
    frame = _frame(rows=400)
    train_frame, test_frame = chronological_split(frame, test_seasons=[2023])

    assert set(test_frame["season"]) == {2023}
    assert 2023 not in set(train_frame["season"])


def test_split_rejects_an_unknown_test_season():
    with pytest.raises(TrainingError, match="No rows found"):
        chronological_split(_frame(), test_seasons=[1999])


def test_split_rejects_an_out_of_range_fraction():
    with pytest.raises(TrainingError, match="test_fraction"):
        chronological_split(_frame(), test_fraction=1.5)


# --- training contract -----------------------------------------------------


def test_training_fails_clearly_when_a_feature_is_missing():
    frame = _frame()
    matrix = feature_matrix(frame).drop(columns="rest_diff")

    with pytest.raises(TrainingError, match="rest_diff"):
        train_model(matrix, target_vector(frame))


def test_training_fails_on_an_unexpected_column():
    frame = _frame()
    matrix = feature_matrix(frame).assign(surprise=1.0)

    with pytest.raises(TrainingError, match="surprise"):
        train_model(matrix, target_vector(frame))


def test_training_rejects_a_length_mismatch():
    frame = _frame()

    with pytest.raises(TrainingError, match="rows"):
        train_model(feature_matrix(frame), target_vector(frame).iloc[:-1])


def test_training_rejects_too_few_rows():
    frame = _frame(rows=40)

    with pytest.raises(TrainingError, match="at least"):
        train_model(feature_matrix(frame), target_vector(frame))


def test_training_rejects_a_single_class():
    frame = _frame()
    labels = pd.Series(1, index=frame.index)

    with pytest.raises(TrainingError, match="single class"):
        train_model(feature_matrix(frame), labels)


def test_training_rejects_a_numpy_matrix():
    """A bare array loses the feature names the contract depends on."""
    frame = _frame()

    with pytest.raises(TrainingError, match="DataFrame"):
        train_model(feature_matrix(frame).to_numpy(), target_vector(frame))


def test_training_reorders_columns_to_the_agreed_order():
    frame = _frame()
    shuffled = feature_matrix(frame)[list(reversed(FEATURES))]

    pipeline = train_model(shuffled, target_vector(frame))

    assert fitted_feature_names(pipeline) == FEATURES


# --- the fitted model ------------------------------------------------------


def test_fitted_model_exposes_predict_and_predict_proba():
    frame = _frame()
    matrix = feature_matrix(frame)
    pipeline = train_model(matrix, target_vector(frame))

    predictions = pipeline.predict(matrix)
    probabilities = pipeline.predict_proba(matrix)

    assert predictions.shape == (len(frame),)
    assert probabilities.shape == (len(frame), 2)
    assert set(np.unique(predictions)).issubset({0, 1})
    assert probabilities.sum(axis=1) == pytest.approx(np.ones(len(frame)))


def test_repeated_training_gives_identical_results():
    frame = _frame()
    matrix = feature_matrix(frame)
    labels = target_vector(frame)

    first = train_model(matrix, labels)
    second = train_model(matrix, labels)

    assert np.array_equal(
        first.named_steps[MODEL_STEP].coef_, second.named_steps[MODEL_STEP].coef_
    )
    assert np.array_equal(first.predict_proba(matrix), second.predict_proba(matrix))


def test_feature_order_is_stored_on_the_fitted_model():
    frame = _frame()
    pipeline = train_model(feature_matrix(frame), target_vector(frame))

    assert fitted_feature_names(pipeline) == FEATURES


def test_predicting_with_reordered_columns_is_rejected():
    """scikit-learn checks the stored order, so a silent mismatch cannot occur."""
    frame = _frame()
    matrix = feature_matrix(frame)
    pipeline = train_model(matrix, target_vector(frame))

    with pytest.raises(ValueError, match="feature names"):
        pipeline.predict(matrix[list(reversed(FEATURES))])


def test_coefficients_map_back_to_feature_names():
    frame = _frame()
    pipeline = train_model(feature_matrix(frame), target_vector(frame))

    table = model_coefficients(pipeline)

    assert set(table["feature"]) == set(FEATURES)
    assert len(table) == len(FEATURES)
    assert table["abs_coefficient"].is_monotonic_decreasing
    assert "intercept" in table.attrs


def test_the_strongest_coefficient_is_the_feature_carrying_the_signal():
    """The fixture builds the label from off_epa_diff_last_5 alone."""
    frame = _frame()
    pipeline = train_model(feature_matrix(frame), target_vector(frame))

    table = model_coefficients(pipeline)

    assert table.iloc[0]["feature"] == "off_epa_diff_last_5"
    assert table.iloc[0]["coefficient"] > 0


def test_odds_ratio_is_the_exponentiated_coefficient():
    frame = _frame()
    pipeline = train_model(feature_matrix(frame), target_vector(frame))
    table = model_coefficients(pipeline)

    assert table["odds_ratio"].to_numpy() == pytest.approx(
        np.exp(table["coefficient"].to_numpy())
    )


def test_coefficients_need_a_fitted_model():
    with pytest.raises(PipelineError, match="not been fitted"):
        model_coefficients(build_pipeline())


# --- persistence -----------------------------------------------------------


def test_saved_model_round_trips_with_its_feature_order(tmp_path):
    frame = _frame()
    matrix = feature_matrix(frame)
    labels = target_vector(frame)
    pipeline = train_model(matrix, labels)
    metadata = training_metadata(pipeline, frame, labels)

    save_model(pipeline, metadata, tmp_path)
    loaded, loaded_metadata = load_model(tmp_path)

    assert fitted_feature_names(loaded) == FEATURES
    assert loaded_metadata.feature_columns == FEATURES
    assert np.array_equal(loaded.predict_proba(matrix), pipeline.predict_proba(matrix))


def test_metadata_records_the_training_set(tmp_path):
    frame = _frame()
    labels = target_vector(frame)
    pipeline = train_model(feature_matrix(frame), labels)

    metadata = training_metadata(pipeline, frame, labels)

    assert metadata.rows == len(frame)
    assert 0.0 < metadata.positive_rate < 1.0
    assert metadata.first_gameday <= metadata.last_gameday


def test_loading_reports_a_missing_model(tmp_path):
    with pytest.raises(TrainingError, match="No saved model"):
        load_model(tmp_path)


def test_loading_detects_metadata_that_disagrees(tmp_path):
    frame = _frame()
    labels = target_vector(frame)
    pipeline = train_model(feature_matrix(frame), labels)
    metadata = training_metadata(pipeline, frame, labels)
    save_model(pipeline, metadata, tmp_path)

    path = tmp_path / METADATA_FILENAME
    import json

    payload = json.loads(path.read_text())
    payload["feature_columns"] = list(reversed(payload["feature_columns"]))
    path.write_text(json.dumps(payload))

    with pytest.raises(TrainingError, match="disagree"):
        load_model(tmp_path)


# --- end to end ------------------------------------------------------------


def test_train_from_matchups_splits_fits_and_describes():
    frame = _frame(rows=400)

    pipeline, train_frame, test_frame, metadata = train_from_matchups(
        frame, test_seasons=[2023]
    )

    assert set(test_frame["season"]) == {2023}
    assert metadata.rows == len(train_frame)
    assert fitted_feature_names(pipeline) == FEATURES
    assert pipeline.predict(feature_matrix(test_frame)).shape == (len(test_frame),)


def test_train_from_matchups_excludes_unsettled_games():
    frame = _frame(rows=400)
    frame.loc[frame.index[:20], "home_cover"] = np.nan

    _, train_frame, test_frame, _ = train_from_matchups(frame, test_fraction=0.25)

    assert len(train_frame) + len(test_frame) == len(frame) - 20


def test_min_training_rows_is_enforced_end_to_end():
    with pytest.raises(TrainingError, match="at least"):
        train_from_matchups(_frame(rows=60), test_fraction=0.5)
