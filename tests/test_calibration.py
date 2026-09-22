"""Tests for calibration assessment and the decision to calibrate."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from gridiron.eda import calibration_charts  # noqa: E402
from gridiron.modeling.calibration import (  # noqa: E402
    BIAS_TOLERANCE,
    BUCKET_LABELS,
    MIN_BUCKET_SAMPLES,
    CalibrationError,
    apply_calibrator,
    calibration_slope,
    compare_calibration,
    decide_calibration,
    discrimination,
    fit_calibrator,
    reliability_table,
    sparse_buckets,
    temporal_calibration_split,
)


def _predictions(
    probability: list[float],
    actual: list[int],
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "game_id": [f"g{index:04d}" for index in range(len(probability))],
            "season": seasons or [2020] * len(probability),
            "week": list(range(1, len(probability) + 1)),
            "home_cover_probability": probability,
            "actual_home_cover": actual,
        }
    )


def _well_calibrated(n_per_bucket: int = 200, seed: int = 4) -> pd.DataFrame:
    """Predictions whose stated probabilities really do hold."""
    rng = np.random.default_rng(seed)
    probability, actual = [], []
    for level in (0.35, 0.45, 0.55, 0.65, 0.75):
        probability.extend([level] * n_per_bucket)
        actual.extend(rng.binomial(1, level, n_per_bucket).tolist())
    return _predictions(probability, actual)


# --- the reliability table -------------------------------------------------


def test_reliability_table_covers_every_bucket():
    table = reliability_table(_well_calibrated())

    assert list(table["bucket"]) == BUCKET_LABELS


def test_every_row_carries_its_sample_count():
    table = reliability_table(_well_calibrated())

    assert "games" in table.columns
    assert table["games"].sum() == 1000


def test_observed_rate_matches_predictions_when_calibrated():
    table = reliability_table(_well_calibrated(n_per_bucket=2000))
    populated = table.loc[table["games"] > 0]

    assert (populated["gap"].abs() < 0.05).all()
    assert not populated["biased"].any()


def test_bias_is_detected_when_the_rate_is_wrong():
    """Predictions of 0.65 that only cover 35% of the time must be flagged."""
    predictions = _predictions([0.65] * 200, [1] * 70 + [0] * 130)

    table = reliability_table(predictions)
    row = table.loc[table["bucket"].eq("0.65+")].iloc[0]

    assert row["games"] == 200
    assert abs(row["gap"]) > BIAS_TOLERANCE
    assert bool(row["biased"])


def test_sparse_buckets_are_identified():
    predictions = _predictions(
        [0.47] * 500 + [0.72] * 5, [1, 0] * 250 + [1, 0, 1, 0, 1]
    )

    table = reliability_table(predictions)
    thin = sparse_buckets(table)

    assert len(thin) == 1
    assert thin.iloc[0]["bucket"] == "0.65+"
    assert thin.iloc[0]["games"] == 5


def test_a_sparse_bucket_is_never_called_biased():
    """Too few games to know means too few games to conclude."""
    predictions = _predictions([0.70] * 3, [0, 0, 0])

    table = reliability_table(predictions)
    row = table.loc[table["bucket"].eq("0.65+")].iloc[0]

    assert bool(row["sparse"])
    assert not bool(row["biased"])
    assert abs(row["gap"]) > BIAS_TOLERANCE


def test_standard_error_shrinks_as_the_bucket_grows():
    small = reliability_table(_predictions([0.47] * 40, [1, 0] * 20))
    large = reliability_table(_predictions([0.47] * 4000, [1, 0] * 2000))

    small_error = small.loc[small["games"] > 0, "standard_error"].iloc[0]
    large_error = large.loc[large["games"] > 0, "standard_error"].iloc[0]

    assert small_error > large_error


def test_reliability_needs_the_expected_columns():
    with pytest.raises(CalibrationError, match="missing column"):
        reliability_table(pd.DataFrame({"game_id": ["a"]}))


def test_reliability_rejects_an_empty_frame():
    with pytest.raises(CalibrationError, match="No predictions"):
        reliability_table(
            pd.DataFrame(columns=["home_cover_probability", "actual_home_cover"])
        )


# --- slope and discrimination ----------------------------------------------


def test_a_well_calibrated_model_has_slope_near_one():
    slope, _ = calibration_slope(_well_calibrated(n_per_bucket=4000))

    assert slope == pytest.approx(1.0, abs=0.2)


def test_an_inverted_model_has_a_negative_slope():
    """Higher predicted probability going with a lower rate is a sign error."""
    rng = np.random.default_rng(2)
    probability, actual = [], []
    for level in (0.35, 0.45, 0.55, 0.65):
        probability.extend([level] * 500)
        actual.extend(rng.binomial(1, 1 - level, 500).tolist())

    slope, _ = calibration_slope(_predictions(probability, actual))

    assert slope < 0


def test_discrimination_is_high_when_scores_rank_well():
    predictions = _predictions([0.2] * 100 + [0.8] * 100, [0] * 100 + [1] * 100)

    assert discrimination(predictions) == pytest.approx(1.0)


def test_discrimination_is_half_when_scores_are_meaningless():
    rng = np.random.default_rng(1)
    probability = rng.uniform(0.4, 0.6, 2000)
    actual = rng.binomial(1, 0.5, 2000)

    assert discrimination(_predictions(list(probability), list(actual))) == (
        pytest.approx(0.5, abs=0.05)
    )


# --- the decision ----------------------------------------------------------


def test_calibration_is_declined_when_scores_do_not_rank():
    """The central case: bias is present but there is nothing to calibrate."""
    rng = np.random.default_rng(7)
    probability = list(rng.uniform(0.40, 0.60, 1500))
    actual = list(rng.binomial(1, 0.49, 1500))

    decision = decide_calibration(_predictions(probability, actual))

    assert not decision.calibrate
    assert any("do not rank" in reason for reason in decision.reasons)


def test_calibration_is_declined_when_the_slope_is_negative():
    rng = np.random.default_rng(3)
    probability, actual = [], []
    for level in (0.30, 0.42, 0.58, 0.70):
        probability.extend([level] * 600)
        actual.extend(rng.binomial(1, 1 - level, 600).tolist())

    decision = decide_calibration(_predictions(probability, actual))

    assert not decision.calibrate
    assert decision.slope < 0


def test_calibration_is_declined_when_there_is_no_bias():
    decision = decide_calibration(_well_calibrated(n_per_bucket=3000))

    assert not decision.calibrate
    assert any("no well-populated bucket" in r.lower() for r in decision.reasons)


def test_calibration_is_recommended_for_a_ranking_but_biased_model():
    """Scores that rank correctly but are systematically shifted should calibrate."""
    rng = np.random.default_rng(11)
    probability, actual = [], []
    # Stated probabilities are far too confident, but ordering is correct.
    for stated, truth in ((0.40, 0.47), (0.50, 0.52), (0.60, 0.70), (0.70, 0.85)):
        probability.extend([stated] * 600)
        actual.extend(rng.binomial(1, truth, 600).tolist())

    decision = decide_calibration(_predictions(probability, actual))

    assert decision.auc > 0.5
    assert decision.slope > 0
    assert decision.biased_buckets > 0
    assert decision.calibrate


def test_the_decision_always_records_its_evidence():
    decision = decide_calibration(_well_calibrated())

    assert len(decision.reasons) >= 3
    assert any("AUC" in reason for reason in decision.reasons)
    assert any("slope" in reason for reason in decision.reasons)


# --- temporal validity -----------------------------------------------------


def test_calibration_data_comes_after_the_fitting_data():
    frame = pd.DataFrame(
        {
            "season": [2016] * 10 + [2017] * 10 + [2018] * 10,
            "game_id": range(30),
        }
    )

    fit, calibrate = temporal_calibration_split(frame)

    assert set(fit["season"]) == {2016, 2017}
    assert set(calibrate["season"]) == {2018}
    assert max(fit["season"]) < min(calibrate["season"])


def test_fitting_and_calibration_rows_are_disjoint():
    frame = pd.DataFrame(
        {"season": [2016] * 10 + [2017] * 10 + [2018] * 10, "game_id": range(30)}
    )

    fit, calibrate = temporal_calibration_split(frame)

    assert not set(fit["game_id"]) & set(calibrate["game_id"])
    assert len(fit) + len(calibrate) == len(frame)


def test_more_calibration_seasons_can_be_held_back():
    frame = pd.DataFrame({"season": [2016] * 5 + [2017] * 5 + [2018] * 5 + [2019] * 5})

    fit, calibrate = temporal_calibration_split(frame, calibration_seasons=2)

    assert set(calibrate["season"]) == {2018, 2019}
    assert max(fit["season"]) < min(calibrate["season"])


def test_a_single_training_season_cannot_be_split():
    frame = pd.DataFrame({"season": [2016] * 10})

    with pytest.raises(CalibrationError, match="Need more than"):
        temporal_calibration_split(frame)


def test_temporal_split_needs_a_season_column():
    with pytest.raises(CalibrationError, match="season"):
        temporal_calibration_split(pd.DataFrame({"game_id": [1]}))


# --- calibrators -----------------------------------------------------------


def test_platt_calibrator_maps_scores_to_probabilities():
    rng = np.random.default_rng(5)
    scores = rng.uniform(0.3, 0.7, 400)
    outcomes = rng.binomial(1, scores)

    calibrator = fit_calibrator(scores, outcomes, "platt")
    calibrated = apply_calibrator(calibrator, scores)

    assert ((calibrated > 0) & (calibrated < 1)).all()
    assert len(calibrated) == len(scores)


def test_isotonic_calibrator_stays_within_bounds():
    rng = np.random.default_rng(6)
    scores = rng.uniform(0.3, 0.7, 400)
    outcomes = rng.binomial(1, scores)

    calibrator = fit_calibrator(scores, outcomes, "isotonic")
    calibrated = apply_calibrator(calibrator, np.array([0.0, 0.5, 1.0]))

    assert ((calibrated > 0) & (calibrated < 1)).all()


def test_an_unknown_calibration_method_is_rejected():
    with pytest.raises(CalibrationError, match="Unknown calibration"):
        fit_calibrator(np.array([0.5]), np.array([1]), "magic")


def test_comparison_includes_the_constant_base_rate():
    """The row that reveals a calibrator collapsing to the base rate."""
    predictions = _well_calibrated()
    calibrated = {"flat": np.full(len(predictions), 0.5)}

    comparison = compare_calibration(predictions, calibrated)

    assert set(comparison["probabilities"]) == {
        "raw",
        "flat",
        "constant base rate",
    }
    assert comparison["log_loss"].notna().all()


# --- charts ----------------------------------------------------------------


def test_all_three_calibration_charts_are_written(tmp_path):
    paths = calibration_charts.build_calibration_charts(_well_calibrated(), tmp_path)

    assert len(paths) == 3
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 5_000


def test_calibration_charts_have_titles_and_labelled_axes(tmp_path, monkeypatch):
    figures = []
    original = calibration_charts._save

    def _capture(figure, output_dir, name):
        figures.append((name, figure))
        return original(figure, output_dir, name)

    monkeypatch.setattr(calibration_charts, "_save", _capture)
    calibration_charts.build_calibration_charts(_well_calibrated(), tmp_path)

    assert len(figures) == 3
    for name, figure in figures:
        axis = figure.axes[0]
        assert axis.get_title(), f"{name} has no title"
        assert axis.get_xlabel().strip(), f"{name} x label empty"
        assert axis.get_ylabel().strip(), f"{name} y label empty"


def test_reliability_chart_shows_sample_counts(tmp_path, monkeypatch):
    """The acceptance criterion: counts must appear on the chart itself."""
    figures = []
    monkeypatch.setattr(
        calibration_charts,
        "_save",
        lambda figure, output_dir, name: figures.append(figure) or tmp_path / name,
    )
    calibration_charts.reliability_diagram(_well_calibrated(), tmp_path)

    texts = [text.get_text() for text in figures[0].axes[0].texts]

    assert any(text.startswith("n=") for text in texts)


def test_bucket_chart_labels_every_bucket_with_its_count(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(
        calibration_charts,
        "_save",
        lambda figure, output_dir, name: figures.append(figure) or tmp_path / name,
    )
    calibration_charts.cover_rate_by_bucket(_well_calibrated(), tmp_path)

    labels = [text.get_text() for text in figures[0].axes[0].get_xticklabels()]

    assert all("n=" in label for label in labels)


def test_sparse_threshold_is_the_documented_value():
    assert MIN_BUCKET_SAMPLES == 30
