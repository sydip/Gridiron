"""Tests for coefficient interpretation."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from gridiron.eda import coefficient_charts  # noqa: E402
from gridiron.modeling.interpretation import (  # noqa: E402
    COEFFICIENT_COLUMNS,
    FEATURE_DIRECTIONS,
    VIF_ELEVATED,
    VIF_SEVERE,
    InterpretationError,
    coefficient_stability,
    coefficient_table,
    model_limitations,
    multicollinearity_warnings,
    variance_inflation_factors,
)
from gridiron.modeling.pipeline import (  # noqa: E402
    MODEL_STEP,
    feature_matrix,
    get_feature_columns,
)
from gridiron.modeling.train import train_model  # noqa: E402

FEATURES = get_feature_columns()


def _frame(rows: int = 600, *, seed: int = 12, collinear: bool = False):
    """A matchup frame whose label depends on one feature only."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "game_id": [f"g{index:04d}" for index in range(rows)],
            "season": np.repeat([2020, 2021, 2022, 2023], rows // 4)[:rows],
            "week": [(index % 17) + 1 for index in range(rows)],
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
            "week_1_flag": np.array(
                [(index % 17) == 0 for index in range(rows)]
            ).astype("int8"),
        }
    )
    if collinear:
        # Make one feature an almost exact copy of another.
        frame["point_margin_diff_last_5"] = frame[
            "win_pct_diff_last_5"
        ] * 20 + rng.normal(0, 0.01, rows)
    signal = frame["off_epa_diff_last_5"] * 6 + rng.normal(0, 0.5, rows)
    frame["home_cover"] = (signal > 0).astype(float)
    frame["is_push"] = False
    return frame


def _fitted(frame: pd.DataFrame, c: float = 1.0):
    features = feature_matrix(frame)
    labels = frame["home_cover"].astype(int)
    return train_model(features, labels, c=c), features


# --- the coefficient table -------------------------------------------------


def test_every_coefficient_maps_to_its_feature():
    """The central correctness requirement."""
    frame = _frame()
    pipeline, features = _fitted(frame)

    table = coefficient_table(pipeline, features)

    assert list(table.columns) == COEFFICIENT_COLUMNS
    assert set(table["feature"]) == set(FEATURES)
    assert len(table) == len(FEATURES)


def test_coefficients_match_the_fitted_classifier_in_order():
    frame = _frame()
    pipeline, features = _fitted(frame)

    table = coefficient_table(pipeline, features).set_index("feature")
    raw = pipeline.named_steps[MODEL_STEP].coef_[0]

    for position, name in enumerate(pipeline.feature_names_in_):
        assert table.loc[name, "coefficient"] == pytest.approx(raw[position])


def test_the_signal_feature_carries_the_largest_coefficient():
    """The fixture builds the label from off_epa_diff_last_5 alone."""
    frame = _frame()
    pipeline, features = _fitted(frame)

    table = coefficient_table(pipeline, features)

    assert table.iloc[0]["feature"] == "off_epa_diff_last_5"
    assert table.iloc[0]["coefficient"] > 0


def test_the_table_is_sorted_by_absolute_magnitude():
    frame = _frame()
    pipeline, features = _fitted(frame)

    table = coefficient_table(pipeline, features)

    assert table["abs_coefficient"].is_monotonic_decreasing


def test_odds_ratio_is_the_exponentiated_coefficient():
    frame = _frame()
    pipeline, features = _fitted(frame)

    table = coefficient_table(pipeline, features)

    assert table["odds_ratio"].to_numpy() == pytest.approx(
        np.exp(table["coefficient"].to_numpy())
    )


def test_odds_change_percent_restates_the_ratio():
    frame = _frame()
    pipeline, features = _fitted(frame)

    table = coefficient_table(pipeline, features)

    assert table["odds_change_pct"].to_numpy() == pytest.approx(
        (table["odds_ratio"].to_numpy() - 1.0) * 100.0
    )


def test_the_intercept_travels_with_the_table():
    frame = _frame()
    pipeline, features = _fitted(frame)

    table = coefficient_table(pipeline, features)

    assert table.attrs["intercept"] == pytest.approx(
        float(pipeline.named_steps[MODEL_STEP].intercept_[0])
    )


# --- direction -------------------------------------------------------------


def test_every_model_feature_has_a_documented_direction():
    assert set(FEATURES).issubset(FEATURE_DIRECTIONS)


def test_direction_follows_the_sign_of_the_coefficient():
    frame = _frame()
    pipeline, features = _fitted(frame)

    table = coefficient_table(pipeline, features)

    for row in table.itertuples():
        if row.coefficient >= 0:
            assert row.direction == "increases"
            assert "higher modelled probability" in row.interpretation
        else:
            assert row.direction == "decreases"
            assert "lower modelled probability" in row.interpretation


def test_interpretations_are_phrased_as_associations_not_causes():
    """No coefficient may be described in causal language."""
    frame = _frame()
    pipeline, features = _fitted(frame)

    table = coefficient_table(pipeline, features)

    banned = ("causes", "caused", "leads to", "results in", "makes the", "because of")
    for row in table.itertuples():
        lowered = row.interpretation.lower()
        assert "associated with" in lowered
        for phrase in banned:
            assert phrase not in lowered


def test_the_confounded_flag_carries_its_caveat():
    frame = _frame()
    pipeline, features = _fitted(frame)

    table = coefficient_table(pipeline, features).set_index("feature")

    assert "confounded" in table.loc["week_1_flag", "interpretation"]


# --- multicollinearity -----------------------------------------------------


def test_vif_is_computed_for_every_feature():
    frame = _frame()
    pipeline, features = _fitted(frame)

    inflation = variance_inflation_factors(pipeline, features)

    assert set(inflation["feature"]) == set(FEATURES)
    assert (inflation["vif"] >= 1.0).all()


def test_independent_features_have_a_vif_near_one():
    frame = _frame()
    pipeline, features = _fitted(frame)

    inflation = variance_inflation_factors(pipeline, features).set_index("feature")

    assert inflation.loc["div_game", "vif"] < 2.0


def test_a_near_duplicate_feature_is_flagged():
    """point_margin is built as a copy of win_pct in this fixture."""
    frame = _frame(collinear=True)
    pipeline, features = _fitted(frame)

    table = coefficient_table(pipeline, features).set_index("feature")

    assert table.loc["point_margin_diff_last_5", "vif"] > VIF_SEVERE
    assert "severe" in table.loc["point_margin_diff_last_5", "collinearity_flag"]
    assert table.loc["win_pct_diff_last_5", "vif"] > VIF_SEVERE


def test_warnings_name_the_collinear_features():
    frame = _frame(collinear=True)
    pipeline, features = _fitted(frame)

    warnings = multicollinearity_warnings(coefficient_table(pipeline, features))

    assert warnings
    assert any("point_margin_diff_last_5" in warning for warning in warnings)
    assert any("should not be read" in warning for warning in warnings)


def test_clean_features_get_a_reassuring_warning_instead():
    frame = _frame()
    pipeline, features = _fitted(frame)

    warnings = multicollinearity_warnings(coefficient_table(pipeline, features))

    assert len(warnings) == 1
    assert "No feature reaches" in warnings[0]


def test_vif_needs_the_features_it_was_fitted_on():
    frame = _frame()
    pipeline, features = _fitted(frame)

    with pytest.raises(InterpretationError, match="missing feature"):
        variance_inflation_factors(pipeline, features.drop(columns="rest_diff"))


def test_a_table_without_features_reports_no_vif():
    frame = _frame()
    pipeline, _ = _fitted(frame)

    table = coefficient_table(pipeline)

    assert table["vif"].isna().all()
    assert multicollinearity_warnings(table) == []


# --- stability -------------------------------------------------------------


def _fold_coefficients() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature": ["steady"] * 3 + ["flipper"] * 3,
            "coefficient": [0.20, 0.22, 0.18, 0.05, -0.04, 0.03],
            "validation_season": [2021, 2022, 2023] * 2,
        }
    )


def test_stability_flags_a_sign_change():
    table = coefficient_stability(_fold_coefficients()).set_index("feature")

    assert not table.loc["steady", "sign_changes"]
    assert table.loc["flipper", "sign_changes"]
    assert table.loc["steady", "stable"]
    assert not table.loc["flipper", "stable"]


def test_stability_reports_the_range_and_spread():
    table = coefficient_stability(_fold_coefficients()).set_index("feature")

    assert table.loc["steady", "min_coefficient"] == pytest.approx(0.18)
    assert table.loc["steady", "max_coefficient"] == pytest.approx(0.22)
    assert table.loc["steady", "folds"] == 3


def test_stability_needs_coefficients():
    with pytest.raises(InterpretationError, match="missing"):
        coefficient_stability(pd.DataFrame({"feature": ["a"]}))


# --- limitations -----------------------------------------------------------


def test_the_causal_caveat_is_always_present():
    frame = _frame()
    pipeline, features = _fitted(frame)

    limitations = model_limitations(coefficient_table(pipeline, features))

    assert limitations
    assert "associations, not causes" in limitations[0].heading


def test_limitations_report_measured_collinearity():
    frame = _frame(collinear=True)
    pipeline, features = _fitted(frame)

    limitations = model_limitations(coefficient_table(pipeline, features))
    headings = [item.heading for item in limitations]

    assert "Some features are largely redundant" in headings


def test_limitations_report_unstable_signs():
    frame = _frame()
    pipeline, features = _fitted(frame)

    limitations = model_limitations(
        coefficient_table(pipeline, features),
        coefficient_stability(_fold_coefficients()),
    )
    headings = [item.heading for item in limitations]

    assert "Several coefficients change sign between folds" in headings


def test_limitations_report_a_model_that_cannot_rank():
    frame = _frame()
    pipeline, features = _fitted(frame)

    limitations = model_limitations(
        coefficient_table(pipeline, features),
        metrics={"roc_auc": 0.48, "log_loss": 0.70},
    )
    headings = [item.heading for item in limitations]

    assert "The model does not rank games" in headings
    assert "The probabilities carry no information" in headings


def test_a_good_model_gets_fewer_limitations():
    """The list shrinks by itself when the measurements improve."""
    frame = _frame()
    pipeline, features = _fitted(frame)
    table = coefficient_table(pipeline, features)

    poor = model_limitations(table, metrics={"roc_auc": 0.48, "log_loss": 0.70})
    good = model_limitations(table, metrics={"roc_auc": 0.72, "log_loss": 0.55})

    assert len(good) < len(poor)


def _table(coefficients: list[float]) -> pd.DataFrame:
    """A coefficient table built by hand, to test the limitation logic alone."""
    frame = pd.DataFrame(
        {
            "feature": [f"f{index}" for index in range(len(coefficients))],
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
            "odds_ratio": np.exp(coefficients),
            "vif": [1.0] * len(coefficients),
        }
    )
    frame["odds_change_pct"] = (frame["odds_ratio"] - 1.0) * 100.0
    return frame


def test_tiny_coefficients_are_called_out():
    """Fired by magnitude alone, so it does not depend on how a fit landed."""
    limitations = model_limitations(_table([0.01, -0.02, 0.005]))
    headings = [item.heading for item in limitations]

    assert "The coefficients are tiny" in headings


def test_substantial_coefficients_are_not_called_tiny():
    limitations = model_limitations(_table([0.9, -1.2, 0.4]))
    headings = [item.heading for item in limitations]

    assert "The coefficients are tiny" not in headings


# --- charts ----------------------------------------------------------------


def test_the_coefficient_plot_is_written(tmp_path):
    frame = _frame()
    pipeline, features = _fitted(frame)

    paths = coefficient_charts.build_coefficient_charts(
        coefficient_table(pipeline, features),
        coefficient_stability(_fold_coefficients()),
        tmp_path,
    )

    assert len(paths) == 2
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 5_000


def test_the_plot_is_drawn_without_stability_too(tmp_path):
    frame = _frame()
    pipeline, features = _fitted(frame)

    paths = coefficient_charts.build_coefficient_charts(
        coefficient_table(pipeline, features), None, tmp_path
    )

    assert len(paths) == 1


def test_coefficient_charts_have_titles_and_labelled_axes(tmp_path, monkeypatch):
    figures = []
    original = coefficient_charts._save

    def _capture(figure, output_dir, name):
        figures.append((name, figure))
        return original(figure, output_dir, name)

    monkeypatch.setattr(coefficient_charts, "_save", _capture)
    frame = _frame()
    pipeline, features = _fitted(frame)
    coefficient_charts.build_coefficient_charts(
        coefficient_table(pipeline, features),
        coefficient_stability(_fold_coefficients()),
        tmp_path,
    )

    assert len(figures) == 2
    for name, figure in figures:
        axis = figure.axes[0]
        assert axis.get_title(), f"{name} has no title"
        assert axis.get_xlabel().strip(), f"{name} x label empty"
        assert axis.get_ylabel().strip(), f"{name} y label empty"


def test_the_plot_says_the_coefficients_are_not_causal(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(
        coefficient_charts,
        "_save",
        lambda figure, output_dir, name: figures.append(figure) or tmp_path / name,
    )
    frame = _frame()
    pipeline, features = _fitted(frame)
    coefficient_charts.coefficient_plot(coefficient_table(pipeline, features), tmp_path)

    text = " ".join(item.get_text() for item in figures[0].texts)

    assert "not causal effects" in text


def test_the_plot_marks_collinear_features(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(
        coefficient_charts,
        "_save",
        lambda figure, output_dir, name: figures.append(figure) or tmp_path / name,
    )
    frame = _frame(collinear=True)
    pipeline, features = _fitted(frame)
    coefficient_charts.coefficient_plot(coefficient_table(pipeline, features), tmp_path)

    labels = [text.get_text() for text in figures[0].axes[0].get_legend().get_texts()]

    assert any(str(int(VIF_ELEVATED)) in label for label in labels)


def test_an_empty_table_is_rejected(tmp_path):
    with pytest.raises(coefficient_charts.CoefficientChartError, match="No coeff"):
        coefficient_charts.coefficient_plot(pd.DataFrame(), tmp_path)
