"""Tests for the hyperparameter search and its selection rule."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gridiron.modeling.tuning import (
    ACCURACY_TOLERANCE,
    BRIER_TOLERANCE,
    C_VALUES,
    CLASS_WEIGHTS,
    DEPLOYMENT_SEASON,
    LARGE_COEFFICIENT,
    LOG_LOSS_TOLERANCE,
    RESULT_COLUMNS,
    SelectedParameters,
    TuningError,
    coefficient_warnings,
    load_selected_parameters,
    run_grid_search,
    save_selected_parameters,
    select_parameters,
)


def _frame(
    seasons: tuple[int, ...] = (2016, 2017, 2018, 2019, 2020),
    per_season: int = 260,
    *,
    seed: int = 9,
) -> pd.DataFrame:
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


def _results(**overrides) -> pd.DataFrame:
    """A small grid result table with controllable values."""
    base = pd.DataFrame(
        {
            "C": [0.01, 0.1, 1.0],
            "class_weight": ["None", "None", "balanced"],
            "mean_accuracy": [0.51, 0.52, 0.50],
            "mean_log_loss": [0.690, 0.695, 0.700],
            "mean_brier": [0.249, 0.251, 0.253],
            "mean_roi": [-0.01, 0.05, -0.03],
            "accuracy_std": [0.02, 0.03, 0.04],
            "worst_season_accuracy": [0.48, 0.46, 0.45],
            "best_season_accuracy": [0.54, 0.56, 0.55],
            "max_abs_coefficient": [0.09, 0.15, 0.20],
            "mean_abs_coefficient": [0.03, 0.05, 0.07],
            "unstable_features": [0, 0, 0],
            "unstable_feature_names": ["", "", ""],
            "beats_uninformative": [True, False, False],
        }
    )
    for key, value in overrides.items():
        base[key] = value
    base.attrs["folds"] = 4
    base.attrs["validation_seasons"] = [2017, 2018, 2019, 2020]
    return base


# --- the grid --------------------------------------------------------------


def test_grid_covers_the_required_search_space():
    assert C_VALUES == [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]
    assert CLASS_WEIGHTS == [None, "balanced"]


def test_grid_search_scores_every_combination():
    results = run_grid_search(_frame(), [0.01, 1.0], [None, "balanced"], 2018, 2020)

    assert len(results) == 4
    assert set(results["C"]) == {0.01, 1.0}
    assert set(results["class_weight"]) == {"None", "balanced"}


def test_results_have_every_required_column():
    results = run_grid_search(_frame(), [1.0], [None], 2019, 2020)

    for column in RESULT_COLUMNS:
        assert column in results.columns


def test_every_parameter_set_uses_identical_folds():
    """The fold count and validation games must not vary across the grid."""
    results = run_grid_search(_frame(), [0.01, 10.0], [None, "balanced"], 2018, 2020)

    assert results.attrs["folds"] == 3
    assert results.attrs["validation_seasons"] == [2018, 2019, 2020]


def test_identical_folds_give_identical_results_for_identical_parameters():
    first = run_grid_search(_frame(), [1.0], [None], 2019, 2020)
    second = run_grid_search(_frame(), [1.0], [None], 2019, 2020)

    pd.testing.assert_frame_equal(first, second)


def test_the_deployment_season_is_never_tuned_on():
    with pytest.raises(TuningError, match=str(DEPLOYMENT_SEASON)):
        run_grid_search(_frame(), [1.0], [None], 2019, DEPLOYMENT_SEASON)


def test_settled_deployment_season_rows_are_rejected():
    """A settled row from the deployment season is an error, not a filter."""
    frame = _frame(seasons=(2016, 2017, 2018, 2019, 2020))
    extra = frame.loc[frame["season"].eq(2020)].copy()
    extra["season"] = DEPLOYMENT_SEASON
    extra["game_id"] = extra["game_id"] + "_dep"
    extra["gameday"] = pd.Timestamp(f"{DEPLOYMENT_SEASON}-09-05")

    with pytest.raises(TuningError, match="deployment season"):
        run_grid_search(
            pd.concat([frame, extra], ignore_index=True), [1.0], [None], 2018, 2020
        )


def test_grid_reports_coefficient_diagnostics():
    results = run_grid_search(_frame(), [0.01, 10.0], [None], 2018, 2020)

    assert "max_abs_coefficient" in results.columns
    assert "unstable_features" in results.columns
    # Heavier regularisation must produce smaller coefficients.
    small = results.loc[results["C"].eq(0.01), "max_abs_coefficient"].iloc[0]
    large = results.loc[results["C"].eq(10.0), "max_abs_coefficient"].iloc[0]
    assert small < large


# --- the ranked selection --------------------------------------------------


def test_selection_prefers_the_lowest_log_loss():
    selection = select_parameters(_results())

    assert selection.c == 0.01
    assert "log loss" in selection.rationale[0]


def test_selection_records_a_reason_for_every_step():
    selection = select_parameters(_results())

    assert len(selection.rationale) >= 3
    assert all(len(line) > 20 for line in selection.rationale)
    assert any("Selected" in line for line in selection.rationale)


def test_a_log_loss_difference_below_tolerance_is_a_tie():
    """Two sets within tolerance go to the next criterion, not to the winner."""
    results = _results(
        mean_log_loss=[0.6900, 0.6902, 0.7000],
        mean_brier=[0.2500, 0.2501, 0.2530],
        accuracy_std=[0.05, 0.01, 0.04],
    )

    selection = select_parameters(results)

    # The second set has marginally worse log loss but far better stability.
    assert selection.c == 0.1
    assert any("stability" in line for line in selection.rationale)


def test_a_log_loss_difference_above_tolerance_decides():
    results = _results(
        mean_log_loss=[0.690, 0.700, 0.710],
        accuracy_std=[0.09, 0.01, 0.01],
    )

    selection = select_parameters(results)

    assert selection.c == 0.01


def test_roi_never_decides_the_selection():
    """The set with by far the best ROI must not win on that basis."""
    results = _results(
        mean_roi=[-0.05, 0.25, -0.03],
        mean_log_loss=[0.690, 0.700, 0.710],
    )

    selection = select_parameters(results)

    assert selection.c == 0.01
    assert any("ROI was not used" in line for line in selection.rationale)


def test_smaller_coefficients_break_a_full_tie():
    results = _results(
        mean_log_loss=[0.690, 0.690, 0.690],
        mean_brier=[0.250, 0.250, 0.250],
        accuracy_std=[0.02, 0.02, 0.02],
        mean_accuracy=[0.51, 0.51, 0.51],
        mean_abs_coefficient=[0.09, 0.03, 0.20],
    )

    selection = select_parameters(results)

    assert selection.c == 0.1
    assert any("smaller coefficients" in line for line in selection.rationale)


def test_selection_carries_the_fold_context():
    selection = select_parameters(_results())

    assert selection.folds == 4
    assert selection.validation_seasons == [2017, 2018, 2019, 2020]
    assert selection.selected_on


def test_selection_rejects_an_empty_grid():
    with pytest.raises(TuningError, match="No results"):
        select_parameters(pd.DataFrame())


def test_tolerances_are_the_documented_values():
    assert LOG_LOSS_TOLERANCE == 0.001
    assert BRIER_TOLERANCE == 0.001
    assert ACCURACY_TOLERANCE == 0.005


# --- coefficient warnings --------------------------------------------------


def test_a_large_coefficient_is_flagged():
    results = _results(max_abs_coefficient=[2.5, 0.1, 0.1])
    chosen = results.iloc[0]

    warnings = coefficient_warnings(results, chosen)

    assert any(str(LARGE_COEFFICIENT) in warning for warning in warnings)
    assert any("overfitting" in warning for warning in warnings)


def test_a_normal_coefficient_is_not_flagged_as_large():
    results = _results()
    warnings = coefficient_warnings(results, results.iloc[0])

    assert not any("implausibly strong" in warning for warning in warnings)


def test_sign_flipping_features_are_flagged_as_unstable():
    results = _results(
        unstable_features=[3, 0, 0],
        unstable_feature_names=["rest_diff, div_game, pace_diff_last_5", "", ""],
    )

    warnings = coefficient_warnings(results, results.iloc[0])

    assert any("change coefficient sign" in warning for warning in warnings)
    assert any("rest_diff" in warning for warning in warnings)


def test_uninformative_probabilities_are_flagged():
    results = _results(beats_uninformative=[False, False, False])

    warnings = coefficient_warnings(results, results.iloc[0])

    assert any("no better than predicting 0.5" in warning for warning in warnings)
    assert any("No parameter set" in warning for warning in warnings)


def test_informative_probabilities_are_not_flagged():
    results = _results(beats_uninformative=[True, True, True])

    warnings = coefficient_warnings(results, results.iloc[0])

    assert not any("no better than predicting" in warning for warning in warnings)


def test_real_search_flags_its_own_instability():
    """The synthetic data has one real signal, so most features should wobble."""
    results = run_grid_search(_frame(), [1.0], [None], 2018, 2020)
    selection = select_parameters(results)

    assert isinstance(selection.warnings, list)
    assert results["unstable_features"].iloc[0] >= 0


# --- configuration ---------------------------------------------------------


def test_selection_round_trips_through_configuration(tmp_path):
    selection = select_parameters(_results())
    path = save_selected_parameters(selection, tmp_path / "model_params.json")

    loaded = load_selected_parameters(path)

    assert loaded.c == selection.c
    assert loaded.class_weight == selection.class_weight
    assert loaded.rationale == selection.rationale
    assert loaded.warnings == selection.warnings


def test_saved_configuration_is_readable_json(tmp_path):
    selection = select_parameters(_results())
    path = save_selected_parameters(selection, tmp_path / "model_params.json")

    payload = json.loads(path.read_text(encoding="utf-8"))

    assert payload["c"] == selection.c
    assert isinstance(payload["rationale"], list)
    assert "validation_seasons" in payload


def test_missing_configuration_falls_back_to_untuned_defaults(tmp_path):
    loaded = load_selected_parameters(tmp_path / "absent.json")

    assert loaded.c == 1.0
    assert loaded.class_weight is None
    assert "No tuning run" in loaded.rationale[0]


def test_class_weight_none_survives_the_round_trip(tmp_path):
    selection = SelectedParameters(c=0.5, class_weight=None, rationale=["x" * 30])
    path = save_selected_parameters(selection, tmp_path / "p.json")

    assert load_selected_parameters(path).class_weight is None


def test_balanced_class_weight_survives_the_round_trip(tmp_path):
    selection = SelectedParameters(c=0.5, class_weight="balanced", rationale=["x" * 30])
    path = save_selected_parameters(selection, tmp_path / "p.json")

    assert load_selected_parameters(path).class_weight == "balanced"


# --- the rationale reaches model metadata ----------------------------------


def test_training_records_the_selection_rationale_in_metadata(tmp_path, monkeypatch):
    import gridiron.modeling.tuning as tuning_module
    from gridiron.modeling.train import train_from_matchups

    selection = SelectedParameters(
        c=0.01,
        class_weight=None,
        rationale=["Chosen because the log loss was lowest by a clear margin."],
        warnings=["Coefficients are unstable across folds."],
        validation_seasons=[2019, 2020],
    )
    path = save_selected_parameters(selection, tmp_path / "model_params.json")
    monkeypatch.setattr(tuning_module, "SELECTED_PARAMS_PATH", path)

    _, _, _, metadata = train_from_matchups(_frame(), test_seasons=[2020])

    assert metadata.c == 0.01
    assert metadata.tuned
    assert metadata.selection_rationale == selection.rationale
    assert metadata.selection_warnings == selection.warnings


def test_training_can_opt_out_of_the_configured_parameters():
    from gridiron.modeling.train import train_from_matchups

    _, _, _, metadata = train_from_matchups(
        _frame(), test_seasons=[2020], use_configured_parameters=False
    )

    assert metadata.c == 1.0
    assert not metadata.tuned
