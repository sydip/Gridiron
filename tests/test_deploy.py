"""Tests for the final deployment model and its artefacts."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from gridiron.modeling.deploy import (
    METADATA_FILENAME,
    MIN_DEPLOYMENT_ROWS,
    MODEL_FILENAME,
    PREDICTION_SEASON,
    SCHEMA_FILENAME,
    TARGET,
    TRAINING_SEASONS,
    DeploymentError,
    assert_hyperparameters_frozen,
    build_training_set,
    deployment_metadata,
    feature_schema,
    fit_deployment_model,
    load_deployment,
    save_deployment,
    smoke_test,
)
from gridiron.modeling.pipeline import fitted_feature_names, get_feature_columns
from gridiron.modeling.tuning import SelectedParameters

FEATURES = get_feature_columns()

# The metadata keys the specification fixes.
REQUIRED_METADATA_KEYS = [
    "model_type",
    "training_seasons",
    "prediction_season",
    "target",
    "feature_names",
    "created_at",
    "n_training_games",
    "excluded_pushes",
    "selected_C",
    "class_weight",
    "validation_summary",
]


def _frozen(**overrides) -> SelectedParameters:
    settings = {
        "c": 0.01,
        "class_weight": None,
        "rationale": ["Chosen because the log loss was lowest."],
        "validation_seasons": [2019, 2020, 2021],
        "folds": 3,
    }
    settings.update(overrides)
    return SelectedParameters(**settings)


def _matchups(
    rows_per_season: int = 260,
    seasons: tuple[int, ...] = (2016, 2017, 2018, 2019, 2020),
    *,
    pushes: int = 20,
    incomplete: int = 0,
    unlined: int = 0,
    seed: int = 31,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for season in seasons:
        for index in range(rows_per_season):
            rows.append(
                {
                    "game_id": f"{season}_{index:03d}",
                    "season": season,
                    "week": (index % 17) + 1,
                    "gameday": pd.Timestamp(f"{season}-09-05")
                    + pd.Timedelta(days=7 * (index % 17)),
                }
            )
    frame = pd.DataFrame(rows)
    count = len(frame)

    frame["home_score"] = rng.integers(10, 35, count).astype(float)
    frame["away_score"] = rng.integers(10, 35, count).astype(float)
    frame["spread_line"] = rng.normal(0, 5, count).round(1)
    frame["spread_line_missing"] = False

    for column, scale in (
        ("off_epa_diff_last_5", 0.2),
        ("def_epa_strength_diff_last_5", 0.2),
        ("pace_diff_last_5", 5.0),
        ("point_margin_diff_last_5", 8.0),
        ("win_pct_diff_last_5", 0.3),
    ):
        frame[column] = rng.normal(0, scale, count)
    frame["rest_diff"] = rng.integers(-7, 8, count).astype(float)
    frame["home_short_week"] = rng.random(count) < 0.15
    frame["away_short_week"] = rng.random(count) < 0.15
    frame["div_game"] = rng.integers(0, 2, count).astype("int8")
    frame["week_1_flag"] = (frame["week"] == 1).astype("int8")

    frame["is_push"] = False
    frame[TARGET] = rng.integers(0, 2, count).astype(float)

    if pushes:
        frame.loc[frame.index[:pushes], "is_push"] = True
        frame.loc[frame.index[:pushes], TARGET] = np.nan
    if incomplete:
        window = frame.index[pushes : pushes + incomplete]
        frame.loc[window, ["home_score", "away_score"]] = np.nan
        frame.loc[window, TARGET] = np.nan
    if unlined:
        start = pushes + incomplete
        window = frame.index[start : start + unlined]
        frame.loc[window, "spread_line"] = np.nan
        frame.loc[window, TARGET] = np.nan

    return frame.drop(columns="spread_line_missing")


# --- frozen hyperparameters ------------------------------------------------


def test_deployment_refuses_unfrozen_hyperparameters():
    """The acceptance criterion: no final fit before the search has run."""
    untuned = SelectedParameters(
        c=1.0, class_weight=None, rationale=["No tuning run found."]
    )

    with pytest.raises(DeploymentError, match="not frozen"):
        assert_hyperparameters_frozen(untuned)


def test_frozen_hyperparameters_are_accepted():
    selection = assert_hyperparameters_frozen(_frozen())

    assert selection.c == 0.01
    assert selection.validation_seasons


def test_fitting_refuses_when_the_search_has_not_run():
    matchups = _matchups()
    untuned = SelectedParameters(c=1.0, class_weight=None)

    with pytest.raises(DeploymentError, match="not frozen"):
        fit_deployment_model(matchups, selection=untuned)


# --- the training set ------------------------------------------------------


def test_only_the_training_seasons_are_used():
    matchups = _matchups(seasons=(2016, 2017, 2018, 2026))
    training, _ = build_training_set(matchups, seasons=[2016, 2017, 2018])

    assert set(training["season"]) == {2016, 2017, 2018}
    assert 2026 not in set(training["season"])


def test_pushes_are_excluded_and_counted():
    matchups = _matchups(pushes=25)

    training, record = build_training_set(
        matchups, seasons=[2016, 2017, 2018, 2019, 2020]
    )

    assert record.excluded_pushes == 25
    assert not training["is_push"].any()


def test_incomplete_games_are_excluded_and_counted():
    matchups = _matchups(pushes=0, incomplete=12)

    _, record = build_training_set(matchups, seasons=[2016, 2017, 2018, 2019, 2020])

    assert record.excluded_not_completed == 12


def test_games_without_a_line_are_excluded_and_counted():
    matchups = _matchups(pushes=0, unlined=9)

    _, record = build_training_set(matchups, seasons=[2016, 2017, 2018, 2019, 2020])

    assert record.excluded_no_spread_line == 9


def test_the_exclusions_account_for_every_game():
    matchups = _matchups(pushes=20, incomplete=10, unlined=7)

    _, record = build_training_set(matchups, seasons=[2016, 2017, 2018, 2019, 2020])

    accounted = (
        record.excluded_not_completed
        + record.excluded_no_spread_line
        + record.excluded_pushes
        + record.excluded_no_label
        + record.training_games
    )
    assert accounted == record.regular_season_games


def test_rows_with_imputable_features_are_kept_and_counted():
    """Missing features are imputed, not a reason to drop a game."""
    matchups = _matchups(pushes=0)
    matchups.loc[matchups.index[:40], "off_epa_diff_last_5"] = np.nan

    training, record = build_training_set(
        matchups, seasons=[2016, 2017, 2018, 2019, 2020]
    )

    assert record.rows_with_imputed_features == 40
    assert record.complete_cases == record.training_games - 40
    assert len(training) == record.training_games


def test_building_needs_the_expected_columns():
    with pytest.raises(DeploymentError, match="missing column"):
        build_training_set(pd.DataFrame({"season": [2016]}))


def test_deployment_refuses_too_few_rows():
    matchups = _matchups(rows_per_season=50, seasons=(2016, 2017))

    with pytest.raises(DeploymentError, match=str(MIN_DEPLOYMENT_ROWS)):
        fit_deployment_model(matchups, seasons=[2016, 2017], selection=_frozen())


# --- the fitted model ------------------------------------------------------


@pytest.fixture(scope="module")
def deployment():
    matchups = _matchups()
    seasons = [2016, 2017, 2018, 2019, 2020]
    pipeline, features, record, selection = fit_deployment_model(
        matchups, seasons=seasons, selection=_frozen()
    )
    schema = feature_schema(pipeline, features)
    metadata = deployment_metadata(
        pipeline,
        record,
        selection,
        {"mean_season_accuracy": 0.51, "roc_auc": 0.49},
        seasons,
    )
    return pipeline, schema, metadata, record


def test_the_model_trains_on_every_included_game(deployment):
    _, _, metadata, record = deployment

    assert metadata.n_training_games == record.training_games


def test_the_frozen_settings_reach_the_fitted_model(deployment):
    pipeline, _, metadata, _ = deployment

    assert pipeline.named_steps["classifier"].C == 0.01
    assert metadata.selected_C == 0.01
    assert metadata.class_weight is None


# --- the artefacts ---------------------------------------------------------


def test_all_three_artefacts_are_written(tmp_path, deployment):
    pipeline, schema, metadata, _ = deployment

    paths = save_deployment(pipeline, schema, metadata, tmp_path)

    assert set(paths) == {"model", "schema", "metadata"}
    for path in paths.values():
        assert path.exists()
    assert (tmp_path / MODEL_FILENAME).exists()
    assert (tmp_path / SCHEMA_FILENAME).exists()
    assert (tmp_path / METADATA_FILENAME).exists()


def test_the_schema_matches_the_model_order_exactly(tmp_path, deployment):
    """The acceptance criterion: schema order is the model's order."""
    pipeline, schema, metadata, _ = deployment
    save_deployment(pipeline, schema, metadata, tmp_path)

    loaded_schema = json.loads((tmp_path / SCHEMA_FILENAME).read_text())

    assert loaded_schema["feature_names"] == fitted_feature_names(pipeline)
    assert loaded_schema["feature_names"] == FEATURES
    assert loaded_schema["n_features"] == len(FEATURES)


def test_saving_rejects_a_schema_that_disagrees(tmp_path, deployment):
    pipeline, schema, metadata, _ = deployment
    broken = dict(schema)
    broken["feature_names"] = list(reversed(schema["feature_names"]))

    with pytest.raises(DeploymentError, match="does not match"):
        save_deployment(pipeline, broken, metadata, tmp_path)


def test_metadata_carries_every_required_key(tmp_path, deployment):
    pipeline, schema, metadata, _ = deployment
    save_deployment(pipeline, schema, metadata, tmp_path)

    payload = json.loads((tmp_path / METADATA_FILENAME).read_text())

    for key in REQUIRED_METADATA_KEYS:
        assert key in payload, key


def test_metadata_reports_exclusions_and_counts(tmp_path, deployment):
    pipeline, schema, metadata, record = deployment
    save_deployment(pipeline, schema, metadata, tmp_path)

    payload = json.loads((tmp_path / METADATA_FILENAME).read_text())

    assert payload["n_training_games"] == record.training_games
    assert payload["excluded_pushes"] == record.excluded_pushes
    assert payload["exclusions"]["regular_season_games"] == (
        record.regular_season_games
    )
    assert payload["exclusions"]["rows_with_imputed_features"] == (
        record.rows_with_imputed_features
    )


def test_metadata_records_the_target_and_seasons(tmp_path, deployment):
    pipeline, schema, metadata, _ = deployment
    save_deployment(pipeline, schema, metadata, tmp_path)

    payload = json.loads((tmp_path / METADATA_FILENAME).read_text())

    assert payload["target"] == TARGET
    assert payload["prediction_season"] == PREDICTION_SEASON
    assert payload["model_type"] == "LogisticRegression"


def test_metadata_carries_the_validation_summary(tmp_path, deployment):
    pipeline, schema, metadata, _ = deployment
    save_deployment(pipeline, schema, metadata, tmp_path)

    payload = json.loads((tmp_path / METADATA_FILENAME).read_text())

    assert payload["validation_summary"]["roc_auc"] == 0.49


# --- reloading -------------------------------------------------------------


def test_artefacts_reload_successfully(tmp_path, deployment):
    pipeline, schema, metadata, _ = deployment
    save_deployment(pipeline, schema, metadata, tmp_path)

    loaded, loaded_schema, loaded_metadata = load_deployment(tmp_path)

    assert fitted_feature_names(loaded) == fitted_feature_names(pipeline)
    assert loaded_schema["feature_names"] == schema["feature_names"]
    assert loaded_metadata.n_training_games == metadata.n_training_games


def test_a_reloaded_model_predicts_identically(tmp_path, deployment):
    pipeline, schema, metadata, _ = deployment
    save_deployment(pipeline, schema, metadata, tmp_path)
    loaded, _, _ = load_deployment(tmp_path)

    probe = pd.DataFrame([dict.fromkeys(FEATURES, 0.5)])[FEATURES]

    assert np.array_equal(loaded.predict_proba(probe), pipeline.predict_proba(probe))


def test_loading_reports_a_missing_artefact(tmp_path, deployment):
    pipeline, schema, metadata, _ = deployment
    save_deployment(pipeline, schema, metadata, tmp_path)
    (tmp_path / SCHEMA_FILENAME).unlink()

    with pytest.raises(DeploymentError, match="Missing schema"):
        load_deployment(tmp_path)


def test_loading_detects_a_schema_that_drifted(tmp_path, deployment):
    """A drifted schema is worse than a missing one."""
    pipeline, schema, metadata, _ = deployment
    save_deployment(pipeline, schema, metadata, tmp_path)

    path = tmp_path / SCHEMA_FILENAME
    payload = json.loads(path.read_text())
    payload["feature_names"] = list(reversed(payload["feature_names"]))
    path.write_text(json.dumps(payload))

    with pytest.raises(DeploymentError, match="disagrees"):
        load_deployment(tmp_path)


# --- the smoke test --------------------------------------------------------


def test_the_smoke_test_produces_probabilities(tmp_path, deployment):
    pipeline, schema, metadata, _ = deployment
    save_deployment(pipeline, schema, metadata, tmp_path)

    result = smoke_test(tmp_path)

    assert result["loaded"]
    assert result["n_features"] == len(FEATURES)
    assert len(result["probabilities"]) == 3
    assert all(0.0 < value < 1.0 for value in result["probabilities"])
    assert set(result["predictions"]).issubset({0, 1})


def test_the_smoke_test_exercises_imputation(tmp_path, deployment):
    """Its probe includes an all-missing row, as a real caller might send."""
    pipeline, schema, metadata, _ = deployment
    save_deployment(pipeline, schema, metadata, tmp_path)

    result = smoke_test(tmp_path)

    assert result["handled_missing_values"]


def test_the_smoke_test_builds_its_input_from_the_schema_alone(tmp_path, deployment):
    """A caller with only the artefacts must be able to use the model."""
    pipeline, schema, metadata, _ = deployment
    save_deployment(pipeline, schema, metadata, tmp_path)

    loaded, loaded_schema, _ = load_deployment(tmp_path)
    probe = pd.DataFrame([dict.fromkeys(loaded_schema["feature_names"], 0.0)])[
        loaded_schema["feature_names"]
    ]

    assert loaded.predict_proba(probe).shape == (1, 2)


def test_the_smoke_test_reports_a_missing_deployment(tmp_path):
    with pytest.raises(DeploymentError, match="Missing"):
        smoke_test(tmp_path)


# --- the shipped artefacts -------------------------------------------------


def test_the_training_window_is_the_specified_one():
    assert list(range(2016, 2026)) == TRAINING_SEASONS
    assert PREDICTION_SEASON == 2026
