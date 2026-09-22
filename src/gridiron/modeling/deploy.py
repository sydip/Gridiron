"""Building and persisting the final deployment model.

This is the last fit: every season of the training window, with the
hyperparameters already frozen by the tuning phase. Nothing here is allowed to
choose a setting -- if the search has not run, this refuses rather than quietly
falling back to a default, because a model fitted on all the data with
parameters picked afterwards has no honest validation behind it.

Three artefacts are written together and must stay consistent: the pickled
pipeline, the feature schema that records exactly what it expects and in what
order, and the metadata that records what it was trained on and what was left
out.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from gridiron.config import DEFAULT_RANDOM_SEED, PROJECT_ROOT
from gridiron.modeling.pipeline import (
    MODEL_STEP,
    feature_matrix,
    fitted_feature_names,
    get_feature_columns,
    target_vector,
)
from gridiron.modeling.train import train_model
from gridiron.modeling.tuning import SelectedParameters, load_selected_parameters

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILENAME = "logistic_regression.joblib"
SCHEMA_FILENAME = "feature_schema.json"
METADATA_FILENAME = "model_metadata.json"

TRAINING_SEASONS = list(range(2016, 2026))
PREDICTION_SEASON = 2026
TARGET = "home_cover"

MIN_DEPLOYMENT_ROWS = 1000


class DeploymentError(ValueError):
    """Raised when a deployment model cannot be built or loaded."""


@dataclass(frozen=True)
class ExclusionRecord:
    """Every game that did not make it into the training set, and why."""

    regular_season_games: int
    excluded_not_completed: int
    excluded_no_spread_line: int
    excluded_pushes: int
    excluded_no_label: int
    training_games: int
    complete_cases: int
    rows_with_imputed_features: int


def assert_hyperparameters_frozen(
    selection: SelectedParameters | None = None,
) -> SelectedParameters:
    """Confirm the settings came from the search, not from a default.

    ``load_selected_parameters`` falls back to untuned defaults when no tuning
    run is found, which is right for exploratory work and wrong here: the final
    model is the one that gets used, and fitting it on every season with
    parameters that were never validated would throw away the only evidence
    that they are reasonable.
    """
    selection = selection or load_selected_parameters()
    if not selection.validation_seasons:
        raise DeploymentError(
            "Hyperparameters are not frozen: no tuning run is recorded in the "
            "configuration. Run scripts/tune_model.py before deploying."
        )
    return selection


def build_training_set(
    matchups: pd.DataFrame,
    seasons: list[int] | None = None,
) -> tuple[pd.DataFrame, ExclusionRecord]:
    """Select the final training rows and record what was dropped.

    Applies the deployment filters in order -- seasons, completed games, a
    valid spread, then pushes -- counting each exclusion as it goes so the
    metadata can report where the rows went rather than only how many arrived.

    Missing *features* are not an exclusion: the pipeline imputes them, which
    is what keeps early-season games in the training set. Rows carrying an
    imputed feature are counted, not removed.
    """
    seasons = seasons or TRAINING_SEASONS
    required = {"season", "home_score", "away_score", "spread_line", TARGET}
    missing = sorted(required.difference(matchups.columns))
    if missing:
        raise DeploymentError(
            "Cannot build the training set; missing column(s): " + ", ".join(missing)
        )

    in_window = matchups.loc[matchups["season"].isin(seasons)]
    regular_season_games = len(in_window)

    completed = in_window.loc[
        in_window["home_score"].notna() & in_window["away_score"].notna()
    ]
    excluded_not_completed = regular_season_games - len(completed)

    lined = completed.loc[completed["spread_line"].notna()]
    excluded_no_spread_line = len(completed) - len(lined)

    pushed = (
        lined["is_push"].fillna(False).astype(bool)
        if "is_push" in lined.columns
        else pd.Series(False, index=lined.index)
    )
    unpushed = lined.loc[~pushed]
    excluded_pushes = int(pushed.sum())

    # A settled game always has a label; anything left without one would mean
    # the target construction disagreed with the push flag, so it is counted
    # rather than silently dropped.
    training = unpushed.loc[unpushed[TARGET].notna()]
    excluded_no_label = len(unpushed) - len(training)

    columns = [column for column in get_feature_columns() if column in training.columns]
    complete = int(training[columns].notna().all(axis=1).sum()) if columns else 0

    record = ExclusionRecord(
        regular_season_games=regular_season_games,
        excluded_not_completed=excluded_not_completed,
        excluded_no_spread_line=excluded_no_spread_line,
        excluded_pushes=excluded_pushes,
        excluded_no_label=excluded_no_label,
        training_games=len(training),
        complete_cases=complete,
        rows_with_imputed_features=len(training) - complete,
    )
    return training.copy(), record


def feature_schema(pipeline: Pipeline, features: pd.DataFrame) -> dict[str, object]:
    """The contract a caller must satisfy to use this model.

    The order is taken from the fitted pipeline rather than from the feature
    list, so the schema records what the model actually expects. scikit-learn
    checks that order on every predict call, and the schema makes it readable
    without unpickling anything.
    """
    names = fitted_feature_names(pipeline)
    return {
        "feature_names": names,
        "n_features": len(names),
        "target": TARGET,
        "dtypes": {name: str(features[name].dtype) for name in names},
        "order_is_significant": True,
        "note": (
            "Columns must be supplied in this exact order. scikit-learn "
            "validates the names on every call and raises on a mismatch."
        ),
    }


@dataclass(frozen=True)
class DeploymentMetadata:
    """What the deployment model is, and what it was built from."""

    model_type: str
    training_seasons: list[int]
    prediction_season: int
    target: str
    feature_names: list[str]
    created_at: str
    n_training_games: int
    excluded_pushes: int
    selected_C: str | float  # noqa: N815 - the metadata key is fixed by the spec
    class_weight: str | None
    validation_summary: dict[str, object] = field(default_factory=dict)
    exclusions: dict[str, int] = field(default_factory=dict)
    hyperparameter_selection: dict[str, object] = field(default_factory=dict)
    random_seed: int = DEFAULT_RANDOM_SEED


def deployment_metadata(
    pipeline: Pipeline,
    record: ExclusionRecord,
    selection: SelectedParameters,
    validation_summary: dict[str, object] | None = None,
    seasons: list[int] | None = None,
) -> DeploymentMetadata:
    """Assemble the metadata that ships beside the model."""
    return DeploymentMetadata(
        model_type=type(pipeline.named_steps[MODEL_STEP]).__name__,
        training_seasons=list(seasons or TRAINING_SEASONS),
        prediction_season=PREDICTION_SEASON,
        target=TARGET,
        feature_names=fitted_feature_names(pipeline),
        created_at=datetime.now(UTC).isoformat(timespec="seconds"),
        n_training_games=record.training_games,
        excluded_pushes=record.excluded_pushes,
        selected_C=selection.c,
        class_weight=selection.class_weight,
        validation_summary=dict(validation_summary or {}),
        exclusions=asdict(record),
        hyperparameter_selection={
            "rationale": list(selection.rationale),
            "warnings": list(selection.warnings),
            "validation_seasons": list(selection.validation_seasons),
            "folds": selection.folds,
            "selected_on": selection.selected_on,
        },
    )


def fit_deployment_model(
    matchups: pd.DataFrame,
    seasons: list[int] | None = None,
    selection: SelectedParameters | None = None,
) -> tuple[Pipeline, pd.DataFrame, ExclusionRecord, SelectedParameters]:
    """Fit the final model on every training season.

    Refuses to run unless the hyperparameters are frozen.
    """
    frozen = assert_hyperparameters_frozen(selection)
    training, record = build_training_set(matchups, seasons)

    if record.training_games < MIN_DEPLOYMENT_ROWS:
        raise DeploymentError(
            f"Refusing to deploy a model trained on {record.training_games} "
            f"games; expected at least {MIN_DEPLOYMENT_ROWS}."
        )

    features = feature_matrix(training)
    labels = target_vector(training)
    pipeline = train_model(
        features, labels, c=frozen.c, class_weight=frozen.class_weight
    )
    return pipeline, features, record, frozen


def save_deployment(
    pipeline: Pipeline,
    schema: dict[str, object],
    metadata: DeploymentMetadata,
    directory: Path | None = None,
) -> dict[str, Path]:
    """Write all three artefacts, after checking they agree with each other."""
    directory = directory or MODEL_DIR
    fitted = fitted_feature_names(pipeline)

    if schema["feature_names"] != fitted:
        raise DeploymentError(
            "Feature schema does not match the fitted model's order: "
            f"{schema['feature_names']} vs {fitted}."
        )
    if metadata.feature_names != fitted:
        raise DeploymentError(
            "Metadata feature names do not match the fitted model's order."
        )

    directory.mkdir(parents=True, exist_ok=True)
    paths = {
        "model": directory / MODEL_FILENAME,
        "schema": directory / SCHEMA_FILENAME,
        "metadata": directory / METADATA_FILENAME,
    }
    joblib.dump(pipeline, paths["model"])
    paths["schema"].write_text(json.dumps(schema, indent=2), encoding="utf-8")
    paths["metadata"].write_text(
        json.dumps(asdict(metadata), indent=2), encoding="utf-8"
    )
    return paths


def load_deployment(
    directory: Path | None = None,
) -> tuple[Pipeline, dict[str, object], DeploymentMetadata]:
    """Load all three artefacts and confirm they still agree.

    A schema that has drifted from the pickle is worse than a missing one: it
    would let a caller build a matrix the model silently misreads.
    """
    directory = directory or MODEL_DIR
    paths = {
        "model": directory / MODEL_FILENAME,
        "schema": directory / SCHEMA_FILENAME,
        "metadata": directory / METADATA_FILENAME,
    }
    for name, path in paths.items():
        if not path.exists():
            raise DeploymentError(f"Missing {name} artefact at {path}.")

    pipeline = joblib.load(paths["model"])
    schema = json.loads(paths["schema"].read_text(encoding="utf-8"))
    metadata = DeploymentMetadata(
        **json.loads(paths["metadata"].read_text(encoding="utf-8"))
    )

    fitted = fitted_feature_names(pipeline)
    if schema.get("feature_names") != fitted:
        raise DeploymentError("Loaded schema disagrees with the model's feature order.")
    if metadata.feature_names != fitted:
        raise DeploymentError(
            "Loaded metadata disagrees with the model's feature order."
        )
    return pipeline, schema, metadata


def smoke_test(directory: Path | None = None) -> dict[str, object]:
    """Reload the saved model and make it produce probabilities.

    Deliberately builds its input from the schema alone, the way a caller with
    no access to the training data would, and includes a row of missing values
    so the imputation step is exercised too.
    """
    pipeline, schema, metadata = load_deployment(directory)
    names = schema["feature_names"]

    probe = pd.DataFrame(
        [
            dict.fromkeys(names, 0.0),
            dict.fromkeys(names, 1.0),
            dict.fromkeys(names, np.nan),
        ]
    )[names]

    probabilities = pipeline.predict_proba(probe)[:, 1]
    predictions = pipeline.predict(probe)

    if probabilities.shape != (3,):
        raise DeploymentError("Reloaded model did not return one probability per row.")
    if not np.all((probabilities > 0) & (probabilities < 1)):
        raise DeploymentError("Reloaded model returned a probability outside (0, 1).")

    return {
        "loaded": True,
        "n_features": len(names),
        "probabilities": [float(value) for value in probabilities],
        "predictions": [int(value) for value in predictions],
        "handled_missing_values": True,
        "model_type": metadata.model_type,
        "training_games": metadata.n_training_games,
    }
