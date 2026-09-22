"""Fitting, saving, and loading the model.

The split is chronological, never random. A random split would put games from
January into the training set and games from the previous September into the
test set, which is a subtler leak than anything the pipeline guards against:
the model would be tested on a season it had already partly seen through its
own rolling features. Time-ordered splits are the only honest option for a
forecasting problem.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from gridiron.config import DEFAULT_RANDOM_SEED, PROJECT_ROOT
from gridiron.modeling.pipeline import (
    PipelineError,
    build_pipeline,
    feature_matrix,
    fitted_feature_names,
    get_feature_columns,
    target_vector,
)

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_FILENAME = "logistic_regression.joblib"
METADATA_FILENAME = "logistic_regression.metadata.json"

MIN_TRAINING_ROWS = 50


class TrainingError(ValueError):
    """Raised when a model cannot be trained from the given data."""


@dataclass(frozen=True)
class TrainingMetadata:
    """What the model was trained on, recorded beside the model itself.

    A fitted pipeline on its own cannot answer "which rows produced this?".
    Storing the feature order, the row count, the class balance, and the date
    range makes a saved model auditable rather than merely reusable.
    """

    feature_columns: list[str]
    rows: int
    positive_rate: float
    seed: int
    trained_at: str
    first_gameday: str | None = None
    last_gameday: str | None = None
    c: float = 1.0
    class_weight: str | None = None


def train_model(
    X_train: pd.DataFrame,  # noqa: N803 - the phase brief fixes this name
    y_train: pd.Series,
) -> Pipeline:
    """Fit the pipeline on the training rows and nothing else.

    ``X_train`` must already hold exactly the model features; a missing one is
    an error naming every column that is absent, rather than a silent fit on a
    narrower matrix.

    The returned pipeline has learned its imputation medians and scaling
    statistics from these rows alone. Nothing that is not in ``X_train`` can
    have influenced them, which is what keeps a later test fold clean.
    """
    if not isinstance(X_train, pd.DataFrame):
        raise TrainingError("X_train must be a DataFrame so feature names are kept.")

    expected = get_feature_columns()
    missing = [column for column in expected if column not in X_train.columns]
    if missing:
        raise TrainingError(
            "Cannot train; missing required feature(s): " + ", ".join(missing)
        )

    extra = [column for column in X_train.columns if column not in expected]
    if extra:
        raise TrainingError(
            "Cannot train; unexpected column(s) in the feature matrix: "
            + ", ".join(sorted(extra))
        )

    if len(X_train) != len(y_train):
        raise TrainingError(
            f"X_train has {len(X_train)} rows but y_train has {len(y_train)}."
        )
    if len(X_train) < MIN_TRAINING_ROWS:
        raise TrainingError(
            f"Refusing to train on {len(X_train)} rows; need at least "
            f"{MIN_TRAINING_ROWS}."
        )

    labels = pd.to_numeric(y_train, errors="coerce")
    if labels.isna().any():
        raise TrainingError("y_train contains missing labels.")
    if labels.nunique() < 2:
        raise TrainingError(
            "y_train has a single class; a classifier cannot be fitted on it."
        )

    ordered = X_train[expected]
    return build_pipeline().fit(ordered, labels.astype(int))


def chronological_split(
    matchups: pd.DataFrame,
    *,
    test_seasons: list[int] | None = None,
    test_fraction: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into earlier training rows and later test rows.

    ``test_seasons`` holds out whole seasons, which is the more honest split
    for this problem: a season is the natural unit over which the market and
    the league change. Without it, the last ``test_fraction`` of games by date
    is held out instead.
    """
    if "gameday" not in matchups.columns:
        raise TrainingError("Cannot split chronologically; missing 'gameday'.")

    ordered = matchups.sort_values(["gameday", "game_id"], kind="stable")

    if test_seasons:
        if "season" not in ordered.columns:
            raise TrainingError("Cannot hold out seasons; missing 'season'.")
        held_out = ordered["season"].isin(test_seasons)
        if not held_out.any():
            raise TrainingError(f"No rows found for test seasons {test_seasons}.")
        return ordered.loc[~held_out].copy(), ordered.loc[held_out].copy()

    if not 0 < test_fraction < 1:
        raise TrainingError(f"test_fraction must be in (0, 1); got {test_fraction}.")

    cut = int(len(ordered) * (1 - test_fraction))
    return ordered.iloc[:cut].copy(), ordered.iloc[cut:].copy()


def training_metadata(
    pipeline: Pipeline,
    frame: pd.DataFrame,
    y_train: pd.Series,
    *,
    c: float = 1.0,
    class_weight: str | None = None,
) -> TrainingMetadata:
    """Describe a fitted model's training set."""
    gamedays = (
        pd.to_datetime(frame["gameday"], errors="coerce")
        if "gameday" in frame.columns
        else None
    )
    return TrainingMetadata(
        feature_columns=fitted_feature_names(pipeline),
        rows=len(frame),
        positive_rate=float(pd.to_numeric(y_train, errors="coerce").mean()),
        seed=DEFAULT_RANDOM_SEED,
        trained_at=datetime.now(UTC).isoformat(timespec="seconds"),
        first_gameday=str(gamedays.min().date()) if gamedays is not None else None,
        last_gameday=str(gamedays.max().date()) if gamedays is not None else None,
        c=c,
        class_weight=class_weight,
    )


def save_model(
    pipeline: Pipeline,
    metadata: TrainingMetadata,
    directory: Path = MODEL_DIR,
) -> tuple[Path, Path]:
    """Write the fitted pipeline and its metadata side by side.

    The feature order travels with the model twice over: inside the pickle, as
    scikit-learn's own ``feature_names_in_``, and in the JSON beside it, where
    it stays readable without unpickling anything.
    """
    directory.mkdir(parents=True, exist_ok=True)
    model_path = directory / MODEL_FILENAME
    metadata_path = directory / METADATA_FILENAME

    joblib.dump(pipeline, model_path)
    metadata_path.write_text(json.dumps(asdict(metadata), indent=2), encoding="utf-8")
    return model_path, metadata_path


def load_model(
    directory: Path = MODEL_DIR,
) -> tuple[Pipeline, TrainingMetadata]:
    """Load a saved pipeline and confirm it matches its recorded features."""
    model_path = directory / MODEL_FILENAME
    metadata_path = directory / METADATA_FILENAME
    if not model_path.exists():
        raise TrainingError(f"No saved model at {model_path}.")
    if not metadata_path.exists():
        raise TrainingError(f"No metadata beside the model at {metadata_path}.")

    pipeline = joblib.load(model_path)
    metadata = TrainingMetadata(**json.loads(metadata_path.read_text(encoding="utf-8")))

    stored = fitted_feature_names(pipeline)
    if stored != metadata.feature_columns:
        raise TrainingError(
            "Saved model and metadata disagree about the feature order: "
            f"{stored} vs {metadata.feature_columns}."
        )
    return pipeline, metadata


def train_from_matchups(
    matchups: pd.DataFrame,
    *,
    test_seasons: list[int] | None = None,
    test_fraction: float = 0.2,
) -> tuple[Pipeline, pd.DataFrame, pd.DataFrame, TrainingMetadata]:
    """Split, fit, and describe in one call.

    Returns the fitted pipeline together with the training and test frames, so
    a caller can score the held-out rows without re-deriving the split.
    """
    from gridiron.modeling.baselines import eligible_games

    eligible = eligible_games(matchups)
    train_frame, test_frame = chronological_split(
        eligible, test_seasons=test_seasons, test_fraction=test_fraction
    )

    try:
        features = feature_matrix(train_frame)
        labels = target_vector(train_frame)
    except PipelineError as error:
        raise TrainingError(str(error)) from error

    pipeline = train_model(features, labels)
    metadata = training_metadata(pipeline, train_frame, labels)
    return pipeline, train_frame, test_frame, metadata
