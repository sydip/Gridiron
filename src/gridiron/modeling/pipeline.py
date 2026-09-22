"""The model pipeline: impute, standardise, then fit logistic regression.

Every transformation lives *inside* the pipeline. That is not a stylistic
preference — it is what makes the model leakage-safe. An imputer fitted outside
the pipeline would learn its medians from whatever frame it was handed, and a
scaler fitted outside would learn its means the same way. Both would quietly
absorb the test fold. Inside a pipeline, ``fit`` sees only the training rows it
is given, and ``transform`` at predict time reuses the statistics learned then.

The order is fixed and matters: impute first, because a scaler cannot compute a
mean over missing values; standardise second, because logistic regression with
an L2 penalty shrinks coefficients toward zero and would otherwise penalise a
feature measured in points far more than one measured in EPA per play.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from gridiron.config import DEFAULT_RANDOM_SEED
from gridiron.features.matchup import (
    FEATURE_COLUMNS,
    POSTGAME_COLUMNS,
    TARGET_COLUMNS,
)

TARGET_COLUMN = "home_cover"

IMPUTE_STEP = "impute"
SCALE_STEP = "scale"
MODEL_STEP = "model"

# lbfgs is deterministic for a given dataset, so repeated fits agree exactly.
# random_state is set regardless, so swapping in a stochastic solver later
# cannot silently make results irreproducible.
SOLVER = "lbfgs"
MAX_ITER = 1000


class PipelineError(ValueError):
    """Raised when a frame cannot be turned into a model matrix."""


def get_feature_columns() -> list[str]:
    """The features the model is trained on, in a fixed order.

    The order is the contract. It is stored on the fitted pipeline and checked
    at predict time, so a frame whose columns arrive in a different order is
    rejected rather than silently scored against the wrong coefficients.

    This delegates to the feature matrix agreed in
    :mod:`gridiron.features.matchup`, so the model cannot drift from the table
    that was built and audited for it.
    """
    return list(FEATURE_COLUMNS)


def build_pipeline(
    c: float = 1.0,
    class_weight: str | None = None,
) -> Pipeline:
    """Assemble the untrained pipeline.

    ``c`` is the inverse regularisation strength passed to the classifier as
    ``C``; smaller values penalise large coefficients harder. ``class_weight``
    is passed through, so ``"balanced"`` is available if a split turns out
    lopsided — the against-the-spread target is close to even, so the default
    of ``None`` is usually right.

    Nothing here is fitted. The returned object learns its imputation medians
    and scaling statistics only when :meth:`fit` is called, from exactly the
    rows it is then given.
    """
    if c <= 0:
        raise PipelineError(f"Regularisation strength must be positive; got {c}.")

    return Pipeline(
        steps=[
            (IMPUTE_STEP, SimpleImputer(strategy="median")),
            (SCALE_STEP, StandardScaler()),
            (
                MODEL_STEP,
                LogisticRegression(
                    C=c,
                    class_weight=class_weight,
                    solver=SOLVER,
                    max_iter=MAX_ITER,
                    random_state=DEFAULT_RANDOM_SEED,
                ),
            ),
        ]
    )


def _missing_features(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column not in frame.columns]


def feature_matrix(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Select the model features, in order, as floats.

    Booleans are cast to floats so the imputer and scaler treat them the same
    way on every call rather than depending on the dtype a CSV round-trip
    happened to produce.

    Raises when a feature is absent, naming every one that is missing, and
    raises when a postgame field is offered — the model must not be trainable
    on a column that would not exist before kickoff.
    """
    selected = columns or get_feature_columns()

    missing = _missing_features(frame, selected)
    if missing:
        raise PipelineError(
            "Cannot build the feature matrix; missing required feature(s): "
            + ", ".join(missing)
        )

    leaked = sorted(set(selected).intersection(POSTGAME_COLUMNS | set(TARGET_COLUMNS)))
    if leaked:
        raise PipelineError(
            "Refusing to build a feature matrix containing postgame or target "
            "field(s): " + ", ".join(leaked)
        )

    matrix = frame[selected].copy()
    for column in selected:
        matrix[column] = pd.to_numeric(matrix[column], errors="coerce").astype(float)
    return matrix


def target_vector(frame: pd.DataFrame) -> pd.Series:
    """The label, as integers, with unsettled games rejected.

    A null label means a push, an unplayed game, or a game with no line. Those
    are filtered upstream by ``eligible_games``; reaching here with one means
    the caller skipped that step, so it is an error rather than something to
    quietly drop.
    """
    if TARGET_COLUMN not in frame.columns:
        raise PipelineError(f"Cannot build the target; missing '{TARGET_COLUMN}'.")

    target = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    if target.isna().any():
        raise PipelineError(
            f"{int(target.isna().sum())} row(s) have no settled result; filter "
            "with eligible_games before training."
        )
    return target.astype(int).rename(TARGET_COLUMN)


def fitted_feature_names(pipeline: Pipeline) -> list[str]:
    """The feature order stored on a fitted pipeline.

    scikit-learn records this during ``fit`` and checks it on every later call,
    so it is the authoritative record of what the model was trained on.
    """
    names = getattr(pipeline, "feature_names_in_", None)
    if names is None:
        raise PipelineError(
            "Pipeline has no stored feature names; it has not been fitted on a "
            "DataFrame."
        )
    return list(names)


def model_coefficients(pipeline: Pipeline) -> pd.DataFrame:
    """Map the fitted coefficients back to their feature names.

    Coefficients are reported on the standardised scale, which is the scale
    they were fitted on: each one is the change in log-odds per one standard
    deviation of that feature. ``odds_ratio`` is the same number exponentiated,
    and ``abs_coefficient`` orders the table by influence.

    Because standardisation happens inside the pipeline, these are directly
    comparable across features measured in different units.
    """
    names = fitted_feature_names(pipeline)
    classifier = pipeline.named_steps.get(MODEL_STEP)
    if classifier is None or not hasattr(classifier, "coef_"):
        raise PipelineError("Pipeline has no fitted classifier to read.")

    coefficients = classifier.coef_.ravel()
    if len(coefficients) != len(names):
        raise PipelineError(
            f"Coefficient count ({len(coefficients)}) does not match feature "
            f"count ({len(names)})."
        )

    table = pd.DataFrame(
        {
            "feature": names,
            "coefficient": coefficients,
            "odds_ratio": np.exp(coefficients),
            "abs_coefficient": np.abs(coefficients),
        }
    )
    table.attrs["intercept"] = float(classifier.intercept_[0])
    return table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)


def imputation_medians(pipeline: Pipeline) -> pd.Series:
    """The medians the pipeline learned, by feature name.

    Exposed so a test can prove they came from the training rows alone.
    """
    imputer = pipeline.named_steps.get(IMPUTE_STEP)
    if imputer is None or not hasattr(imputer, "statistics_"):
        raise PipelineError("Pipeline has no fitted imputer to read.")
    return pd.Series(imputer.statistics_, index=fitted_feature_names(pipeline))


def scaling_statistics(pipeline: Pipeline) -> pd.DataFrame:
    """The means and scales the pipeline learned, by feature name."""
    scaler = pipeline.named_steps.get(SCALE_STEP)
    if scaler is None or not hasattr(scaler, "mean_"):
        raise PipelineError("Pipeline has no fitted scaler to read.")
    return pd.DataFrame(
        {"mean": scaler.mean_, "scale": scaler.scale_},
        index=fitted_feature_names(pipeline),
    )
