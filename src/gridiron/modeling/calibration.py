"""Calibration analysis for out-of-sample predictions.

A calibrated binary classifier is one whose stated probabilities match observed
rates: among the games it calls at 0.80, about 80% should cover. This module
measures that against the walk-forward predictions, and decides -- on evidence,
not on availability -- whether a post-hoc calibrator is worth adding.

The distinction that decides it is between **calibration** and
**discrimination**. Calibration is whether the numbers mean what they say;
discrimination is whether they rank one game above another at all. A
calibrator can fix the first and can never supply the second. Applied to a
score that does not rank, it collapses every prediction to the base rate --
which is not a calibrated model, it is the absence of one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

# The buckets the phase brief asks for, with a below-0.40 bucket added because
# the model does produce predictions there and silently dropping them would
# misrepresent the tail.
BUCKET_EDGES = [0.0, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 1.0]
BUCKET_LABELS = [
    "<0.40",
    "0.40-0.45",
    "0.45-0.50",
    "0.50-0.55",
    "0.55-0.60",
    "0.60-0.65",
    "0.65+",
]

# Below this many games, an observed rate says almost nothing: the standard
# error on a proportion at n=30 is about nine percentage points, which is wider
# than any miscalibration worth acting on.
MIN_BUCKET_SAMPLES = 30

# A bucket is called biased when its observed rate misses its predicted rate by
# more than this, and it has enough games for that gap to mean anything.
BIAS_TOLERANCE = 0.05

# A calibration slope of 1.0 is perfect. Below 1 means predictions are spread
# too widely; at or below 0 the scores carry no usable ordering at all.
MIN_USEFUL_SLOPE = 0.0

# Discrimination floor. An AUC of 0.5 is chance -- but a *measured* AUC wanders
# around 0.5 by sampling noise alone, so 0.5122 from 1,500 coin flips is not
# evidence of ranking. The floor is therefore 0.5 plus this many null standard
# errors, which scales with the sample rather than being a fixed guess.
MIN_USEFUL_AUC = 0.5
AUC_NULL_MARGIN = 2.0

PROBABILITY_COLUMN = "home_cover_probability"
ACTUAL_COLUMN = "actual_home_cover"


class CalibrationError(ValueError):
    """Raised when calibration cannot be assessed."""


@dataclass(frozen=True)
class CalibrationDecision:
    """Whether to calibrate, and the evidence behind the answer."""

    calibrate: bool
    reasons: list[str] = field(default_factory=list)
    auc: float = float("nan")
    slope: float = float("nan")
    intercept: float = float("nan")
    biased_buckets: int = 0
    sparse_buckets: int = 0


def _require(predictions: pd.DataFrame) -> None:
    missing = [
        column
        for column in (PROBABILITY_COLUMN, ACTUAL_COLUMN)
        if column not in predictions.columns
    ]
    if missing:
        raise CalibrationError(
            "Cannot assess calibration; missing column(s): " + ", ".join(missing)
        )
    if predictions.empty:
        raise CalibrationError("No predictions to assess.")


def reliability_table(predictions: pd.DataFrame) -> pd.DataFrame:
    """Observed cover rate against predicted probability, by bucket.

    Every row carries its sample count, because a reliability figure without
    one invites reading a three-game bucket as though it meant something.
    ``sparse`` marks the buckets where it does not.
    """
    _require(predictions)

    probability = pd.to_numeric(predictions[PROBABILITY_COLUMN], errors="coerce")
    actual = pd.to_numeric(predictions[ACTUAL_COLUMN], errors="coerce")
    buckets = pd.cut(
        probability,
        BUCKET_EDGES,
        labels=BUCKET_LABELS,
        include_lowest=True,
        right=False,
    )

    frame = pd.DataFrame(
        {"bucket": buckets, "probability": probability, "actual": actual}
    )
    grouped = frame.groupby("bucket", observed=False).agg(
        games=("actual", "size"),
        mean_predicted=("probability", "mean"),
        observed_rate=("actual", "mean"),
    )
    grouped = grouped.reset_index()
    grouped["gap"] = grouped["observed_rate"] - grouped["mean_predicted"]
    grouped["sparse"] = grouped["games"] < MIN_BUCKET_SAMPLES
    # A rate from n games has this much standard error; shown so a reader can
    # see immediately whether a gap is larger than the noise around it.
    grouped["standard_error"] = np.sqrt(
        grouped["observed_rate"].clip(0, 1)
        * (1 - grouped["observed_rate"].clip(0, 1))
        / grouped["games"].replace(0, np.nan)
    )
    grouped["biased"] = grouped["gap"].abs().gt(BIAS_TOLERANCE) & ~grouped["sparse"]
    return grouped


def sparse_buckets(table: pd.DataFrame) -> pd.DataFrame:
    """The buckets too thin to support a conclusion."""
    return table.loc[table["sparse"] & table["games"].gt(0)].reset_index(drop=True)


def calibration_slope(predictions: pd.DataFrame) -> tuple[float, float]:
    """Slope and intercept of observed outcome on the predicted log-odds.

    Slope 1.0 with intercept 0.0 is perfect calibration. A slope below 1 means
    the predictions are spread wider than the evidence supports. A *negative*
    slope means they point the wrong way: higher predicted probability goes
    with a lower observed rate.
    """
    _require(predictions)

    probability = pd.to_numeric(predictions[PROBABILITY_COLUMN], errors="coerce").clip(
        1e-6, 1 - 1e-6
    )
    actual = pd.to_numeric(predictions[ACTUAL_COLUMN], errors="coerce")

    logit = np.log(probability / (1 - probability)).to_numpy().reshape(-1, 1)
    # C is set very large to approximate an unpenalised fit; the slope is being
    # measured, not regularised.
    model = LogisticRegression(C=1e12, solver="lbfgs", max_iter=1000)
    model.fit(logit, actual.astype(int))
    return float(model.coef_[0][0]), float(model.intercept_[0])


def discrimination(predictions: pd.DataFrame) -> float:
    """Area under the ROC curve: whether the scores rank games at all."""
    _require(predictions)
    actual = pd.to_numeric(predictions[ACTUAL_COLUMN], errors="coerce").astype(int)
    if actual.nunique() < 2:
        return float("nan")
    return float(roc_auc_score(actual, pd.to_numeric(predictions[PROBABILITY_COLUMN])))


def auc_null_threshold(predictions: pd.DataFrame) -> float:
    """The AUC a set of scores must clear to count as ranking anything.

    Under the null hypothesis of no discrimination, the measured AUC has a
    standard error of roughly ``sqrt((n1 + n0 + 1) / (12 * n1 * n0))``. The
    threshold is 0.5 plus :data:`AUC_NULL_MARGIN` of those, so a model is only
    credited with ranking when it does so by more than chance would supply at
    the sample size actually available.
    """
    _require(predictions)
    actual = pd.to_numeric(predictions[ACTUAL_COLUMN], errors="coerce").astype(int)
    positives = int(actual.sum())
    negatives = len(actual) - positives
    if not positives or not negatives:
        return float("nan")

    standard_error = np.sqrt(
        (positives + negatives + 1) / (12.0 * positives * negatives)
    )
    return MIN_USEFUL_AUC + AUC_NULL_MARGIN * standard_error


def decide_calibration(predictions: pd.DataFrame) -> CalibrationDecision:
    """Decide whether post-hoc calibration is worth adding.

    Systematic bias alone is not sufficient grounds. The scores must also rank
    games, because a calibrator maps scores to probabilities and cannot invent
    an ordering that is not there. When discrimination is absent, calibration
    collapses every prediction to the base rate -- available already, without
    a model.
    """
    table = reliability_table(predictions)
    auc = discrimination(predictions)
    threshold = auc_null_threshold(predictions)
    slope, intercept = calibration_slope(predictions)

    biased = int(table["biased"].sum())
    sparse = int((table["sparse"] & table["games"].gt(0)).sum())
    ranks = bool(auc > threshold)

    reasons: list[str] = []
    reasons.append(
        f"{biased} well-populated bucket(s) miss their predicted rate by more "
        f"than {BIAS_TOLERANCE:.2f}"
    )
    reasons.append(
        f"Discrimination: AUC {auc:.4f} against a chance threshold of "
        f"{threshold:.4f} ({'ranks games' if ranks else 'does not rank games'})"
    )
    reasons.append(
        f"Calibration slope {slope:+.4f}, intercept {intercept:+.4f} "
        f"(1.0 would be perfectly calibrated)"
    )

    if not ranks:
        reasons.append(
            "Declined: the scores do not rank games by more than chance would "
            "supply at this sample size, so a calibrator would map them all to "
            "the base rate. That is not a calibrated model, it is the absence "
            "of one."
        )
        return CalibrationDecision(
            False, reasons, auc, slope, intercept, biased, sparse
        )

    if slope <= MIN_USEFUL_SLOPE:
        reasons.append(
            "Declined: the calibration slope is not positive, so a fitted "
            "calibrator would invert the predictions. That is learning the "
            "sign of noise, not correcting a bias."
        )
        return CalibrationDecision(
            False, reasons, auc, slope, intercept, biased, sparse
        )

    if not biased:
        reasons.append(
            "Declined: no well-populated bucket shows bias worth correcting."
        )
        return CalibrationDecision(
            False, reasons, auc, slope, intercept, biased, sparse
        )

    reasons.append(
        "Calibration is warranted: the scores rank games and the mapping from "
        "score to probability is systematically off."
    )
    return CalibrationDecision(True, reasons, auc, slope, intercept, biased, sparse)


def temporal_calibration_split(
    train_frame: pd.DataFrame,
    calibration_seasons: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a fold's training rows into fit and calibration halves.

    The calibration rows are the *latest* seasons of the training window, so
    they sit after the rows the model is fitted on and before the validation
    season. Fitting a calibrator on the same rows that fitted the model would
    measure how well it memorised them; fitting it on later rows keeps the
    whole procedure pointing forward in time.
    """
    if "season" not in train_frame.columns:
        raise CalibrationError("Cannot split for calibration; missing 'season'.")

    seasons = sorted(train_frame["season"].unique())
    if len(seasons) <= calibration_seasons:
        raise CalibrationError(
            f"Need more than {calibration_seasons} training season(s) to hold "
            f"one back for calibration; got {len(seasons)}."
        )

    held = seasons[-calibration_seasons:]
    fit = train_frame.loc[~train_frame["season"].isin(held)]
    calibrate = train_frame.loc[train_frame["season"].isin(held)]
    return fit, calibrate


def fit_calibrator(
    scores: np.ndarray,
    outcomes: np.ndarray,
    method: str = "platt",
):
    """Fit a post-hoc calibrator on held-back scores.

    ``platt`` fits a logistic curve to the score; ``isotonic`` fits a
    non-decreasing step function, which is far more flexible and therefore far
    easier to overfit on a single season.
    """
    if method == "platt":
        return LogisticRegression(solver="lbfgs", max_iter=1000).fit(
            np.asarray(scores).reshape(-1, 1), np.asarray(outcomes)
        )
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(scores, outcomes)
    raise CalibrationError(f"Unknown calibration method {method!r}.")


def apply_calibrator(calibrator, scores: np.ndarray) -> np.ndarray:
    """Apply a fitted calibrator, keeping probabilities inside (0, 1)."""
    scores = np.asarray(scores)
    if isinstance(calibrator, IsotonicRegression):
        return np.clip(calibrator.predict(scores), 1e-6, 1 - 1e-6)
    return calibrator.predict_proba(scores.reshape(-1, 1))[:, 1]


def compare_calibration(
    predictions: pd.DataFrame,
    calibrated: dict[str, np.ndarray],
) -> pd.DataFrame:
    """Score raw and calibrated probabilities side by side.

    The base-rate row is the comparison that matters: a calibrator that merely
    matches it has not improved the model, it has replaced it.
    """
    _require(predictions)
    actual = pd.to_numeric(predictions[ACTUAL_COLUMN], errors="coerce").astype(int)
    raw = pd.to_numeric(predictions[PROBABILITY_COLUMN], errors="coerce")

    rows = [
        {
            "probabilities": "raw",
            "log_loss": float(log_loss(actual, raw, labels=[0, 1])),
            "brier": float(brier_score_loss(actual, raw)),
        }
    ]
    for name, values in calibrated.items():
        clipped = np.clip(values, 1e-6, 1 - 1e-6)
        rows.append(
            {
                "probabilities": name,
                "log_loss": float(log_loss(actual, clipped, labels=[0, 1])),
                "brier": float(brier_score_loss(actual, clipped)),
            }
        )

    constant = np.full(len(actual), float(actual.mean()))
    rows.append(
        {
            "probabilities": "constant base rate",
            "log_loss": float(log_loss(actual, constant, labels=[0, 1])),
            "brier": float(brier_score_loss(actual, constant)),
        }
    )
    return pd.DataFrame(rows)
