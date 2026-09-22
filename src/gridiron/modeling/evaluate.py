"""Running the walk-forward backtest and scoring what comes out.

Each fold trains a fresh pipeline and predicts one unseen season. The fold
predictions are concatenated into a single out-of-sample table, so every game
from ``first_validation_season`` onwards appears exactly once, predicted by a
model that had never seen it or anything after it.

Accuracy is reported as a spread across seasons, not only as a pooled number. A
strategy that goes 55% one year and 45% the next averages to something
respectable and is worthless; the standard deviation and the worst season are
the honest summary.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.pipeline import Pipeline

from gridiron.modeling.metrics import WIN_PROFIT, roi
from gridiron.modeling.pipeline import feature_matrix, target_vector
from gridiron.modeling.splits import (
    FINAL_VALIDATION_SEASON,
    FIRST_VALIDATION_SEASON,
    SeasonSplit,
    season_walk_forward_splits,
)
from gridiron.modeling.train import train_model

HOME = "HOME"
AWAY = "AWAY"

# What a model that knows nothing scores. Predicting 0.5 for every game gives a
# log loss of ln(2) and a Brier score of 0.25. Any model scoring *above* these
# is worse than useless: its probabilities are actively misleading, not merely
# uninformative. They are reported alongside the model's own figures so the
# comparison cannot be skipped.
UNINFORMATIVE_LOG_LOSS = float(np.log(2.0))
UNINFORMATIVE_BRIER = 0.25

PREDICTION_COLUMNS = [
    "game_id",
    "season",
    "week",
    "actual_home_cover",
    "predicted_home_cover",
    "home_cover_probability",
    "predicted_side",
    "confidence",
]

AGGREGATE_ROWS = [
    "mean_season_accuracy",
    "median_season_accuracy",
    "worst_season_accuracy",
    "best_season_accuracy",
    "accuracy_std",
    "pooled_accuracy",
    "log_loss",
    "brier_score",
    "roi",
]


class EvaluationError(ValueError):
    """Raised when a backtest cannot be run or scored."""


def _fold_predictions(
    frame: pd.DataFrame,
    pipeline: Pipeline,
    split: SeasonSplit,
) -> pd.DataFrame:
    """Score one fold's validation season into the per-game output."""
    validation = frame.loc[split.validation_indices]
    features = feature_matrix(validation)

    probability = pipeline.predict_proba(features)[:, 1]
    predicted = (probability >= 0.5).astype(int)

    return pd.DataFrame(
        {
            "game_id": validation["game_id"].to_numpy(),
            "season": validation["season"].to_numpy(),
            "week": validation["week"].to_numpy(),
            "actual_home_cover": target_vector(validation).to_numpy(),
            "predicted_home_cover": predicted,
            "home_cover_probability": probability,
            "predicted_side": np.where(predicted == 1, HOME, AWAY),
            # The probability assigned to the side actually picked. 0.5 means
            # the model is indifferent; 1.0 would mean certainty.
            "confidence": np.maximum(probability, 1.0 - probability),
        },
        index=validation.index,
    )


def run_walk_forward(
    df: pd.DataFrame,
    first_validation_season: int = FIRST_VALIDATION_SEASON,
    final_validation_season: int = FINAL_VALIDATION_SEASON,
    *,
    splits: list[SeasonSplit] | None = None,
    c: float = 1.0,
    class_weight: str | None = None,
    collect_coefficients: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train and validate fold by fold; return the predictions and fold summary.

    A fresh pipeline is fitted inside every fold. Reusing one would carry the
    imputation medians and scaling statistics of an earlier fold into a later
    one, which is the same leak the pipeline exists to prevent, moved up a
    level.

    ``splits`` accepts folds computed once elsewhere. The hyperparameter search
    passes the same list to every parameter set, so a difference in the results
    is a difference in the parameters and can never be a difference in the
    folds. ``collect_coefficients`` attaches each fold's fitted coefficients to
    the fold summary, which is how coefficient stability is measured.
    """
    from gridiron.modeling.baselines import eligible_games

    eligible = eligible_games(df).sort_values(["gameday", "game_id"], kind="stable")

    folds = (
        splits
        if splits is not None
        else list(
            season_walk_forward_splits(
                eligible, first_validation_season, final_validation_season
            )
        )
    )

    frames = []
    fold_rows = []
    coefficient_rows = []
    for split in folds:
        train_frame = eligible.loc[split.train_indices]
        pipeline = train_model(
            feature_matrix(train_frame),
            target_vector(train_frame),
            c=c,
            class_weight=class_weight,
        )
        if collect_coefficients:
            from gridiron.modeling.pipeline import model_coefficients

            fold_coefficients = model_coefficients(pipeline)
            fold_coefficients["validation_season"] = split.validation_season
            coefficient_rows.append(fold_coefficients)
        predictions = _fold_predictions(eligible, pipeline, split)
        frames.append(predictions)

        correct = int(
            (
                predictions["predicted_home_cover"] == predictions["actual_home_cover"]
            ).sum()
        )
        fold_rows.append(
            {
                "validation_season": split.validation_season,
                "train_seasons": f"{split.train_seasons[0]}-{split.train_seasons[-1]}",
                "train_games": split.train_size,
                "validation_games": split.validation_size,
                "wins": correct,
                "losses": split.validation_size - correct,
                "accuracy": correct / split.validation_size,
                "roi": roi(correct, split.validation_size - correct),
            }
        )

    if not frames:
        raise EvaluationError("No folds produced predictions.")

    out_of_sample = pd.concat(frames)[PREDICTION_COLUMNS]
    duplicated = out_of_sample["game_id"].duplicated()
    if duplicated.any():
        raise EvaluationError(
            f"{int(duplicated.sum())} game(s) were predicted by more than one "
            "fold; the validation seasons overlap."
        )
    summary = pd.DataFrame(fold_rows)
    if collect_coefficients and coefficient_rows:
        summary.attrs["coefficients"] = pd.concat(coefficient_rows, ignore_index=True)
    return out_of_sample.reset_index(drop=True), summary


def season_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """Accuracy, record, ROI, log loss, and Brier score for each season."""
    _require(predictions)

    rows = []
    for season, group in predictions.groupby("season", sort=True):
        correct = int(
            (group["predicted_home_cover"] == group["actual_home_cover"]).sum()
        )
        losses = len(group) - correct
        rows.append(
            {
                "season": int(season),
                "games": len(group),
                "wins": correct,
                "losses": losses,
                "accuracy": correct / len(group),
                "roi": roi(correct, losses),
                "log_loss": _safe_log_loss(group),
                "brier_score": brier_score_loss(
                    group["actual_home_cover"], group["home_cover_probability"]
                ),
                "mean_confidence": float(group["confidence"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _require(predictions: pd.DataFrame) -> None:
    missing = [
        column for column in PREDICTION_COLUMNS if column not in predictions.columns
    ]
    if missing:
        raise EvaluationError(
            "Predictions are missing column(s): " + ", ".join(missing)
        )
    if predictions.empty:
        raise EvaluationError("No predictions to score.")


def _safe_log_loss(frame: pd.DataFrame) -> float:
    """Log loss, with both classes declared.

    ``log_loss`` infers the label set from the data, so a season in which every
    game went one way would otherwise raise. Declaring the labels keeps the
    metric defined.
    """
    return float(
        log_loss(
            frame["actual_home_cover"],
            frame["home_cover_probability"],
            labels=[0, 1],
        )
    )


def aggregate_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    """The overall summary, one metric per row.

    Accuracy is summarised across seasons -- mean, median, worst, best, and
    standard deviation -- because the spread matters more than the average. Log
    loss, Brier score, and ROI are pooled over every out-of-sample game.
    """
    _require(predictions)

    seasons = season_metrics(predictions)
    accuracies = seasons["accuracy"]

    correct = int(
        (predictions["predicted_home_cover"] == predictions["actual_home_cover"]).sum()
    )
    losses = len(predictions) - correct

    values = {
        "mean_season_accuracy": float(accuracies.mean()),
        "median_season_accuracy": float(accuracies.median()),
        "worst_season_accuracy": float(accuracies.min()),
        "best_season_accuracy": float(accuracies.max()),
        "accuracy_std": float(accuracies.std(ddof=1)) if len(accuracies) > 1 else 0.0,
        "pooled_accuracy": correct / len(predictions),
        "log_loss": _safe_log_loss(predictions),
        "brier_score": float(
            brier_score_loss(
                predictions["actual_home_cover"],
                predictions["home_cover_probability"],
            )
        ),
        "roi": roi(correct, losses),
    }

    table = pd.DataFrame(
        {"metric": AGGREGATE_ROWS, "value": [values[key] for key in AGGREGATE_ROWS]}
    )
    table.attrs["worst_season"] = int(seasons.loc[accuracies.idxmin(), "season"])
    table.attrs["best_season"] = int(seasons.loc[accuracies.idxmax(), "season"])
    table.attrs["games"] = len(predictions)
    table.attrs["uninformative_log_loss"] = UNINFORMATIVE_LOG_LOSS
    table.attrs["uninformative_brier"] = UNINFORMATIVE_BRIER
    table.attrs["beats_uninformative"] = bool(
        values["log_loss"] < UNINFORMATIVE_LOG_LOSS
        and values["brier_score"] < UNINFORMATIVE_BRIER
    )
    return table


def probability_quality(predictions: pd.DataFrame) -> pd.DataFrame:
    """Compare the model's probabilities against knowing nothing.

    A model whose log loss exceeds ln(2), or whose Brier score exceeds 0.25, is
    not merely adding no information: it would be improved by replacing every
    prediction with a flat 0.5.
    """
    _require(predictions)
    model_log_loss = _safe_log_loss(predictions)
    model_brier = float(
        brier_score_loss(
            predictions["actual_home_cover"],
            predictions["home_cover_probability"],
        )
    )
    return pd.DataFrame(
        [
            {
                "metric": "log_loss",
                "model": model_log_loss,
                "uninformative_0.5": UNINFORMATIVE_LOG_LOSS,
                "model_is_better": model_log_loss < UNINFORMATIVE_LOG_LOSS,
            },
            {
                "metric": "brier_score",
                "model": model_brier,
                "uninformative_0.5": UNINFORMATIVE_BRIER,
                "model_is_better": model_brier < UNINFORMATIVE_BRIER,
            },
        ]
    )


def profit_by_season(predictions: pd.DataFrame) -> pd.Series:
    """Cumulative profit in units across the whole out-of-sample run."""
    _require(predictions)
    ordered = predictions.sort_values(["season", "week"], kind="stable")
    outcome = np.where(
        ordered["predicted_home_cover"] == ordered["actual_home_cover"],
        WIN_PROFIT,
        -1.0,
    )
    return pd.Series(np.cumsum(outcome), index=ordered.index, name="profit_units")
