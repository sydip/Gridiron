"""Scoring for against-the-spread strategies.

Accuracy alone is misleading at a bookmaker's price: a strategy can be right
more often than not and still lose money. Every strategy here is therefore
reported with its record, its return on investment at the standard price, and
the worst peak-to-trough run it would have put a bankroll through.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Standard American odds of -110: stake 110 to win 100. A winning unit bet
# returns 100/110 of a unit in profit; a losing one costs the whole unit. This
# is why 50% accuracy is not break-even -- 52.38% is.
STANDARD_PRICE = -110.0
WIN_PROFIT = 100.0 / 110.0
BREAK_EVEN_ACCURACY = 110.0 / 210.0

TARGET_COLUMN = "home_cover"

SUMMARY_COLUMNS = [
    "strategy",
    "games",
    "wins",
    "losses",
    "ats_record",
    "accuracy",
    "roi",
    "profit_units",
    "max_drawdown_units",
    "beats_break_even",
]


class MetricError(ValueError):
    """Raised when predictions and outcomes cannot be scored."""


def _aligned(
    matchups: pd.DataFrame, predictions: pd.Series
) -> tuple[pd.Series, pd.Series]:
    """Return the actual and predicted series, checked for alignment."""
    if TARGET_COLUMN not in matchups.columns:
        raise MetricError(f"Cannot score; missing '{TARGET_COLUMN}'.")
    if len(predictions) != len(matchups):
        raise MetricError(
            f"Predictions ({len(predictions)}) and games ({len(matchups)}) "
            "differ in length."
        )
    if not predictions.index.equals(matchups.index):
        raise MetricError("Predictions are not aligned to the games frame.")

    actual = pd.to_numeric(matchups[TARGET_COLUMN], errors="coerce")
    if actual.isna().any():
        raise MetricError(
            "Unsettled games reached scoring; filter with eligible_games first."
        )
    predicted = pd.to_numeric(predictions, errors="coerce")
    if predicted.isna().any():
        raise MetricError("Predictions contain missing values.")
    return actual.astype(int), predicted.astype(int)


def wins_and_losses(matchups: pd.DataFrame, predictions: pd.Series) -> tuple[int, int]:
    """Correct and incorrect picks. Pushes never reach here."""
    actual, predicted = _aligned(matchups, predictions)
    correct = int((actual == predicted).sum())
    return correct, len(actual) - correct


def accuracy(matchups: pd.DataFrame, predictions: pd.Series) -> float:
    """Share of picks that were correct."""
    wins, losses = wins_and_losses(matchups, predictions)
    total = wins + losses
    return float(wins / total) if total else float("nan")


def roi(wins: int, losses: int, win_profit: float = WIN_PROFIT) -> float:
    """Return per unit staked at the standard price.

    Each pick stakes one unit. A win returns ``win_profit``; a loss costs one.
    """
    total = wins + losses
    if not total:
        return float("nan")
    return float((wins * win_profit - losses) / total)


def profit_curve(
    matchups: pd.DataFrame,
    predictions: pd.Series,
    win_profit: float = WIN_PROFIT,
) -> pd.Series:
    """Cumulative profit in units, in the order the games were played.

    Sorted by date first: a drawdown is a statement about a sequence, so it is
    meaningless on rows in an arbitrary order.
    """
    actual, predicted = _aligned(matchups, predictions)
    order = (
        matchups.sort_values(["gameday", "game_id"], kind="stable").index
        if {"gameday", "game_id"}.issubset(matchups.columns)
        else matchups.index
    )
    outcome = np.where(actual.loc[order] == predicted.loc[order], win_profit, -1.0)
    return pd.Series(np.cumsum(outcome), index=order, name="profit_units")


def max_drawdown(curve: pd.Series) -> float:
    """Largest peak-to-trough fall in the profit curve, in units.

    Reported as a positive number. The running peak starts at zero, so a
    strategy that never rises above its starting bankroll still reports the
    full depth of its decline rather than zero.
    """
    if curve.empty:
        return float("nan")
    values = curve.to_numpy(dtype=float)
    running_peak = np.maximum.accumulate(np.concatenate([[0.0], values]))[1:]
    return float(np.max(running_peak - values))


def season_accuracy(matchups: pd.DataFrame, predictions: pd.Series) -> pd.DataFrame:
    """Accuracy, record, and ROI for each season."""
    if "season" not in matchups.columns:
        raise MetricError("Cannot break down by season; missing 'season'.")
    actual, predicted = _aligned(matchups, predictions)

    frame = pd.DataFrame(
        {
            "season": matchups["season"].to_numpy(),
            "correct": (actual == predicted).to_numpy(),
        }
    )
    grouped = frame.groupby("season", sort=True)["correct"].agg(["size", "sum"])
    grouped.columns = ["games", "wins"]
    grouped["losses"] = grouped["games"] - grouped["wins"]
    grouped["accuracy"] = grouped["wins"] / grouped["games"]
    grouped["roi"] = [
        roi(int(row.wins), int(row.losses)) for row in grouped.itertuples()
    ]
    return grouped.reset_index()


def evaluate(
    matchups: pd.DataFrame,
    predictions: pd.Series,
    name: str,
) -> dict[str, object]:
    """Score one strategy into a single summary row."""
    wins, losses = wins_and_losses(matchups, predictions)
    curve = profit_curve(matchups, predictions)
    strategy_roi = roi(wins, losses)
    return {
        "strategy": name,
        "games": wins + losses,
        "wins": wins,
        "losses": losses,
        "ats_record": f"{wins}-{losses}",
        "accuracy": accuracy(matchups, predictions),
        "roi": strategy_roi,
        "profit_units": float(curve.iloc[-1]) if len(curve) else float("nan"),
        "max_drawdown_units": max_drawdown(curve),
        "beats_break_even": bool(strategy_roi > 0),
    }


def comparison_table(
    matchups: pd.DataFrame,
    strategies: dict[str, pd.Series],
) -> pd.DataFrame:
    """Score every strategy on the same games, best ROI first.

    Every strategy is handed the identical frame, so a difference in the table
    is a difference in the picks and never a difference in the sample.
    """
    if not strategies:
        raise MetricError("No strategies to compare.")
    rows = [
        evaluate(matchups, predictions, name)
        for name, predictions in strategies.items()
    ]
    table = pd.DataFrame(rows, columns=SUMMARY_COLUMNS)
    return table.sort_values("roi", ascending=False).reset_index(drop=True)


def season_table(
    matchups: pd.DataFrame,
    strategies: dict[str, pd.Series],
) -> pd.DataFrame:
    """Season-by-season accuracy for every strategy, one row per season."""
    frames = []
    for name, predictions in strategies.items():
        seasons = season_accuracy(matchups, predictions)
        seasons.insert(0, "strategy", name)
        frames.append(seasons)
    return pd.concat(frames, ignore_index=True)
