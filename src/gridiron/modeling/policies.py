"""Backtesting confidence-filtered betting policies.

A policy is a rule for which predictions to act on. The simplest bets every
game; the rest bet only where the model claims an edge over a coin flip. The
question is whether filtering on the model's own confidence improves results.

Three things make this easy to get wrong, and are handled explicitly:

* **Pushes are real.** A game that lands exactly on the number refunds the
  stake. It is neither a win nor a loss, and counting it as either would
  distort every rate on the page. Push games are bet on and reported, never
  silently dropped -- which means the backtest predicts them too, even though
  they cannot be trained on.
* **A threshold chosen by its backtest result is not a finding.** Every policy
  here is scored on out-of-sample fold predictions, and nothing selects a
  threshold by looking at them.
* **A high return on a handful of bets is noise.** Sample size and a confidence
  interval sit beside every figure, and thin policies are flagged.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from gridiron.modeling.metrics import BREAK_EVEN_ACCURACY, WIN_PROFIT

# Thresholds on the model's edge over a coin flip, in percentage points:
# an edge of 2 means the model put the game at 0.52 or at 0.48.
POLICY_THRESHOLDS = [0.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]

# Below this many bets, a win rate says almost nothing: the 95% interval on a
# 50% rate over 100 bets is roughly +/-10 points, which is wider than any edge
# a model of this kind could plausibly have.
MIN_BETS_FOR_A_CLAIM = 100

# A second, harsher line. Under this, the figures are reported but should not
# be read as measurements at all.
TINY_SAMPLE = 30

# Threshold comparisons need a tolerance. In binary floating point
# abs(0.45 - 0.5) * 100 is 4.999999999999999 while abs(0.55 - 0.5) * 100 is
# 5.000000000000004, so a bare ``>=`` would admit a prediction of 0.55 to a
# 5-point policy and turn away its mirror image at 0.45. The two sides of the
# threshold must be treated alike.
THRESHOLD_TOLERANCE = 1e-9

RESULT_COLUMNS = [
    "policy",
    "threshold_pp",
    "bets",
    "wins",
    "losses",
    "pushes",
    "win_rate",
    "win_rate_ci_low",
    "win_rate_ci_high",
    "units_won",
    "roi",
    "max_drawdown_units",
    "longest_losing_streak",
    "profitable_seasons",
    "seasons",
    "sample_warning",
]

WIN = "win"
LOSS = "loss"
PUSH = "push"


class PolicyError(ValueError):
    """Raised when a policy backtest cannot be run."""


@dataclass(frozen=True)
class OddsAssumption:
    """The price every bet is assumed to be struck at.

    American -110: stake 110 to win 100. A winning unit returns
    ``win_profit`` in profit, a loser costs the full unit, and a push returns
    the stake with no profit or loss. Break-even is 52.38% of resolved bets,
    not 50%.

    This is the standard price for a point spread. Real books move it, and a
    backtest at a fixed price is therefore an approximation that flatters
    nothing but assumes nothing worse either.
    """

    name: str = "American -110"
    win_profit: float = WIN_PROFIT
    loss_cost: float = 1.0
    push_profit: float = 0.0
    break_even_rate: float = BREAK_EVEN_ACCURACY


DEFAULT_ODDS = OddsAssumption()


def edge_percentage_points(probabilities: pd.Series) -> pd.Series:
    """How far each prediction sits from a coin flip, in percentage points."""
    return (pd.to_numeric(probabilities, errors="coerce") - 0.5).abs() * 100.0


def settle(
    predictions: pd.DataFrame,
    actual_column: str = "actual_home_cover",
    predicted_column: str = "predicted_home_cover",
    push_column: str = "is_push",
) -> pd.Series:
    """Label every game a win, a loss, or a push.

    A push is decided by the game, not by the prediction: the result landed on
    the number, so the bet is refunded whichever side was taken.
    """
    missing = [
        column
        for column in (actual_column, predicted_column)
        if column not in predictions.columns
    ]
    if missing:
        raise PolicyError("Cannot settle bets; missing: " + ", ".join(missing))

    pushed = (
        predictions[push_column].fillna(False).astype(bool)
        if push_column in predictions.columns
        else pd.Series(False, index=predictions.index)
    )
    correct = pd.to_numeric(predictions[predicted_column], errors="coerce").eq(
        pd.to_numeric(predictions[actual_column], errors="coerce")
    )
    return pd.Series(
        np.where(pushed, PUSH, np.where(correct, WIN, LOSS)),
        index=predictions.index,
        name="settlement",
    )


def _profit(settlements: pd.Series, odds: OddsAssumption) -> pd.Series:
    return pd.Series(
        np.select(
            [settlements.eq(WIN), settlements.eq(LOSS)],
            [odds.win_profit, -odds.loss_cost],
            default=odds.push_profit,
        ),
        index=settlements.index,
        name="profit_units",
    )


def wilson_interval(
    wins: int, resolved: int, confidence: float = 0.95
) -> tuple[float, float]:
    """A confidence interval on a win rate that behaves at small samples.

    The Wilson interval is used rather than the textbook normal approximation
    because the normal one produces nonsense at the sample sizes the strictest
    policies reach -- intervals that run past 0 and 1, and that are far too
    narrow when only a handful of bets were placed.
    """
    if resolved <= 0:
        return (float("nan"), float("nan"))

    z = 1.959963984540054 if confidence == 0.95 else 1.6448536269514722
    rate = wins / resolved
    denominator = 1 + z**2 / resolved
    centre = rate + z**2 / (2 * resolved)
    spread = z * np.sqrt(rate * (1 - rate) / resolved + z**2 / (4 * resolved**2))
    return (
        float(max(0.0, (centre - spread) / denominator)),
        float(min(1.0, (centre + spread) / denominator)),
    )


def longest_losing_streak(settlements: pd.Series) -> int:
    """The worst run of consecutive losses. Pushes break nothing."""
    longest = current = 0
    for settlement in settlements:
        if settlement == LOSS:
            current += 1
            longest = max(longest, current)
        elif settlement == WIN:
            current = 0
        # A push leaves the streak where it stands: no money changed hands.
    return longest


def max_drawdown(profit: pd.Series) -> float:
    """Largest peak-to-trough fall in cumulative profit, as a positive number."""
    if profit.empty:
        return 0.0
    curve = profit.cumsum().to_numpy(dtype=float)
    peak = np.maximum.accumulate(np.concatenate([[0.0], curve]))[1:]
    return float(np.max(peak - curve))


def apply_policy(
    predictions: pd.DataFrame,
    threshold_pp: float,
) -> pd.DataFrame:
    """The subset of games a policy would have bet, in the order played."""
    if "home_cover_probability" not in predictions.columns:
        raise PolicyError("Cannot apply a policy; missing probabilities.")

    frame = predictions.copy()
    frame["edge_pp"] = edge_percentage_points(frame["home_cover_probability"])
    selected = frame.loc[frame["edge_pp"] >= threshold_pp - THRESHOLD_TOLERANCE].copy()

    sort_columns = [
        column for column in ("season", "week", "game_id") if column in selected
    ]
    return selected.sort_values(sort_columns, kind="stable")


def backtest_policy(
    predictions: pd.DataFrame,
    threshold_pp: float,
    odds: OddsAssumption = DEFAULT_ODDS,
) -> dict[str, object]:
    """Score one policy into a single row of results."""
    bets = apply_policy(predictions, threshold_pp)
    name = "bet every prediction" if threshold_pp == 0 else f">= {threshold_pp}pp"

    if bets.empty:
        return {
            "policy": name,
            "threshold_pp": threshold_pp,
            "bets": 0,
            "wins": 0,
            "losses": 0,
            "pushes": 0,
            "win_rate": float("nan"),
            "win_rate_ci_low": float("nan"),
            "win_rate_ci_high": float("nan"),
            "units_won": 0.0,
            "roi": float("nan"),
            "max_drawdown_units": 0.0,
            "longest_losing_streak": 0,
            "profitable_seasons": 0,
            "seasons": 0,
            "sample_warning": "no bets placed",
        }

    settlements = settle(bets)
    profit = _profit(settlements, odds)

    wins = int(settlements.eq(WIN).sum())
    losses = int(settlements.eq(LOSS).sum())
    pushes = int(settlements.eq(PUSH).sum())
    resolved = wins + losses

    low, high = wilson_interval(wins, resolved)

    seasons = 0
    profitable = 0
    if "season" in bets.columns:
        by_season = profit.groupby(bets["season"]).sum()
        seasons = int(len(by_season))
        profitable = int((by_season > 0).sum())

    return {
        "policy": name,
        "threshold_pp": threshold_pp,
        "bets": len(bets),
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "win_rate": float(wins / resolved) if resolved else float("nan"),
        "win_rate_ci_low": low,
        "win_rate_ci_high": high,
        "units_won": float(profit.sum()),
        # ROI is per *resolved* bet. A push returns the stake, so it neither
        # earns nor risks anything and does not belong in the denominator.
        "roi": float(profit.sum() / resolved) if resolved else float("nan"),
        "max_drawdown_units": max_drawdown(profit),
        "longest_losing_streak": longest_losing_streak(settlements),
        "profitable_seasons": profitable,
        "seasons": seasons,
        "sample_warning": sample_warning(len(bets), resolved),
    }


def sample_warning(bets: int, resolved: int) -> str:
    """Say plainly when a policy has too few bets to support a conclusion."""
    if resolved == 0:
        return "no resolved bets"
    if bets < TINY_SAMPLE:
        return (
            f"only {bets} bets: far below the {MIN_BETS_FOR_A_CLAIM} needed for "
            "a win rate to mean anything; treat every figure on this row as "
            "noise"
        )
    if bets < MIN_BETS_FOR_A_CLAIM:
        return (
            f"only {bets} bets, under the {MIN_BETS_FOR_A_CLAIM} needed for a "
            "win rate to be informative"
        )
    return ""


def backtest_policies(
    predictions: pd.DataFrame,
    thresholds: list[float] | None = None,
    odds: OddsAssumption = DEFAULT_ODDS,
) -> pd.DataFrame:
    """Score every policy on the same out-of-sample predictions."""
    rows = [
        backtest_policy(predictions, threshold, odds)
        for threshold in (thresholds or POLICY_THRESHOLDS)
    ]
    return pd.DataFrame(rows, columns=RESULT_COLUMNS)


def flag_unreliable_results(results: pd.DataFrame) -> pd.DataFrame:
    """Policies whose headline number rests on too little evidence.

    The case this exists for: a policy that bet seven times, won five, and
    shows a spectacular return. Flagging it in the table is the difference
    between a backtest and a sales pitch.
    """
    if results.empty:
        return results

    flagged = results.loc[results["sample_warning"].ne("")].copy()
    flagged["interval_width"] = flagged["win_rate_ci_high"] - flagged["win_rate_ci_low"]
    return flagged.reset_index(drop=True)


def cumulative_units(
    predictions: pd.DataFrame,
    threshold_pp: float,
    odds: OddsAssumption = DEFAULT_ODDS,
) -> pd.DataFrame:
    """The running bankroll for one policy, in the order the games were played."""
    bets = apply_policy(predictions, threshold_pp)
    if bets.empty:
        return pd.DataFrame(
            columns=["season", "week", "profit_units", "cumulative_units"]
        )

    settlements = settle(bets)
    profit = _profit(settlements, odds)
    return pd.DataFrame(
        {
            "season": bets.get("season"),
            "week": bets.get("week"),
            "game_id": bets.get("game_id"),
            "settlement": settlements,
            "profit_units": profit,
            "cumulative_units": profit.cumsum(),
        }
    ).reset_index(drop=True)


def season_units(
    predictions: pd.DataFrame,
    threshold_pp: float,
    odds: OddsAssumption = DEFAULT_ODDS,
) -> pd.DataFrame:
    """Units won per season for one policy."""
    bets = apply_policy(predictions, threshold_pp)
    if bets.empty or "season" not in bets.columns:
        return pd.DataFrame(columns=["season", "bets", "units_won", "roi"])

    settlements = settle(bets)
    profit = _profit(settlements, odds)
    frame = pd.DataFrame(
        {
            "season": bets["season"].to_numpy(),
            "profit": profit.to_numpy(),
            "resolved": settlements.ne(PUSH).to_numpy(),
        }
    )
    grouped = frame.groupby("season", sort=True).agg(
        bets=("profit", "size"),
        units_won=("profit", "sum"),
        resolved=("resolved", "sum"),
    )
    grouped["roi"] = grouped["units_won"] / grouped["resolved"].replace(0, np.nan)
    return grouped.reset_index()


def win_rate_by_confidence(
    predictions: pd.DataFrame,
    bin_edges: list[float] | None = None,
) -> pd.DataFrame:
    """Observed win rate by how much edge the model claimed.

    If confidence filtering worked, the win rate would rise with the edge.
    Each row carries its bet count and interval so a rise built on four bets
    is visible as such.
    """
    edges = bin_edges or [0.0, 2.0, 3.0, 4.0, 5.0, 7.5, 100.0]
    frame = predictions.copy()
    frame["edge_pp"] = edge_percentage_points(frame["home_cover_probability"])
    frame["settlement"] = settle(frame)
    frame["bucket"] = pd.cut(frame["edge_pp"], edges, right=False)

    rows = []
    for bucket, group in frame.groupby("bucket", observed=True):
        wins = int(group["settlement"].eq(WIN).sum())
        losses = int(group["settlement"].eq(LOSS).sum())
        resolved = wins + losses
        low, high = wilson_interval(wins, resolved)
        rows.append(
            {
                "edge_bucket": str(bucket),
                "bets": len(group),
                "wins": wins,
                "losses": losses,
                "pushes": int(group["settlement"].eq(PUSH).sum()),
                "win_rate": wins / resolved if resolved else float("nan"),
                "ci_low": low,
                "ci_high": high,
                "sparse": len(group) < MIN_BETS_FOR_A_CLAIM,
            }
        )
    return pd.DataFrame(rows)


def bettable_predictions(
    matchups: pd.DataFrame,
    first_validation_season: int | None = None,
    final_validation_season: int | None = None,
) -> pd.DataFrame:
    """Walk-forward predictions over every game that could have been bet.

    This differs from the walk-forward evaluation in one way that matters for a
    betting backtest: it predicts push games too. A push cannot be a training
    label -- there is no right answer to learn -- but it is a game the policy
    would have staked money on, and leaving it out would understate the number
    of bets and hide the refunds.

    So each fold trains on the settled games of its earlier seasons and then
    predicts every completed, priced game of its validation season, pushes
    included.
    """
    from gridiron.modeling.baselines import eligible_games
    from gridiron.modeling.pipeline import feature_matrix, target_vector
    from gridiron.modeling.splits import (
        FINAL_VALIDATION_SEASON,
        FIRST_VALIDATION_SEASON,
        season_walk_forward_splits,
    )
    from gridiron.modeling.train import train_model
    from gridiron.modeling.tuning import load_selected_parameters

    first = first_validation_season or FIRST_VALIDATION_SEASON
    final = final_validation_season or FINAL_VALIDATION_SEASON
    selection = load_selected_parameters()

    required = {"home_score", "away_score", "spread_line", "season", "game_id"}
    missing = sorted(required.difference(matchups.columns))
    if missing:
        raise PolicyError(
            "Cannot build bettable predictions; missing: " + ", ".join(missing)
        )

    # Every completed game that carried a price is bettable, whether or not it
    # settled to a winner.
    bettable = matchups.loc[
        matchups["home_score"].notna()
        & matchups["away_score"].notna()
        & matchups["spread_line"].notna()
    ].copy()
    bettable["is_push"] = (
        bettable["is_push"].fillna(False).astype(bool)
        if "is_push" in bettable.columns
        else False
    )

    settled = eligible_games(matchups).sort_values(
        ["gameday", "game_id"], kind="stable"
    )

    frames = []
    for split in season_walk_forward_splits(settled, first, final):
        train_frame = settled.loc[split.train_indices]
        pipeline = train_model(
            feature_matrix(train_frame),
            target_vector(train_frame),
            c=selection.c,
            class_weight=selection.class_weight,
        )

        season_games = bettable.loc[bettable["season"].eq(split.validation_season)]
        if season_games.empty:
            continue

        probability = pipeline.predict_proba(feature_matrix(season_games))[:, 1]
        # A push has no actual cover, so it is held as a null rather than
        # forced to 0 or 1; settle() decides it from is_push regardless.
        actual = pd.to_numeric(season_games["home_cover"], errors="coerce")

        frames.append(
            pd.DataFrame(
                {
                    "game_id": season_games["game_id"].to_numpy(),
                    "season": season_games["season"].to_numpy(),
                    "week": season_games["week"].to_numpy(),
                    "actual_home_cover": actual.to_numpy(),
                    "predicted_home_cover": (probability >= 0.5).astype(int),
                    "home_cover_probability": probability,
                    "is_push": season_games["is_push"].to_numpy(),
                }
            )
        )

    if not frames:
        raise PolicyError("No folds produced bettable predictions.")

    combined = pd.concat(frames, ignore_index=True)
    if combined["game_id"].duplicated().any():
        raise PolicyError("A game was predicted by more than one fold.")
    return combined.sort_values(["season", "week", "game_id"]).reset_index(drop=True)
