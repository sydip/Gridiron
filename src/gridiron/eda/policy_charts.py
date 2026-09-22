"""Charts for the betting-policy backtest.

Several of these show one quantity across seven policies. Rather than invent
four more categorical hues, the per-policy views are drawn as small multiples:
the same axes repeated, which compares more honestly than seven lines fighting
over one panel and keeps colour doing only the job it is good at.

Every chart that shows a rate also shows the bets behind it. At the strictest
threshold a policy places seven bets, and a chart that renders that at the same
visual weight as eighteen hundred is not reporting, it is advertising.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridiron.eda.theme import (
    CATEGORICAL,
    DIVERGING_CMAP,
    TEXT_MUTED,
    TEXT_SECONDARY,
    apply_theme,
)
from gridiron.modeling.policies import (
    DEFAULT_ODDS,
    MIN_BETS_FOR_A_CLAIM,
    POLICY_THRESHOLDS,
    cumulative_units,
    season_units,
    win_rate_by_confidence,
)


class PolicyChartError(ValueError):
    """Raised when a policy chart cannot be drawn."""


def _save(figure: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def _policy_label(threshold: float) -> str:
    return "bet every prediction" if threshold == 0 else f">= {threshold}pp"


def _grid(count: int) -> tuple[int, int]:
    columns = 4 if count > 3 else count
    return int(np.ceil(count / columns)), columns


def cumulative_units_chart(
    predictions: pd.DataFrame,
    output_dir: Path,
    thresholds: list[float] | None = None,
) -> Path:
    """Running bankroll for each policy, in the order the games were played."""
    apply_theme()
    thresholds = thresholds or POLICY_THRESHOLDS
    rows, columns = _grid(len(thresholds))

    figure, axes = plt.subplots(
        rows, columns, figsize=(4.0 * columns, 3.1 * rows), sharey=True
    )
    flat = np.atleast_1d(axes).ravel()

    for axis, threshold in zip(flat, thresholds, strict=False):
        curve = cumulative_units(predictions, threshold)
        axis.axhline(0, color=TEXT_SECONDARY, linewidth=1.0)
        if curve.empty:
            axis.text(0.5, 0.5, "no bets", ha="center", transform=axis.transAxes)
        else:
            final = float(curve["cumulative_units"].iloc[-1])
            axis.plot(
                np.arange(len(curve)),
                curve["cumulative_units"],
                color=CATEGORICAL[0] if final >= 0 else CATEGORICAL[1],
                linewidth=1.8,
            )
            axis.annotate(
                f"{final:+.1f}u over {len(curve)} bets",
                (0.04, 0.06),
                xycoords="axes fraction",
                fontsize=8.5,
                color=TEXT_MUTED,
            )
        axis.set_title(_policy_label(threshold), fontsize=10.5)
        axis.set_xlabel("Bet number, in date order")
        axis.set_ylabel("Cumulative units")

    for axis in flat[len(thresholds) :]:
        axis.set_visible(False)

    figure.suptitle(
        f"Cumulative units by policy ({DEFAULT_ODDS.name})",
        fontsize=14,
        fontweight="semibold",
    )
    figure.tight_layout()
    return _save(figure, output_dir, "14_cumulative_units")


def roi_by_threshold_chart(results: pd.DataFrame, output_dir: Path) -> Path:
    """Return per resolved bet at each threshold, with the bet counts on it."""
    apply_theme()
    if results.empty:
        raise PolicyChartError("No results to plot.")

    figure, axis = plt.subplots(figsize=(10.5, 5.6))
    colours = [
        CATEGORICAL[0] if value >= 0 else CATEGORICAL[1] for value in results["roi"]
    ]
    labels = [f"{row.policy}\nn={int(row.bets)}" for row in results.itertuples()]
    axis.bar(labels, results["roi"], color=colours, width=0.64)
    axis.axhline(0, color=TEXT_SECONDARY, linewidth=1.4, label="break even")

    for index, row in enumerate(results.itertuples()):
        if row.sample_warning:
            axis.annotate(
                "small sample",
                (index, row.roi),
                textcoords="offset points",
                xytext=(0, 10 if row.roi >= 0 else -16),
                ha="center",
                fontsize=8,
                color=CATEGORICAL[1],
                fontweight="semibold",
            )

    axis.set_title("Return on investment by confidence threshold")
    axis.set_xlabel("Policy")
    axis.set_ylabel("ROI per resolved bet")
    axis.legend(loc="upper left")
    figure.text(
        0.01,
        0.015,
        f"{DEFAULT_ODDS.name}. Policies marked 'small sample' have fewer than "
        f"{MIN_BETS_FOR_A_CLAIM} bets; their ROI is not a measurement.",
        fontsize=8.5,
        color=TEXT_MUTED,
    )
    plt.setp(axis.get_xticklabels(), rotation=15, ha="right")
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(figure, output_dir, "15_roi_by_threshold")


def bets_by_threshold_chart(results: pd.DataFrame, output_dir: Path) -> Path:
    """How many bets survive each threshold.

    Drawn on a log scale: the counts run from 1,871 down to 7, and on a linear
    axis the strict policies would be invisible -- which is the opposite of the
    point being made.
    """
    apply_theme()
    if results.empty:
        raise PolicyChartError("No results to plot.")

    figure, axis = plt.subplots(figsize=(10.5, 5.6))
    axis.bar(
        results["policy"],
        results["bets"].clip(lower=0.9),
        color=CATEGORICAL[0],
        width=0.64,
    )
    axis.axhline(
        MIN_BETS_FOR_A_CLAIM,
        color=TEXT_SECONDARY,
        linewidth=1.4,
        label=f"{MIN_BETS_FOR_A_CLAIM} bets: minimum for a claim",
    )
    axis.set_yscale("log")

    for index, row in enumerate(results.itertuples()):
        axis.annotate(
            f"{int(row.bets)}",
            (index, max(row.bets, 1)),
            textcoords="offset points",
            xytext=(0, 5),
            ha="center",
            fontsize=9,
            color=TEXT_MUTED,
        )

    axis.set_title("Bets available at each confidence threshold")
    axis.set_xlabel("Policy")
    axis.set_ylabel("Bets (log scale)")
    axis.legend(loc="upper right")
    plt.setp(axis.get_xticklabels(), rotation=15, ha="right")
    figure.tight_layout()
    return _save(figure, output_dir, "16_bets_by_threshold")


def win_rate_by_confidence_chart(predictions: pd.DataFrame, output_dir: Path) -> Path:
    """Observed win rate by claimed edge, with intervals and counts.

    If confidence filtering worked, this would slope upward.
    """
    apply_theme()
    table = win_rate_by_confidence(predictions)
    if table.empty:
        raise PolicyChartError("No confidence buckets to plot.")

    figure, axis = plt.subplots(figsize=(10.5, 5.8))
    positions = np.arange(len(table))
    errors = np.vstack(
        [
            table["win_rate"] - table["ci_low"],
            table["ci_high"] - table["win_rate"],
        ]
    )
    colours = [
        CATEGORICAL[1] if sparse else CATEGORICAL[0] for sparse in table["sparse"]
    ]

    axis.bar(positions, table["win_rate"], color=colours, width=0.62)
    axis.errorbar(
        positions,
        table["win_rate"],
        yerr=errors,
        fmt="none",
        ecolor=TEXT_SECONDARY,
        elinewidth=1.3,
        capsize=5,
    )
    axis.axhline(
        DEFAULT_ODDS.break_even_rate,
        color=TEXT_SECONDARY,
        linewidth=1.5,
        label=f"break even ({DEFAULT_ODDS.break_even_rate:.4f})",
    )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"{row.edge_bucket}\nn={int(row.bets)}" for row in table.itertuples()],
        fontsize=8.5,
    )
    axis.set_title("Win rate by the edge the model claimed")
    axis.set_xlabel("Claimed edge over a coin flip, percentage points")
    axis.set_ylabel("Win rate on resolved bets")
    axis.set_ylim(0, 1.0)
    axis.legend(loc="upper right")
    figure.text(
        0.01,
        0.015,
        "Bars are 95% Wilson intervals. Orange marks buckets with fewer than "
        f"{MIN_BETS_FOR_A_CLAIM} bets.",
        fontsize=8.5,
        color=TEXT_MUTED,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(figure, output_dir, "17_win_rate_by_confidence")


def seasonal_units_chart(
    predictions: pd.DataFrame,
    output_dir: Path,
    thresholds: list[float] | None = None,
) -> Path:
    """Units won by season and policy.

    Diverging and centred on zero, so a losing season reads as the opposite of
    a winning one rather than as a smaller version of it.
    """
    apply_theme()
    thresholds = thresholds or POLICY_THRESHOLDS

    frames = {}
    for threshold in thresholds:
        seasons = season_units(predictions, threshold)
        if not seasons.empty:
            frames[_policy_label(threshold)] = seasons.set_index("season")["units_won"]
    if not frames:
        raise PolicyChartError("No seasonal results to plot.")

    grid = pd.DataFrame(frames).T
    limit = float(np.nanmax(np.abs(grid.to_numpy()))) or 1.0

    figure, axis = plt.subplots(figsize=(11, 5.4))
    mesh = axis.imshow(
        grid.to_numpy(dtype=float),
        cmap=DIVERGING_CMAP,
        vmin=-limit,
        vmax=limit,
        aspect="auto",
    )
    for row_index in range(grid.shape[0]):
        for column_index in range(grid.shape[1]):
            value = grid.to_numpy(dtype=float)[row_index, column_index]
            if not np.isnan(value):
                axis.text(
                    column_index,
                    row_index,
                    f"{value:+.1f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                )

    axis.set_xticks(range(grid.shape[1]))
    axis.set_xticklabels([str(int(season)) for season in grid.columns])
    axis.set_yticks(range(grid.shape[0]))
    axis.set_yticklabels(grid.index)
    axis.set_title("Units won by season and policy")
    axis.set_xlabel("Season")
    axis.set_ylabel("Policy")
    axis.grid(False)
    figure.colorbar(mesh, ax=axis, label="Units won", shrink=0.8)
    figure.tight_layout()
    return _save(figure, output_dir, "18_seasonal_units")


def drawdown_chart(
    predictions: pd.DataFrame,
    output_dir: Path,
    thresholds: list[float] | None = None,
) -> Path:
    """How far below its own high-water mark each policy fell, over time."""
    apply_theme()
    thresholds = thresholds or POLICY_THRESHOLDS
    rows, columns = _grid(len(thresholds))

    figure, axes = plt.subplots(
        rows, columns, figsize=(4.0 * columns, 3.1 * rows), sharey=True
    )
    flat = np.atleast_1d(axes).ravel()

    for axis, threshold in zip(flat, thresholds, strict=False):
        curve = cumulative_units(predictions, threshold)
        if curve.empty:
            axis.text(0.5, 0.5, "no bets", ha="center", transform=axis.transAxes)
        else:
            values = curve["cumulative_units"].to_numpy(dtype=float)
            peak = np.maximum.accumulate(np.concatenate([[0.0], values]))[1:]
            drawdown = values - peak
            axis.fill_between(
                np.arange(len(drawdown)),
                drawdown,
                0,
                color=CATEGORICAL[1],
                alpha=0.55,
                linewidth=0,
            )
            axis.annotate(
                f"worst {abs(float(drawdown.min())):.1f}u",
                (0.04, 0.08),
                xycoords="axes fraction",
                fontsize=8.5,
                color=TEXT_MUTED,
            )
        axis.set_title(_policy_label(threshold), fontsize=10.5)
        axis.set_xlabel("Bet number, in date order")
        axis.set_ylabel("Units below high-water mark")

    for axis in flat[len(thresholds) :]:
        axis.set_visible(False)

    figure.suptitle("Drawdown by policy", fontsize=14, fontweight="semibold")
    figure.tight_layout()
    return _save(figure, output_dir, "19_drawdown")


def build_policy_charts(
    predictions: pd.DataFrame,
    results: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """Draw all six policy charts."""
    return [
        cumulative_units_chart(predictions, output_dir),
        roi_by_threshold_chart(results, output_dir),
        bets_by_threshold_chart(results, output_dir),
        win_rate_by_confidence_chart(predictions, output_dir),
        seasonal_units_chart(predictions, output_dir),
        drawdown_chart(predictions, output_dir),
    ]
