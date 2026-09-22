"""Calibration charts: reliability, prediction spread, and rate by bucket.

These are model diagnostics rather than feature exploration, so they live apart
from the exploratory charts, but they draw on the same validated theme.

Every one of them carries its sample counts. A reliability point built from
three games looks identical to one built from a thousand unless the chart says
otherwise, and on this data the difference between those two cases is the whole
story.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gridiron.eda.theme import (
    CATEGORICAL,
    TEXT_MUTED,
    TEXT_SECONDARY,
    apply_theme,
)
from gridiron.modeling.calibration import (
    MIN_BUCKET_SAMPLES,
    PROBABILITY_COLUMN,
    reliability_table,
)


class CalibrationChartError(ValueError):
    """Raised when a calibration chart cannot be drawn."""


def _save(figure: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def reliability_diagram(predictions: pd.DataFrame, output_dir: Path) -> Path:
    """Observed cover rate against predicted probability, with counts.

    The diagonal is perfect calibration. Points are sized by the games behind
    them and annotated with the count, and the sparse ones are drawn hollow so
    a three-game bucket cannot be mistaken for evidence.
    """
    apply_theme()
    table = reliability_table(predictions)
    populated = table.loc[table["games"] > 0].copy()
    if populated.empty:
        raise CalibrationChartError("No populated buckets to plot.")

    figure, axis = plt.subplots(figsize=(8.5, 7))

    axis.plot(
        [0.3, 0.75],
        [0.3, 0.75],
        color=TEXT_SECONDARY,
        linewidth=1.4,
        label="perfect calibration",
        zorder=1,
    )

    solid = populated.loc[~populated["sparse"]]
    thin = populated.loc[populated["sparse"]]

    axis.errorbar(
        solid["mean_predicted"],
        solid["observed_rate"],
        yerr=solid["standard_error"],
        fmt="o",
        markersize=9,
        color=CATEGORICAL[0],
        ecolor=CATEGORICAL[0],
        elinewidth=1.4,
        capsize=4,
        label=f"bucket (n >= {MIN_BUCKET_SAMPLES})",
        zorder=3,
    )
    if not thin.empty:
        axis.errorbar(
            thin["mean_predicted"],
            thin["observed_rate"],
            yerr=thin["standard_error"],
            fmt="o",
            markersize=9,
            markerfacecolor="none",
            color=CATEGORICAL[1],
            ecolor=CATEGORICAL[1],
            elinewidth=1.2,
            capsize=4,
            label=f"sparse bucket (n < {MIN_BUCKET_SAMPLES})",
            zorder=2,
        )

    for row in populated.itertuples():
        # Anchor the count above the top of its error bar, not above the
        # marker: on the thin buckets the bar is far taller than any fixed
        # offset and the label would otherwise sit on top of it.
        top = row.observed_rate + (
            row.standard_error if pd.notna(row.standard_error) else 0.0
        )
        axis.annotate(
            f"n={int(row.games)}",
            (row.mean_predicted, min(top, 0.97)),
            textcoords="offset points",
            xytext=(0, 7),
            ha="center",
            fontsize=8.5,
            color=TEXT_MUTED,
        )

    axis.set_title("Reliability: observed cover rate against predicted probability")
    axis.set_xlabel("Mean predicted probability of a home cover")
    axis.set_ylabel("Observed home cover rate")
    axis.set_xlim(0.3, 0.75)
    axis.set_ylim(0.0, 1.0)
    axis.legend(loc="upper left")
    figure.text(
        0.01,
        0.015,
        "Error bars are one standard error on the observed rate. Out-of-sample "
        "walk-forward predictions only.",
        fontsize=8.5,
        color=TEXT_MUTED,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    return _save(figure, output_dir, "11_reliability_diagram")


def prediction_histogram(predictions: pd.DataFrame, output_dir: Path) -> Path:
    """Where the predicted probabilities actually fall.

    The bucket edges are drawn on, so the reliability diagram's thin buckets
    can be seen to be thin rather than merely asserted to be.
    """
    apply_theme()
    if PROBABILITY_COLUMN not in predictions.columns:
        raise CalibrationChartError(f"Missing '{PROBABILITY_COLUMN}'.")

    probability = pd.to_numeric(
        predictions[PROBABILITY_COLUMN], errors="coerce"
    ).dropna()

    figure, axis = plt.subplots(figsize=(10, 5.4))
    axis.hist(probability, bins=50, color=CATEGORICAL[0], edgecolor="none")

    for edge in (0.40, 0.45, 0.50, 0.55, 0.60, 0.65):
        axis.axvline(edge, color=TEXT_SECONDARY, linewidth=0.8, alpha=0.5)

    axis.axvline(
        0.5,
        color=CATEGORICAL[1],
        linewidth=1.6,
        label="0.50 decision threshold",
    )

    axis.set_title("Distribution of predicted home-cover probabilities")
    axis.set_xlabel("Predicted probability of a home cover")
    axis.set_ylabel("Games")
    axis.legend(loc="upper right")
    figure.text(
        0.01,
        0.015,
        f"{len(probability)} out-of-sample games. Thin vertical rules mark the "
        f"reporting buckets; range {probability.min():.3f} to "
        f"{probability.max():.3f}.",
        fontsize=8.5,
        color=TEXT_MUTED,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(figure, output_dir, "12_prediction_histogram")


def cover_rate_by_bucket(predictions: pd.DataFrame, output_dir: Path) -> Path:
    """Observed cover rate per bucket, beside what the model predicted."""
    apply_theme()
    table = reliability_table(predictions)
    populated = table.loc[table["games"] > 0].copy()
    if populated.empty:
        raise CalibrationChartError("No populated buckets to plot.")

    figure, axis = plt.subplots(figsize=(11, 5.6))
    positions = np.arange(len(populated))
    width = 0.38

    axis.bar(
        positions - width / 2,
        populated["mean_predicted"],
        width=width * 0.94,
        color=CATEGORICAL[0],
        label="predicted",
    )
    axis.bar(
        positions + width / 2,
        populated["observed_rate"],
        width=width * 0.94,
        color=CATEGORICAL[1],
        label="observed",
    )
    axis.errorbar(
        positions + width / 2,
        populated["observed_rate"],
        yerr=populated["standard_error"],
        fmt="none",
        ecolor=TEXT_SECONDARY,
        elinewidth=1.2,
        capsize=4,
    )

    for index, row in enumerate(populated.itertuples()):
        if not row.sparse:
            continue
        # Above the group rather than inside it: at the baseline the label is
        # overdrawn by the bars themselves and reads as clipped text.
        error = row.standard_error if pd.notna(row.standard_error) else 0.0
        top = max(row.mean_predicted, row.observed_rate + error)
        axis.annotate(
            "sparse",
            (index, min(top + 0.03, 0.95)),
            ha="center",
            fontsize=8,
            color=CATEGORICAL[1],
            fontweight="semibold",
        )

    axis.set_xticks(positions)
    axis.set_xticklabels(
        [f"{row.bucket}\nn={int(row.games)}" for row in populated.itertuples()]
    )
    axis.axhline(0.5, color=TEXT_SECONDARY, linewidth=1.0, alpha=0.6)

    axis.set_title("Predicted against observed cover rate, by probability bucket")
    axis.set_xlabel("Predicted probability bucket")
    axis.set_ylabel("Home cover rate")
    axis.set_ylim(0, 1.0)
    axis.legend(loc="upper left", ncols=2)
    figure.text(
        0.01,
        0.015,
        f"Buckets with fewer than {MIN_BUCKET_SAMPLES} games are marked sparse; "
        "their observed rates carry error bars wider than any gap worth acting "
        "on.",
        fontsize=8.5,
        color=TEXT_MUTED,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(figure, output_dir, "13_cover_rate_by_bucket")


def build_calibration_charts(predictions: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Draw all three calibration charts."""
    return [
        reliability_diagram(predictions, output_dir),
        prediction_histogram(predictions, output_dir),
        cover_rate_by_bucket(predictions, output_dir),
    ]
