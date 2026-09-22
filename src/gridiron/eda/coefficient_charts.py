"""Coefficient charts for the fitted logistic regression.

A coefficient plot invites the reader to rank features by bar length, so these
charts carry the things that make that ranking unsafe: the collinearity flag on
each bar, and the fold-to-fold range where it is available. A bar drawn without
them is a claim the model cannot support.
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
from gridiron.modeling.interpretation import VIF_ELEVATED, VIF_SEVERE


class CoefficientChartError(ValueError):
    """Raised when a coefficient chart cannot be drawn."""


def _save(figure: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def coefficient_plot(table: pd.DataFrame, output_dir: Path) -> Path:
    """Horizontal bars, largest absolute coefficient at the top.

    Diverging by sign: a bar to the right raises the modelled probability of a
    home cover, one to the left lowers it. Features whose variance inflation
    factor is elevated are drawn in the second hue and marked, because their
    bar length is shared with the features they overlap.
    """
    if table.empty:
        raise CoefficientChartError("No coefficients to plot.")
    apply_theme()

    ordered = table.sort_values("abs_coefficient")
    positions = np.arange(len(ordered))

    has_vif = "vif" in ordered.columns and ordered["vif"].notna().any()
    colours = []
    for row in ordered.itertuples():
        vif = getattr(row, "vif", float("nan"))
        elevated = has_vif and pd.notna(vif) and vif >= VIF_ELEVATED
        colours.append(CATEGORICAL[1] if elevated else CATEGORICAL[0])

    figure, axis = plt.subplots(figsize=(10.5, 6.4))
    axis.barh(positions, ordered["coefficient"], color=colours, height=0.62)
    axis.axvline(0, color=TEXT_SECONDARY, linewidth=1.4)

    axis.set_yticks(positions)
    axis.set_yticklabels(ordered["feature"], fontsize=9.5)

    limit = float(ordered["abs_coefficient"].max()) * 1.45 or 1.0
    axis.set_xlim(-limit, limit)

    for index, row in enumerate(ordered.itertuples()):
        offset = 6 if row.coefficient >= 0 else -6
        alignment = "left" if row.coefficient >= 0 else "right"
        label = f"{row.odds_change_pct:+.1f}%"
        vif = getattr(row, "vif", float("nan"))
        if has_vif and pd.notna(vif) and vif >= VIF_SEVERE:
            label += "  (VIF high)"
        axis.annotate(
            label,
            (row.coefficient, index),
            textcoords="offset points",
            xytext=(offset, 0),
            va="center",
            ha=alignment,
            fontsize=8.5,
            color=TEXT_MUTED,
        )

    axis.set_title("Standardised coefficients: association with a home cover")
    axis.set_xlabel("Coefficient (log-odds per standard deviation)")
    axis.set_ylabel("Feature")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=CATEGORICAL[0]),
        plt.Rectangle((0, 0), 1, 1, color=CATEGORICAL[1]),
    ]
    axis.legend(
        handles,
        ["independent enough to read", f"VIF >= {VIF_ELEVATED:.0f}: shared"],
        loc="lower right",
        fontsize=8.5,
    )

    intercept = table.attrs.get("intercept")
    note = (
        "Labels give the change in odds per one standard deviation. These are "
        "associations in the fitted model, not causal effects."
    )
    if intercept is not None:
        note += f" Intercept {intercept:+.4f}."
    figure.text(0.01, 0.015, note, fontsize=8.5, color=TEXT_MUTED)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(figure, output_dir, "20_model_coefficients")


def coefficient_stability_plot(stability: pd.DataFrame, output_dir: Path) -> Path:
    """Each coefficient's range across the walk-forward folds.

    A bar that crosses zero is a feature whose direction the model could not
    agree on from one validation season to the next.
    """
    if stability.empty:
        raise CoefficientChartError("No stability data to plot.")
    apply_theme()

    ordered = stability.sort_values("std_coefficient")
    positions = np.arange(len(ordered))
    colours = [
        CATEGORICAL[1] if flips else CATEGORICAL[0] for flips in ordered["sign_changes"]
    ]

    figure, axis = plt.subplots(figsize=(10.5, 6.4))
    axis.hlines(
        positions,
        ordered["min_coefficient"],
        ordered["max_coefficient"],
        color=colours,
        linewidth=3.0,
    )
    axis.scatter(
        ordered["mean_coefficient"],
        positions,
        color=colours,
        s=42,
        zorder=3,
        label="mean across folds",
    )
    axis.axvline(0, color=TEXT_SECONDARY, linewidth=1.4, label="zero")

    axis.set_yticks(positions)
    axis.set_yticklabels(ordered["feature"], fontsize=9.5)
    axis.set_title("Coefficient range across the walk-forward folds")
    axis.set_xlabel("Coefficient (log-odds per standard deviation)")
    axis.set_ylabel("Feature")
    axis.legend(loc="lower right", fontsize=8.5)

    flipped = int(ordered["sign_changes"].sum())
    figure.text(
        0.01,
        0.015,
        f"{flipped} of {len(ordered)} features change sign between folds "
        "(orange). A direction that will not hold still is not a finding.",
        fontsize=8.5,
        color=TEXT_MUTED,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(figure, output_dir, "21_coefficient_stability")


def build_coefficient_charts(
    table: pd.DataFrame,
    stability: pd.DataFrame | None,
    output_dir: Path,
) -> list[Path]:
    """Draw the coefficient plot, and the stability plot when available."""
    paths = [coefficient_plot(table, output_dir)]
    if stability is not None and not stability.empty:
        paths.append(coefficient_stability_plot(stability, output_dir))
    return paths
