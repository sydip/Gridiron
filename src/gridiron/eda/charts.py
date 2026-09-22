"""The ten exploratory charts, each saved as a titled, fully labelled figure.

Every function takes a frame and an output directory, writes one PNG, and
returns its path. Colour follows the job the chart is doing -- diverging where a
value has a sign, sequential where it has only a magnitude, categorical only for
identity -- and the theme keeps grid and axis lines recessive.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gridiron.eda.correlation import correlation_matrix, numeric_fields
from gridiron.eda.theme import (
    CATEGORICAL,
    DIVERGING_CMAP,
    SEQUENTIAL_CMAP,
    TEXT_MUTED,
    TEXT_SECONDARY,
    apply_theme,
)

# A reference line at the break-even cover rate. Against the spread, 50% is the
# only number that matters; without it a bar chart of rates is unreadable.
BREAK_EVEN = 0.5

DISTRIBUTION_FEATURES = (
    "spread_line",
    "off_epa_diff_last_5",
    "def_epa_strength_diff_last_5",
    "pace_diff_last_5",
    "rest_diff",
    "point_margin_diff_last_5",
    "win_pct_diff_last_5",
)


class ChartError(ValueError):
    """Raised when a frame cannot produce a requested chart."""


def _require(frame: pd.DataFrame, columns: set[str], chart: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ChartError(
            f"Cannot draw {chart}; missing column(s): " + ", ".join(missing)
        )


def _save(figure: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def _bin_tick_labels(index, counts: pd.Series) -> list[str]:
    """Bin label with its sample size underneath.

    The count belongs on the axis rather than above the bar: floated above, it
    collides with the break-even rule wherever a rate sits near 50%, which is
    exactly where these rates sit.
    """
    return [
        f"{label}\nn={int(count)}" for label, count in zip(index, counts, strict=True)
    ]


def correlation_heatmap(
    matchups: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
) -> Path:
    """Correlation between the numeric model fields.

    Diverging, centred on zero: a correlation has a sign, so no relationship
    must read as the neutral midpoint rather than as a colour.
    """
    apply_theme()
    matrix = correlation_matrix(matchups, feature_columns)

    figure, axis = plt.subplots(figsize=(8.5, 7.6))
    # k=0 masks the diagonal as well as the upper triangle: a variable's
    # correlation with itself is always +1 and only draws the eye.
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)
    sns.heatmap(
        matrix,
        mask=mask,
        cmap=DIVERGING_CMAP,
        vmin=-1,
        vmax=1,
        center=0,
        annot=True,
        fmt="+.2f",
        annot_kws={"size": 8},
        linewidths=2,
        linecolor=plt.rcParams["figure.facecolor"],
        cbar_kws={"label": "Pearson correlation", "shrink": 0.8},
        ax=axis,
    )
    axis.set_title("Correlation between numeric model features")
    axis.set_xlabel("Model feature")
    axis.set_ylabel("Model feature")
    axis.grid(False)
    plt.setp(axis.get_xticklabels(), rotation=40, ha="right")

    blank = int(matrix.isna().to_numpy()[~mask].sum())
    note = (
        "Diagonal and upper triangle omitted. "
        "Blank cells are undefined, not zero: week_1_flag is constant "
        "wherever a rolling feature is present, so the pair has no "
        "overlapping observations."
        if blank
        else "Diagonal and upper triangle omitted."
    )
    figure.text(0.01, 0.015, note, fontsize=8.5, color=TEXT_MUTED, wrap=True)
    figure.tight_layout(rect=(0, 0.045, 1, 1))
    return _save(figure, output_dir, "01_correlation_heatmap")


def feature_distributions(
    matchups: pd.DataFrame,
    output_dir: Path,
    features: tuple[str, ...] = DISTRIBUTION_FEATURES,
) -> Path:
    """Small multiples of each feature's distribution.

    One hue throughout: these are seven views of different quantities, not seven
    categories to tell apart, so colour carries no identity here.
    """
    apply_theme()
    available = [column for column in features if column in matchups.columns]
    if not available:
        raise ChartError("Cannot draw feature distributions; no features present.")

    columns = 3
    rows = int(np.ceil(len(available) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(13, 3.3 * rows))
    flat = np.atleast_1d(axes).ravel()

    for axis, column in zip(flat, available, strict=False):
        values = pd.to_numeric(matchups[column], errors="coerce").dropna()
        sns.histplot(values, bins=40, color=CATEGORICAL[0], ax=axis, edgecolor="none")
        axis.axvline(values.mean(), color=CATEGORICAL[1], linewidth=1.6, label="mean")
        axis.set_title(column, fontsize=11)
        axis.set_xlabel(column.replace("_", " "))
        axis.set_ylabel("Games")
        axis.legend(loc="upper right")

    for axis in flat[len(available) :]:
        axis.set_visible(False)

    figure.suptitle(
        "Distribution of each model feature", fontsize=14, fontweight="semibold"
    )
    figure.tight_layout()
    return _save(figure, output_dir, "02_feature_distributions")


def _team_season_heatmap(
    team_games: pd.DataFrame,
    value_column: str,
    output_dir: Path,
    *,
    name: str,
    title: str,
    legend: str,
) -> Path:
    _require(team_games, {"team", "season", value_column}, title)
    apply_theme()

    grid = team_games.pivot_table(
        index="team", columns="season", values=value_column, aggfunc="mean"
    ).sort_index()
    limit = float(np.nanmax(np.abs(grid.to_numpy())))

    figure, axis = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        grid,
        cmap=DIVERGING_CMAP,
        center=0,
        vmin=-limit,
        vmax=limit,
        annot=True,
        fmt=".02f",
        annot_kws={"size": 7},
        linewidths=2,
        linecolor=plt.rcParams["figure.facecolor"],
        cbar_kws={"label": legend, "shrink": 0.7},
        ax=axis,
    )
    axis.set_title(title)
    axis.set_xlabel("Season")
    axis.set_ylabel("Team")
    axis.grid(False)
    figure.tight_layout()
    return _save(figure, output_dir, name)


def offensive_epa_by_team_season(team_games: pd.DataFrame, output_dir: Path) -> Path:
    """Mean offensive EPA per play, by team and season. Higher is better."""
    return _team_season_heatmap(
        team_games,
        "offensive_epa_per_play",
        output_dir,
        name="03_offensive_epa_by_team_season",
        title="Offensive EPA per play by team and season (higher is better)",
        legend="Offensive EPA per play",
    )


def defensive_epa_by_team_season(team_games: pd.DataFrame, output_dir: Path) -> Path:
    """Mean defensive EPA strength, by team and season. Higher is better.

    This is the sign-adjusted view, so it points the same way as the offensive
    chart: a strong defence is a high number in both.
    """
    return _team_season_heatmap(
        team_games,
        "defensive_epa_strength_per_play",
        output_dir,
        name="04_defensive_epa_by_team_season",
        title="Defensive EPA strength by team and season (higher is better)",
        legend="Defensive EPA strength per play",
    )


def pace_distribution(team_games: pd.DataFrame, output_dir: Path) -> Path:
    """Team-game pace, as plays run and as seconds between snaps."""
    _require(team_games, {"pace_plays_per_game"}, "pace distribution")
    apply_theme()

    has_seconds = "seconds_per_play" in team_games.columns
    figure, axes = plt.subplots(1, 2 if has_seconds else 1, figsize=(12, 4.6))
    axes = np.atleast_1d(axes)

    plays = pd.to_numeric(team_games["pace_plays_per_game"], errors="coerce").dropna()
    sns.histplot(plays, bins=40, color=CATEGORICAL[0], ax=axes[0], edgecolor="none")
    axes[0].axvline(plays.median(), color=CATEGORICAL[1], linewidth=1.6, label="median")
    axes[0].set_title("Offensive plays per team-game", fontsize=11)
    axes[0].set_xlabel("Eligible offensive plays")
    axes[0].set_ylabel("Team-games")
    axes[0].legend(loc="upper right")

    if has_seconds:
        seconds = pd.to_numeric(
            team_games["seconds_per_play"], errors="coerce"
        ).dropna()
        sns.histplot(
            seconds, bins=40, color=CATEGORICAL[0], ax=axes[1], edgecolor="none"
        )
        axes[1].axvline(
            seconds.median(), color=CATEGORICAL[1], linewidth=1.6, label="median"
        )
        axes[1].set_title("Seconds between snaps", fontsize=11)
        axes[1].set_xlabel("Seconds per play")
        axes[1].set_ylabel("Team-games")
        axes[1].legend(loc="upper right")

    figure.suptitle("Pace distribution", fontsize=14, fontweight="semibold")
    figure.tight_layout()
    return _save(figure, output_dir, "05_pace_distribution")


def _rate_by_bin(
    matchups: pd.DataFrame,
    bins: pd.Series,
    output_dir: Path,
    *,
    name: str,
    title: str,
    xlabel: str,
    note: str,
) -> Path:
    apply_theme()
    labelled = matchups.loc[matchups["home_cover"].notna()].copy()
    labelled["_bin"] = bins.loc[labelled.index]

    grouped = labelled.groupby("_bin", observed=True)["home_cover"].agg(
        ["size", "mean"]
    )
    grouped = grouped.loc[grouped["size"] > 0]

    figure, axis = plt.subplots(figsize=(10, 5.4))
    labels = _bin_tick_labels([str(value) for value in grouped.index], grouped["size"])
    axis.bar(labels, grouped["mean"], color=CATEGORICAL[0], width=0.68)
    axis.axhline(
        BREAK_EVEN,
        color=TEXT_SECONDARY,
        linewidth=1.4,
        label="break-even (50%)",
    )

    axis.set_title(title)
    axis.set_xlabel(xlabel)
    axis.set_ylabel("Home cover rate")
    axis.set_ylim(0, max(0.75, float(grouped["mean"].max()) + 0.12))
    axis.legend(loc="upper right")
    figure.text(0.01, 0.015, note, fontsize=8.5, color=TEXT_MUTED)
    plt.setp(axis.get_xticklabels(), rotation=0)
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    return _save(figure, output_dir, name)


def rest_advantage_vs_cover(matchups: pd.DataFrame, output_dir: Path) -> Path:
    """Home cover rate by how much more rest the home side had."""
    _require(matchups, {"rest_diff", "home_cover"}, "rest advantage vs cover")
    bins = pd.cut(
        pd.to_numeric(matchups["rest_diff"], errors="coerce"),
        [-np.inf, -7, -3, -0.5, 0.5, 3, 7, np.inf],
        labels=["<= -7", "-6 to -3", "-2 to -1", "0", "+1 to +2", "+3 to +6", ">= +7"],
    )
    return _rate_by_bin(
        matchups,
        bins,
        output_dir,
        name="06_rest_advantage_vs_cover",
        title="Home cover rate by rest advantage (home rest minus away rest)",
        xlabel="Rest advantage, days",
        note="n = labelled games in each bin; pushes are excluded.",
    )


def spread_bins_vs_cover(matchups: pd.DataFrame, output_dir: Path) -> Path:
    """Home cover rate by the size of the line.

    A positive spread means the home team is favoured, following the nflverse
    convention used throughout this project.
    """
    _require(matchups, {"spread_line", "home_cover"}, "spread bins vs cover")
    bins = pd.cut(
        pd.to_numeric(matchups["spread_line"], errors="coerce"),
        [-np.inf, -10, -6.5, -3, 0, 3, 6.5, 10, np.inf],
    )
    return _rate_by_bin(
        matchups,
        bins,
        output_dir,
        name="07_spread_bins_vs_cover",
        title="Home cover rate by spread bin (positive = home favoured)",
        xlabel="Spread line bin, points",
        note="n = labelled games in each bin; pushes are excluded.",
    )


def epa_differential_vs_cover(matchups: pd.DataFrame, output_dir: Path) -> Path:
    """Home cover rate by the rolling offensive EPA differential."""
    _require(
        matchups, {"off_epa_diff_last_5", "home_cover"}, "EPA differential vs cover"
    )
    values = pd.to_numeric(matchups["off_epa_diff_last_5"], errors="coerce")
    bins = pd.qcut(values, 5, duplicates="drop")
    return _rate_by_bin(
        matchups,
        bins,
        output_dir,
        name="08_epa_differential_vs_cover",
        title="Home cover rate by rolling offensive EPA differential (last 5)",
        xlabel="off_epa_diff_last_5 quintile",
        note="n = labelled games in each quintile; pushes are excluded.",
    )


def missing_value_heatmap(
    matchups: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
) -> Path:
    """Percentage of each feature missing, by season and week.

    Sequential, one hue: missingness is a magnitude with no sign, so a
    two-ended scale would imply a polarity that does not exist.
    """
    _require(matchups, {"season", "week"}, "missing-value heatmap")
    apply_theme()
    columns = [column for column in feature_columns if column in matchups.columns]

    by_season = matchups.groupby("season")[columns].apply(
        lambda frame: frame.isna().mean() * 100
    )
    early = matchups.loc[matchups["week"].le(8)]
    by_week = early.groupby("week")[columns].apply(
        lambda frame: frame.isna().mean() * 100
    )

    figure, axes = plt.subplots(
        2, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [1, 1]}
    )
    for axis, grid, index_label, title in (
        (axes[0], by_season, "Season", "Missing values by season"),
        (axes[1], by_week, "Week", "Missing values by week (weeks 1-8)"),
    ):
        sns.heatmap(
            grid,
            cmap=SEQUENTIAL_CMAP,
            vmin=0,
            vmax=100,
            annot=True,
            fmt=".0f",
            annot_kws={"size": 7},
            linewidths=2,
            linecolor=plt.rcParams["figure.facecolor"],
            cbar_kws={"label": "Missing, %", "shrink": 0.8},
            ax=axis,
        )
        axis.set_title(title, fontsize=11)
        axis.set_xlabel("Model feature")
        axis.set_ylabel(index_label)
        axis.grid(False)
        plt.setp(axis.get_xticklabels(), rotation=35, ha="right")

    figure.suptitle(
        "Missing values across the feature matrix", fontsize=14, fontweight="semibold"
    )
    figure.tight_layout()
    return _save(figure, output_dir, "09_missing_value_heatmap")


def ats_class_balance(matchups: pd.DataFrame, output_dir: Path) -> Path:
    """Home cover, away cover, and push counts by season.

    Two categorical slots carry the covered/not-covered identity; pushes are
    drawn in the third and directly labelled, since that slot sits below the
    contrast floor against the surface.
    """
    _require(matchups, {"season", "home_cover"}, "ATS class balance")
    apply_theme()

    frame = matchups.copy()
    pushes = (
        frame["is_push"].fillna(False).astype(bool)
        if "is_push" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    summary = pd.DataFrame(
        {
            "Home cover": frame.loc[frame["home_cover"].eq(1)].groupby("season").size(),
            "Away cover": frame.loc[frame["home_cover"].eq(0)].groupby("season").size(),
            "Push": frame.loc[pushes].groupby("season").size(),
        }
    ).fillna(0)
    summary = summary.loc[summary.sum(axis=1) > 0]

    figure, axes = plt.subplots(1, 2, figsize=(13.5, 5))

    positions = np.arange(len(summary))
    width = 0.27
    for offset, (label, colour) in enumerate(
        zip(summary.columns, CATEGORICAL, strict=True)
    ):
        axes[0].bar(
            positions + (offset - 1) * width,
            summary[label],
            width=width * 0.92,
            label=label,
            color=colour,
        )
    axes[0].set_xticks(positions)
    axes[0].set_xticklabels([str(int(value)) for value in summary.index], rotation=0)
    axes[0].set_title("Games by outcome", fontsize=11)
    axes[0].set_xlabel("Season")
    axes[0].set_ylabel("Games")
    # Headroom first, so the legend sits above the bars instead of over them.
    axes[0].set_ylim(0, float(summary.to_numpy().max()) * 1.28)
    axes[0].legend(loc="upper left", ncols=3)

    rate = summary["Home cover"] / (summary["Home cover"] + summary["Away cover"])
    axes[1].plot(
        [str(int(value)) for value in summary.index],
        rate,
        marker="o",
        color=CATEGORICAL[0],
        label="home cover rate",
    )
    axes[1].axhline(
        BREAK_EVEN, color=TEXT_SECONDARY, linewidth=1.4, label="break-even (50%)"
    )
    axes[1].set_ylim(0.35, 0.65)
    axes[1].set_title("Home cover rate, excluding pushes", fontsize=11)
    axes[1].set_xlabel("Season")
    axes[1].set_ylabel("Home cover rate")
    axes[1].legend(loc="upper right")

    figure.suptitle(
        "Against-the-spread class balance by season",
        fontsize=14,
        fontweight="semibold",
    )
    figure.tight_layout()
    return _save(figure, output_dir, "10_ats_class_balance")


def build_all_charts(
    matchups: pd.DataFrame,
    team_games: pd.DataFrame,
    feature_columns: list[str],
    output_dir: Path,
) -> list[Path]:
    """Draw every required chart and return the paths written."""
    numeric = numeric_fields(matchups, feature_columns)
    return [
        correlation_heatmap(matchups, numeric, output_dir),
        feature_distributions(matchups, output_dir),
        offensive_epa_by_team_season(team_games, output_dir),
        defensive_epa_by_team_season(team_games, output_dir),
        pace_distribution(team_games, output_dir),
        rest_advantage_vs_cover(matchups, output_dir),
        spread_bins_vs_cover(matchups, output_dir),
        epa_differential_vs_cover(matchups, output_dir),
        missing_value_heatmap(matchups, feature_columns, output_dir),
        ats_class_balance(matchups, output_dir),
    ]
