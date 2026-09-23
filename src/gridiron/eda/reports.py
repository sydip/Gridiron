"""Visual reports for readers who are not going to read the code.

Two reports, kept visually distinct because they answer different questions:

* the **historical** report is a record of what the model did on seasons it had
  never seen, and every panel on it is settled fact,
* the **weekly** report is a set of predictions about games that have not been
  played, and nothing on it is a result.

Confusing the two is the single easiest way for a reader to come away with a
wrong impression, so they use different accent colours, different headers, and
say which they are in their titles.

Everything is drawn from the saved processed tables rather than recomputed, so
what a reader sees is what the pipeline actually produced.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from gridiron.eda.theme import (
    CATEGORICAL,
    DIVERGING_CMAP,
    DIVERGING_HIGH,
    DIVERGING_LOW,
    TEXT_MUTED,
    TEXT_SECONDARY,
    apply_theme,
)

# Plain-English names. A reader who knows football but not this codebase should
# not have to decode a column name to read a chart.
FRIENDLY_NAMES: dict[str, str] = {
    "spread_line": "Betting line (+ = home favoured)",
    "off_epa_diff_last_5": "Offence edge, last 5 games",
    "def_epa_strength_diff_last_5": "Defence edge, last 5 games",
    "pace_diff_last_5": "Pace edge, last 5 games",
    "rest_diff": "Rest advantage (days)",
    "point_margin_diff_last_5": "Scoring margin edge, last 5",
    "win_pct_diff_last_5": "Win rate edge, last 5",
    "home_short_week": "Home on a short week",
    "away_short_week": "Away on a short week",
    "div_game": "Divisional matchup",
    "week_1_flag": "Season opener",
    "off_epa_diff_last_3": "Offence edge, last 3 games",
    "def_epa_strength_diff_last_3": "Defence edge, last 3 games",
    "pace_diff_last_3": "Pace edge, last 3 games",
    "point_margin_diff_last_3": "Scoring margin edge, last 3",
    "off_epa_diff_season": "Offence edge, season to date",
    "def_epa_strength_diff_season": "Defence edge, season to date",
    "pace_diff_season": "Pace edge, season to date",
    "ats_margin_diff_last_5": "Margin vs the line, last 5",
    "cover_rate_diff_last_5": "Cover rate edge, last 5",
}

BASELINE_NAMES: dict[str, str] = {
    "home_every_game": "Always back the home team",
    "away_every_game": "Always back the away team",
    "favorite": "Always back the favourite",
    "underdog": "Always back the underdog",
    "better_epa": "Back the better recent offence",
    "random": "Coin flip",
}

# Historical panels use the blue accent; anything about unplayed games uses the
# orange one, so the two can never be mistaken for each other at a glance.
HISTORICAL_ACCENT = CATEGORICAL[0]
UPCOMING_ACCENT = CATEGORICAL[1]

BREAK_EVEN_RATE = 110.0 / 210.0


class ReportError(ValueError):
    """Raised when a report cannot be built from the saved tables."""


@dataclass(frozen=True)
class ReportInputs:
    """The saved tables a report is drawn from."""

    walk_forward: pd.DataFrame
    by_season: pd.DataFrame
    aggregate: pd.DataFrame
    baselines: pd.DataFrame
    policies: pd.DataFrame
    reliability: pd.DataFrame
    coefficients: pd.DataFrame
    correlations: pd.DataFrame
    confidence_buckets: pd.DataFrame


def friendly(name: str) -> str:
    """A football-friendly label for a column name."""
    return FRIENDLY_NAMES.get(name, name.replace("_", " ").capitalize())


def as_percent(value: float, places: int = 1) -> str:
    """One formatting rule for percentages, used everywhere.

    Mixing '52%', '0.524' and '52.38%' across a report makes numbers look
    inconsistent even when they agree, so every percentage goes through here.
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value * 100:.{places}f}%"


def load_report_inputs(report_dir: Path) -> ReportInputs:
    """Read the saved tables, naming any that are missing."""
    wanted = {
        "walk_forward": "walk_forward_predictions.csv",
        "by_season": "walk_forward_by_season.csv",
        "aggregate": "walk_forward_aggregate.csv",
        "baselines": "walk_forward_vs_baselines.csv",
        "policies": "policy_backtest.csv",
        "reliability": "calibration_reliability.csv",
        "coefficients": "model_coefficients.csv",
        "correlations": "eda_correlation_model_fields.csv",
        "confidence_buckets": "policy_win_rate_by_confidence.csv",
    }
    frames: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for key, filename in wanted.items():
        path = report_dir / filename
        if not path.exists():
            missing.append(filename)
            continue
        frames[key] = pd.read_csv(path, index_col=0 if key == "correlations" else None)

    if missing:
        raise ReportError(
            "Cannot build the report; missing saved table(s): "
            + ", ".join(missing)
            + ". Run the pipeline scripts first."
        )
    return ReportInputs(**frames)


def _headline(figure: plt.Figure, title: str, subtitle: str, accent: str) -> None:
    """A coloured banner so the two report types are distinguishable."""
    figure.suptitle(title, fontsize=17, fontweight="semibold", y=0.985)
    figure.text(0.5, 0.958, subtitle, ha="center", fontsize=10.5, color=TEXT_SECONDARY)
    figure.patches.append(
        plt.Rectangle(
            (0.0, 0.995), 1.0, 0.005, transform=figure.transFigure, color=accent
        )
    )


def historical_report(inputs: ReportInputs, output_dir: Path) -> Path:
    """One page covering everything the model did on unseen seasons."""
    apply_theme()
    figure = plt.figure(figsize=(19, 22))
    grid = figure.add_gridspec(4, 3, hspace=0.42, wspace=0.28, top=0.93, bottom=0.05)

    _record_panel(figure.add_subplot(grid[0, 0]), inputs)
    _accuracy_by_season_panel(figure.add_subplot(grid[0, 1:]), inputs)
    _baseline_panel(figure.add_subplot(grid[1, 0:2]), inputs)
    _probability_score_panel(figure.add_subplot(grid[1, 2]), inputs)
    _roi_drawdown_panel(figure.add_subplot(grid[2, 0:2]), inputs)
    _calibration_panel(figure.add_subplot(grid[2, 2]), inputs)
    _coefficient_panel(figure.add_subplot(grid[3, 0]), inputs)
    _correlation_panel(figure.add_subplot(grid[3, 1]), inputs)
    _confidence_bucket_panel(figure.add_subplot(grid[3, 2]), inputs)

    games = len(inputs.walk_forward)
    seasons = inputs.by_season["season"]
    _headline(
        figure,
        "Gridiron: how the model performed on seasons it had never seen",
        f"{games:,} games, {seasons.min()}-{seasons.max()}. "
        "Every panel is a settled result, not a prediction.",
        HISTORICAL_ACCENT,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "22_historical_report.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def _record_panel(axis: plt.Axes, inputs: ReportInputs) -> None:
    """The overall against-the-spread record, as a stat block."""
    seasons = inputs.by_season
    wins = int(seasons["wins"].sum())
    losses = int(seasons["losses"].sum())
    total = wins + losses
    accuracy = wins / total if total else float("nan")

    axis.axis("off")
    axis.set_title("Overall against-the-spread record", loc="left")
    lines = [
        (f"{wins}-{losses}", "record"),
        (as_percent(accuracy), "win rate"),
        (as_percent(BREAK_EVEN_RATE), "needed to break even at -110"),
    ]
    for index, (value, label) in enumerate(lines):
        colour = HISTORICAL_ACCENT if index != 2 else TEXT_SECONDARY
        axis.text(
            0.02,
            0.74 - index * 0.28,
            value,
            fontsize=30 if index < 2 else 20,
            fontweight="semibold",
            color=colour,
            transform=axis.transAxes,
        )
        axis.text(
            0.02,
            0.66 - index * 0.28,
            label,
            fontsize=10.5,
            color=TEXT_MUTED,
            transform=axis.transAxes,
        )
    verdict = "below break-even" if accuracy < BREAK_EVEN_RATE else "above break-even"
    axis.text(
        0.02,
        0.02,
        f"The model finished {verdict}.",
        fontsize=10,
        color=TEXT_SECONDARY,
        transform=axis.transAxes,
    )


def _accuracy_by_season_panel(axis: plt.Axes, inputs: ReportInputs) -> None:
    seasons = inputs.by_season.copy()
    seasons["season"] = seasons["season"].astype(int).astype(str)

    sns.lineplot(
        data=seasons,
        x="season",
        y="accuracy",
        marker="o",
        color=HISTORICAL_ACCENT,
        ax=axis,
        label="Model win rate",
    )
    axis.axhline(
        BREAK_EVEN_RATE,
        color=TEXT_SECONDARY,
        linewidth=1.4,
        label=f"Break even ({as_percent(BREAK_EVEN_RATE, 2)})",
    )
    axis.set_title("Win rate by season", loc="left")
    axis.set_xlabel("Season")
    axis.set_ylabel("Share of games called correctly")
    axis.set_ylim(0.35, 0.65)
    axis.yaxis.set_major_formatter(lambda value, _: as_percent(value, 0))
    axis.legend(loc="upper right")


def _baseline_panel(axis: plt.Axes, inputs: ReportInputs) -> None:
    table = inputs.baselines.copy()
    table["label"] = [
        "THE MODEL"
        if str(name).startswith("MODEL")
        else BASELINE_NAMES.get(str(name), str(name))
        for name in table["strategy"]
    ]
    table = table.sort_values("roi")
    colours = [
        HISTORICAL_ACCENT if label == "THE MODEL" else TEXT_SECONDARY
        for label in table["label"]
    ]

    sns.barplot(
        data=table,
        y="label",
        x="roi",
        palette=colours,
        hue="label",
        legend=False,
        ax=axis,
    )
    axis.axvline(0, color=TEXT_SECONDARY, linewidth=1.4)
    axis.set_title(
        "Return per bet: the model against strategies needing no model",
        loc="left",
    )
    axis.set_xlabel("Return on each unit staked")
    axis.set_ylabel("")
    axis.xaxis.set_major_formatter(lambda value, _: as_percent(value, 1))
    for index, row in enumerate(table.itertuples()):
        axis.annotate(
            f"{as_percent(row.roi, 1)}  ({row.ats_record})",
            (0.0, index),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=8.5,
            color=TEXT_MUTED,
        )


def _probability_score_panel(axis: plt.Axes, inputs: ReportInputs) -> None:
    """Log loss and Brier, each against what knowing nothing would score."""
    values = inputs.aggregate.set_index("metric")["value"]
    rows = pd.DataFrame(
        {
            "measure": ["Log loss", "Log loss", "Brier score", "Brier score"],
            "who": ["Model", "Knowing nothing"] * 2,
            "score": [
                float(values.get("log_loss", np.nan)),
                float(np.log(2)),
                float(values.get("brier_score", np.nan)),
                0.25,
            ],
        }
    )
    sns.barplot(
        data=rows,
        x="measure",
        y="score",
        hue="who",
        palette=[HISTORICAL_ACCENT, TEXT_SECONDARY],
        ax=axis,
    )
    axis.set_title("Probability quality (lower is better)", loc="left")
    axis.set_xlabel("")
    axis.set_ylabel("Score")
    axis.set_ylim(0, 0.9)
    axis.legend(loc="upper right", fontsize=8.5)
    axis.annotate(
        "A bar taller than its grey partner means the model's\n"
        "probabilities are worse than guessing 50% every time.",
        (0.0, -0.17),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8.5,
        color=TEXT_MUTED,
    )


def _roi_drawdown_panel(axis: plt.Axes, inputs: ReportInputs) -> None:
    table = inputs.policies.copy()
    table["label"] = [
        "Bet every game"
        if row.threshold_pp == 0
        else f"Only if edge >= {row.threshold_pp}pp"
        for row in table.itertuples()
    ]

    sns.barplot(data=table, x="label", y="units_won", color=HISTORICAL_ACCENT, ax=axis)
    axis.axhline(0, color=TEXT_SECONDARY, linewidth=1.4)
    axis.set_title(
        "Units won and worst losing run, by how selective the policy is",
        loc="left",
    )
    axis.set_xlabel("Betting policy")
    axis.set_ylabel("Units won (1 unit staked per bet)")
    plt.setp(axis.get_xticklabels(), rotation=18, ha="right", fontsize=8.5)

    # Every policy lost, so the bars run down; make room under the deepest one.
    floor = float(table["units_won"].min())
    axis.set_ylim(floor * 1.30 if floor < 0 else -1, 4)

    for index, row in enumerate(table.itertuples()):
        axis.annotate(
            f"{row.bets} bets, worst dip {row.max_drawdown_units:.0f}u",
            (index, row.units_won),
            xytext=(0, -11),
            textcoords="offset points",
            ha="center",
            va="top",
            fontsize=7.5,
            color=TEXT_MUTED,
        )


def _calibration_panel(axis: plt.Axes, inputs: ReportInputs) -> None:
    table = inputs.reliability.loc[inputs.reliability["games"] > 0].copy()

    axis.plot(
        [0.35, 0.7],
        [0.35, 0.7],
        color=TEXT_SECONDARY,
        linewidth=1.4,
        label="Perfectly calibrated",
    )
    sns.scatterplot(
        data=table,
        x="mean_predicted",
        y="observed_rate",
        size="games",
        sizes=(40, 380),
        color=HISTORICAL_ACCENT,
        alpha=0.85,
        ax=axis,
        legend="brief",
    )
    axis.set_title("Do the stated chances come true?", loc="left")
    axis.set_xlabel("Chance the model gave the home team")
    axis.set_ylabel("How often the home team actually covered")
    axis.set_xlim(0.3, 0.75)
    axis.set_ylim(0.0, 1.0)
    axis.xaxis.set_major_formatter(lambda value, _: as_percent(value, 0))
    axis.yaxis.set_major_formatter(lambda value, _: as_percent(value, 0))
    axis.legend(loc="upper left", fontsize=8, title="Games in bucket")


def _coefficient_panel(axis: plt.Axes, inputs: ReportInputs) -> None:
    table = inputs.coefficients.copy().head(8)
    table["label"] = [friendly(name) for name in table["feature"]]
    table = table.sort_values("coefficient")

    colours = [
        DIVERGING_HIGH if value >= 0 else DIVERGING_LOW
        for value in table["coefficient"]
    ]
    sns.barplot(
        data=table,
        y="label",
        x="coefficient",
        palette=colours,
        hue="label",
        legend=False,
        ax=axis,
    )
    axis.axvline(0, color=TEXT_SECONDARY, linewidth=1.4)
    axis.set_title("What the model leans on (top 8)", loc="left")
    axis.set_xlabel("Effect on the home team's chance")
    axis.set_ylabel("")
    axis.annotate(
        "Right = pushes toward a home cover. Patterns the model found, not causes.",
        (0.0, -0.20),
        xycoords="axes fraction",
        fontsize=8,
        color=TEXT_MUTED,
    )


def _correlation_panel(axis: plt.Axes, inputs: ReportInputs) -> None:
    matrix = inputs.correlations.copy()
    matrix.index = [friendly(name) for name in matrix.index]
    matrix.columns = [friendly(name) for name in matrix.columns]

    mask = np.triu(np.ones_like(matrix, dtype=bool), k=0)
    sns.heatmap(
        matrix,
        mask=mask,
        cmap=DIVERGING_CMAP,
        vmin=-1,
        vmax=1,
        center=0,
        square=False,
        linewidths=1.5,
        linecolor=plt.rcParams["figure.facecolor"],
        cbar_kws={"label": "Correlation", "shrink": 0.7},
        ax=axis,
    )
    axis.set_title("Which inputs move together", loc="left")
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.grid(False)
    plt.setp(axis.get_xticklabels(), rotation=42, ha="right", fontsize=7.5)
    plt.setp(axis.get_yticklabels(), fontsize=7.5)


def _confidence_bucket_panel(axis: plt.Axes, inputs: ReportInputs) -> None:
    table = inputs.confidence_buckets.copy()
    table["label"] = table["edge_bucket"].astype(str)

    sns.barplot(data=table, x="label", y="win_rate", color=HISTORICAL_ACCENT, ax=axis)
    axis.axhline(
        BREAK_EVEN_RATE,
        color=TEXT_SECONDARY,
        linewidth=1.4,
        label=f"Break even ({as_percent(BREAK_EVEN_RATE, 1)})",
    )
    axis.set_title("Does the model win more when it is surer?", loc="left")
    axis.set_xlabel("How confident the model was (percentage points over 50%)")
    axis.set_ylabel("Win rate")
    axis.set_ylim(0, 0.75)
    axis.yaxis.set_major_formatter(lambda value, _: as_percent(value, 0))
    axis.legend(loc="upper right", fontsize=8.5)
    plt.setp(axis.get_xticklabels(), rotation=18, ha="right", fontsize=8)

    for index, row in enumerate(table.itertuples()):
        # Inside the bar: above it the labels run into the break-even rule,
        # which sits right where these win rates do.
        axis.annotate(
            f"{int(row.bets)} games",
            (index, 0.02),
            ha="center",
            va="bottom",
            fontsize=8,
            color="white",
        )


# --- the weekly report -----------------------------------------------------

# The comparisons a reader wants beside each pick, in plain terms.
COMPARISON_ROWS = (
    ("off_epa_diff_last_5", "Offence edge"),
    ("def_epa_strength_diff_last_5", "Defence edge"),
    ("pace_diff_last_5", "Pace edge"),
    ("rest_diff", "Rest advantage"),
)


def prediction_contributions(
    features: pd.DataFrame,
    pipeline,
) -> pd.DataFrame:
    """How much each input pushed each game's probability, and which way.

    For a standardised logistic regression the log-odds is the intercept plus
    the sum of coefficient times scaled value, exactly. Splitting that sum back
    out is therefore not an approximation: it is the arithmetic the model did,
    which is what lets a reader see *why* a side was picked rather than being
    asked to take it on trust.
    """
    from gridiron.modeling.pipeline import (
        IMPUTE_STEP,
        MODEL_STEP,
        SCALE_STEP,
        fitted_feature_names,
    )

    names = fitted_feature_names(pipeline)
    scaled = pipeline.named_steps[SCALE_STEP].transform(
        pipeline.named_steps[IMPUTE_STEP].transform(features[names])
    )
    coefficients = pipeline.named_steps[MODEL_STEP].coef_.ravel()
    return pd.DataFrame(scaled * coefficients, columns=names, index=features.index)


def weekly_report(
    table: pd.DataFrame,
    matchups: pd.DataFrame,
    contributions: pd.DataFrame,
    output_dir: Path,
    season: int,
    week: int,
) -> Path:
    """One page per week: the picks, the comparisons, and the reasoning."""
    apply_theme()
    priced = table.loc[table["home_cover_probability"].notna()].copy()
    if priced.empty:
        raise ReportError(
            f"No priced games in season {season} week {week}; nothing to draw."
        )

    figure = plt.figure(figsize=(19, 6 + 0.52 * len(priced)))
    grid = figure.add_gridspec(
        3,
        2,
        hspace=0.45,
        wspace=0.22,
        top=0.90,
        bottom=0.06,
        height_ratios=[1.35, 1.0, 1.0],
    )

    _picks_panel(figure.add_subplot(grid[0, :]), priced)
    _comparison_panel(figure.add_subplot(grid[1, 0]), priced, matchups)
    _confidence_spread_panel(figure.add_subplot(grid[1, 1]), priced)
    _reasoning_panel(figure.add_subplot(grid[2, :]), priced, contributions)

    _headline(
        figure,
        f"Gridiron picks: {season} week {week}",
        f"{len(priced)} unplayed games. These are predictions, not results, "
        "and the model has no demonstrated edge.",
        UPCOMING_ACCENT,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"23_week_{week:02d}_report.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def _matchup_label(row) -> str:
    return f"{row.away_team} at {row.home_team}"


# The picks panel is split between bars and a column of labels. The bars get
# this share of the width; the labels start just past it.
BAR_SHARE_OF_PANEL = 0.62
ANNOTATION_COLUMN = 0.66


def _picks_panel(axis: plt.Axes, priced: pd.DataFrame) -> None:
    """Each game's home-cover chance, with the pick and the line beside it."""
    frame = priced.copy()
    frame["label"] = [_matchup_label(row) for row in frame.itertuples()]
    frame = frame.sort_values("home_cover_probability")

    colours = [
        UPCOMING_ACCENT if probability >= 0.5 else CATEGORICAL[0]
        for probability in frame["home_cover_probability"]
    ]
    sns.barplot(
        data=frame,
        y="label",
        x="home_cover_probability",
        palette=colours,
        hue="label",
        legend=False,
        ax=axis,
    )
    axis.axvline(0.5, color=TEXT_SECONDARY, linewidth=1.6)
    axis.set_title("Chance the home team covers", loc="left")
    axis.set_xlabel("Home-team cover probability")
    axis.set_ylabel("")

    # The right of the panel is a reserved column for the per-game labels, so
    # the upper limit is set from the longest bar rather than fixed: with a
    # fixed limit, a confident week draws bars straight through its own text.
    lower = 0.35
    longest = float(frame["home_cover_probability"].max())
    upper = max(0.65, lower + (longest - lower) / BAR_SHARE_OF_PANEL)
    axis.set_xlim(lower, upper)
    axis.xaxis.set_major_formatter(lambda value, _: as_percent(value, 0))
    # Labelled in place rather than through a legend: a legend box lands on the
    # annotation column to the right of the bars.
    # Anchored in axes fraction vertically: at a data y above the top bar this
    # falls outside the limits and is clipped away silently.
    axis.annotate(
        "even (50%)",
        (0.5, 0.99),
        xycoords=("data", "axes fraction"),
        xytext=(4, 0),
        textcoords="offset points",
        ha="left",
        va="top",
        fontsize=8.5,
        color=TEXT_SECONDARY,
    )

    for index, row in enumerate(frame.itertuples()):
        axis.annotate(
            f"line {row.spread_line_used:+.1f}   pick {row.predicted_side}"
            f"   {as_percent(row.confidence, 1)} over even",
            (ANNOTATION_COLUMN, index),
            xycoords=("axes fraction", "data"),
            va="center",
            fontsize=8.5,
            color=TEXT_MUTED,
            annotation_clip=False,
        )


def _comparison_panel(
    axis: plt.Axes, priced: pd.DataFrame, matchups: pd.DataFrame
) -> None:
    """Offence, defence, pace and rest edges, as a home-minus-away heatmap."""
    joined = priced.merge(
        matchups[["game_id", *[name for name, _ in COMPARISON_ROWS]]],
        on="game_id",
        how="left",
    )
    grid = joined[[name for name, _ in COMPARISON_ROWS]].copy()
    # Assigning the index directly: set_index would read these strings as
    # column names and fail.
    grid.index = [_matchup_label(row) for row in joined.itertuples()]
    grid.columns = [label for _, label in COMPARISON_ROWS]

    # Each column is on its own scale, so compare within a column by ranking
    # rather than across columns by raw size.
    ranked = grid.rank(pct=True) * 2 - 1

    sns.heatmap(
        ranked,
        cmap=DIVERGING_CMAP,
        vmin=-1,
        vmax=1,
        center=0,
        annot=grid.round(2),
        fmt="",
        annot_kws={"size": 7.5},
        linewidths=1.5,
        linecolor=plt.rcParams["figure.facecolor"],
        cbar_kws={"label": "Home advantage within this week", "shrink": 0.7},
        ax=axis,
    )
    axis.set_title(
        "Team comparisons: home minus away (numbers are the raw edge)",
        loc="left",
    )
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.grid(False)
    plt.setp(axis.get_yticklabels(), fontsize=8)
    plt.setp(axis.get_xticklabels(), rotation=0, fontsize=8.5)


def _confidence_spread_panel(axis: plt.Axes, priced: pd.DataFrame) -> None:
    """Where this week's confidence sits, against the model's usual range."""
    sns.histplot(
        priced["confidence"] * 100,
        bins=12,
        color=UPCOMING_ACCENT,
        edgecolor="none",
        ax=axis,
    )
    axis.set_title("How confident the model is this week", loc="left")
    axis.set_xlabel("Percentage points above an even call")
    axis.set_ylabel("Games")
    # Below the axes, not inside them: a two-line caption in the plot area
    # lands on the bars whenever the week's confidences are spread out.
    axis.annotate(
        "In backtesting, being surer did not mean being righter.\n"
        "Treat these as the model's own certainty, not evidence.",
        (0.0, -0.17),
        xycoords="axes fraction",
        ha="left",
        va="top",
        fontsize=8.5,
        color=TEXT_MUTED,
    )


def _reasoning_panel(
    axis: plt.Axes, priced: pd.DataFrame, contributions: pd.DataFrame
) -> None:
    """Why each pick was made, as the model's own arithmetic."""
    available = [name for name in contributions.columns if name in FRIENDLY_NAMES]
    frame = contributions.loc[priced.index, available].copy()
    frame.index = [_matchup_label(row) for row in priced.itertuples()]
    frame.columns = [friendly(name) for name in available]

    order = priced.sort_values("home_cover_probability").index
    frame = frame.loc[[_matchup_label(row) for row in priced.loc[order].itertuples()]]

    limit = float(np.abs(frame.to_numpy()).max()) or 1.0
    sns.heatmap(
        frame,
        cmap=DIVERGING_CMAP,
        vmin=-limit,
        vmax=limit,
        center=0,
        linewidths=1.5,
        linecolor=plt.rcParams["figure.facecolor"],
        cbar_kws={"label": "Push toward home (right) or away (left)", "shrink": 0.6},
        ax=axis,
    )
    axis.set_title(
        "Why: how much each input moved the pick, for every game", loc="left"
    )
    axis.set_xlabel("")
    axis.set_ylabel("")
    axis.grid(False)
    plt.setp(axis.get_xticklabels(), rotation=28, ha="right", fontsize=8)
    plt.setp(axis.get_yticklabels(), fontsize=8)
