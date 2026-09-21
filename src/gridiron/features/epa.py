"""Game-level offensive and defensive EPA aggregation from play-by-play data."""

from __future__ import annotations

import pandas as pd

SCRIMMAGE_PLAY_TYPES = frozenset({"pass", "run"})
DEAD_BALL_INDICATORS = ("qb_kneel", "qb_spike")
MIN_OFFENSIVE_PLAYS = 30
MIN_DEFENSIVE_PLAYS = MIN_OFFENSIVE_PLAYS
RECONCILIATION_TOLERANCE = 1e-6

REQUIRED_PBP_COLUMNS = frozenset(
    {"season", "week", "game_id", "posteam", "defteam", "play_type", "epa"}
)
GROUP_COLUMNS = ["season", "week", "game_id", "team"]

OFFENSE_COLUMNS = [
    "season",
    "week",
    "game_id",
    "team",
    "opponent",
    "offensive_plays",
    "offensive_epa",
    "offensive_epa_per_play",
    "offensive_success_rate",
    "pass_plays",
    "pass_epa",
    "pass_epa_per_play",
    "rush_plays",
    "rush_epa",
    "rush_epa_per_play",
    "valid_game",
]

DEFENSE_COLUMNS = [
    "season",
    "week",
    "game_id",
    "team",
    "opponent",
    "defensive_plays",
    "defensive_epa_allowed",
    "defensive_epa_allowed_per_play",
    "defensive_success_rate_allowed",
    "defensive_epa_strength",
    "defensive_epa_strength_per_play",
    "pass_plays_faced",
    "pass_epa_allowed",
    "pass_epa_allowed_per_play",
    "rush_plays_faced",
    "rush_epa_allowed",
    "rush_epa_allowed_per_play",
    "valid_game",
]

RECONCILIATION_COLUMNS = [
    "season",
    "week",
    "game_id",
    "offense",
    "defense",
    "offensive_plays",
    "defensive_plays",
    "plays_difference",
    "offensive_epa",
    "defensive_epa_allowed",
    "epa_difference",
    "offensive_success_rate",
    "defensive_success_rate_allowed",
    "success_rate_difference",
    "paired",
    "reconciled",
]

_OFFENSE_NAMES = {
    "plays": "offensive_plays",
    "epa": "offensive_epa",
    "epa_per_play": "offensive_epa_per_play",
    "success_rate": "offensive_success_rate",
    "pass_plays": "pass_plays",
    "pass_epa": "pass_epa",
    "pass_epa_per_play": "pass_epa_per_play",
    "rush_plays": "rush_plays",
    "rush_epa": "rush_epa",
    "rush_epa_per_play": "rush_epa_per_play",
}

_DEFENSE_NAMES = {
    "plays": "defensive_plays",
    "epa": "defensive_epa_allowed",
    "epa_per_play": "defensive_epa_allowed_per_play",
    "success_rate": "defensive_success_rate_allowed",
    "pass_plays": "pass_plays_faced",
    "pass_epa": "pass_epa_allowed",
    "pass_epa_per_play": "pass_epa_allowed_per_play",
    "rush_plays": "rush_plays_faced",
    "rush_epa": "rush_epa_allowed",
    "rush_epa_per_play": "rush_epa_allowed_per_play",
}

_UNIT_AGGREGATIONS = {
    "opponent": ("opponent", "first"),
    "distinct_opponents": ("opponent", "nunique"),
    "plays": ("epa", "size"),
    "epa": ("epa", "sum"),
    "epa_per_play": ("epa", "mean"),
    "success_rate": ("success", "mean"),
    "pass_plays": ("is_pass", "sum"),
    "pass_epa": ("pass_play_epa", "sum"),
    "pass_epa_per_play": ("pass_play_epa", "mean"),
    "rush_plays": ("is_rush", "sum"),
    "rush_epa": ("rush_play_epa", "sum"),
    "rush_epa_per_play": ("rush_play_epa", "mean"),
}


class OffensiveEpaError(ValueError):
    """Raised when play-by-play data cannot produce game-level EPA."""


def _blank_to_na(values: pd.Series) -> pd.Series:
    """Treat whitespace-only team codes the same as missing ones."""
    if values.dtype == object or isinstance(values.dtype, pd.StringDtype):
        return values.astype("string").str.strip().replace("", pd.NA)
    return values


def _usable_indicator(plays: pd.DataFrame, column: str) -> pd.Series | None:
    """Return a numeric indicator column, or ``None`` when it is not reliable.

    A column counts as reliable only when it is present and carries at least one
    non-null numeric value; an all-null or non-numeric column is ignored rather
    than silently dropping every play.
    """
    if column not in plays.columns:
        return None
    indicator = pd.to_numeric(plays[column], errors="coerce")
    if not indicator.notna().any():
        return None
    return indicator


def _play_flags(plays: pd.DataFrame, kind: str) -> pd.Series:
    """Flag pass or rush plays, falling back to ``play_type`` when needed."""
    indicator = _usable_indicator(plays, kind)
    if indicator is not None:
        return indicator.eq(1)
    play_type = "pass" if kind == "pass" else "run"
    return plays["play_type"].eq(play_type)


def _success_flags(plays: pd.DataFrame) -> pd.Series:
    """Return per-play success, deriving it from EPA when the column is absent.

    nflverse defines a successful play as one with positive EPA, so the fallback
    reproduces the shipped ``success`` column rather than approximating it.
    """
    indicator = _usable_indicator(plays, "success")
    if indicator is not None:
        return indicator.eq(1)
    return plays["epa"].gt(0)


def filter_offensive_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Keep the scrimmage plays that can carry EPA credit.

    Retains ``pass`` and ``run`` plays that have an EPA value and a known
    offense and defense. That drops administrative rows (``no_play`` penalties,
    timeouts, end-of-quarter markers), special teams, and kneel-downs and spikes
    -- the latter are removed again by the ``qb_kneel`` and ``qb_spike``
    indicators whenever those fields are present and populated.

    Two-point conversions are kept: they are genuine scrimmage plays with a real
    EPA value, even though they carry no ``down``.

    Both the offensive and defensive aggregations start here, so every play
    credited to an offense is charged to exactly one defense.
    """
    missing = sorted(REQUIRED_PBP_COLUMNS.difference(pbp.columns))
    if missing:
        raise OffensiveEpaError(
            "Cannot aggregate EPA; missing required column(s): " + ", ".join(missing)
        )

    plays = pbp.copy()
    plays["epa"] = pd.to_numeric(plays["epa"], errors="coerce")
    plays["posteam"] = _blank_to_na(plays["posteam"])
    plays["defteam"] = _blank_to_na(plays["defteam"])

    keep = (
        plays["play_type"].isin(SCRIMMAGE_PLAY_TYPES)
        & plays["epa"].notna()
        & plays["posteam"].notna()
        & plays["defteam"].notna()
    )
    for column in DEAD_BALL_INDICATORS:
        indicator = _usable_indicator(plays, column)
        if indicator is not None:
            keep &= indicator.ne(1)

    return plays.loc[keep]


def eligible_plays(pbp: pd.DataFrame) -> pd.DataFrame:
    """Normalise the eligible plays into the shared offense/defense frame.

    Every play appears exactly once, labelled with the ``offense`` that ran it
    and the ``defense`` that faced it. Aggregating this frame by either side
    uses an identical play set, which is what lets the two aggregations
    reconcile game by game.
    """
    plays = filter_offensive_plays(pbp)

    frame = pd.DataFrame(
        {
            "season": plays["season"],
            "week": plays["week"],
            "game_id": plays["game_id"],
            "offense": plays["posteam"],
            "defense": plays["defteam"],
            "epa": plays["epa"],
            "success": _success_flags(plays),
            "is_pass": _play_flags(plays, "pass"),
            "is_rush": _play_flags(plays, "rush"),
        }
    )
    frame["pass_play_epa"] = frame["epa"].where(frame["is_pass"])
    frame["rush_play_epa"] = frame["epa"].where(frame["is_rush"])
    return frame


def _aggregate_unit(
    frame: pd.DataFrame,
    *,
    team_column: str,
    opponent_column: str,
    names: dict[str, str],
    min_plays: int,
) -> pd.DataFrame:
    """Aggregate the shared play frame from one unit's point of view."""
    unit = frame.rename(columns={team_column: "team", opponent_column: "opponent"})
    aggregated = (
        unit.groupby(GROUP_COLUMNS, dropna=False, sort=True)
        .agg(**_UNIT_AGGREGATIONS)
        .reset_index()
    )

    ambiguous = aggregated.loc[aggregated["distinct_opponents"].gt(1), "game_id"]
    if not ambiguous.empty:
        raise OffensiveEpaError(
            f"Cannot aggregate EPA; a {team_column} faces more than one "
            f"{opponent_column} in game(s): {sorted(set(ambiguous))[:5]}"
        )

    aggregated = aggregated.drop(columns="distinct_opponents").rename(columns=names)
    for column in (names["plays"], names["pass_plays"], names["rush_plays"]):
        aggregated[column] = aggregated[column].astype("int64")
    aggregated["valid_game"] = aggregated[names["plays"]].ge(min_plays)
    return aggregated


def aggregate_offensive_epa(
    pbp: pd.DataFrame,
    *,
    min_plays: int = MIN_OFFENSIVE_PLAYS,
) -> pd.DataFrame:
    """Build one offensive-efficiency row per team per game.

    ``posteam`` becomes ``team`` and ``defteam`` becomes ``opponent``, so a
    completed game yields two rows. EPA is reported both as a game total
    (``offensive_epa``) and per play (``offensive_epa_per_play``); only the
    former is needed by the baseline model, but the rest supports later work.

    ``valid_game`` is false when a team-game has fewer than ``min_plays``
    eligible plays. Those rows are flagged, never dropped -- an implausible play
    count usually means incomplete source data and deserves review.
    """
    offense = _aggregate_unit(
        eligible_plays(pbp),
        team_column="offense",
        opponent_column="defense",
        names=_OFFENSE_NAMES,
        min_plays=min_plays,
    )
    return offense[OFFENSE_COLUMNS].reset_index(drop=True)


def aggregate_defensive_epa(
    pbp: pd.DataFrame,
    *,
    min_plays: int = MIN_DEFENSIVE_PLAYS,
) -> pd.DataFrame:
    """Build one defensive-efficiency row per team per game.

    The mirror of :func:`aggregate_offensive_epa`: ``defteam`` becomes ``team``
    and ``posteam`` becomes ``opponent``, over the same eligible play set. A
    defense's totals are therefore what it *allowed*, and low EPA allowed is a
    good defensive game.

    ``defensive_epa_strength`` negates ``defensive_epa_allowed`` so the measure
    points the same way as the offensive one: higher is a stronger unit. Only
    play-level EPA feeds this -- no score, spread, or result of the game in
    question is read.
    """
    defense = _aggregate_unit(
        eligible_plays(pbp),
        team_column="defense",
        opponent_column="offense",
        names=_DEFENSE_NAMES,
        min_plays=min_plays,
    )
    defense["defensive_epa_strength"] = -defense["defensive_epa_allowed"]
    defense["defensive_epa_strength_per_play"] = -defense[
        "defensive_epa_allowed_per_play"
    ]
    return defense[DEFENSE_COLUMNS].reset_index(drop=True)


def reconcile_offense_and_defense(
    offense: pd.DataFrame,
    defense: pd.DataFrame,
    *,
    tolerance: float = RECONCILIATION_TOLERANCE,
) -> pd.DataFrame:
    """Pair every offense with the defense that faced it and compare totals.

    Within a game the home offense's EPA is by definition the away defense's EPA
    allowed, and vice versa, because both come from the same eligible plays. Any
    non-zero difference means the two sides were built from different play sets
    -- a filtering divergence -- so it is surfaced rather than tolerated.

    Returns one row per offense/defense pairing. ``paired`` is false when a side
    has no counterpart at all; ``reconciled`` is false for that case and for any
    mismatch in play count, EPA, or success rate beyond ``tolerance``.
    """
    offense_side = offense[
        [
            "season",
            "week",
            "game_id",
            "team",
            "opponent",
            "offensive_plays",
            "offensive_epa",
            "offensive_success_rate",
        ]
    ].rename(columns={"team": "offense", "opponent": "defense"})
    defense_side = defense[
        [
            "season",
            "week",
            "game_id",
            "team",
            "opponent",
            "defensive_plays",
            "defensive_epa_allowed",
            "defensive_success_rate_allowed",
        ]
    ].rename(columns={"team": "defense", "opponent": "offense"})

    report = offense_side.merge(
        defense_side,
        on=["season", "week", "game_id", "offense", "defense"],
        how="outer",
        indicator="_pairing",
    )

    report["plays_difference"] = report["offensive_plays"] - report["defensive_plays"]
    report["epa_difference"] = report["offensive_epa"] - report["defensive_epa_allowed"]
    report["success_rate_difference"] = (
        report["offensive_success_rate"] - report["defensive_success_rate_allowed"]
    )
    report["paired"] = report["_pairing"].eq("both")
    report["reconciled"] = (
        report["paired"]
        & report["plays_difference"].eq(0)
        & report["epa_difference"].abs().le(tolerance)
        & report["success_rate_difference"].abs().le(tolerance)
    )

    return (
        report[RECONCILIATION_COLUMNS]
        .sort_values(["season", "week", "game_id", "offense"])
        .reset_index(drop=True)
    )


def epa_reconciliation_failures(report: pd.DataFrame) -> pd.DataFrame:
    """Return only the pairings that failed to reconcile, worst EPA gap first."""
    failures = report.loc[~report["reconciled"]].copy()
    failures["_gap"] = failures["epa_difference"].abs().fillna(float("inf"))
    return (
        failures.sort_values(["_gap", "game_id", "offense"], ascending=False)
        .drop(columns="_gap")
        .reset_index(drop=True)
    )


def low_volume_team_games(team_games: pd.DataFrame) -> pd.DataFrame:
    """Return the flagged team-games, ordered by play count, for manual review.

    Accepts either an offensive or a defensive aggregation.
    """
    play_column = (
        "offensive_plays"
        if "offensive_plays" in team_games.columns
        else "defensive_plays"
    )
    if play_column not in team_games.columns:
        raise OffensiveEpaError(
            "Expected an offensive or defensive aggregation with a play count."
        )
    flagged = team_games.loc[~team_games["valid_game"]]
    return flagged.sort_values([play_column, "game_id", "team"]).reset_index(drop=True)
