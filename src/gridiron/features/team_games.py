"""Chronological team-game history and rest days from a canonical schedule.

A schedule stores one row per game, with the two teams in separate columns.
Almost every team-level question -- form, rest, travel, rolling strength -- wants
the opposite shape: one row per team per game, ordered in time. This module
performs that transformation and derives each team's rest from it.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_SCHEDULE_COLUMNS = frozenset(
    {
        "game_id",
        "season",
        "week",
        "gameday",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "spread_line",
    }
)

# Rest for a season opener is asserted rather than measured: the gap to the
# previous season's final game is an offseason, not rest, and differs by months
# between teams. A normal week keeps openers on the same scale as everything
# else and leaves week_1_flag to mark them for any model that wants to.
SEASON_OPENER_REST_DAYS = 7

NORMAL_REST_DAYS = 7
# A bye is a scheduled week with no game, so it is detected from the week
# counter rather than from elapsed days. Days alone cannot separate a true bye
# from the 10-to-12 day gap a team gets after playing on a Thursday.
BYE_WEEK_GAP = 2

TEAM_GAME_COLUMNS = [
    "season",
    "week",
    "game_id",
    "gameday",
    "team",
    "opponent",
    "is_home",
    "points_for",
    "points_against",
    "point_margin",
    "team_spread",
    "covered",
    "is_completed",
]

REST_COLUMNS = [
    "previous_game_date",
    "rest_days",
    "rest_advantage_placeholder",
    "short_week",
    "extra_rest",
    "bye_week_rest",
    "week_1_flag",
]


class TeamGameError(ValueError):
    """Raised when a schedule cannot produce a team-game history."""


def _require_columns(schedule: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_SCHEDULE_COLUMNS.difference(schedule.columns))
    if missing:
        raise TeamGameError(
            "Cannot build team-game history; missing required column(s): "
            + ", ".join(missing)
        )


def _one_side(schedule: pd.DataFrame, *, home: bool) -> pd.DataFrame:
    """Project the schedule onto one team's point of view.

    ``team_spread`` follows the nflverse convention that the source spread uses:
    a positive number means *this* team is favoured by that many points. It is
    therefore the home spread for the home row and its negation for the away
    row, so the two rows of a game always sum to zero. Note this is the opposite
    sign to betting-sheet notation, where a favourite is quoted negative.
    """
    team_column, opponent_column = (
        ("home_team", "away_team") if home else ("away_team", "home_team")
    )
    points_for_column, points_against_column = (
        ("home_score", "away_score") if home else ("away_score", "home_score")
    )

    side = pd.DataFrame(
        {
            "season": schedule["season"],
            "week": schedule["week"],
            "game_id": schedule["game_id"],
            "gameday": schedule["gameday"],
            "team": schedule[team_column],
            "opponent": schedule[opponent_column],
            "is_home": home,
            "points_for": schedule[points_for_column],
            "points_against": schedule[points_against_column],
        }
    )
    side["point_margin"] = side["points_for"] - side["points_against"]
    side["team_spread"] = schedule["spread_line"] if home else -schedule["spread_line"]
    return side


def build_team_games(schedule: pd.DataFrame) -> pd.DataFrame:
    """Turn one row per game into two rows, one per team.

    Every schedule row becomes a home row and an away row, so a completed game
    always yields exactly two records whose ``point_margin`` values sum to zero
    and whose ``team_spread`` values sum to zero.

    ``covered`` is nullable: 1 when the team beat its own spread, 0 when it did
    not, and null for a push or for any game without a score or a line. A push
    is a real outcome, not a loss against the spread, so it is never recorded as
    a 0 -- this matches the convention in
    :func:`~gridiron.features.target.add_ats_target`.

    Unplayed games are kept. They carry null scores and are marked by
    ``is_completed``, so the same history serves both training and prediction.
    """
    _require_columns(schedule)

    frame = schedule.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="coerce")

    team_games = pd.concat(
        [_one_side(frame, home=True), _one_side(frame, home=False)],
        ignore_index=True,
    )

    against_spread = team_games["point_margin"] - team_games["team_spread"]
    team_games["covered"] = (
        pd.Series(pd.NA, index=team_games.index, dtype="Int8")
        .mask(against_spread.gt(0), 1)
        .mask(against_spread.lt(0), 0)
    )
    team_games["is_completed"] = (
        team_games["points_for"].notna() & team_games["points_against"].notna()
    )

    return (
        team_games[TEAM_GAME_COLUMNS]
        .sort_values(["team", "season", "gameday", "game_id"], kind="stable")
        .reset_index(drop=True)
    )


def add_rest_days(team_games: pd.DataFrame) -> pd.DataFrame:
    """Add rest days and its derived flags to a team-game history.

    Rest is measured within a team and season, from that team's own previous
    game. Rows are ordered by date first, so the previous game is always
    genuinely earlier -- no rest value can be derived from a future date.

    A season opener has no previous game in its season. Rather than reach back
    across the offseason, those rows take
    :data:`SEASON_OPENER_REST_DAYS` and are marked by ``week_1_flag``.

    ``week_1_flag`` marks a team's *first game of the season*, which is week 1
    in ordinary seasons but not always: Miami and Tampa Bay opened 2017 in week
    2 after Hurricane Irma cancelled their week 1 game. Defining the flag by
    first appearance rather than by ``week == 1`` keeps those openers from
    falling through with undefined rest.

    ``rest_advantage_placeholder`` is deliberately null. Rest advantage is a
    team-versus-opponent comparison that a later phase will fill; the column is
    reserved here so downstream schemas are stable, and an empty column is
    honest where an invented one would not be.
    """
    if "gameday" not in team_games.columns:
        raise TeamGameError("Expected a team-game history with a 'gameday' column.")

    history = team_games.copy()
    history["gameday"] = pd.to_datetime(history["gameday"], errors="coerce")
    history = history.sort_values(
        ["team", "season", "gameday", "game_id"], kind="stable"
    ).reset_index(drop=True)

    grouped = history.groupby(["team", "season"], sort=False, dropna=False)
    history["previous_game_date"] = grouped["gameday"].shift(1)
    previous_week = grouped["week"].shift(1)

    measured_rest = (history["gameday"] - history["previous_game_date"]).dt.days
    is_opener = history["previous_game_date"].isna()

    history["week_1_flag"] = is_opener.astype("int8")
    history["rest_days"] = measured_rest.where(
        ~is_opener, SEASON_OPENER_REST_DAYS
    ).astype("Int64")

    week_gap = history["week"] - previous_week
    history["short_week"] = history["rest_days"].lt(NORMAL_REST_DAYS).fillna(False)
    history["extra_rest"] = history["rest_days"].gt(NORMAL_REST_DAYS).fillna(False)
    history["bye_week_rest"] = week_gap.ge(BYE_WEEK_GAP).fillna(False)
    history["rest_advantage_placeholder"] = pd.Series(
        pd.NA, index=history.index, dtype="Float64"
    )

    ordered = [
        *[column for column in team_games.columns if column in history.columns],
        *[column for column in REST_COLUMNS if column not in team_games.columns],
    ]
    return history[ordered]


def team_game_history(schedule: pd.DataFrame) -> pd.DataFrame:
    """Build the team-game history and its rest columns in one step."""
    return add_rest_days(build_team_games(schedule))


def team_game_reconciliation(team_games: pd.DataFrame) -> pd.DataFrame:
    """Check each game's two rows against each other, one row per game.

    Surfaces the invariants that the transformation must preserve: a completed
    game has exactly two rows, its margins cancel, and its spreads cancel.
    ``reconciled`` is true only when all three hold.
    """
    required = {"game_id", "point_margin", "team_spread", "is_completed"}
    missing = sorted(required.difference(team_games.columns))
    if missing:
        raise TeamGameError(
            "Cannot reconcile team games; missing column(s): " + ", ".join(missing)
        )

    grouped = team_games.groupby("game_id", sort=True)
    report = pd.DataFrame(
        {
            "rows": grouped.size(),
            "distinct_teams": grouped["team"].nunique(),
            "margin_sum": grouped["point_margin"].sum(min_count=1),
            "spread_sum": grouped["team_spread"].sum(min_count=1),
            "is_completed": grouped["is_completed"].any(),
        }
    ).reset_index()

    report["two_rows"] = report["rows"].eq(2) & report["distinct_teams"].eq(2)
    report["margins_cancel"] = report["margin_sum"].fillna(0).abs().le(1e-9)
    report["spreads_cancel"] = report["spread_sum"].fillna(0).abs().le(1e-9)
    report["reconciled"] = (
        report["two_rows"] & report["margins_cancel"] & report["spreads_cancel"]
    )
    return report


def team_game_reconciliation_failures(report: pd.DataFrame) -> pd.DataFrame:
    """Return only the games that failed reconciliation."""
    return report.loc[~report["reconciled"]].reset_index(drop=True)


def rest_distribution(team_games: pd.DataFrame) -> pd.DataFrame:
    """Summarise rest days and the share of games carrying each rest flag."""
    if "rest_days" not in team_games.columns:
        raise TeamGameError("Expected a history with rest days; call add_rest_days.")

    rest = pd.to_numeric(team_games["rest_days"], errors="coerce").dropna()
    rows = [
        {
            "metric": "rest_days",
            "count": int(rest.size),
            "mean": rest.mean(),
            "median": rest.median(),
            "std": rest.std(),
            "min": rest.min(),
            "max": rest.max(),
        }
    ]
    summary = pd.DataFrame(rows)

    flags = [
        column
        for column in ("short_week", "extra_rest", "bye_week_rest", "week_1_flag")
        if column in team_games.columns
    ]
    shares = pd.DataFrame(
        {
            "metric": flags,
            "count": [int(team_games[column].astype(bool).sum()) for column in flags],
            "share": [
                float(team_games[column].astype(bool).mean()) for column in flags
            ],
        }
    )
    return pd.concat([summary, shares], ignore_index=True)
