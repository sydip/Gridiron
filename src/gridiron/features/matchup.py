"""Game-level matchup records built from both teams' pregame features.

Phase 9 produces one row per team per game. A model needs one row per *game*,
carrying what each side brought into it. This module joins the home and away
views onto a single row, prefixes them, and reduces the pairs to differentials.

Every column here is knowable before kickoff. The one thing that is not -- the
outcome -- is kept in clearly named target columns, and
:func:`assert_no_postgame_fields` exists to make that separation enforceable
rather than merely intended.
"""

from __future__ import annotations

import pandas as pd

from gridiron.data.team_names import TEAM_DIVISION, normalize_team_names
from gridiron.features.rolling import ROLLING_FEATURES

HOME_PREFIX = "home_"
AWAY_PREFIX = "away_"

GAME_KEYS = ["season", "week", "game_id"]

# Per-team columns carried onto the game row from each side's history.
SIDE_COLUMNS = (
    *ROLLING_FEATURES,
    "rest_days",
    "short_week",
    "extra_rest",
    "bye_week_rest",
    "week_1_flag",
    "prior_games_this_season",
)

# Each differential is home minus away, without exception.
DIFFERENTIAL_SOURCES = (
    ("off_epa_diff_last_5", "off_epa_last_5"),
    ("def_epa_strength_diff_last_5", "def_epa_strength_last_5"),
    ("pace_diff_last_5", "pace_last_5"),
    ("point_margin_diff_last_5", "point_margin_last_5"),
    ("win_pct_diff_last_5", "win_pct_last_5"),
    ("off_epa_diff_last_3", "off_epa_last_3"),
    ("def_epa_strength_diff_last_3", "def_epa_strength_last_3"),
    ("pace_diff_last_3", "pace_last_3"),
    ("point_margin_diff_last_3", "point_margin_last_3"),
    ("off_epa_diff_season", "off_epa_season"),
    ("def_epa_strength_diff_season", "def_epa_strength_season"),
    ("pace_diff_season", "pace_season"),
    ("ats_margin_diff_last_5", "ats_margin_last_5"),
    ("cover_rate_diff_last_5", "cover_rate_last_5"),
    ("rest_diff", "rest_days"),
)

IDENTIFIER_COLUMNS = [
    "game_id",
    "season",
    "week",
    "gameday",
    "home_team",
    "away_team",
]

# The feature matrix. Everything here is available before kickoff.
FEATURE_COLUMNS = [
    "spread_line",
    "off_epa_diff_last_5",
    "def_epa_strength_diff_last_5",
    "pace_diff_last_5",
    "rest_diff",
    "point_margin_diff_last_5",
    "win_pct_diff_last_5",
    "home_short_week",
    "away_short_week",
    "div_game",
    "week_1_flag",
]

TARGET_COLUMNS = ["home_cover", "is_push"]

MODEL_ROW_COLUMNS = [*IDENTIFIER_COLUMNS, *FEATURE_COLUMNS, *TARGET_COLUMNS]

# Fields that are only knowable after kickoff. Matched by exact name, never by
# substring: ``point_margin_diff_last_5`` is a pregame rolling feature and must
# not be caught by a rule aimed at ``point_margin``.
POSTGAME_COLUMNS = frozenset(
    {
        "home_score",
        "away_score",
        "result",
        "total",
        "overtime",
        "home_margin",
        "adjusted_home_margin",
        "ats_result",
        "point_margin",
        "points_for",
        "points_against",
        "covered",
        "covered_value",
        "win_value",
        "ats_margin",
        "offensive_epa_per_play",
        "defensive_epa_strength_per_play",
        "pace_plays_per_game",
        "offensive_plays",
        "defensive_plays",
        "is_completed",
        *(
            f"{prefix}{column}"
            for prefix in (HOME_PREFIX, AWAY_PREFIX)
            for column in (
                "point_margin",
                "points_for",
                "points_against",
                "covered",
                "covered_value",
                "win_value",
                "ats_margin",
                "offensive_epa_per_play",
                "defensive_epa_strength_per_play",
                "pace_plays_per_game",
                "is_completed",
            )
        ),
    }
)


class MatchupError(ValueError):
    """Raised when team-game features cannot be assembled into matchups."""


def _require_columns(frame: pd.DataFrame, columns: set[str], purpose: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise MatchupError(
            f"Cannot {purpose}; missing required column(s): " + ", ".join(missing)
        )


def _side(features: pd.DataFrame, *, home: bool) -> pd.DataFrame:
    """One side's columns, prefixed and keyed for the join.

    The team column is renamed to ``home_team`` or ``away_team`` so the merge
    asserts the identity of the side it is attaching, rather than trusting row
    order or an ``is_home`` flag.
    """
    prefix = HOME_PREFIX if home else AWAY_PREFIX
    team_key = f"{prefix}team"

    available = [column for column in SIDE_COLUMNS if column in features.columns]
    side = features[[*GAME_KEYS, "team", *available]].copy()
    side["team"] = normalize_team_names(side["team"])
    return side.rename(
        columns={
            "team": team_key,
            **{column: f"{prefix}{column}" for column in available},
        }
    )


def build_matchups(
    features: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Join both teams' pregame features onto one row per game.

    ``schedule`` is the game-level spine, supplying the identifiers, the spread,
    and which team was at home. ``features`` is the Phase 9 team-game frame. The
    home side is joined on ``(season, week, game_id, home_team)`` and the away
    side on ``(season, week, game_id, away_team)``, so a row can only be built
    when both sides are present and their team codes agree with the schedule.

    Differentials are all home minus away. ``div_game`` is derived from the two
    teams' divisions, and ``week_1_flag`` is set when *either* team is playing
    its season opener -- that is the condition under which a matchup has a side
    with no current-season history, which is what the flag exists to warn about.
    """
    _require_columns(
        schedule,
        {
            "game_id",
            "season",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "spread_line",
        },
        "build matchups",
    )
    _require_columns(features, {*GAME_KEYS, "team"}, "build matchups")

    spine = schedule[
        [
            "game_id",
            "season",
            "week",
            "gameday",
            "home_team",
            "away_team",
            "spread_line",
            *[
                column
                for column in ("home_score", "away_score", "is_completed")
                if column in schedule.columns
            ],
        ]
    ].copy()
    spine["home_team"] = normalize_team_names(spine["home_team"])
    spine["away_team"] = normalize_team_names(spine["away_team"])

    matchups = spine.merge(
        _side(features, home=True),
        on=[*GAME_KEYS, "home_team"],
        how="left",
        validate="one_to_one",
    ).merge(
        _side(features, home=False),
        on=[*GAME_KEYS, "away_team"],
        how="left",
        validate="one_to_one",
    )

    for output, source in DIFFERENTIAL_SOURCES:
        home_column = f"{HOME_PREFIX}{source}"
        away_column = f"{AWAY_PREFIX}{source}"
        if home_column in matchups.columns and away_column in matchups.columns:
            matchups[output] = pd.to_numeric(
                matchups[home_column], errors="coerce"
            ) - pd.to_numeric(matchups[away_column], errors="coerce")

    matchups["div_game"] = (
        matchups["home_team"].map(TEAM_DIVISION)
        == matchups["away_team"].map(TEAM_DIVISION)
    ).astype("int8")

    openers = [
        matchups[column].fillna(0).astype(int)
        for column in (f"{HOME_PREFIX}week_1_flag", f"{AWAY_PREFIX}week_1_flag")
        if column in matchups.columns
    ]
    matchups["week_1_flag"] = (
        pd.concat(openers, axis=1).max(axis=1).astype("int8")
        if openers
        else pd.Series(0, index=matchups.index, dtype="int8")
    )

    for prefix in (HOME_PREFIX, AWAY_PREFIX):
        column = f"{prefix}short_week"
        if column in matchups.columns:
            matchups[column] = matchups[column].fillna(False).astype(bool)

    return matchups.sort_values(["season", "week", "gameday", "game_id"]).reset_index(
        drop=True
    )


def add_matchup_target(matchups: pd.DataFrame) -> pd.DataFrame:
    """Attach ``home_cover`` and ``is_push`` from the realised result.

    These are the labels, not features. A push is null in ``home_cover`` and
    true in ``is_push``; a game with no score or no line is null in both, so an
    unplayed 2026 fixture carries features and no target.
    """
    from gridiron.features.target import add_ats_target

    _require_columns(
        matchups, {"home_score", "away_score", "spread_line"}, "attach the target"
    )

    labelled = add_ats_target(matchups)
    labelled["is_push"] = labelled["is_push"].fillna(False).astype(bool)
    return labelled


def model_rows(matchups: pd.DataFrame) -> pd.DataFrame:
    """Reduce the wide matchup frame to the model row.

    Any expected column that was not produced is reported rather than silently
    omitted, so a missing feature cannot slip through as an absent column.
    """
    missing = [column for column in MODEL_ROW_COLUMNS if column not in matchups.columns]
    if missing:
        raise MatchupError(
            "Cannot build model rows; missing column(s): " + ", ".join(missing)
        )
    return matchups[MODEL_ROW_COLUMNS].copy()


def assert_no_postgame_fields(
    frame: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> None:
    """Raise if any postgame field has reached the feature matrix.

    Checks exact names against :data:`POSTGAME_COLUMNS`, and separately that no
    target column is being passed off as a feature.
    """
    columns = feature_columns or FEATURE_COLUMNS
    present = [column for column in columns if column in frame.columns]

    leaked = sorted(set(present).intersection(POSTGAME_COLUMNS))
    if leaked:
        raise MatchupError(
            "Postgame field(s) present in the feature matrix: " + ", ".join(leaked)
        )

    targets = sorted(set(present).intersection(TARGET_COLUMNS))
    if targets:
        raise MatchupError(
            "Target column(s) present in the feature matrix: " + ", ".join(targets)
        )


def build_model_table(
    features: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble matchups, attach the target, and return the model row."""
    matchups = add_matchup_target(build_matchups(features, schedule))
    rows = model_rows(matchups)
    assert_no_postgame_fields(rows)
    return rows


def missing_value_report(
    matchups: pd.DataFrame,
    feature_columns: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Missing-value percentages by feature, season, week, and team.

    Returns four frames under the keys ``feature``, ``season``, ``week`` and
    ``team``. Early-season rows are expected to be sparse and are reported, not
    dropped: the point of this report is to make that sparsity visible before
    anything decides to discard it.
    """
    columns = [
        column
        for column in (feature_columns or FEATURE_COLUMNS)
        if column in matchups.columns
    ]
    if not columns:
        raise MatchupError("No feature columns present to report on.")

    missing = matchups[columns].isna()

    by_feature = (
        pd.DataFrame(
            {
                "feature": columns,
                "missing": [int(missing[column].sum()) for column in columns],
                "rows": len(matchups),
            }
        )
        .assign(missing_pct=lambda frame: 100.0 * frame["missing"] / frame["rows"])
        .sort_values("missing_pct", ascending=False)
        .reset_index(drop=True)
    )

    def _grouped(key: pd.Series, name: str) -> pd.DataFrame:
        report = (
            missing.groupby(key)
            .mean()
            .mul(100.0)
            .round(3)
            .reset_index()
            .rename(columns={key.name: name})
        )
        report.insert(1, "rows", matchups.groupby(key).size().to_numpy())
        report.insert(
            2,
            "any_missing_pct",
            missing.any(axis=1).groupby(key).mean().mul(100.0).round(3).to_numpy(),
        )
        return report

    teams = pd.concat(
        [
            matchups[["home_team", *columns]].rename(columns={"home_team": "team"}),
            matchups[["away_team", *columns]].rename(columns={"away_team": "team"}),
        ],
        ignore_index=True,
    )
    team_missing = teams[columns].isna()
    by_team = (
        team_missing.groupby(teams["team"])
        .mean()
        .mul(100.0)
        .round(3)
        .reset_index()
        .rename(columns={"team": "team"})
    )
    by_team.insert(1, "rows", teams.groupby("team").size().to_numpy())
    by_team.insert(
        2,
        "any_missing_pct",
        team_missing.any(axis=1)
        .groupby(teams["team"])
        .mean()
        .mul(100.0)
        .round(3)
        .to_numpy(),
    )

    return {
        "feature": by_feature,
        "season": _grouped(matchups["season"], "season"),
        "week": _grouped(matchups["week"], "week"),
        "team": by_team,
    }
