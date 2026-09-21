"""Leakage-free rolling team form features.

Every feature here describes what a team had done *before* the game it is
attached to. The order of operations is what guarantees that:

1. sort each team's games chronologically,
2. shift by one game, so the current game is excluded,
3. take the rolling or expanding mean of what remains,
4. attach the result to the upcoming game.

Shifting before the window is the whole point. ``s.rolling(5).mean()`` includes
the current game and leaks the outcome into its own predictor;
``s.shift(1).rolling(5).mean()`` does not.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from gridiron.data.team_names import normalize_team_names

# One prior game is enough to produce a value. Week 1 has none and is left null
# for the imputer rather than being filled here, so that the absence stays
# visible to anything downstream.
MIN_PERIODS = 1

SHORT_WINDOW = 3
LONG_WINDOW = 5

# Weeks with little or no current-season history behind them.
EARLY_SEASON_WEEKS = 4

# Season is part of the grouping key, so every rolling and season-to-date value
# restarts in week 1. Last year's closing form never reaches this year.
GROUP_KEYS = ["team", "season"]
SORT_KEYS = ["team", "season", "gameday", "week", "game_id"]

MERGE_KEYS = ["season", "week", "game_id", "team"]

OFFENSIVE_EPA_SOURCE = "offensive_epa_per_play"
DEFENSIVE_EPA_SOURCE = "defensive_epa_strength_per_play"
PACE_SOURCE = "pace_plays_per_game"

CONTEXT_COLUMNS = [
    "week_number",
    "is_early_season",
    "prior_games_this_season",
]


@dataclass(frozen=True)
class RollingSpec:
    """One rolling feature: where it comes from and how far back it looks.

    ``window`` of ``None`` means season-to-date, which expands from the start of
    the season instead of holding a fixed length.
    """

    output: str
    source: str
    window: int | None

    @property
    def description(self) -> str:
        span = "season to date" if self.window is None else f"last {self.window}"
        return f"{self.source} ({span}, excluding the current game)"


ROLLING_SPECS: tuple[RollingSpec, ...] = (
    RollingSpec("off_epa_last_3", OFFENSIVE_EPA_SOURCE, SHORT_WINDOW),
    RollingSpec("off_epa_last_5", OFFENSIVE_EPA_SOURCE, LONG_WINDOW),
    RollingSpec("off_epa_season", OFFENSIVE_EPA_SOURCE, None),
    RollingSpec("def_epa_strength_last_3", DEFENSIVE_EPA_SOURCE, SHORT_WINDOW),
    RollingSpec("def_epa_strength_last_5", DEFENSIVE_EPA_SOURCE, LONG_WINDOW),
    RollingSpec("def_epa_strength_season", DEFENSIVE_EPA_SOURCE, None),
    RollingSpec("pace_last_3", PACE_SOURCE, SHORT_WINDOW),
    RollingSpec("pace_last_5", PACE_SOURCE, LONG_WINDOW),
    RollingSpec("pace_season", PACE_SOURCE, None),
    RollingSpec("point_margin_last_3", "point_margin", SHORT_WINDOW),
    RollingSpec("point_margin_last_5", "point_margin", LONG_WINDOW),
    RollingSpec("win_pct_last_5", "win_value", LONG_WINDOW),
    RollingSpec("ats_margin_last_5", "ats_margin", LONG_WINDOW),
    RollingSpec("cover_rate_last_5", "covered_value", LONG_WINDOW),
)

ROLLING_FEATURES = tuple(spec.output for spec in ROLLING_SPECS)

# Derived per-game inputs that no upstream module owns.
DERIVED_SOURCES = ("win_value", "ats_margin", "covered_value")


class RollingFeatureError(ValueError):
    """Raised when a team-game frame cannot produce rolling features."""


def _require_columns(frame: pd.DataFrame, columns: set[str], purpose: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise RollingFeatureError(
            f"Cannot {purpose}; missing required column(s): " + ", ".join(missing)
        )


def _normalized(frame: pd.DataFrame, *columns: str) -> pd.DataFrame:
    """Return a copy with the given team columns mapped to canonical codes.

    Play-by-play and schedule sources do not agree on historical franchise
    codes -- nflverse play-by-play still calls the Rams ``LA`` while the cleaned
    schedule uses ``LAR``. Normalising both sides before the join keeps those
    team-games from silently losing their EPA and pace.
    """
    result = frame.copy()
    for column in columns:
        if column in result.columns:
            result[column] = normalize_team_names(result[column])
    return result


def assemble_team_game_inputs(
    history: pd.DataFrame,
    offense: pd.DataFrame,
    defense: pd.DataFrame,
    pace: pd.DataFrame,
) -> pd.DataFrame:
    """Join the per-game measurements onto the team-game history.

    ``history`` is the chronological spine; the EPA and pace aggregations supply
    the per-game values that the rolling windows average. Team codes are
    normalised on both sides first. The join is a left join on the history, so
    a game without play-by-play keeps its row and carries nulls.
    """
    _require_columns(
        history, {"season", "week", "game_id", "team", "gameday"}, "assemble inputs"
    )

    frame = _normalized(history, "team", "opponent")
    sources = (
        (offense, OFFENSIVE_EPA_SOURCE),
        (defense, DEFENSIVE_EPA_SOURCE),
        (pace, PACE_SOURCE),
    )
    for source_frame, column in sources:
        _require_columns(source_frame, {*MERGE_KEYS, column}, f"join {column}")
        side = _normalized(source_frame, "team")[[*MERGE_KEYS, column]]
        frame = frame.merge(side, on=MERGE_KEYS, how="left", validate="one_to_one")

    return frame


def add_derived_sources(team_games: pd.DataFrame) -> pd.DataFrame:
    """Add the per-game quantities the rolling windows average.

    ``win_value`` scores a tie as half a win, which is how win percentage has
    always treated one. ``ats_margin`` is the team's margin relative to its own
    spread, positive when it beat the number. ``covered_value`` is the cover
    indicator as a float, with pushes left null so they neither raise nor lower
    a cover rate.
    """
    _require_columns(
        team_games, {"point_margin", "team_spread"}, "derive rolling inputs"
    )

    frame = team_games.copy()
    margin = pd.to_numeric(frame["point_margin"], errors="coerce")

    frame["win_value"] = (
        pd.Series(pd.NA, index=frame.index, dtype="Float64")
        .mask(margin.gt(0), 1.0)
        .mask(margin.lt(0), 0.0)
        .mask(margin.eq(0), 0.5)
    )
    frame["win_value"] = frame["win_value"].astype("float64")

    frame["ats_margin"] = margin - pd.to_numeric(frame["team_spread"], errors="coerce")
    frame["covered_value"] = (
        pd.to_numeric(frame["covered"], errors="coerce").astype("float64")
        if "covered" in frame.columns
        else pd.Series(float("nan"), index=frame.index)
    )
    return frame


def _prior_mean(series: pd.Series, window: int | None) -> pd.Series:
    """Mean of the values strictly before each row.

    The ``shift(1)`` is what excludes the current game. Without it the window
    would include the row it is describing, and the feature would leak.
    """
    shifted = series.shift(1)
    if window is None:
        return shifted.expanding(min_periods=MIN_PERIODS).mean()
    return shifted.rolling(window, min_periods=MIN_PERIODS).mean()


def add_rolling_features(
    team_games: pd.DataFrame,
    specs: tuple[RollingSpec, ...] = ROLLING_SPECS,
) -> pd.DataFrame:
    """Attach leakage-free rolling form to each team-game.

    Rows are sorted chronologically within a team and season, then every feature
    is built as ``shift(1)`` followed by a rolling or expanding mean. Because
    ``season`` is part of the grouping key, both the fixed windows and the
    season-to-date values reset in week 1 -- last season's closing form cannot
    reach this season.

    Week 1 has no prior game, so its rolling features are null by construction.
    They are deliberately left that way for an imputer to handle;
    ``prior_games_this_season`` records how much history stands behind each row
    so a caller can tell a thin value from a full one.
    """
    required = {"team", "season", "gameday", "week", "game_id"}
    _require_columns(team_games, required, "add rolling features")

    frame = team_games.copy()
    frame["gameday"] = pd.to_datetime(frame["gameday"], errors="coerce")
    if not set(DERIVED_SOURCES).issubset(frame.columns):
        frame = add_derived_sources(frame)

    frame = frame.sort_values(SORT_KEYS, kind="stable").reset_index(drop=True)

    missing_sources = sorted({spec.source for spec in specs}.difference(frame.columns))
    if missing_sources:
        raise RollingFeatureError(
            "Cannot add rolling features; missing source column(s): "
            + ", ".join(missing_sources)
        )

    grouped = frame.groupby(GROUP_KEYS, sort=False, dropna=False)
    for spec in specs:
        frame[spec.output] = grouped[spec.source].transform(
            lambda series, window=spec.window: _prior_mean(series, window)
        )

    frame["week_number"] = frame["week"]
    frame["is_early_season"] = frame["week"].le(EARLY_SEASON_WEEKS)
    frame["prior_games_this_season"] = grouped.cumcount()
    return frame


def build_rolling_features(
    history: pd.DataFrame,
    offense: pd.DataFrame,
    defense: pd.DataFrame,
    pace: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the inputs and attach rolling features in one step."""
    inputs = assemble_team_game_inputs(history, offense, defense, pace)
    return add_rolling_features(add_derived_sources(inputs))


def rolling_feature_coverage(features: pd.DataFrame) -> pd.DataFrame:
    """Report how populated each rolling feature is, overall and in week 1.

    Week 1 nulls are expected, not a fault, so they are reported separately from
    nulls elsewhere -- the latter point at a genuine gap in the source data.
    """
    _require_columns(features, {"week"}, "report coverage")

    week_one = features["week"].eq(1)
    rows = []
    for spec in ROLLING_SPECS:
        if spec.output not in features.columns:
            continue
        values = features[spec.output]
        rows.append(
            {
                "feature": spec.output,
                "source": spec.source,
                "window": "season" if spec.window is None else spec.window,
                "present": int(values.notna().sum()),
                "missing": int(values.isna().sum()),
                "missing_week_1": int((values.isna() & week_one).sum()),
                "missing_after_week_1": int((values.isna() & ~week_one).sum()),
            }
        )
    return pd.DataFrame(rows)
