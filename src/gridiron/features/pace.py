"""Team-game offensive pace from play-by-play data.

Pace answers how many snaps an offense ran and how quickly it ran them. The
baseline statistic is a volume count; the optional timing statistics describe
tempo. Both are built from the same eligible-play set that the EPA aggregation
uses, so a team-game's pace play count and its offensive EPA play count are
guaranteed to agree.
"""

from __future__ import annotations

import pandas as pd

from gridiron.features.epa import (
    MIN_OFFENSIVE_PLAYS,
    OffensiveEpaError,
    filter_offensive_plays,
)

MIN_PACE_PLAYS = MIN_OFFENSIVE_PLAYS

# The play clock is 40 seconds, so a genuine snap-to-snap interval cannot run
# much past the low 50s once the previous play's duration is included. A longer
# gap means the game clock stopped -- timeout, injury, review, two-minute
# warning, change of quarter -- and would measure the stoppage, not the tempo.
MAX_SNAP_INTERVAL_SECONDS = 60.0

# Neutral tempo excludes the situations where the score dictates the clock: a
# margin inside one possession, and outside the closing two minutes of a half.
NEUTRAL_SCORE_MARGIN = 8.0
NEUTRAL_MIN_HALF_SECONDS = 120.0

# Below this many qualifying intervals the neutral mean is noise: observed
# dispersion roughly doubles under ten intervals compared with ten or more.
MIN_NEUTRAL_INTERVALS = 10

TIMING_COLUMNS = frozenset(
    {"game_seconds_remaining", "half_seconds_remaining", "fixed_drive", "play_id"}
)
NEUTRAL_COLUMNS = frozenset({"score_differential", "half_seconds_remaining"})
NO_HUDDLE_COLUMN = "no_huddle"

BASELINE_COLUMNS = [
    "season",
    "week",
    "game_id",
    "team",
    "opponent",
    "offensive_play_count",
    "pace_plays_per_game",
    "valid_pace",
]

TIMING_OUTPUT_COLUMNS = [
    "seconds_per_play",
    "pace_intervals",
    "neutral_seconds_per_play",
    "neutral_pace_intervals",
    "valid_neutral_pace",
    "no_huddle_rate",
]

DISTRIBUTION_STATISTICS = [
    "count",
    "missing",
    "mean",
    "median",
    "std",
    "min",
    "p01",
    "p99",
    "max",
]

# Bounds for the outlier report. These are review triggers, not filters: a row
# outside them is surfaced for inspection and always retained. They are set from
# what football allows rather than from the observed range, so they stay a real
# test -- the single-game record for offensive plays by one team is around 100,
# and a snap-to-snap mean outside 18-45 seconds implies a clock or ordering
# fault rather than an unusually fast or slow offense.
PLAUSIBLE_PLAY_COUNT = (30, 100)
PLAUSIBLE_SECONDS_PER_PLAY = (18.0, 45.0)


class PaceError(ValueError):
    """Raised when play-by-play data cannot produce team-game pace."""


def timing_fields_available(pbp: pd.DataFrame) -> bool:
    """Report whether the optional timing statistics can be computed.

    The advanced statistics are gated on this rather than assumed: a source
    without usable clock fields still produces the baseline play count instead
    of failing or, worse, emitting a fabricated tempo.
    """
    if not TIMING_COLUMNS.issubset(pbp.columns):
        return False
    return all(
        pd.to_numeric(pbp[column], errors="coerce").notna().any()
        for column in sorted(TIMING_COLUMNS)
    )


def _snap_intervals(plays: pd.DataFrame) -> pd.Series:
    """Return seconds elapsed between consecutive snaps of the same drive.

    Plays are ordered within a drive by ``play_id`` and differenced on the game
    clock. The first snap of each drive has no predecessor, and intervals that
    are non-positive (clock artefacts in the source) or longer than
    :data:`MAX_SNAP_INTERVAL_SECONDS` (the clock stopped) become null, so they
    are excluded from the mean rather than distorting it.
    """
    ordered = plays.sort_values(["game_id", "play_id"])
    previous = ordered.groupby(
        ["game_id", "posteam", "fixed_drive"], sort=False, dropna=False
    )["game_seconds_remaining"].shift(1)
    elapsed = previous - ordered["game_seconds_remaining"]
    usable = elapsed.gt(0) & elapsed.le(MAX_SNAP_INTERVAL_SECONDS)
    return elapsed.where(usable).reindex(plays.index)


def _neutral_flags(plays: pd.DataFrame) -> pd.Series:
    """Flag plays run in a game state that does not itself dictate tempo."""
    margin = pd.to_numeric(plays["score_differential"], errors="coerce").abs()
    half_remaining = pd.to_numeric(plays["half_seconds_remaining"], errors="coerce")
    return margin.le(NEUTRAL_SCORE_MARGIN) & half_remaining.gt(NEUTRAL_MIN_HALF_SECONDS)


def aggregate_pace(
    pbp: pd.DataFrame,
    *,
    min_plays: int = MIN_PACE_PLAYS,
    include_timing: bool = True,
) -> pd.DataFrame:
    """Build one pace row per team per game.

    ``pace_plays_per_game`` is the baseline statistic, defined as the number of
    eligible offensive plays the team ran. It is identical by construction to
    ``offensive_play_count``, which is carried alongside it so the definition
    stays legible at the point of use. Eligibility is delegated wholesale to
    :func:`~gridiron.features.epa.filter_offensive_plays`, so pace counts only
    offensive scrimmage plays and cannot drift from the EPA play set.

    The timing statistics are added only when ``include_timing`` is true *and*
    the clock fields are usable. When they are not, the columns are omitted
    entirely rather than filled with a placeholder.

    ``valid_pace`` is false below ``min_plays``; ``valid_neutral_pace`` is false
    below :data:`MIN_NEUTRAL_INTERVALS` qualifying neutral intervals. Both flag
    rows for review or imputation -- no row is dropped, and a team-game with no
    measurable tempo keeps a null rather than an invented number.
    """
    try:
        plays = filter_offensive_plays(pbp)
    except OffensiveEpaError as error:
        raise PaceError(str(error)) from error

    frame = pd.DataFrame(
        {
            "season": plays["season"],
            "week": plays["week"],
            "game_id": plays["game_id"],
            "team": plays["posteam"],
            "opponent": plays["defteam"],
        }
    )

    with_timing = include_timing and timing_fields_available(pbp)
    has_no_huddle = False
    if with_timing:
        frame["interval"] = _snap_intervals(plays)
        neutral = (
            _neutral_flags(plays)
            if NEUTRAL_COLUMNS.issubset(plays.columns)
            else pd.Series(False, index=plays.index)
        )
        frame["neutral_interval"] = frame["interval"].where(neutral)
        if NO_HUDDLE_COLUMN in plays.columns:
            no_huddle = pd.to_numeric(plays[NO_HUDDLE_COLUMN], errors="coerce")
            has_no_huddle = bool(no_huddle.notna().any())
            if has_no_huddle:
                frame["no_huddle"] = no_huddle.eq(1)

    aggregations: dict[str, tuple[str, str]] = {
        "opponent": ("opponent", "first"),
        "distinct_opponents": ("opponent", "nunique"),
        "offensive_play_count": ("team", "size"),
    }
    if with_timing:
        aggregations |= {
            "seconds_per_play": ("interval", "mean"),
            "pace_intervals": ("interval", "count"),
            "neutral_seconds_per_play": ("neutral_interval", "mean"),
            "neutral_pace_intervals": ("neutral_interval", "count"),
        }
        if has_no_huddle:
            aggregations["no_huddle_rate"] = ("no_huddle", "mean")

    grouped = (
        frame.groupby(["season", "week", "game_id", "team"], dropna=False, sort=True)
        .agg(**aggregations)
        .reset_index()
    )

    ambiguous = grouped.loc[grouped["distinct_opponents"].gt(1), "game_id"]
    if not ambiguous.empty:
        raise PaceError(
            "Cannot aggregate pace; a team faces more than one opponent in "
            f"game(s): {sorted(set(ambiguous))[:5]}"
        )
    grouped = grouped.drop(columns="distinct_opponents")

    grouped["offensive_play_count"] = grouped["offensive_play_count"].astype("int64")
    grouped["pace_plays_per_game"] = grouped["offensive_play_count"]
    grouped["valid_pace"] = grouped["offensive_play_count"].ge(min_plays)

    columns = list(BASELINE_COLUMNS)
    if with_timing:
        grouped["pace_intervals"] = grouped["pace_intervals"].astype("int64")
        grouped["neutral_pace_intervals"] = grouped["neutral_pace_intervals"].astype(
            "int64"
        )
        grouped["valid_neutral_pace"] = (
            grouped["neutral_pace_intervals"].ge(MIN_NEUTRAL_INTERVALS)
            & grouped["neutral_seconds_per_play"].notna()
        )
        columns += [
            column for column in TIMING_OUTPUT_COLUMNS if column in grouped.columns
        ]

    return grouped[columns].reset_index(drop=True)


def pace_distribution_report(
    team_games: pd.DataFrame,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    """Summarise each pace statistic's distribution, one row per statistic.

    ``missing`` sits next to ``count`` so a mostly-null metric is not mistaken
    for a well-populated one, and the 1st and 99th percentiles give the tails
    that :func:`flag_pace_outliers` reports against.
    """
    candidates = columns or ["pace_plays_per_game", *TIMING_OUTPUT_COLUMNS]
    available = [
        column
        for column in candidates
        if column in team_games.columns
        and pd.api.types.is_numeric_dtype(team_games[column])
        and not pd.api.types.is_bool_dtype(team_games[column])
    ]
    if not available:
        raise PaceError("No numeric pace columns found to summarise.")

    rows = []
    for column in available:
        values = pd.to_numeric(team_games[column], errors="coerce")
        present = values.dropna()
        rows.append(
            {
                "statistic": column,
                "count": int(present.size),
                "missing": int(values.isna().sum()),
                "mean": present.mean(),
                "median": present.median(),
                "std": present.std(),
                "min": present.min(),
                "p01": present.quantile(0.01),
                "p99": present.quantile(0.99),
                "max": present.max(),
            }
        )
    return pd.DataFrame(rows, columns=["statistic", *DISTRIBUTION_STATISTICS])


def _append_reason(reasons: pd.Series, mask: pd.Series, reason: str) -> None:
    """Record a review reason against the rows selected by ``mask``, in place."""
    selected = mask.fillna(False)
    if not selected.any():
        return
    reasons.loc[selected] = reasons.loc[selected].apply(lambda items: [*items, reason])


def flag_pace_outliers(team_games: pd.DataFrame) -> pd.DataFrame:
    """Return the team-games worth inspecting, with the reason for each.

    A row is flagged when its play count or seconds per play falls outside the
    plausible bounds, when it sits below the volume threshold, or when its
    neutral tempo rests on too few intervals to carry meaning. Flagging is
    advisory: every input row is preserved in the aggregation itself.
    """
    if "pace_plays_per_game" not in team_games.columns:
        raise PaceError("Expected a pace aggregation with 'pace_plays_per_game'.")

    flagged = team_games.copy()
    reasons = pd.Series(
        [[] for _ in range(len(flagged))], index=flagged.index, dtype=object
    )

    low, high = PLAUSIBLE_PLAY_COUNT
    counts = flagged["pace_plays_per_game"]
    _append_reason(reasons, counts.lt(low), f"play count below {low}")
    _append_reason(reasons, counts.gt(high), f"play count above {high}")
    if "valid_pace" in flagged.columns:
        _append_reason(
            reasons,
            ~flagged["valid_pace"].astype(bool),
            "below minimum play volume",
        )

    if "seconds_per_play" in flagged.columns:
        fast, slow = PLAUSIBLE_SECONDS_PER_PLAY
        seconds = flagged["seconds_per_play"]
        _append_reason(
            reasons,
            seconds.notna() & (seconds.lt(fast) | seconds.gt(slow)),
            f"seconds per play outside {fast}-{slow}",
        )
        _append_reason(reasons, seconds.isna(), "no measurable snap interval")

    if "valid_neutral_pace" in flagged.columns:
        _append_reason(
            reasons,
            ~flagged["valid_neutral_pace"].astype(bool),
            f"fewer than {MIN_NEUTRAL_INTERVALS} neutral intervals",
        )

    flagged["flag_reasons"] = reasons.apply("; ".join)
    return (
        flagged.loc[reasons.str.len().gt(0)]
        .sort_values(["pace_plays_per_game", "game_id", "team"])
        .reset_index(drop=True)
    )
