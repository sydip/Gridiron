"""Trivial against-the-spread strategies to measure a model against.

A model is only worth something if it beats the strategies that cost nothing to
invent. These are those strategies. Each one takes the game-level frame and
returns a predicted ``home_cover`` -- 1 when it expects the home side to cover,
0 when it expects the away side -- aligned to the frame's index.

Every baseline scores **every eligible game**. A baseline that quietly declined
the games it found hard would be compared against the others on a different
sample, so where a rule is genuinely undefined the fallback is stated here
rather than left to chance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HOME = 1
AWAY = 0

DEFAULT_SEED = 42

# The feature the EPA baseline reads. Named here so the eligibility helper and
# the baseline cannot drift apart.
EPA_COLUMN = "off_epa_diff_last_5"
SPREAD_COLUMN = "spread_line"
TARGET_COLUMN = "home_cover"


class BaselineError(ValueError):
    """Raised when a frame cannot support a baseline strategy."""


def _require(frame: pd.DataFrame, columns: set[str], strategy: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise BaselineError(
            f"Cannot run {strategy}; missing column(s): " + ", ".join(missing)
        )


def eligible_games(
    matchups: pd.DataFrame,
    *,
    require_epa: bool = False,
) -> pd.DataFrame:
    """The games every strategy is scored on.

    A game is eligible when it carries a settled against-the-spread result.
    ``home_cover`` is null for a push, for an unplayed game, and for a game
    without a line, so this single condition excludes all three -- and excludes
    them identically for every baseline and for the model, which is the only way
    the comparison means anything.

    ``require_epa`` additionally demands a rolling EPA differential. That is not
    the default: the model imputes early-season features rather than dropping
    those games, so dropping them here would score the baselines on a different
    sample than the model. It exists so the comparison can be re-run on
    complete cases and the effect of the fallback can be seen.
    """
    _require(matchups, {TARGET_COLUMN}, "eligibility")

    eligible = matchups.loc[matchups[TARGET_COLUMN].notna()].copy()
    if require_epa and EPA_COLUMN in eligible.columns:
        eligible = eligible.loc[eligible[EPA_COLUMN].notna()]
    return eligible


def _constant(frame: pd.DataFrame, value: int) -> pd.Series:
    return pd.Series(value, index=frame.index, dtype="int8", name="prediction")


def predict_home_every_game(df: pd.DataFrame) -> pd.Series:
    """Always back the home side against the spread."""
    return _constant(df, HOME)


def predict_away_every_game(df: pd.DataFrame) -> pd.Series:
    """Always back the away side against the spread."""
    return _constant(df, AWAY)


def predict_favorite(df: pd.DataFrame) -> pd.Series:
    """Always back the favourite.

    A positive spread means the home side is favoured, following the nflverse
    convention used throughout this project.

    A pick'em game has no favourite. Rather than drop those games -- which would
    score this baseline on a smaller sample than the others -- they resolve to
    the home side, which is the conventional edge when the market sees two equal
    teams. This affects 4 of 2,574 settled games in the current data, and the
    same tiebreak is applied in reverse by :func:`predict_underdog`.
    """
    _require(df, {SPREAD_COLUMN}, "predict_favorite")
    spread = pd.to_numeric(df[SPREAD_COLUMN], errors="coerce")
    return pd.Series(
        np.where(spread.lt(0), AWAY, HOME), index=df.index, dtype="int8"
    ).rename("prediction")


def predict_underdog(df: pd.DataFrame) -> pd.Series:
    """Always back the underdog: the exact inverse of :func:`predict_favorite`."""
    favourite = predict_favorite(df)
    return (1 - favourite).astype("int8").rename("prediction")


def predict_better_epa(df: pd.DataFrame, *, fallback: int = HOME) -> pd.Series:
    """Back whichever side brought the better rolling offensive EPA.

    The differential is home minus away, so a positive value favours the home
    side. An exact tie resolves to ``fallback``.

    Where the differential is missing -- season openers, which have no
    current-season history -- there is no EPA signal to act on, so this also
    falls back. The fallback is the home side by default, matching the
    convention the other baselines use, and it is recorded rather than hidden:
    :func:`epa_fallback_count` reports how many games it decided.
    """
    _require(df, {EPA_COLUMN}, "predict_better_epa")
    differential = pd.to_numeric(df[EPA_COLUMN], errors="coerce")
    predictions = np.where(
        differential.isna(),
        fallback,
        np.where(
            differential.gt(0), HOME, np.where(differential.lt(0), AWAY, fallback)
        ),
    )
    return pd.Series(predictions, index=df.index, dtype="int8", name="prediction")


def epa_fallback_count(df: pd.DataFrame) -> int:
    """How many games :func:`predict_better_epa` decided without an EPA signal."""
    if EPA_COLUMN not in df.columns:
        return len(df)
    return int(pd.to_numeric(df[EPA_COLUMN], errors="coerce").isna().sum())


def predict_random(df: pd.DataFrame, seed: int = DEFAULT_SEED) -> pd.Series:
    """Pick a side at random, reproducibly.

    The generator is seeded per call and draws in the frame's row order, so the
    same frame and seed always give the same picks -- across processes and
    machines, since ``default_rng`` is not affected by global state.
    """
    generator = np.random.default_rng(seed)
    draws = generator.integers(0, 2, size=len(df))
    return pd.Series(draws, index=df.index, dtype="int8", name="prediction")


BASELINES = {
    "home_every_game": predict_home_every_game,
    "away_every_game": predict_away_every_game,
    "favorite": predict_favorite,
    "underdog": predict_underdog,
    "better_epa": predict_better_epa,
    "random": predict_random,
}
