"""Season-based walk-forward splits.

The question a backtest has to answer is "what would this have done on a season
it had never seen?" — so each fold trains on every season before a target season
and validates on that season alone. The training window expands forward one
season at a time, which is how the model would actually have been used: at the
start of 2022 you have 2016 through 2021 and nothing else.

A single random split cannot answer that question, and neither can one held-out
season: one season is 250-odd near-coinflip games, which is far too few to tell
a real edge from a lucky year. Seven folds give seven independent-ish readings
and, more usefully, a spread.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

import pandas as pd

LOGGER = logging.getLogger(__name__)

FIRST_VALIDATION_SEASON = 2019
FINAL_VALIDATION_SEASON = 2025

# A fold with too few training rows produces a model that says nothing useful.
MIN_TRAIN_ROWS = 200


class SplitError(ValueError):
    """Raised when walk-forward splits cannot be built from a frame."""


@dataclass(frozen=True)
class SeasonSplit:
    """One fold: the seasons trained on, and the single season validated."""

    train_indices: pd.Index
    validation_indices: pd.Index
    train_seasons: tuple[int, ...]
    validation_season: int

    @property
    def train_size(self) -> int:
        return len(self.train_indices)

    @property
    def validation_size(self) -> int:
        return len(self.validation_indices)

    def __repr__(self) -> str:
        span = (
            f"{self.train_seasons[0]}-{self.train_seasons[-1]}"
            if self.train_seasons
            else "none"
        )
        return (
            f"SeasonSplit(train={span} [{self.train_size}], "
            f"validate={self.validation_season} [{self.validation_size}])"
        )


def _check_chronology(frame: pd.DataFrame, split: SeasonSplit) -> None:
    """Confirm no training game was played after any validation game.

    Season boundaries make this true in practice -- a season's last game falls
    in early January, the next season's first in September -- but it is the
    property the whole backtest rests on, so it is checked rather than assumed.
    """
    if "gameday" not in frame.columns:
        return

    gamedays = pd.to_datetime(frame["gameday"], errors="coerce")
    latest_train = gamedays.loc[split.train_indices].max()
    earliest_validation = gamedays.loc[split.validation_indices].min()

    if (
        pd.notna(latest_train)
        and pd.notna(earliest_validation)
        and latest_train >= earliest_validation
    ):
        raise SplitError(
            f"Fold validating {split.validation_season} has a training game "
            f"on {latest_train.date()} at or after its first validation "
            f"game on {earliest_validation.date()}."
        )


def season_walk_forward_splits(
    df: pd.DataFrame,
    first_validation_season: int = FIRST_VALIDATION_SEASON,
    final_validation_season: int = FINAL_VALIDATION_SEASON,
) -> Iterator[SeasonSplit]:
    """Yield one expanding-window fold per validation season.

    Fold *n* trains on every season strictly before its validation season and
    validates on that season alone, so a season can never appear on both sides
    of one fold, and the training set grows by one season per fold.

    Seasons that are requested but absent from ``df`` are skipped with a log
    line rather than yielding an empty fold, and a season with too little
    history behind it is skipped for the same reason.
    """
    if "season" not in df.columns:
        raise SplitError("Cannot build splits; missing 'season'.")
    if first_validation_season > final_validation_season:
        raise SplitError(
            f"first_validation_season ({first_validation_season}) is after "
            f"final_validation_season ({final_validation_season})."
        )

    seasons = pd.to_numeric(df["season"], errors="coerce")
    available = sorted({int(season) for season in seasons.dropna().unique()})
    if not available:
        raise SplitError("Cannot build splits; no seasons present.")

    yielded = 0
    for validation_season in range(
        first_validation_season, final_validation_season + 1
    ):
        if validation_season not in available:
            LOGGER.info("Skipping %d; no rows for that season.", validation_season)
            continue

        train_seasons = tuple(
            season for season in available if season < validation_season
        )
        if not train_seasons:
            LOGGER.info(
                "Skipping %d; no earlier season to train on.", validation_season
            )
            continue

        train_mask = seasons.isin(train_seasons)
        validation_mask = seasons.eq(validation_season)

        train_indices = df.index[train_mask]
        if len(train_indices) < MIN_TRAIN_ROWS:
            LOGGER.info(
                "Skipping %d; only %d training rows available.",
                validation_season,
                len(train_indices),
            )
            continue

        split = SeasonSplit(
            train_indices=train_indices,
            validation_indices=df.index[validation_mask],
            train_seasons=train_seasons,
            validation_season=validation_season,
        )
        _check_chronology(df, split)
        yielded += 1
        yield split

    if not yielded:
        raise SplitError(
            f"No usable folds between {first_validation_season} and "
            f"{final_validation_season}; available seasons are {available}."
        )


def describe_splits(splits: list[SeasonSplit]) -> pd.DataFrame:
    """Summarise the folds, so the expanding window is visible at a glance."""
    return pd.DataFrame(
        [
            {
                "validation_season": split.validation_season,
                "train_seasons": (
                    f"{split.train_seasons[0]}-{split.train_seasons[-1]}"
                    if split.train_seasons
                    else ""
                ),
                "train_season_count": len(split.train_seasons),
                "train_games": split.train_size,
                "validation_games": split.validation_size,
            }
            for split in splits
        ]
    )
