"""The feature pipeline, from raw parquet to the model table.

This is the one place the pipeline is assembled. Both ``build_features`` (the
workflow command) and ``matchup_report`` (the phase-10 report) call it, so
there is no way for the table the model trains on and the table the report
describes to drift apart.

The order is fixed and matters: team-game history first, because rolling form
is defined over a team's own sequence of games; then the per-play aggregates;
then the shift-and-roll; and only then the merge into one row per game. Every
rolling column is built from games strictly before the one it describes, which
is what makes the merged table safe to train on.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from gridiron.config import PROJECT_ROOT
from gridiron.data.clean import clean_schedules
from gridiron.features.epa import aggregate_defensive_epa, aggregate_offensive_epa
from gridiron.features.matchup import (
    add_matchup_target,
    assert_no_postgame_fields,
    build_matchups,
    missing_value_report,
    model_rows,
)
from gridiron.features.pace import aggregate_pace
from gridiron.features.rolling import build_rolling_features
from gridiron.features.team_games import team_game_history

LOGGER = logging.getLogger(__name__)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

PBP_PATH = RAW_DIR / "pbp_2016_2025.parquet"
SCHEDULE_PATH = RAW_DIR / "schedules_2016_2026.parquet"

MATCHUP_FILENAME = "matchups_wide.csv"
MODEL_TABLE_FILENAME = "model_table.csv"

# The play-by-play columns the aggregations read. Named so the parquet is
# loaded to one shape, and so a schema change surfaces here rather than as a
# KeyError three functions deep.
PBP_COLUMNS = [
    "season",
    "week",
    "game_id",
    "posteam",
    "defteam",
    "play_type",
    "epa",
    "success",
    "pass",
    "rush",
    "qb_kneel",
    "qb_spike",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "no_huddle",
    "fixed_drive",
    "play_id",
    "score_differential",
]


def feature_output_paths(directory: Path | None = None) -> list[Path]:
    """Every file :func:`write_feature_tables` writes, in order.

    Callers use this to warn before overwriting and to decide whether the
    tables are already up to date, so it must stay in step with the writer.
    """
    directory = Path(directory or REPORT_DIR)
    paths = [directory / MATCHUP_FILENAME, directory / MODEL_TABLE_FILENAME]
    paths += [
        directory / f"matchup_missing_by_{name}.csv"
        for name in ("feature", "week", "season", "team")
    ]
    return paths


def load_sources(
    pbp_path: Path | None = None,
    schedule_path: Path | None = None,
    seasons: Sequence[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read and clean the two raw inputs, optionally narrowed to seasons."""
    pbp_path = Path(pbp_path or PBP_PATH)
    schedule_path = Path(schedule_path or SCHEDULE_PATH)

    pbp = pd.read_parquet(pbp_path, columns=PBP_COLUMNS)
    schedule = clean_schedules(pd.read_parquet(schedule_path))

    if seasons:
        wanted = sorted(set(seasons))
        pbp = pbp.loc[pbp["season"].isin(wanted)]
        schedule = schedule.loc[schedule["season"].isin(wanted)]
        LOGGER.info("Filtered to seasons %s", wanted)

    return pbp, schedule


def assemble_matchups(pbp: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    """Run the pipeline and return one row per game, with the target attached."""
    features = build_rolling_features(
        team_game_history(schedule),
        aggregate_offensive_epa(pbp),
        aggregate_defensive_epa(pbp),
        aggregate_pace(pbp),
    )
    return add_matchup_target(build_matchups(features, schedule))


def build_feature_tables(
    pbp: pd.DataFrame, schedule: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """The matchup table, the model rows, and the missing-value breakdowns.

    ``assert_no_postgame_fields`` runs here rather than in the caller so that
    no path can write a model table carrying a field from the game it is meant
    to predict.
    """
    matchups = assemble_matchups(pbp, schedule)
    rows = model_rows(matchups)
    assert_no_postgame_fields(rows)
    return matchups, rows, missing_value_report(matchups)


def write_feature_tables(
    matchups: pd.DataFrame,
    rows: pd.DataFrame,
    report: dict[str, pd.DataFrame],
    directory: Path | None = None,
) -> list[Path]:
    """Write the tables and return the paths, in the documented order."""
    directory = Path(directory or REPORT_DIR)
    directory.mkdir(parents=True, exist_ok=True)

    written = [directory / MATCHUP_FILENAME, directory / MODEL_TABLE_FILENAME]
    matchups.to_csv(written[0], index=False)
    rows.to_csv(written[1], index=False)

    for name, frame in report.items():
        path = directory / f"matchup_missing_by_{name}.csv"
        frame.to_csv(path, index=False)
        written.append(path)

    return written
