"""Download and validate the nflverse source datasets.

Fetches play-by-play, team stats and schedules into ``data/raw`` as Parquet,
then prints the validation summary that says whether what arrived is usable.

Usage::

    python scripts/download_data.py
    python scripts/download_data.py --seasons 2023 2024 --prediction-season 2025
    python scripts/download_data.py --force-refresh
    python -m gridiron.cli download

Files already on disk are reused. This is the one command that moves hundreds
of megabytes over the network, so re-downloading is opt-in through
``--force-refresh`` rather than the default.
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from gridiron.data.download import RAW_DATA_DIR, cache_paths, load_or_download_data
from gridiron.workflow import (
    RunRecord,
    WorkflowError,
    add_common_arguments,
    check_dependencies,
    print_artifacts,
    warn_on_overwrite,
)

LOGGER = logging.getLogger("download_data")

COMMAND = "download"

DEFAULT_HISTORICAL_SEASONS = list(range(2016, 2026))
DEFAULT_PREDICTION_SEASON = 2026


def _season_values(df: pd.DataFrame) -> list[int]:
    return sorted(
        set(pd.to_numeric(df["season"], errors="coerce").dropna().astype(int))
    )


def print_validation(data: dict[str, pd.DataFrame]) -> None:
    """Print the requested validation summary."""
    pbp = data["pbp"]
    schedules = data["schedules"]
    schedule_seasons = _season_values(schedules)

    print(f"PBP seasons loaded: {_season_values(pbp)}")
    print(f"Schedule seasons loaded: {schedule_seasons}")

    regular = schedules.loc[schedules["game_type"].eq("REG")]
    regular_counts = regular.groupby("season", dropna=False).size()
    print("Number of regular-season games per season:")
    for season in schedule_seasons:
        print(f"  {season}: {int(regular_counts.get(season, 0))}")

    missing_spreads = schedules.groupby("season")["spread_line"].apply(
        lambda values: values.isna().mean() * 100
    )
    print("Missing spread-line percentage by season:")
    for season in schedule_seasons:
        print(f"  {season}: {missing_spreads.get(season, float('nan')):.2f}%")

    duplicate_game_ids = int(schedules["game_id"].duplicated(keep=False).sum())
    print(f"Duplicate game IDs: {duplicate_game_ids}")

    dates = pd.to_datetime(schedules["gameday"], errors="coerce").dropna()
    if dates.empty:
        print("Earliest and latest game date: unavailable")
    else:
        print(
            "Earliest and latest game date: "
            f"{dates.min().date().isoformat()} to {dates.max().date().isoformat()}"
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        nargs="+",
        type=int,
        default=DEFAULT_HISTORICAL_SEASONS,
        help="Historical seasons (default: 2016 through 2025).",
    )
    parser.add_argument(
        "--prediction-season",
        type=int,
        default=DEFAULT_PREDICTION_SEASON,
        help="Upcoming schedule season (default: 2026).",
    )
    add_common_arguments(parser, RAW_DATA_DIR, "Directory for the raw Parquet files.")
    return parser.parse_args(argv)


def _run(args: argparse.Namespace, record: RunRecord) -> int:
    # nflreadpy is only needed here, so it is checked here rather than being
    # made a hard requirement of every command.
    check_dependencies(("nflreadpy",))

    paths = cache_paths(args.seasons, args.prediction_season, args.output_path)
    if args.force_refresh:
        LOGGER.warning("--force-refresh: every source will be downloaded again.")
        warn_on_overwrite(paths.values(), LOGGER)

    try:
        data = load_or_download_data(
            seasons=args.seasons,
            prediction_season=args.prediction_season,
            force_refresh=args.force_refresh,
            directory=args.output_path,
        )
    except (OSError, ValueError, TypeError, KeyError) as error:
        raise WorkflowError(
            f"Download failed: {error}. Check the network connection and that "
            "the requested seasons have been published by nflverse."
        ) from error

    record.add_artifacts(paths.values(), "raw data")

    print(f"\n=== {COMMAND} ===")
    for name, frame in data.items():
        print(f"  {name:12s} {len(frame):>10,} rows")

    print()
    print_validation(data)
    print_artifacts(paths.values(), "Raw files")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args(argv)

    record = RunRecord(COMMAND, argv, args.run_record_dir)
    record.configure(args)

    try:
        code = _run(args, record)
    except WorkflowError as error:
        LOGGER.error("%s", error)
        code = 1

    print(f"\nRun record: {record.finish(code)}")
    return code


if __name__ == "__main__":
    sys.exit(main())
