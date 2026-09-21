"""Download and validate nflverse source datasets."""

from __future__ import annotations

import argparse
import logging

import pandas as pd

from gridiron.data.download import load_or_download_data

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


def parse_args() -> argparse.Namespace:
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
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Download all sources even when raw Parquet files exist.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    data = load_or_download_data(
        seasons=args.seasons,
        prediction_season=args.prediction_season,
        force_refresh=args.force_refresh,
    )
    print_validation(data)


if __name__ == "__main__":
    main()
