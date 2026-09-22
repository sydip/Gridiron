"""Build rolling team features and report their coverage.

Assembles the team-game history with EPA and pace, attaches the leakage-free
rolling features, and writes the feature table plus a coverage report to
``outputs/reports``.

Usage::

    python scripts/rolling_report.py
    python scripts/rolling_report.py --seasons 2023 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from gridiron.config import PROJECT_ROOT, configure_environment
from gridiron.data.clean import clean_schedules
from gridiron.features.epa import aggregate_defensive_epa, aggregate_offensive_epa
from gridiron.features.pace import aggregate_pace
from gridiron.features.rolling import (
    ROLLING_FEATURES,
    build_rolling_features,
    rolling_feature_coverage,
)
from gridiron.features.team_games import team_game_history

LOGGER = logging.getLogger("rolling_report")

PBP_PATH = PROJECT_ROOT / "data" / "raw" / "pbp_2016_2025.parquet"
SCHEDULE_PATH = PROJECT_ROOT / "data" / "raw" / "schedules_2016_2026.parquet"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="*", default=None)
    parser.add_argument("--pbp-path", type=Path, default=PBP_PATH)
    parser.add_argument("--schedule-path", type=Path, default=SCHEDULE_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    configure_environment()

    for path in (args.pbp_path, args.schedule_path):
        if not path.exists():
            LOGGER.error("Required data not found at %s", path)
            LOGGER.error("Run 'python -m gridiron.cli download' first.")
            return 1

    pbp = pd.read_parquet(args.pbp_path, columns=PBP_COLUMNS)
    history = team_game_history(clean_schedules(pd.read_parquet(args.schedule_path)))
    if args.seasons:
        pbp = pbp.loc[pbp["season"].isin(args.seasons)]
        history = history.loc[history["season"].isin(args.seasons)]
        LOGGER.info("Filtered to seasons %s", sorted(args.seasons))

    features = build_rolling_features(
        history,
        aggregate_offensive_epa(pbp),
        aggregate_defensive_epa(pbp),
        aggregate_pace(pbp),
    )
    coverage = rolling_feature_coverage(features)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output_dir / "rolling_features.csv", index=False)
    coverage.to_csv(args.output_dir / "rolling_feature_coverage.csv", index=False)

    LOGGER.info("Rows: %d  features: %d", len(features), len(ROLLING_FEATURES))
    print("\n=== Coverage ===")
    print(coverage.to_string(index=False))
    print("\n=== Sample (one team's season) ===")
    sample = features.loc[features["team"].eq("KC") & features["season"].eq(2023)].head(
        6
    )
    columns = ["week", "off_epa_last_3", "off_epa_season", "win_pct_last_5"]
    print(sample[columns].to_string(index=False))
    print(f"\nWrote reports to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
