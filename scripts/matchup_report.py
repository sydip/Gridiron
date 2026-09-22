"""Assemble the game-level model table and report its missing values.

Runs the full feature pipeline -- schedule, team-game history, EPA, pace,
rolling form -- then merges both sides into one row per game and writes the
model table plus four missing-value breakdowns to ``outputs/reports``.

Usage::

    python scripts/matchup_report.py
    python scripts/matchup_report.py --seasons 2023 2024
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
from gridiron.features.matchup import (
    FEATURE_COLUMNS,
    TARGET_COLUMNS,
    add_matchup_target,
    assert_no_postgame_fields,
    build_matchups,
    missing_value_report,
    model_rows,
)
from gridiron.features.pace import aggregate_pace
from gridiron.features.rolling import build_rolling_features
from gridiron.features.team_games import team_game_history

LOGGER = logging.getLogger("matchup_report")

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
            LOGGER.error("Run 'python scripts/download_data.py' first.")
            return 1

    pbp = pd.read_parquet(args.pbp_path, columns=PBP_COLUMNS)
    schedule = clean_schedules(pd.read_parquet(args.schedule_path))
    if args.seasons:
        pbp = pbp.loc[pbp["season"].isin(args.seasons)]
        schedule = schedule.loc[schedule["season"].isin(args.seasons)]
        LOGGER.info("Filtered to seasons %s", sorted(args.seasons))

    features = build_rolling_features(
        team_game_history(schedule),
        aggregate_offensive_epa(pbp),
        aggregate_defensive_epa(pbp),
        aggregate_pace(pbp),
    )
    matchups = add_matchup_target(build_matchups(features, schedule))
    rows = model_rows(matchups)
    assert_no_postgame_fields(rows)

    report = missing_value_report(matchups)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    matchups.to_csv(args.output_dir / "matchups_wide.csv", index=False)
    rows.to_csv(args.output_dir / "model_table.csv", index=False)
    for name, frame in report.items():
        frame.to_csv(args.output_dir / f"matchup_missing_by_{name}.csv", index=False)

    labelled = rows["home_cover"].notna().sum()
    LOGGER.info(
        "Games: %d  features: %d  labelled: %d",
        len(rows),
        len(FEATURE_COLUMNS),
        labelled,
    )
    print("\n=== Model row ===")
    print(
        f"identifiers + {len(FEATURE_COLUMNS)} features + {len(TARGET_COLUMNS)} targets"
    )
    print("\n=== Missing by feature ===")
    print(report["feature"].to_string(index=False))
    print("\n=== Missing by week (first 6) ===")
    columns = ["week", "rows", "any_missing_pct", "off_epa_diff_last_5"]
    print(report["week"][columns].head(6).to_string(index=False))
    print("\n=== Missing by season ===")
    columns = [
        "season",
        "rows",
        "any_missing_pct",
        "off_epa_diff_last_5",
        "spread_line",
    ]
    print(report["season"][columns].to_string(index=False))
    print(f"\nWrote reports to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
