"""Build the team-game pace table and its distribution/outlier reports.

Reads the raw play-by-play parquet, aggregates pace, and writes three CSVs to
``outputs/reports``: the team-game pace table, the distribution summary, and
the flagged rows for inspection.

Usage::

    python scripts/pace_report.py
    python scripts/pace_report.py --seasons 2023 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from gridiron.config import PROJECT_ROOT, configure_environment
from gridiron.features.pace import (
    aggregate_pace,
    flag_pace_outliers,
    pace_distribution_report,
    timing_fields_available,
)

LOGGER = logging.getLogger("pace_report")

PBP_PATH = PROJECT_ROOT / "data" / "raw" / "pbp_2016_2025.parquet"
SCHEDULE_PATH = PROJECT_ROOT / "data" / "raw" / "schedules_2016_2026.parquet"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

# Only the columns pace needs, so the 182MB parquet is not read in full.
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
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="*",
        default=None,
        help="Limit the report to these seasons (default: all available).",
    )
    parser.add_argument(
        "--pbp-path", type=Path, default=PBP_PATH, help="Raw play-by-play parquet."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPORT_DIR, help="Where to write the CSVs."
    )
    return parser.parse_args(argv)


def coverage_summary(pace: pd.DataFrame, schedule_path: Path) -> pd.DataFrame | None:
    """Compare pace coverage against the completed games in the schedule.

    Returns ``None`` when the schedule is unavailable, so the report still runs
    from play-by-play alone.
    """
    if not schedule_path.exists():
        LOGGER.warning("Schedule not found at %s; skipping coverage.", schedule_path)
        return None

    from gridiron.data.clean import clean_schedules

    schedule = clean_schedules(pd.read_parquet(schedule_path))
    completed = schedule.loc[schedule["is_completed"]]
    teams_per_game = pace.groupby("game_id")["team"].nunique()
    both_sides = set(teams_per_game.loc[teams_per_game.eq(2)].index)

    rows = []
    for season, group in completed.groupby("season", sort=True):
        game_ids = set(group["game_id"])
        covered = len(game_ids & both_sides)
        rows.append(
            {
                "season": int(season),
                "completed_games": len(game_ids),
                "games_with_both_teams": covered,
                "coverage_pct": round(100.0 * covered / max(len(game_ids), 1), 3),
            }
        )
    return pd.DataFrame(rows)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    configure_environment()

    if not args.pbp_path.exists():
        LOGGER.error("Play-by-play parquet not found at %s", args.pbp_path)
        LOGGER.error("Run 'python -m gridiron.cli download' first.")
        return 1

    LOGGER.info("Reading %s", args.pbp_path)
    pbp = pd.read_parquet(args.pbp_path, columns=PBP_COLUMNS)
    if args.seasons:
        pbp = pbp.loc[pbp["season"].isin(args.seasons)]
        LOGGER.info("Filtered to seasons %s", sorted(args.seasons))

    LOGGER.info(
        "Timing fields usable: %s", "yes" if timing_fields_available(pbp) else "no"
    )

    pace = aggregate_pace(pbp)
    distribution = pace_distribution_report(pace)
    outliers = flag_pace_outliers(pace)
    coverage = coverage_summary(pace, SCHEDULE_PATH)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    pace.to_csv(args.output_dir / "pace_team_game.csv", index=False)
    distribution.to_csv(args.output_dir / "pace_distribution.csv", index=False)
    outliers.to_csv(args.output_dir / "pace_outliers.csv", index=False)
    if coverage is not None:
        coverage.to_csv(args.output_dir / "pace_coverage.csv", index=False)

    LOGGER.info("Team-game pace rows: %d", len(pace))
    print("\n=== Distribution ===")
    print(distribution.to_string(index=False))
    if coverage is not None:
        print("\n=== Coverage vs completed schedule ===")
        print(coverage.to_string(index=False))
    print(f"\n=== Flagged for inspection: {len(outliers)} rows ===")
    if not outliers.empty:
        print(outliers["flag_reasons"].value_counts().to_string())
    print(f"\nWrote reports to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
