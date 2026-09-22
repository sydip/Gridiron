"""Build the team-game history and report on its rest columns.

Reads the raw schedule, builds one chronological row per team per game, derives
rest, and writes three CSVs to ``outputs/reports``: the history itself, the rest
distribution, and any games that failed reconciliation.

Usage::

    python scripts/team_game_report.py
    python scripts/team_game_report.py --seasons 2023 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from gridiron.config import PROJECT_ROOT, configure_environment
from gridiron.data.clean import clean_schedules
from gridiron.features.team_games import (
    rest_distribution,
    team_game_history,
    team_game_reconciliation,
    team_game_reconciliation_failures,
)

LOGGER = logging.getLogger("team_game_report")

SCHEDULE_PATH = PROJECT_ROOT / "data" / "raw" / "schedules_2016_2026.parquet"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


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
        "--schedule-path", type=Path, default=SCHEDULE_PATH, help="Raw schedule."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPORT_DIR, help="Where to write the CSVs."
    )
    return parser.parse_args(argv)


def weekday_rest_summary(history: pd.DataFrame) -> pd.DataFrame:
    """Rest by the weekday a game was played, which is where short weeks show."""
    measured = history.loc[history["week_1_flag"].eq(0)].copy()
    measured["weekday"] = measured["gameday"].dt.day_name()
    summary = (
        measured.groupby("weekday")
        .agg(
            games=("rest_days", "size"),
            mean_rest=("rest_days", "mean"),
            median_rest=("rest_days", "median"),
            short_week_share=("short_week", "mean"),
            extra_rest_share=("extra_rest", "mean"),
        )
        .reset_index()
    )
    return summary.sort_values("games", ascending=False).reset_index(drop=True)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    configure_environment()

    if not args.schedule_path.exists():
        LOGGER.error("Schedule parquet not found at %s", args.schedule_path)
        LOGGER.error("Run 'python -m gridiron.cli download' first.")
        return 1

    schedule = clean_schedules(pd.read_parquet(args.schedule_path))
    if args.seasons:
        schedule = schedule.loc[schedule["season"].isin(args.seasons)]
        LOGGER.info("Filtered to seasons %s", sorted(args.seasons))

    history = team_game_history(schedule)
    report = team_game_reconciliation(history)
    failures = team_game_reconciliation_failures(report)
    rest = rest_distribution(history)
    weekday = weekday_rest_summary(history)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    history.to_csv(args.output_dir / "team_game_history.csv", index=False)
    rest.to_csv(args.output_dir / "team_game_rest_distribution.csv", index=False)
    weekday.to_csv(args.output_dir / "team_game_rest_by_weekday.csv", index=False)
    failures.to_csv(
        args.output_dir / "team_game_reconciliation_failures.csv", index=False
    )

    completed = history.loc[history["is_completed"]]
    LOGGER.info(
        "Team-game rows: %d (%d completed, across %d games)",
        len(history),
        len(completed),
        completed["game_id"].nunique(),
    )
    print("\n=== Reconciliation ===")
    print(f"games: {len(report)}  reconciled: {int(report['reconciled'].sum())}")
    print(f"failures: {len(failures)}")
    print("\n=== Rest distribution ===")
    print(rest.to_string(index=False))
    print("\n=== Rest by weekday ===")
    print(weekday.to_string(index=False))
    print(f"\nWrote reports to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
