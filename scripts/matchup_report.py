"""Report the game-level model table and its missing values.

The pipeline itself lives in :mod:`gridiron.features.build`, which this script
and ``build_features.py`` both call, so the table this report describes is the
same table the model trains on.

For the workflow command, use ``build_features``. This script stays because
the phase-10 report prints breakdowns by week and by team that the workflow
summary leaves out.

Usage::

    python scripts/matchup_report.py
    python scripts/matchup_report.py --seasons 2023 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gridiron.config import configure_environment
from gridiron.features.build import (
    PBP_PATH,
    REPORT_DIR,
    SCHEDULE_PATH,
    build_feature_tables,
    load_sources,
    write_feature_tables,
)
from gridiron.features.matchup import FEATURE_COLUMNS, TARGET_COLUMNS

LOGGER = logging.getLogger("matchup_report")


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

    pbp, schedule = load_sources(args.pbp_path, args.schedule_path, args.seasons)
    matchups, rows, report = build_feature_tables(pbp, schedule)
    write_feature_tables(matchups, rows, report, args.output_dir)

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
