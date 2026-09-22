"""Build the model table from the downloaded raw data.

Runs the whole feature pipeline -- schedule cleaning, team-game history, EPA,
pace, rolling form -- and writes the matchup table, the model rows, and the
four missing-value breakdowns.

Usage::

    python scripts/build_features.py
    python scripts/build_features.py --seasons 2023 2024
    python scripts/build_features.py --force-refresh
    python -m gridiron.cli build-features

Re-running with unchanged inputs is a no-op: the command says the tables are
already current and exits 0 rather than spending a minute reproducing them
byte for byte. ``--force-refresh`` rebuilds regardless.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gridiron.config import configure_environment, seed_everything
from gridiron.features.build import (
    PBP_PATH,
    REPORT_DIR,
    SCHEDULE_PATH,
    build_feature_tables,
    feature_output_paths,
    load_sources,
    write_feature_tables,
)
from gridiron.features.matchup import FEATURE_COLUMNS, TARGET_COLUMNS
from gridiron.workflow import (
    RunRecord,
    WorkflowError,
    add_common_arguments,
    check_dependencies,
    is_up_to_date,
    print_artifacts,
    print_up_to_date,
    require_inputs,
    warn_on_overwrite,
)

LOGGER = logging.getLogger("build_features")

COMMAND = "build-features"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        type=int,
        nargs="*",
        default=None,
        help="Restrict the build to these seasons (default: all available).",
    )
    parser.add_argument("--pbp-path", type=Path, default=PBP_PATH)
    parser.add_argument("--schedule-path", type=Path, default=SCHEDULE_PATH)
    add_common_arguments(parser, REPORT_DIR, "Directory for the feature tables.")
    return parser.parse_args(argv)


def _run(args: argparse.Namespace, record: RunRecord) -> int:
    check_dependencies()
    require_inputs(
        {"play-by-play": args.pbp_path, "schedule": args.schedule_path},
        remedy="Run 'python -m gridiron.cli download' first.",
    )
    record.add_input(args.pbp_path, "play-by-play")
    record.add_input(args.schedule_path, "schedule")

    outputs = feature_output_paths(args.output_path)
    inputs = [args.pbp_path, args.schedule_path]

    # A season filter produces a different table from the same inputs, so the
    # timestamp comparison cannot be trusted; always rebuild in that case.
    if not args.force_refresh and not args.seasons and is_up_to_date(inputs, outputs):
        record.note("Skipped: outputs were newer than inputs.")
        record.add_artifacts(outputs)
        print_up_to_date(outputs, COMMAND)
        return 0

    warn_on_overwrite(outputs, LOGGER)

    pbp, schedule = load_sources(args.pbp_path, args.schedule_path, args.seasons)
    LOGGER.info("Loaded %s plays and %s games.", f"{len(pbp):,}", f"{len(schedule):,}")

    matchups, rows, report = build_feature_tables(pbp, schedule)
    written = write_feature_tables(matchups, rows, report, args.output_path)
    record.add_artifacts(written)

    labelled = int(rows["home_cover"].notna().sum())
    seasons = sorted(set(rows["season"]))

    print(f"\n=== {COMMAND} ===")
    print(f"  seasons            {seasons[0]}-{seasons[-1]} ({len(seasons)} seasons)")
    print(f"  games              {len(rows):,}")
    print(f"  labelled games     {labelled:,}")
    print(f"  features           {len(FEATURE_COLUMNS)}")
    print(f"  targets            {len(TARGET_COLUMNS)}")

    print("\n=== Missing by feature ===")
    print(report["feature"].to_string(index=False))

    print("\n=== Missing by season ===")
    columns = ["season", "rows", "any_missing_pct", "off_epa_diff_last_5"]
    print(report["season"][columns].to_string(index=False))

    print_artifacts(written)
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    configure_environment()
    seed_everything()

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
