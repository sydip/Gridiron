"""Generate predictions for one week of the upcoming season.

Rebuilds the feature pipeline from source, loads the deployed model, and
writes a prediction CSV and a formatted report.

Usage::

    python scripts/predict_week.py --season 2026 --week 3
    python scripts/predict_week.py --season 2026 --week 3 --refresh-data
    python -m gridiron.cli predict --season 2026 --week 3

Unlike the other commands this one never short-circuits. Rolling features for
week *n* depend on every completed game before it, so a run made after last
week's results landed must produce different numbers than one made before;
recomputing every time is the only way that happens reliably. Here
``--force-refresh`` therefore means "download the latest results first", and
is a synonym for ``--refresh-data``.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from gridiron.config import configure_environment, seed_everything
from gridiron.modeling.deploy import DeploymentError, load_deployment
from gridiron.prediction.weekly import (
    PREDICTION_DIR,
    PREDICTION_SEASON,
    STATUS_PREDICTED,
    WeeklyPredictionError,
    build_prediction_features,
    format_report,
    predict_week,
    refresh_source_data,
    save_predictions,
    source_state,
)
from gridiron.workflow import (
    RunRecord,
    WorkflowError,
    add_common_arguments,
    check_dependencies,
    warn_on_overwrite,
)

LOGGER = logging.getLogger("predict_week")

COMMAND = "predict"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=PREDICTION_SEASON)
    parser.add_argument("--week", type=int, required=True)
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Download the latest schedule and in-progress play-by-play first.",
    )
    parser.add_argument("--model-dir", type=Path, default=None)
    add_common_arguments(
        parser, PREDICTION_DIR, "Directory for the prediction CSV and report."
    )
    return parser.parse_args(argv)


def _run(args: argparse.Namespace, run: RunRecord) -> int:
    # nflreadpy is only reached when refreshing, so it is only required then.
    refresh = args.refresh_data or args.force_refresh
    check_dependencies(("nflreadpy",) if refresh else ())

    if refresh:
        LOGGER.info("Refreshing schedule and %d play-by-play...", args.season)
        record = refresh_source_data(args.season)
    else:
        record = source_state(args.season)
        LOGGER.info(
            "Using data on disk from %s; pass --refresh-data to update.",
            record.data_updated_at or "an unknown time",
        )
    run.add_input(Path(record.schedule_path), "schedule")

    try:
        pipeline, schema, metadata = load_deployment(args.model_dir)
    except DeploymentError as error:
        raise WorkflowError(
            f"{error} Run 'python -m gridiron.cli deploy' first."
        ) from error

    try:
        matchups = build_prediction_features(args.season)
        table = predict_week(
            matchups,
            args.season,
            args.week,
            pipeline,
            schema,
            record.data_updated_at,
        )
    except WeeklyPredictionError as error:
        raise WorkflowError(str(error)) from error

    report = format_report(table, record, metadata)
    expected = [
        args.output_path / f"week_{args.week:02d}_predictions.csv",
        args.output_path / f"week_{args.week:02d}_report.txt",
    ]
    warn_on_overwrite(expected, LOGGER)
    paths = save_predictions(table, report, args.season, args.week, args.output_path)
    run.add_artifacts(paths.values())

    print()
    print(report)
    print()
    predicted = int(table["status"].eq(STATUS_PREDICTED).sum())
    print(f"Wrote {predicted} prediction(s) and {len(table)} row(s):")
    for name, path in paths.items():
        print(f"  {name:6s} {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    configure_environment()
    seed_everything()

    run = RunRecord(COMMAND, argv, args.run_record_dir)
    run.configure(args)

    try:
        code = _run(args, run)
    except WorkflowError as error:
        LOGGER.error("%s", error)
        code = 1

    print(f"\nRun record: {run.finish(code)}")
    return code


if __name__ == "__main__":
    sys.exit(main())
