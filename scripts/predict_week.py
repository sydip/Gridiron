"""Generate predictions for one week of the upcoming season.

Rebuilds the feature pipeline from source, loads the deployed model, and
writes a prediction CSV and a formatted report.

Usage::

    python scripts/predict_week.py --season 2026 --week 3
    python scripts/predict_week.py --season 2026 --week 3 --refresh-data
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

LOGGER = logging.getLogger("predict_week")


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
    parser.add_argument("--output-dir", type=Path, default=PREDICTION_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    configure_environment()
    seed_everything()

    if args.refresh_data:
        LOGGER.info("Refreshing schedule and %d play-by-play...", args.season)
        record = refresh_source_data(args.season)
    else:
        record = source_state(args.season)
        LOGGER.info(
            "Using data on disk from %s; pass --refresh-data to update.",
            record.data_updated_at or "an unknown time",
        )

    try:
        pipeline, schema, metadata = load_deployment(args.model_dir)
    except DeploymentError as error:
        LOGGER.error("%s", error)
        LOGGER.error("Run 'python scripts/deploy_model.py' first.")
        return 1

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
        LOGGER.error("%s", error)
        return 1

    report = format_report(table, record, metadata)
    paths = save_predictions(table, report, args.season, args.week, args.output_dir)

    print()
    print(report)
    print()
    predicted = int(table["status"].eq(STATUS_PREDICTED).sum())
    print(f"Wrote {predicted} prediction(s) and {len(table)} row(s):")
    for name, path in paths.items():
        print(f"  {name:6s} {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
