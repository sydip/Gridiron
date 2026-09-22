"""Build the visual reports.

Draws the historical report from the saved processed tables, and a weekly
report for the requested week from the saved prediction CSV.

    python scripts/build_reports.py
    python scripts/build_reports.py --week 3
    python scripts/build_reports.py --historical-only

Nothing here recomputes a metric. Every number on a chart is read from a table
the pipeline already wrote, so the report cannot disagree with the run that
produced it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from gridiron.config import (  # noqa: E402
    PROJECT_ROOT,
    configure_environment,
    seed_everything,
)
from gridiron.eda.reports import (  # noqa: E402
    ReportError,
    historical_report,
    load_report_inputs,
    prediction_contributions,
    weekly_report,
)
from gridiron.modeling.deploy import DeploymentError, load_deployment  # noqa: E402
from gridiron.modeling.pipeline import feature_matrix  # noqa: E402
from gridiron.prediction.weekly import (  # noqa: E402
    PREDICTION_SEASON,
    build_prediction_features,
)

LOGGER = logging.getLogger("build_reports")

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", type=int, default=PREDICTION_SEASON)
    parser.add_argument(
        "--week",
        type=int,
        default=None,
        help="Week to draw a prediction report for (default: the latest saved).",
    )
    parser.add_argument("--historical-only", action="store_true")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--prediction-dir", type=Path, default=PREDICTION_DIR)
    return parser.parse_args(argv)


def _latest_week(directory: Path) -> int | None:
    """The highest week number with a saved prediction CSV."""
    weeks = []
    for path in directory.glob("week_*_predictions.csv"):
        try:
            weeks.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            continue
    return max(weeks) if weeks else None


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    configure_environment()
    seed_everything()

    written: list[Path] = []

    try:
        inputs = load_report_inputs(args.report_dir)
    except ReportError as error:
        LOGGER.error("%s", error)
        return 1

    written.append(historical_report(inputs, args.figure_dir))
    print("\n=== Historical report ===")
    print(f"  {written[-1]}")
    print(
        "  Covers seasons the model never saw during training. "
        "Every panel is a settled result."
    )

    if args.historical_only:
        return 0

    week = args.week or _latest_week(args.prediction_dir)
    if week is None:
        LOGGER.warning(
            "No saved predictions in %s; run scripts/predict_week.py first.",
            args.prediction_dir,
        )
        return 0

    prediction_path = args.prediction_dir / f"week_{week:02d}_predictions.csv"
    if not prediction_path.exists():
        LOGGER.error("No saved predictions at %s", prediction_path)
        return 1

    table = pd.read_csv(prediction_path)
    priced = table.loc[table["home_cover_probability"].notna()]
    if priced.empty:
        LOGGER.warning("Week %d has no priced games; skipping the weekly report.", week)
        return 0

    try:
        pipeline, _, _ = load_deployment()
    except DeploymentError as error:
        LOGGER.error("%s", error)
        LOGGER.error("Run 'python scripts/deploy_model.py' first.")
        return 1

    matchups = build_prediction_features(args.season)
    selected = matchups.loc[matchups["game_id"].isin(set(table["game_id"]))]
    ordered = selected.set_index("game_id").loc[table["game_id"]].reset_index()

    contributions = prediction_contributions(feature_matrix(ordered), pipeline)
    contributions.index = table.index

    written.append(
        weekly_report(table, ordered, contributions, args.figure_dir, args.season, week)
    )
    print(f"\n=== Week {week} report ===")
    print(f"  {written[-1]}")
    print(
        f"  {len(priced)} unplayed games. Predictions, not results -- drawn in "
        "the second accent colour so it cannot be mistaken for the historical "
        "page."
    )

    print("\nAll figures:")
    for path in written:
        print(f"  {path.name}  ({path.stat().st_size:,} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
