"""Fit and persist the final deployment model.

Trains on every season of the window with the hyperparameters already frozen,
writes the three artefacts, reloads them, and proves the reloaded model can
produce probabilities.

Refuses to run if the hyperparameter search has not been recorded: a model
fitted on all the data with settings chosen afterwards has no validation
behind it.

Usage::

    python scripts/deploy_model.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from gridiron.config import PROJECT_ROOT, configure_environment, seed_everything
from gridiron.modeling.deploy import (
    MODEL_DIR,
    PREDICTION_SEASON,
    TRAINING_SEASONS,
    DeploymentError,
    deployment_metadata,
    feature_schema,
    fit_deployment_model,
    save_deployment,
    smoke_test,
)
from gridiron.modeling.evaluate import aggregate_metrics, run_walk_forward

LOGGER = logging.getLogger("deploy_model")

MATCHUP_PATH = PROJECT_ROOT / "outputs" / "reports" / "matchups_wide.csv"

SUMMARY_METRICS = (
    "mean_season_accuracy",
    "worst_season_accuracy",
    "best_season_accuracy",
    "accuracy_std",
    "roc_auc",
    "log_loss",
    "brier_score",
    "roi",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matchup-path", type=Path, default=MATCHUP_PATH)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument(
        "--skip-validation-summary",
        action="store_true",
        help="Skip the walk-forward run that fills validation_summary.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    configure_environment()
    seed_everything()

    if not args.matchup_path.exists():
        LOGGER.error("Matchup table not found at %s", args.matchup_path)
        LOGGER.error("Run 'python scripts/matchup_report.py' first.")
        return 1

    matchups = pd.read_csv(args.matchup_path, parse_dates=["gameday"])

    try:
        pipeline, features, record, selection = fit_deployment_model(matchups)
    except DeploymentError as error:
        LOGGER.error("%s", error)
        return 1

    summary: dict[str, object] = {}
    if not args.skip_validation_summary:
        predictions, _ = run_walk_forward(
            matchups, c=selection.c, class_weight=selection.class_weight
        )
        metrics = aggregate_metrics(predictions).set_index("metric")["value"].to_dict()
        summary = {
            name: float(metrics[name]) for name in SUMMARY_METRICS if name in metrics
        }
        summary["out_of_sample_games"] = int(len(predictions))
        summary["scheme"] = "expanding-window walk-forward by season"
        summary["validation_seasons"] = list(selection.validation_seasons)

    schema = feature_schema(pipeline, features)
    metadata = deployment_metadata(pipeline, record, selection, summary)
    paths = save_deployment(pipeline, schema, metadata, args.model_dir)

    print("\n=== Training set ===")
    for key, value in asdict(record).items():
        print(f"  {key.replace('_', ' '):34s} {value}")

    print("\n=== Frozen hyperparameters ===")
    print(f"  C            {selection.c}")
    print(f"  class_weight {selection.class_weight}")
    print(f"  chosen over  {selection.folds} validation folds")
    for warning in selection.warnings:
        print(f"  ! {warning}")

    print("\n=== Feature schema ===")
    for index, name in enumerate(schema["feature_names"], start=1):
        print(f"  {index:2d}. {name}  ({schema['dtypes'][name]})")

    print("\n=== Artefacts ===")
    for name, path in paths.items():
        print(f"  {name:9s} {path}  ({path.stat().st_size:,} bytes)")

    print("\n=== Smoke test: reload and predict ===")
    result = smoke_test(args.model_dir)
    print(f"  reloaded          {result['loaded']}")
    print(f"  features expected {result['n_features']}")
    print(f"  probabilities     {[round(p, 5) for p in result['probabilities']]}")
    print(f"  predictions       {result['predictions']}")
    print(f"  missing handled   {result['handled_missing_values']}")

    if summary:
        print("\n=== Validation summary recorded in metadata ===")
        for key, value in summary.items():
            printable = f"{value:.5f}" if isinstance(value, float) else str(value)
            print(f"  {key:24s} {printable}")

    print(
        f"\nTrained on {min(TRAINING_SEASONS)}-{max(TRAINING_SEASONS)}; "
        f"intended for {PREDICTION_SEASON}."
    )
    print(
        "The validation summary above is what this model did on unseen "
        "seasons. It has no demonstrated edge, and shipping it is not a "
        "recommendation to bet with it."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
