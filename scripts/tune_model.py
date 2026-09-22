"""Search the hyperparameter grid and record the chosen settings.

Writes ``outputs/reports/hyperparameter_results.csv`` and saves the selection,
with its reasoning, to ``config/model_params.json`` where later training runs
read it.

Selection follows a ranked decision process -- log loss, then Brier, then
stability, then accuracy, then coefficient size -- and never ROI alone.

Usage::

    python scripts/tune_model.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from gridiron.config import PROJECT_ROOT, configure_environment, seed_everything
from gridiron.modeling.evaluate import UNINFORMATIVE_BRIER, UNINFORMATIVE_LOG_LOSS
from gridiron.modeling.tuning import (
    DEPLOYMENT_SEASON,
    RESULT_COLUMNS,
    SELECTED_PARAMS_PATH,
    run_grid_search,
    save_selected_parameters,
    select_parameters,
)

LOGGER = logging.getLogger("tune_model")

MATCHUP_PATH = PROJECT_ROOT / "outputs" / "reports" / "matchups_wide.csv"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matchup-path", type=Path, default=MATCHUP_PATH)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--config-path", type=Path, default=SELECTED_PARAMS_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    configure_environment()
    seed_everything()

    if not args.matchup_path.exists():
        LOGGER.error("Matchup table not found at %s", args.matchup_path)
        LOGGER.error("Run 'python -m gridiron.cli build-features' first.")
        return 1

    matchups = pd.read_csv(args.matchup_path, parse_dates=["gameday"])
    results = run_grid_search(matchups)
    selection = select_parameters(results)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "hyperparameter_results.csv", index=False)
    config_path = save_selected_parameters(selection, args.config_path)

    folds = results.attrs["folds"]
    seasons = results.attrs["validation_seasons"]
    print(f"\n=== Grid: {len(results)} parameter sets over {folds} identical folds ===")
    print(f"validation seasons: {seasons}")
    print(f"deployment season {DEPLOYMENT_SEASON} was not scored.")

    print("\n=== Results (required columns) ===")
    print(results[RESULT_COLUMNS].round(5).to_string(index=False))

    print("\n=== Coefficient diagnostics ===")
    diagnostics = results[
        [
            "C",
            "class_weight",
            "max_abs_coefficient",
            "mean_abs_coefficient",
            "unstable_features",
            "beats_uninformative",
        ]
    ]
    print(diagnostics.round(5).to_string(index=False))

    print("\n=== Selection ===")
    for step, line in enumerate(selection.rationale, start=1):
        print(f"  {step}. {line}")
    print(
        f"\nChosen: C={selection.c}, class_weight={selection.class_weight}  "
        f"(log loss {selection.mean_log_loss:.5f}, Brier "
        f"{selection.mean_brier:.5f}, accuracy {selection.mean_accuracy:.5f} "
        f"+/- {selection.accuracy_std:.5f})"
    )

    if selection.warnings:
        print("\n=== Warnings ===")
        for warning in selection.warnings:
            print(f"  ! {warning}")

    informative = int(results["beats_uninformative"].sum())
    print(
        f"\n{informative} of {len(results)} parameter sets beat a flat 0.5 "
        f"prediction (log loss {UNINFORMATIVE_LOG_LOSS:.5f}, Brier "
        f"{UNINFORMATIVE_BRIER})."
    )

    print(f"\nWrote {args.output_dir / 'hyperparameter_results.csv'}")
    print(f"Saved selection to {config_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
