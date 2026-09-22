"""Assess calibration on the out-of-sample walk-forward predictions.

Draws the three calibration charts, writes the reliability table, and runs a
temporally-valid calibration trial so the decision to calibrate or not rests on
measurement rather than on the method being available.

Usage::

    python scripts/calibration_report.py
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from gridiron.config import (  # noqa: E402
    PROJECT_ROOT,
    configure_environment,
    seed_everything,
)
from gridiron.eda.calibration_charts import build_calibration_charts  # noqa: E402
from gridiron.modeling.baselines import eligible_games  # noqa: E402
from gridiron.modeling.calibration import (  # noqa: E402
    MIN_BUCKET_SAMPLES,
    apply_calibrator,
    compare_calibration,
    decide_calibration,
    fit_calibrator,
    reliability_table,
    sparse_buckets,
    temporal_calibration_split,
)
from gridiron.modeling.evaluate import run_walk_forward  # noqa: E402
from gridiron.modeling.pipeline import feature_matrix, target_vector  # noqa: E402
from gridiron.modeling.splits import season_walk_forward_splits  # noqa: E402
from gridiron.modeling.train import train_model  # noqa: E402
from gridiron.modeling.tuning import load_selected_parameters  # noqa: E402

LOGGER = logging.getLogger("calibration_report")

MATCHUP_PATH = PROJECT_ROOT / "outputs" / "reports" / "matchups_wide.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matchup-path", type=Path, default=MATCHUP_PATH)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    return parser.parse_args(argv)


def calibration_trial(matchups: pd.DataFrame) -> pd.DataFrame:
    """Run a temporally-valid calibration over the same walk-forward folds.

    Within each fold the model is fitted on the earlier training seasons and
    the calibrator on the latest training season alone. The two are disjoint,
    and both sit entirely before the validation season, so nothing about the
    procedure looks forward in time.
    """
    selection = load_selected_parameters()
    eligible = eligible_games(matchups).sort_values(
        ["gameday", "game_id"], kind="stable"
    )

    rows = []
    for split in season_walk_forward_splits(eligible):
        train_frame = eligible.loc[split.train_indices]
        try:
            fit_frame, calibration_frame = temporal_calibration_split(train_frame)
        except Exception as error:  # noqa: BLE001 - reported, not swallowed
            LOGGER.info(
                "Fold %d skipped for calibration: %s",
                split.validation_season,
                error,
            )
            continue

        validation = eligible.loc[split.validation_indices]
        pipeline = train_model(
            feature_matrix(fit_frame),
            target_vector(fit_frame),
            c=selection.c,
            class_weight=selection.class_weight,
        )

        calibration_scores = pipeline.predict_proba(feature_matrix(calibration_frame))[
            :, 1
        ]
        calibration_outcomes = target_vector(calibration_frame).to_numpy()
        validation_scores = pipeline.predict_proba(feature_matrix(validation))[:, 1]

        platt = fit_calibrator(calibration_scores, calibration_outcomes, "platt")
        isotonic = fit_calibrator(calibration_scores, calibration_outcomes, "isotonic")

        rows.append(
            pd.DataFrame(
                {
                    "season": validation["season"].to_numpy(),
                    "actual_home_cover": target_vector(validation).to_numpy(),
                    "home_cover_probability": validation_scores,
                    "platt": apply_calibrator(platt, validation_scores),
                    "isotonic": apply_calibrator(isotonic, validation_scores),
                    "platt_coefficient": float(platt.coef_[0][0]),
                }
            )
        )

    if not rows:
        raise SystemExit("No folds had enough history for a calibration split.")
    return pd.concat(rows, ignore_index=True)


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
    selection = load_selected_parameters()
    predictions, _ = run_walk_forward(
        matchups, c=selection.c, class_weight=selection.class_weight
    )

    table = reliability_table(predictions)
    thin = sparse_buckets(table)
    decision = decide_calibration(predictions)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.report_dir / "calibration_reliability.csv", index=False)
    paths = build_calibration_charts(predictions, args.figure_dir)

    print(f"\n=== Out-of-sample predictions: {len(predictions)} games ===")
    print("Walk-forward only; no in-sample prediction is scored here.")

    print("\n=== Reliability by bucket ===")
    print(table.round(4).to_string(index=False))

    print(f"\n=== Sparse buckets (fewer than {MIN_BUCKET_SAMPLES} games) ===")
    if thin.empty:
        print("  none")
    else:
        print(
            thin[["bucket", "games", "observed_rate", "standard_error"]]
            .round(4)
            .to_string(index=False)
        )
        print(
            f"  {int(thin['games'].sum())} of {len(predictions)} games sit in "
            "buckets too thin to support a conclusion."
        )

    print("\n=== Calibration trial (temporally valid) ===")
    trial = calibration_trial(matchups)
    comparison = compare_calibration(
        trial,
        {
            "platt (temporal)": trial["platt"].to_numpy(),
            "isotonic (temporal)": trial["isotonic"].to_numpy(),
        },
    )
    comparison.to_csv(args.report_dir / "calibration_comparison.csv", index=False)
    print(comparison.round(5).to_string(index=False))

    coefficients = trial.groupby("season")["platt_coefficient"].first()
    print("\nPlatt coefficient per fold (negative inverts the score):")
    print("  " + ", ".join(f"{value:+.2f}" for value in coefficients))
    flips = int(np.sign(coefficients).nunique() > 1)
    print(f"  sign changes across folds: {'yes' if flips else 'no'}")

    print("\n=== Decision ===")
    for index, reason in enumerate(decision.reasons, start=1):
        print(f"  {index}. {reason}")
    print(f"\nCalibration adopted: {'YES' if decision.calibrate else 'NO'}")

    print(f"\nCharts written to {args.figure_dir}:")
    for path in paths:
        print(f"  {path.name}")
    print(f"Reports written to {args.report_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
