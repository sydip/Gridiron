"""Train the logistic regression and score it against the baselines.

The model is never reported alone. It is scored on the held-out seasons with
the same metrics as the trivial strategies and printed in the same table, so
the only available reading is a comparative one.

Usage::

    python scripts/train_model.py
    python scripts/train_model.py --test-seasons 2024 2025
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from gridiron.config import PROJECT_ROOT, configure_environment, seed_everything
from gridiron.modeling.baselines import BASELINES, eligible_games
from gridiron.modeling.metrics import (
    BREAK_EVEN_ACCURACY,
    comparison_table,
    season_table,
)
from gridiron.modeling.pipeline import feature_matrix, model_coefficients
from gridiron.modeling.train import (
    METADATA_FILENAME,
    MODEL_DIR,
    MODEL_FILENAME,
    save_model,
    train_from_matchups,
)
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

LOGGER = logging.getLogger("train_model")

MATCHUP_PATH = PROJECT_ROOT / "outputs" / "reports" / "matchups_wide.csv"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

DEFAULT_TEST_SEASONS = [2024, 2025]

COMMAND = "train"

REPORT_FILENAMES = [
    "model_coefficients.csv",
    "model_vs_baselines.csv",
    "model_vs_baselines_by_season.csv",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matchup-path", type=Path, default=MATCHUP_PATH)
    parser.add_argument(
        "--test-seasons",
        type=int,
        nargs="*",
        default=DEFAULT_TEST_SEASONS,
        help="Seasons held out entirely from training.",
    )
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    add_common_arguments(parser, REPORT_DIR, "Directory for the training reports.")
    return parser.parse_args(argv)


def _expected_outputs(args: argparse.Namespace) -> list[Path]:
    return [args.output_path / name for name in REPORT_FILENAMES] + [
        args.model_dir / MODEL_FILENAME,
        args.model_dir / METADATA_FILENAME,
    ]


def _run(args: argparse.Namespace, record: RunRecord) -> int:
    check_dependencies()
    require_inputs(
        {"matchup table": args.matchup_path},
        remedy="Run 'python -m gridiron.cli build-features' first.",
    )
    record.add_input(args.matchup_path, "matchup table")

    outputs = _expected_outputs(args)
    if not args.force_refresh and is_up_to_date([args.matchup_path], outputs):
        record.note("Skipped: the fitted model was newer than the matchup table.")
        record.add_artifacts(outputs)
        print_up_to_date(outputs, COMMAND)
        return 0

    warn_on_overwrite(outputs, LOGGER)
    args.output_path.mkdir(parents=True, exist_ok=True)

    matchups = pd.read_csv(args.matchup_path, parse_dates=["gameday"])
    pipeline, train_frame, test_frame, metadata = train_from_matchups(
        matchups, test_seasons=args.test_seasons
    )
    save_model(pipeline, metadata, args.model_dir)

    print("\n=== Training set ===")
    print(f"rows: {metadata.rows}  seasons: {sorted(set(train_frame['season']))}")
    print(f"dates: {metadata.first_gameday} to {metadata.last_gameday}")
    print(f"home cover rate in training: {metadata.positive_rate:.4f}")
    print("\n=== Held out ===")
    print(f"rows: {len(test_frame)}  seasons: {sorted(set(test_frame['season']))}")

    coefficients = model_coefficients(pipeline)
    coefficients.to_csv(args.output_path / "model_coefficients.csv", index=False)
    print("\n=== Coefficients (standardised scale, log-odds per SD) ===")
    print(coefficients.to_string(index=False))
    print(f"intercept: {coefficients.attrs['intercept']:+.5f}")

    # The model and every baseline are scored on exactly the same held-out
    # games, through the same metric code.
    held_out = eligible_games(test_frame)
    predictions = {name: function(held_out) for name, function in BASELINES.items()}
    predictions["MODEL logistic_regression"] = pd.Series(
        pipeline.predict(feature_matrix(held_out)),
        index=held_out.index,
        dtype="int8",
    )

    summary = comparison_table(held_out, predictions)
    seasons = season_table(held_out, predictions)
    summary.to_csv(args.output_path / "model_vs_baselines.csv", index=False)
    seasons.to_csv(args.output_path / "model_vs_baselines_by_season.csv", index=False)

    print(f"\n=== Held-out comparison ({len(held_out)} games) ===")
    print(summary.to_string(index=False))

    model_row = summary.loc[summary["strategy"].str.startswith("MODEL")].iloc[0]
    best_baseline = summary.loc[~summary["strategy"].str.startswith("MODEL")].iloc[0]
    beats_baseline = model_row["roi"] > best_baseline["roi"]
    beats_market = model_row["accuracy"] > BREAK_EVEN_ACCURACY

    print("\n=== Verdict ===")
    print(
        f"model:         {model_row['accuracy']:.4f} accuracy, "
        f"{model_row['roi']:+.4f} ROI"
    )
    print(
        f"best baseline: {best_baseline['accuracy']:.4f} accuracy, "
        f"{best_baseline['roi']:+.4f} ROI  ({best_baseline['strategy']})"
    )
    print(f"break-even accuracy at -110: {BREAK_EVEN_ACCURACY:.4f}")
    print(
        f"\nBeats the best baseline: {'YES' if beats_baseline else 'NO'}. "
        f"Beats break-even: {'YES' if beats_market else 'NO'}."
    )
    if not (beats_baseline and beats_market):
        print(
            "This model has NOT been shown to add value. Do not describe it as "
            "useful on the strength of its accuracy alone."
        )

    record.add_artifacts(outputs)
    print_artifacts(outputs)
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
