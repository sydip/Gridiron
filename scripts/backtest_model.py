"""Backtest the model: walk-forward validation, then betting policies.

This is the one command for "how did it actually do on games it had not
seen". It runs both halves of the backtest in order and prints a combined
verdict:

* **Walk-forward** -- a fresh model per fold, each predicting one unseen
  season, scored for accuracy, discrimination and calibration.
* **Policies** -- the same out-of-sample predictions filtered by confidence
  and staked at -110, reported as ROI and drawdown.

Usage::

    python scripts/backtest_model.py
    python scripts/backtest_model.py --force-refresh
    python -m gridiron.cli backtest

Neither half chooses anything by looking at its own results. The folds and the
seven policy thresholds are both fixed in advance, and every one is reported,
including the losers.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from gridiron.config import PROJECT_ROOT, configure_environment, seed_everything
from gridiron.modeling.metrics import BREAK_EVEN_ACCURACY
from gridiron.modeling.policies import MIN_BETS_FOR_A_CLAIM
from gridiron.workflow import (
    RunRecord,
    WorkflowError,
    add_common_arguments,
    check_dependencies,
    is_up_to_date,
    load_script,
    print_artifacts,
    print_up_to_date,
    require_inputs,
    warn_on_overwrite,
)

LOGGER = logging.getLogger("backtest_model")

COMMAND = "backtest"

MATCHUP_PATH = PROJECT_ROOT / "outputs" / "reports" / "matchups_wide.csv"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

WALK_FORWARD_OUTPUTS = [
    "walk_forward_predictions.csv",
    "walk_forward_folds.csv",
    "walk_forward_by_season.csv",
    "walk_forward_aggregate.csv",
]

POLICY_OUTPUTS = [
    "policy_backtest.csv",
    "policy_by_season.csv",
    "policy_win_rate_by_confidence.csv",
    "policy_bettable_predictions.csv",
]

HEADLINE_METRICS = (
    "accuracy",
    "mean_season_accuracy",
    "worst_season_accuracy",
    "roc_auc",
    "log_loss",
    "brier_score",
    "roi",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matchup-path", type=Path, default=MATCHUP_PATH)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument(
        "--skip-policies",
        action="store_true",
        help="Run only the walk-forward half, and draw no charts.",
    )
    add_common_arguments(parser, REPORT_DIR, "Directory for the backtest tables.")
    return parser.parse_args(argv)


def _expected_outputs(args: argparse.Namespace) -> list[Path]:
    names = list(WALK_FORWARD_OUTPUTS)
    if not args.skip_policies:
        names += POLICY_OUTPUTS
    return [args.output_path / name for name in names]


def _print_walk_forward(directory: Path) -> None:
    aggregate = pd.read_csv(directory / "walk_forward_aggregate.csv")
    metrics = aggregate.set_index("metric")["value"].to_dict()

    print("\n=== Walk-forward, out of sample ===")
    for name in HEADLINE_METRICS:
        if name in metrics:
            print(f"  {name:24s} {float(metrics[name]):+.5f}")

    accuracy = float(metrics.get("accuracy", float("nan")))
    print(f"  {'break-even at -110':24s} {BREAK_EVEN_ACCURACY:+.5f}")
    print(
        "  verdict                  "
        + ("above break-even" if accuracy > BREAK_EVEN_ACCURACY else "below break-even")
    )


def _print_policies(directory: Path) -> None:
    results = pd.read_csv(directory / "policy_backtest.csv")
    view = [
        column
        for column in ("policy", "bets", "win_rate", "roi", "max_drawdown_units")
        if column in results.columns
    ]

    print("\n=== Betting policies at -110 ===")
    print(results[view].round(4).to_string(index=False))

    credible = results.loc[results["bets"] >= MIN_BETS_FOR_A_CLAIM]
    profitable = credible.loc[credible["roi"] > 0]
    if profitable.empty:
        print(
            f"\nNo policy with at least {MIN_BETS_FOR_A_CLAIM} bets is "
            "profitable. Nothing here supports betting this model."
        )
    else:
        print(
            "\nProfitable on a sample large enough to discuss: "
            + ", ".join(profitable["policy"].astype(str))
            + ". Check the confidence interval before reading that as an edge."
        )


def _run(args: argparse.Namespace, record: RunRecord) -> int:
    extra = () if args.skip_policies else ("matplotlib", "seaborn")
    check_dependencies(extra)
    require_inputs(
        {"matchup table": args.matchup_path},
        remedy="Run 'python -m gridiron.cli build-features' first.",
    )
    record.add_input(args.matchup_path, "matchup table")

    outputs = _expected_outputs(args)
    if not args.force_refresh and is_up_to_date([args.matchup_path], outputs):
        record.note("Skipped: outputs were newer than the matchup table.")
        record.add_artifacts(outputs)
        print_up_to_date(outputs, COMMAND)
        return 0

    warn_on_overwrite(outputs, LOGGER)
    args.output_path.mkdir(parents=True, exist_ok=True)

    walk_forward = load_script("walk_forward_report")
    code = walk_forward.main(
        [
            "--matchup-path",
            str(args.matchup_path),
            "--output-dir",
            str(args.output_path),
        ]
    )
    if code != 0:
        raise WorkflowError("The walk-forward half of the backtest failed.")
    record.add_artifacts(
        [args.output_path / name for name in WALK_FORWARD_OUTPUTS], "walk-forward"
    )

    if not args.skip_policies:
        policies = load_script("policy_backtest")
        code = policies.main(
            [
                "--matchup-path",
                str(args.matchup_path),
                "--report-dir",
                str(args.output_path),
                "--figure-dir",
                str(args.figure_dir),
            ]
        )
        if code != 0:
            raise WorkflowError("The policy half of the backtest failed.")
        record.add_artifacts(
            [args.output_path / name for name in POLICY_OUTPUTS], "policies"
        )

    print(f"\n\n=== {COMMAND}: combined summary ===")
    _print_walk_forward(args.output_path)
    if not args.skip_policies:
        _print_policies(args.output_path)

    print_artifacts(record.artifact_paths)
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
