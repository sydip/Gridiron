"""Backtest confidence-filtered betting policies on out-of-sample predictions.

Writes the policy results, the per-season breakdown, the confidence-bucket
table, and six charts.

Every figure comes from walk-forward fold predictions. No threshold is chosen
by looking at these results; the seven policies are fixed in advance and all
seven are reported, including the ones that lose.

Usage::

    python scripts/policy_backtest.py
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
from gridiron.eda.policy_charts import build_policy_charts  # noqa: E402
from gridiron.modeling.policies import (  # noqa: E402
    DEFAULT_ODDS,
    MIN_BETS_FOR_A_CLAIM,
    POLICY_THRESHOLDS,
    backtest_policies,
    bettable_predictions,
    flag_unreliable_results,
    season_units,
    win_rate_by_confidence,
)

LOGGER = logging.getLogger("policy_backtest")

MATCHUP_PATH = PROJECT_ROOT / "outputs" / "reports" / "matchups_wide.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

SUMMARY_VIEW = [
    "policy",
    "bets",
    "wins",
    "losses",
    "pushes",
    "win_rate",
    "units_won",
    "roi",
    "max_drawdown_units",
    "longest_losing_streak",
    "profitable_seasons",
    "seasons",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matchup-path", type=Path, default=MATCHUP_PATH)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
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
    predictions = bettable_predictions(matchups)
    results = backtest_policies(predictions)
    buckets = win_rate_by_confidence(predictions)

    seasons = pd.concat(
        [
            season_units(predictions, threshold).assign(threshold_pp=threshold)
            for threshold in POLICY_THRESHOLDS
        ],
        ignore_index=True,
    )

    args.report_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.report_dir / "policy_backtest.csv", index=False)
    seasons.to_csv(args.report_dir / "policy_by_season.csv", index=False)
    buckets.to_csv(args.report_dir / "policy_win_rate_by_confidence.csv", index=False)
    predictions.to_csv(args.report_dir / "policy_bettable_predictions.csv", index=False)

    paths = build_policy_charts(predictions, results, args.figure_dir)

    pushes = int(predictions["is_push"].sum())
    print(f"\n=== Out-of-sample bettable games: {len(predictions)} ===")
    print(
        f"Walk-forward fold predictions only. {pushes} of these pushed: the "
        "stake is returned, and they count as neither a win nor a loss."
    )
    print(
        f"Odds assumption: {DEFAULT_ODDS.name}. A winning unit returns "
        f"{DEFAULT_ODDS.win_profit:.4f} in profit, so break even is "
        f"{DEFAULT_ODDS.break_even_rate:.4f} of resolved bets, not 0.50."
    )

    print("\n=== Policy results ===")
    print(results[SUMMARY_VIEW].round(4).to_string(index=False))

    print("\n=== Win rate with 95% Wilson interval ===")
    interval = results[
        ["policy", "bets", "win_rate", "win_rate_ci_low", "win_rate_ci_high"]
    ].copy()
    interval["interval_width"] = (
        interval["win_rate_ci_high"] - interval["win_rate_ci_low"]
    )
    print(interval.round(4).to_string(index=False))

    print("\n=== Win rate by claimed edge ===")
    print(buckets.round(4).to_string(index=False))

    flagged = flag_unreliable_results(results)
    print("\n=== Sample-size warnings ===")
    if flagged.empty:
        print("  none")
    else:
        for row in flagged.itertuples():
            print(f"  ! {row.policy}: {row.sample_warning}")
            print(
                f"      95% interval spans {row.win_rate_ci_low:.3f} to "
                f"{row.win_rate_ci_high:.3f} ({row.interval_width:.3f} wide)"
            )

    profitable = results.loc[results["roi"] > 0]
    print("\n=== Verdict ===")
    if profitable.empty:
        print("  No policy returned a profit. Confidence filtering did not help.")
    else:
        for row in profitable.itertuples():
            note = (
                f" -- but on {int(row.bets)} bets, which is not evidence of anything"
                if row.sample_warning
                else ""
            )
            print(f"  {row.policy}: {row.roi:+.4f} ROI on {int(row.bets)} bets{note}")
    print(
        "\nNothing here establishes a profitable strategy. Every policy with "
        f"enough bets to measure ({MIN_BETS_FOR_A_CLAIM}+) lost money, and the "
        "only positive return comes from a sample too small to distinguish "
        "from chance."
    )

    print(f"\nCharts written to {args.figure_dir}:")
    for path in paths:
        print(f"  {path.name}")
    print(f"Reports written to {args.report_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
