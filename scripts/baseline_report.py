"""Score every baseline strategy and write the comparison table.

Runs each trivial strategy over the same eligible games and writes the summary,
the season-by-season breakdown, and the profit curves to ``outputs/reports``.

The table is the bar any model has to clear. A model that does not appear in a
table beside these numbers has not been shown to be worth anything.

Usage::

    python scripts/baseline_report.py
    python scripts/baseline_report.py --require-epa
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from gridiron.config import PROJECT_ROOT, configure_environment
from gridiron.modeling.baselines import BASELINES, eligible_games, epa_fallback_count
from gridiron.modeling.metrics import (
    BREAK_EVEN_ACCURACY,
    comparison_table,
    profit_curve,
    season_table,
)

LOGGER = logging.getLogger("baseline_report")

MATCHUP_PATH = PROJECT_ROOT / "outputs" / "reports" / "matchups_wide.csv"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--matchup-path",
        type=Path,
        default=MATCHUP_PATH,
        help="Wide matchup table written by scripts/matchup_report.py.",
    )
    parser.add_argument(
        "--require-epa",
        action="store_true",
        help="Restrict to games with a rolling EPA differential (complete cases).",
    )
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    configure_environment()

    if not args.matchup_path.exists():
        LOGGER.error("Matchup table not found at %s", args.matchup_path)
        LOGGER.error("Run 'python scripts/matchup_report.py' first.")
        return 1

    matchups = pd.read_csv(args.matchup_path, parse_dates=["gameday"])
    eligible = eligible_games(matchups, require_epa=args.require_epa)

    predictions = {name: function(eligible) for name, function in BASELINES.items()}
    summary = comparison_table(eligible, predictions)
    seasons = season_table(eligible, predictions)
    curves = pd.DataFrame(
        {name: profit_curve(eligible, picks) for name, picks in predictions.items()}
    )
    curves.insert(0, "game_id", eligible.loc[curves.index, "game_id"].to_numpy())
    curves.insert(1, "gameday", eligible.loc[curves.index, "gameday"].to_numpy())

    suffix = "_complete_cases" if args.require_epa else ""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / f"baseline_comparison{suffix}.csv", index=False)
    seasons.to_csv(args.output_dir / f"baseline_by_season{suffix}.csv", index=False)
    curves.to_csv(args.output_dir / f"baseline_profit_curves{suffix}.csv", index=False)

    total = matchups["home_cover"].notna().sum()
    print(f"\n=== Eligible games: {len(eligible)} of {total} settled ===")
    print(
        "Excluded: pushes, unplayed games, and games without a line "
        "(all three are null in home_cover)."
    )
    if args.require_epa:
        print("Also excluded: games without a rolling EPA differential.")
    else:
        fallback = epa_fallback_count(eligible)
        print(
            f"better_epa decided {fallback} game(s) by fallback rather than by "
            "an EPA signal; re-run with --require-epa to exclude them."
        )

    print("\n=== Baseline comparison (break-even ROI = 0, at -110) ===")
    print(summary.to_string(index=False))

    best = summary.iloc[0]
    print(
        f"\nBest baseline: {best['strategy']} at {best['accuracy']:.4f} accuracy "
        f"and {best['roi']:+.4f} ROI."
    )
    print(
        f"Break-even accuracy at -110 is {BREAK_EVEN_ACCURACY:.4f}. "
        f"{'No baseline clears it.' if not summary['beats_break_even'].any() else ''}"
    )
    print(
        "\nAny model must be reported beside this table. Beating 50% accuracy is "
        "not evidence of value; beating the best baseline's ROI is the bar."
    )

    print("\n=== Accuracy by season ===")
    pivot = seasons.pivot(index="season", columns="strategy", values="accuracy")
    print(pivot.round(4).to_string())
    print(f"\nWrote reports to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
