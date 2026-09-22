"""Run the season walk-forward backtest and report it.

Trains a fresh model per fold, predicts one unseen season at a time, and writes
the concatenated out-of-sample predictions, the fold summary, the per-season
metrics, and the aggregate metrics to ``outputs/reports``.

The out-of-sample predictions are also scored against the Phase 12 baselines on
exactly the same games, because an accuracy figure on its own says nothing.

Usage::

    python scripts/walk_forward_report.py
    python scripts/walk_forward_report.py --first 2020 --final 2025
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

from gridiron.config import PROJECT_ROOT, configure_environment, seed_everything
from gridiron.modeling.baselines import BASELINES, eligible_games
from gridiron.modeling.evaluate import (
    UNINFORMATIVE_BRIER,
    UNINFORMATIVE_LOG_LOSS,
    aggregate_metrics,
    probability_quality,
    run_walk_forward,
    season_metrics,
)
from gridiron.modeling.metrics import BREAK_EVEN_ACCURACY, comparison_table
from gridiron.modeling.splits import (
    FINAL_VALIDATION_SEASON,
    FIRST_VALIDATION_SEASON,
)
from gridiron.modeling.tuning import load_selected_parameters

LOGGER = logging.getLogger("walk_forward_report")

MATCHUP_PATH = PROJECT_ROOT / "outputs" / "reports" / "matchups_wide.csv"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matchup-path", type=Path, default=MATCHUP_PATH)
    parser.add_argument("--first", type=int, default=FIRST_VALIDATION_SEASON)
    parser.add_argument("--final", type=int, default=FINAL_VALIDATION_SEASON)
    parser.add_argument("--output-dir", type=Path, default=REPORT_DIR)
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
    # Use the hyperparameters the tuning phase chose, so this report describes
    # the model the project actually ships rather than an untuned variant.
    selection = load_selected_parameters()
    predictions, folds = run_walk_forward(
        matchups,
        args.first,
        args.final,
        c=selection.c,
        class_weight=selection.class_weight,
    )
    seasons = season_metrics(predictions)
    aggregate = aggregate_metrics(predictions)
    quality = probability_quality(predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(args.output_dir / "walk_forward_predictions.csv", index=False)
    folds.to_csv(args.output_dir / "walk_forward_folds.csv", index=False)
    seasons.to_csv(args.output_dir / "walk_forward_by_season.csv", index=False)
    aggregate.to_csv(args.output_dir / "walk_forward_aggregate.csv", index=False)

    print("\n=== Folds: training expands one season at a time ===")
    print(folds.round(4).to_string(index=False))

    print(f"\n=== Out-of-sample games: {len(predictions)} ===")
    print(
        "Every game predicted once, by a model that saw nothing from its "
        "season or later."
    )

    print("\n=== By season ===")
    print(seasons.round(4).to_string(index=False))

    print("\n=== Aggregate ===")
    print(aggregate.round(5).to_string(index=False))
    print(
        f"worst season: {aggregate.attrs['worst_season']}  "
        f"best season: {aggregate.attrs['best_season']}"
    )

    print("\n=== Are the probabilities worth anything? ===")
    print(quality.round(5).to_string(index=False))
    if not aggregate.attrs["beats_uninformative"]:
        print(
            f"\nThe model's probabilities are WORSE than predicting 0.5 for "
            f"every game (log loss {UNINFORMATIVE_LOG_LOSS:.4f}, Brier "
            f"{UNINFORMATIVE_BRIER:.2f}). They carry no usable information."
        )

    # The same out-of-sample games, scored for every trivial strategy.
    scored = eligible_games(matchups)
    scored = scored.loc[scored["game_id"].isin(set(predictions["game_id"]))]
    ordered = predictions.set_index("game_id").loc[scored["game_id"]]

    strategies = {name: function(scored) for name, function in BASELINES.items()}
    strategies["MODEL walk_forward"] = pd.Series(
        ordered["predicted_home_cover"].to_numpy(), index=scored.index, dtype="int8"
    )
    comparison = comparison_table(scored, strategies)
    comparison.to_csv(args.output_dir / "walk_forward_vs_baselines.csv", index=False)

    print(f"\n=== Against the baselines, same {len(scored)} games ===")
    print(comparison.to_string(index=False))

    model_row = comparison.loc[comparison["strategy"].str.startswith("MODEL")].iloc[0]
    best_baseline = comparison.loc[
        ~comparison["strategy"].str.startswith("MODEL")
    ].iloc[0]

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

    if model_row["roi"] <= best_baseline["roi"] or not model_row["beats_break_even"]:
        print(
            "\nAcross seven unseen seasons the model does NOT beat the trivial "
            "strategies and does not clear break-even. It has not been shown to "
            "add value and should not be described as if it had."
        )

    print(f"\nWrote reports to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
