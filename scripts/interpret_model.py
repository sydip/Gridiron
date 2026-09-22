"""Interpret the fitted model: coefficients, odds ratios, and the caveats.

Fits the deployment model on the full training window (2016-2025, as the
specification prescribes), then writes the coefficient table, the odds-ratio
view, the collinearity diagnostics, the fold-to-fold stability of each
coefficient, and the horizontal coefficient plot.

The limitations are printed with the table rather than after it, because a
coefficient table read on its own from this model would mislead.

Usage::

    python scripts/interpret_model.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from gridiron.config import (  # noqa: E402
    PROJECT_ROOT,
    configure_environment,
    seed_everything,
)
from gridiron.eda.coefficient_charts import build_coefficient_charts  # noqa: E402
from gridiron.modeling.baselines import eligible_games  # noqa: E402
from gridiron.modeling.evaluate import (  # noqa: E402
    aggregate_metrics,
    run_walk_forward,
)
from gridiron.modeling.interpretation import (  # noqa: E402
    coefficient_stability,
    coefficient_table,
    model_limitations,
    multicollinearity_warnings,
    variance_inflation_factors,
)
from gridiron.modeling.pipeline import (  # noqa: E402
    feature_matrix,
    target_vector,
)
from gridiron.modeling.train import train_model  # noqa: E402
from gridiron.modeling.tuning import load_selected_parameters  # noqa: E402

LOGGER = logging.getLogger("interpret_model")

MATCHUP_PATH = PROJECT_ROOT / "outputs" / "reports" / "matchups_wide.csv"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

TRAINING_SEASONS = list(range(2016, 2026))

TABLE_VIEW = [
    "feature",
    "coefficient",
    "odds_ratio",
    "odds_change_pct",
    "vif",
    "collinearity_flag",
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
    selection = load_selected_parameters()

    training = eligible_games(matchups)
    training = training.loc[training["season"].isin(TRAINING_SEASONS)]
    features = feature_matrix(training)
    labels = target_vector(training)

    pipeline = train_model(
        features, labels, c=selection.c, class_weight=selection.class_weight
    )

    table = coefficient_table(pipeline, features)
    inflation = variance_inflation_factors(pipeline, features)

    # Fold-to-fold coefficients come from the same walk-forward machinery the
    # backtest uses, so stability is measured on unseen seasons.
    _, folds = run_walk_forward(
        matchups,
        c=selection.c,
        class_weight=selection.class_weight,
        collect_coefficients=True,
    )
    stability = coefficient_stability(folds.attrs["coefficients"])

    predictions, _ = run_walk_forward(
        matchups, c=selection.c, class_weight=selection.class_weight
    )
    metrics = aggregate_metrics(predictions).set_index("metric")["value"].to_dict()

    warnings = multicollinearity_warnings(table)
    limitations = model_limitations(table, stability, metrics)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.report_dir / "model_coefficients.csv", index=False)
    table[["feature", "odds_ratio", "odds_change_pct", "interpretation"]].to_csv(
        args.report_dir / "model_odds_ratios.csv", index=False
    )
    inflation.to_csv(args.report_dir / "model_collinearity.csv", index=False)
    stability.to_csv(args.report_dir / "model_coefficient_stability.csv", index=False)
    (args.report_dir / "model_limitations.json").write_text(
        json.dumps([asdict(item) for item in limitations], indent=2),
        encoding="utf-8",
    )

    paths = build_coefficient_charts(table, stability, args.figure_dir)

    print(
        f"\n=== Deployment model: trained on {len(training)} games, "
        f"{min(TRAINING_SEASONS)}-{max(TRAINING_SEASONS)} ==="
    )
    print(f"C={selection.c}, class_weight={selection.class_weight}")

    print("\n=== Coefficients, largest absolute first ===")
    print(table[TABLE_VIEW].round(5).to_string(index=False))
    print(f"intercept: {table.attrs['intercept']:+.5f}")
    print(
        "\nA coefficient is the change in log-odds of a home cover per one "
        "standard deviation of that feature; odds_change_pct is the same thing "
        "as a percentage change in the odds."
    )

    print("\n=== What each direction means ===")
    for row in table.itertuples():
        print(f"\n  {row.feature} ({row.coefficient:+.5f})")
        print(f"    {row.interpretation}")

    print("\n=== Multicollinearity ===")
    for warning in warnings:
        print(f"  ! {warning}")

    print("\n=== Coefficient stability across folds ===")
    print(
        stability[
            [
                "feature",
                "folds",
                "mean_coefficient",
                "std_coefficient",
                "sign_changes",
            ]
        ]
        .round(5)
        .to_string(index=False)
    )

    print("\n=== Limitations ===")
    for index, item in enumerate(limitations, start=1):
        print(f"  {index}. {item.heading}")
        print(f"     {item.detail}")

    print("\nCharts:")
    for path in paths:
        print(f"  {path.name}")
    print(f"Reports written to {args.report_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
