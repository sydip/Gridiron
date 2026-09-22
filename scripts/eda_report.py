"""Run the exploratory analysis: ten charts plus the correlation review.

Charts are written to ``outputs/figures``; the correlation matrix, the flagged
pairs, and the decision record are written to ``outputs/reports``.

Usage::

    python scripts/eda_report.py
    python scripts/eda_report.py --seasons 2023 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")

from gridiron.config import PROJECT_ROOT, configure_environment  # noqa: E402
from gridiron.data.clean import clean_schedules  # noqa: E402
from gridiron.eda.charts import build_all_charts  # noqa: E402
from gridiron.eda.correlation import (  # noqa: E402
    CORRELATION_THRESHOLD,
    DEFENSIVE_EPA_DECISION,
    correlated_pairs,
    correlation_decisions,
    correlation_matrix,
    numeric_fields,
)
from gridiron.features.epa import (  # noqa: E402
    aggregate_defensive_epa,
    aggregate_offensive_epa,
)
from gridiron.features.matchup import (  # noqa: E402
    FEATURE_COLUMNS,
    add_matchup_target,
    build_matchups,
)
from gridiron.features.pace import aggregate_pace  # noqa: E402
from gridiron.features.rolling import build_rolling_features  # noqa: E402
from gridiron.features.team_games import team_game_history  # noqa: E402

LOGGER = logging.getLogger("eda_report")

PBP_PATH = PROJECT_ROOT / "data" / "raw" / "pbp_2016_2025.parquet"
SCHEDULE_PATH = PROJECT_ROOT / "data" / "raw" / "schedules_2016_2026.parquet"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"

PBP_COLUMNS = [
    "season",
    "week",
    "game_id",
    "posteam",
    "defteam",
    "play_type",
    "epa",
    "success",
    "pass",
    "rush",
    "qb_kneel",
    "qb_spike",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "no_huddle",
    "fixed_drive",
    "play_id",
    "score_differential",
]

# The wider candidate set the correlation policy asks about: every differential
# available, not only the eleven that reached the model row.
CANDIDATE_PREFIXES = (
    "off_",
    "def_",
    "pace_",
    "point_",
    "win_",
    "ats_",
    "cover_",
    "rest_",
)
CANDIDATE_SUFFIXES = ("_last_3", "_last_5", "_season", "_diff")


def candidate_differentials(matchups: pd.DataFrame) -> list[str]:
    """Every numeric differential in the wide matchup frame."""
    return numeric_fields(
        matchups,
        [
            column
            for column in matchups.columns
            if column.startswith(CANDIDATE_PREFIXES)
            and column.endswith(CANDIDATE_SUFFIXES)
        ],
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seasons", type=int, nargs="*", default=None)
    parser.add_argument("--pbp-path", type=Path, default=PBP_PATH)
    parser.add_argument("--schedule-path", type=Path, default=SCHEDULE_PATH)
    parser.add_argument("--figure-dir", type=Path, default=FIGURE_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    configure_environment()

    for path in (args.pbp_path, args.schedule_path):
        if not path.exists():
            LOGGER.error("Required data not found at %s", path)
            LOGGER.error("Run 'python -m gridiron.cli download' first.")
            return 1

    pbp = pd.read_parquet(args.pbp_path, columns=PBP_COLUMNS)
    schedule = clean_schedules(pd.read_parquet(args.schedule_path))
    if args.seasons:
        pbp = pbp.loc[pbp["season"].isin(args.seasons)]
        schedule = schedule.loc[schedule["season"].isin(args.seasons)]

    team_games = build_rolling_features(
        team_game_history(schedule),
        aggregate_offensive_epa(pbp),
        aggregate_defensive_epa(pbp),
        aggregate_pace(pbp),
    )
    team_games = team_games.merge(
        aggregate_pace(pbp)[
            ["season", "week", "game_id", "team", "seconds_per_play"]
        ].assign(team=lambda frame: frame["team"].astype("string")),
        on=["season", "week", "game_id", "team"],
        how="left",
    )
    matchups = add_matchup_target(build_matchups(team_games, schedule))

    paths = build_all_charts(matchups, team_games, FEATURE_COLUMNS, args.figure_dir)

    model_numeric = numeric_fields(matchups, FEATURE_COLUMNS)
    model_matrix = correlation_matrix(matchups, model_numeric)
    model_pairs = correlated_pairs(model_matrix, CORRELATION_THRESHOLD)

    candidates = candidate_differentials(matchups)
    wide_matrix = correlation_matrix(matchups, candidates)
    wide_pairs = correlated_pairs(wide_matrix, CORRELATION_THRESHOLD)

    decisions = correlation_decisions(wide_pairs, extra=(DEFENSIVE_EPA_DECISION,))

    args.report_dir.mkdir(parents=True, exist_ok=True)
    model_matrix.to_csv(args.report_dir / "eda_correlation_model_fields.csv")
    wide_matrix.to_csv(args.report_dir / "eda_correlation_all_differentials.csv")
    wide_pairs.to_csv(args.report_dir / "eda_correlated_pairs.csv", index=False)
    decisions.to_csv(args.report_dir / "eda_correlation_decisions.csv", index=False)

    print(f"\n=== Charts written to {args.figure_dir} ===")
    for path in paths:
        print(f"  {path.name}")

    print(f"\n=== Model fields at |r| >= {CORRELATION_THRESHOLD} ===")
    if model_pairs.empty:
        # Report how close the set came, so "no pairs" is informative rather
        # than merely empty.
        near = correlated_pairs(model_matrix, 0.0)
        strongest = near.iloc[0]
        print(
            f"  none. Strongest pair is {strongest['left']} ~ "
            f"{strongest['right']} at r = {strongest['correlation']:+.3f}"
        )
    else:
        print(model_pairs.to_string(index=False))

    print(f"\n=== All differentials at |r| >= {CORRELATION_THRESHOLD} ===")
    print(wide_pairs.to_string(index=False))

    print("\n=== Decisions recorded ===")
    print(
        decisions[["left", "right", "correlation", "decision"]].to_string(index=False)
    )
    print(f"\nWrote correlation reports to {args.report_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
