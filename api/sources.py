"""Every file the API reads, and the only place it reads them.

The rule this module exists to enforce: the serving layer computes no
statistics. It opens files the pipeline already wrote, reshapes them into
JSON, and says where each one came from. If a number is wrong here, it is
wrong in the CSV, and the fix belongs in the pipeline rather than in the API.

Nothing in this module writes to the project. The one exception in the whole
service is the refresh endpoint, which runs the existing CLI as a subprocess
and lets *that* write, exactly as it would from a terminal.

File names below are the real ones on disk. Several differ from the names in
the original request -- there is no ``model_metrics.json`` or
``season_metrics.csv`` in this project -- so the mapping is written out
explicitly rather than guessed at each call site.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# api/ sits at the project root, beside src/, outputs/ and models/.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"
MODEL_DIR = PROJECT_ROOT / "models"

PREDICTION_SEASON = 2026

# Logical name -> file on disk. The frontend never sees these paths, but every
# response carries the ones it was built from, so any figure on the page can be
# traced back to a file without reading this source.
SOURCES: dict[str, Path] = {
    "matchups": REPORT_DIR / "matchups_wide.csv",
    "walk_forward_aggregate": REPORT_DIR / "walk_forward_aggregate.csv",
    "walk_forward_by_season": REPORT_DIR / "walk_forward_by_season.csv",
    "walk_forward_vs_baselines": REPORT_DIR / "walk_forward_vs_baselines.csv",
    "policy_backtest": REPORT_DIR / "policy_backtest.csv",
    "policy_win_rate_by_confidence": REPORT_DIR / "policy_win_rate_by_confidence.csv",
    "policy_by_season": REPORT_DIR / "policy_by_season.csv",
    "policy_bettable_predictions": REPORT_DIR / "policy_bettable_predictions.csv",
    "calibration_reliability": REPORT_DIR / "calibration_reliability.csv",
    "calibration_comparison": REPORT_DIR / "calibration_comparison.csv",
    "coefficients": REPORT_DIR / "model_coefficients.csv",
    "odds_ratios": REPORT_DIR / "model_odds_ratios.csv",
    "collinearity": REPORT_DIR / "model_collinearity.csv",
    "coefficient_stability": REPORT_DIR / "model_coefficient_stability.csv",
    "limitations": REPORT_DIR / "model_limitations.json",
    "correlation_model_fields": REPORT_DIR / "eda_correlation_model_fields.csv",
    "correlation_decisions": REPORT_DIR / "eda_correlation_decisions.csv",
    "model_metadata": MODEL_DIR / "model_metadata.json",
}

# The command that produces each source, quoted back to the caller when a file
# is missing. A 503 that says "run this" beats one that says "not found".
REMEDIES: dict[str, str] = {
    "matchups": "python -m gridiron.cli build-features",
    "walk_forward_aggregate": "python -m gridiron.cli backtest",
    "walk_forward_by_season": "python -m gridiron.cli backtest",
    "walk_forward_vs_baselines": "python -m gridiron.cli backtest",
    "policy_backtest": "python -m gridiron.cli backtest",
    "policy_win_rate_by_confidence": "python -m gridiron.cli backtest",
    "policy_by_season": "python -m gridiron.cli backtest",
    "policy_bettable_predictions": "python -m gridiron.cli backtest",
    "calibration_reliability": "python scripts/calibration_report.py",
    "calibration_comparison": "python scripts/calibration_report.py",
    "coefficients": "python -m gridiron.cli train",
    "odds_ratios": "python scripts/interpret_model.py",
    "collinearity": "python scripts/interpret_model.py",
    "coefficient_stability": "python scripts/interpret_model.py",
    "limitations": "python scripts/interpret_model.py",
    "correlation_model_fields": "python scripts/eda_report.py",
    "correlation_decisions": "python scripts/eda_report.py",
    "model_metadata": "python -m gridiron.cli deploy",
}


class SourceMissingError(RuntimeError):
    """A required artifact has not been generated yet."""

    def __init__(self, name: str) -> None:
        path = SOURCES[name]
        remedy = REMEDIES.get(name, "run the pipeline")
        super().__init__(
            f"{path.relative_to(PROJECT_ROOT).as_posix()} has not been "
            f"generated. Run '{remedy}' first."
        )
        self.name = name
        self.path = path


@dataclass(frozen=True)
class Source:
    """Where a payload came from, carried back with the payload."""

    name: str
    path: str
    modified_at: str | None

    @classmethod
    def of(cls, name: str) -> Source:
        path = SOURCES[name]
        stamp = None
        if path.exists():
            stamp = pd.Timestamp(path.stat().st_mtime, unit="s", tz="UTC").isoformat()
        return cls(
            name=name,
            path=path.relative_to(PROJECT_ROOT).as_posix(),
            modified_at=stamp,
        )


def provenance(*names: str) -> list[dict[str, str | None]]:
    """The source block every response carries."""
    return [vars(Source.of(name)) for name in names]


# --- reading ---------------------------------------------------------------


def read_csv(name: str, **kwargs) -> pd.DataFrame:
    """Load one of the known sources, or say what has not been run."""
    path = SOURCES[name]
    if not path.exists():
        raise SourceMissingError(name)
    return pd.read_csv(path, **kwargs)


def read_json(name: str):
    path = SOURCES[name]
    if not path.exists():
        raise SourceMissingError(name)
    return json.loads(path.read_text(encoding="utf-8"))


def _scalar(value: object) -> object:
    """One cell, as a JSON-safe Python value, with its value unchanged.

    Deliberately not ``DataFrame.to_json``: that defaults to
    ``double_precision=10`` and silently rounds every float, so a probability
    of 0.5002683352080302 is served as 0.5002683352. Rounding a served value
    is exactly what this layer must not do. ``numpy.generic.item()`` converts
    to the Python equivalent without touching the value, and Python's JSON
    encoder writes floats at full round-trip precision.
    """
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or value is pd.NaT:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def records(frame: pd.DataFrame) -> list[dict]:
    """A frame as JSON-safe records, with no values altered."""
    return [
        {str(key): _scalar(value) for key, value in row.items()}
        for row in frame.to_dict(orient="records")
    ]


def matrix(frame: pd.DataFrame) -> list[list[float | None]]:
    """A square frame as nested lists, values untouched."""
    return [[_scalar(value) for value in row] for row in frame.to_numpy().tolist()]


def scalar(value: object) -> object:
    """Public form of :func:`_scalar`, for values read outside a frame."""
    return _scalar(value)


# --- the 2026 season -------------------------------------------------------

# Columns carried from the schedule for every game, whether or not a
# prediction exists for it.
SCHEDULE_FIELDS = [
    "game_id",
    "season",
    "week",
    "gameday",
    "away_team",
    "home_team",
    "spread_line",
    "div_game",
]

# Columns carried from a week's prediction CSV, verbatim.
PREDICTION_FIELDS = [
    "away_cover_probability",
    "home_cover_probability",
    "predicted_side",
    "confidence",
    "recommendation_tier",
    "status",
    "spread_line_used",
    "spread_source",
    "spread_timestamp",
    "data_updated_at",
    "prior_games_this_season",
]

# Columns carried from the matchup table once a game has been played.
RESULT_FIELDS = [
    "home_score",
    "away_score",
    "home_margin",
    "adjusted_home_margin",
    "home_cover",
]

STATUS_COMPLETED = "completed"
STATUS_UPCOMING = "upcoming"
STATUS_PENDING_SPREAD = "pending_spread"


def season_schedule(season: int = PREDICTION_SEASON) -> pd.DataFrame:
    """Every game of the season, from the matchup table."""
    frame = read_csv("matchups")
    season_rows = frame.loc[frame["season"].eq(season)].copy()
    if season_rows.empty:
        raise SourceMissingError("matchups")
    return season_rows


def prediction_path(week: int, season: int = PREDICTION_SEASON) -> Path:
    """Where ``predict_week`` writes a week, whether or not it has run."""
    return PREDICTION_DIR / f"week_{week:02d}_predictions.csv"


def report_path(week: int, season: int = PREDICTION_SEASON) -> Path:
    return PREDICTION_DIR / f"week_{week:02d}_report.txt"


def read_week_predictions(week: int) -> pd.DataFrame | None:
    """A week's saved predictions, or None if the week has never been run."""
    path = prediction_path(week)
    if not path.exists():
        return None
    return pd.read_csv(path)


def week_refreshed_at(predictions: pd.DataFrame | None) -> str | None:
    """When the data behind a week's predictions was last refreshed.

    Read from the ``data_updated_at`` column the prediction CSV already
    carries, not from the file's mtime: the column records when the *source
    data* was pulled, which is what a reader wants to know. A file rewritten
    from stale data is not fresher for having been rewritten.
    """
    if predictions is None or predictions.empty:
        return None
    if "data_updated_at" not in predictions.columns:
        return None

    values = predictions["data_updated_at"].dropna()
    if values.empty:
        return None
    return str(values.iloc[0])


def week_status(games: pd.DataFrame) -> str:
    """Classify a week from the schedule fields already in the table.

    Three states, decided only by what the file says: every game has a score
    (completed), no game has a line yet (pending_spread), or neither
    (upcoming). Weeks part-way through a state resolve to ``upcoming``, which
    is the one the frontend renders with both settled and unsettled rows.
    """
    played = int(games["home_score"].notna().sum())
    priced = int(games["spread_line"].notna().sum())

    if played == len(games) and len(games) > 0:
        return STATUS_COMPLETED
    if priced == 0:
        return STATUS_PENDING_SPREAD
    return STATUS_UPCOMING


def figure_files() -> list[dict[str, object]]:
    """The PNGs on disk, listed so the frontend can link to them if wanted."""
    if not FIGURE_DIR.exists():
        return []
    return [
        {
            "name": path.name,
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
        }
        for path in sorted(FIGURE_DIR.glob("*.png"))
    ]
