"""Read-only HTTP access to what the Gridiron pipeline has already produced.

    uvicorn api.main:app --reload --port 8000

This service does not model anything. Every number it returns was read from a
file in ``outputs/`` or ``models/``; the only work done here is selecting
columns, joining a week's predictions to that week's results, and shaping the
whole into JSON. Each response carries a ``sources`` block naming the files it
was built from and when they were last written, so any figure shown in the
browser can be traced to a file without reading this code.

The single exception to read-only is ``POST /api/refresh/{week}``, which runs
the existing CLI as a subprocess. It passes arguments and reports the exit
code. It contains no prediction logic of its own, and could be replaced by
typing the same command in a terminal.

A note on what this cannot do. ``predict_week`` refuses games that have already
been played -- a finished game has a result, not a recommendation -- so weeks
that finished before the model was pointed at them have no prediction file and
never will. Those weeks are served with their results and a null prediction,
and the frontend is expected to say so rather than imply the model called them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi import Path as PathParam
from fastapi.middleware.cors import CORSMiddleware

from api import sources
from api.sources import (
    PREDICTION_FIELDS,
    PREDICTION_SEASON,
    PROJECT_ROOT,
    RESULT_FIELDS,
    SCHEDULE_FIELDS,
    SourceMissingError,
    provenance,
    records,
)

app = FastAPI(
    title="Gridiron API",
    version="1.0.0",
    description=(
        "Read-only access to the Gridiron NFL against-the-spread pipeline. "
        "Serves what the pipeline wrote; computes nothing."
    ),
)

# Local frontend dev servers. The regex covers whichever port Vite settles on
# without listing them all; nothing here is intended to be exposed publicly.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


def _missing(error: SourceMissingError) -> HTTPException:
    """A missing artifact is a server-state problem, not a bad request."""
    return HTTPException(status_code=503, detail=str(error))


# --- weeks -----------------------------------------------------------------


@app.get("/api/weeks", tags=["weeks"])
def list_weeks(season: int = PREDICTION_SEASON) -> dict[str, Any]:
    """Every week of the season, with its status and what exists for it."""
    try:
        schedule = sources.season_schedule(season)
    except SourceMissingError as error:
        raise _missing(error) from error

    weeks = []
    for week, games in schedule.groupby("week", sort=True):
        predictions = sources.read_week_predictions(int(week))
        weeks.append(
            {
                "week": int(week),
                "season": season,
                "status": sources.week_status(games),
                "games": int(len(games)),
                "games_played": int(games["home_score"].notna().sum()),
                "games_with_spread": int(games["spread_line"].notna().sum()),
                "first_gameday": games["gameday"].min(),
                "last_gameday": games["gameday"].max(),
                "has_predictions": predictions is not None,
                "predictions_available": (
                    0 if predictions is None else int(len(predictions))
                ),
                # Straight from the prediction CSV's own column, so the strip
                # can say when a week was last generated without opening it.
                "data_updated_at": sources.week_refreshed_at(predictions),
            }
        )

    return {
        "season": season,
        "weeks": weeks,
        "sources": provenance("matchups"),
    }


@app.get("/api/weeks/{week}", tags=["weeks"])
def get_week(
    week: int = PathParam(ge=1, le=22),
    season: int = PREDICTION_SEASON,
) -> dict[str, Any]:
    """One week's games: the schedule, any prediction, and any result.

    Games are always listed. A week with no prediction file still returns its
    matchups with ``prediction: null``, so the season can be browsed end to
    end rather than only where the model happened to run.
    """
    try:
        schedule = sources.season_schedule(season)
    except SourceMissingError as error:
        raise _missing(error) from error

    games = schedule.loc[schedule["week"].eq(week)].copy()
    if games.empty:
        raise HTTPException(
            status_code=404,
            detail=f"Season {season} has no week {week} in the schedule.",
        )

    predictions = sources.read_week_predictions(week)
    predicted_by_game: dict[str, dict] = {}
    if predictions is not None:
        for row in records(predictions):
            predicted_by_game[row["game_id"]] = row

    matchups = []
    for row in records(games.sort_values(["gameday", "game_id"])):
        prediction = predicted_by_game.get(row["game_id"])
        result = (
            {field: row.get(field) for field in RESULT_FIELDS}
            if row.get("home_score") is not None
            else None
        )

        matchups.append(
            {
                **{field: row.get(field) for field in SCHEDULE_FIELDS},
                "prediction": (
                    {field: prediction.get(field) for field in PREDICTION_FIELDS}
                    if prediction
                    else None
                ),
                "result": result,
                "pick_correct": _pick_correct(prediction, result, row["home_team"]),
            }
        )

    payload = {
        "season": season,
        "week": week,
        "status": sources.week_status(games),
        "has_predictions": predictions is not None,
        "matchups": matchups,
        "sources": provenance("matchups"),
    }

    report = sources.report_path(week)
    if report.exists():
        payload["report_text"] = report.read_text(encoding="utf-8")
    if predictions is not None:
        payload["sources"].append(
            {
                "name": "predictions",
                "path": sources.prediction_path(week)
                .relative_to(PROJECT_ROOT)
                .as_posix(),
                "modified_at": pd.Timestamp(
                    sources.prediction_path(week).stat().st_mtime, unit="s", tz="UTC"
                ).isoformat(),
            }
        )
    return payload


def _pick_correct(
    prediction: dict | None, result: dict | None, home_team: str
) -> bool | None:
    """Whether the predicted side covered.

    This is the one derived field in the service, and it derives nothing new:
    it compares two values that are already returned beside it --
    ``predicted_side`` from the prediction file and ``home_cover`` from the
    matchup table. Null when either is absent, and null on a push, where
    ``home_cover`` is itself null because there is no side that covered.
    """
    if not prediction or not result:
        return None
    side = prediction.get("predicted_side")
    covered = result.get("home_cover")
    if side is None or covered is None:
        return None
    return (side == home_team) == (float(covered) == 1.0)


# --- metrics ---------------------------------------------------------------


@app.get("/api/metrics/summary", tags=["metrics"])
def metrics_summary() -> dict[str, Any]:
    """Headline out-of-sample metrics, by season, and against the baselines.

    Sourced from ``walk_forward_aggregate.csv`` and
    ``walk_forward_by_season.csv``; this project has no ``model_metrics.json``
    or ``season_metrics.csv``.
    """
    try:
        aggregate = sources.read_csv("walk_forward_aggregate")
        seasons = sources.read_csv("walk_forward_by_season")
        baselines = sources.read_csv("walk_forward_vs_baselines")
        policies = sources.read_csv("policy_backtest")
        metadata = sources.read_json("model_metadata")
    except SourceMissingError as error:
        raise _missing(error) from error

    return {
        "overall": {
            str(metric): sources.scalar(value)
            for metric, value in zip(
                aggregate["metric"], aggregate["value"], strict=True
            )
        },
        "by_season": records(seasons),
        "baselines": records(baselines),
        "policies": records(policies),
        "model": {
            "model_type": metadata.get("model_type"),
            "training_seasons": metadata.get("training_seasons"),
            "prediction_season": metadata.get("prediction_season"),
            "n_training_games": metadata.get("n_training_games"),
            "excluded_pushes": metadata.get("excluded_pushes"),
            "selected_C": metadata.get("selected_C"),
            "class_weight": metadata.get("class_weight"),
            "created_at": metadata.get("created_at"),
            "random_seed": metadata.get("random_seed"),
            "feature_names": metadata.get("feature_names"),
            "exclusions": metadata.get("exclusions"),
            "validation_summary": metadata.get("validation_summary"),
        },
        "sources": provenance(
            "walk_forward_aggregate",
            "walk_forward_by_season",
            "walk_forward_vs_baselines",
            "policy_backtest",
            "model_metadata",
        ),
    }


@app.get("/api/metrics/calibration", tags=["metrics"])
def metrics_calibration() -> dict[str, Any]:
    """The numbers behind the reliability diagram, plus the calibrator trial."""
    try:
        reliability = sources.read_csv("calibration_reliability")
        comparison = sources.read_csv("calibration_comparison")
        buckets = sources.read_csv("policy_win_rate_by_confidence")
    except SourceMissingError as error:
        raise _missing(error) from error

    return {
        "reliability": records(reliability),
        "comparison": records(comparison),
        "win_rate_by_confidence": records(buckets),
        "sources": provenance(
            "calibration_reliability",
            "calibration_comparison",
            "policy_win_rate_by_confidence",
        ),
    }


@app.get("/api/metrics/backtest", tags=["metrics"])
def metrics_backtest() -> dict[str, Any]:
    """Everything the ROI page draws: per-bet, per-season and per-policy.

    The per-bet table is served whole -- 1,871 rows -- because a cumulative
    units curve needs every bet in order, and summarising it here would be the
    layer computing a statistic. The frontend runs the flat -110 arithmetic and
    checks its own total against ``policy_by_season.csv``, which this endpoint
    also returns for exactly that purpose.
    """
    try:
        bets = sources.read_csv("policy_bettable_predictions")
        seasons = sources.read_csv("policy_by_season")
        policies = sources.read_csv("policy_backtest")
        buckets = sources.read_csv("policy_win_rate_by_confidence")
    except SourceMissingError as error:
        raise _missing(error) from error

    return {
        "bets": records(bets),
        "by_season": records(seasons),
        "policies": records(policies),
        "win_rate_by_confidence": records(buckets),
        "sources": provenance(
            "policy_bettable_predictions",
            "policy_by_season",
            "policy_backtest",
            "policy_win_rate_by_confidence",
        ),
    }


@app.get("/api/metrics/coefficients", tags=["metrics"])
def metrics_coefficients() -> dict[str, Any]:
    """Coefficients, odds ratios, collinearity and cross-fold stability.

    The intercept is not served: it lives on a DataFrame attribute in the
    pipeline and is not written to ``model_coefficients.csv``, and this layer
    does not compute values that are not in a file.
    """
    try:
        coefficients = sources.read_csv("coefficients")
        odds_ratios = sources.read_csv("odds_ratios")
        collinearity = sources.read_csv("collinearity")
        stability = sources.read_csv("coefficient_stability")
        limitations = sources.read_json("limitations")
    except SourceMissingError as error:
        raise _missing(error) from error

    return {
        "coefficients": records(coefficients),
        "odds_ratios": records(odds_ratios),
        "collinearity": records(collinearity),
        "stability": records(stability),
        "limitations": limitations,
        "sources": provenance(
            "coefficients",
            "odds_ratios",
            "collinearity",
            "coefficient_stability",
            "limitations",
        ),
    }


@app.get("/api/metrics/correlation", tags=["metrics"])
def metrics_correlation() -> dict[str, Any]:
    """The correlation matrix behind the heatmap, as features plus values."""
    try:
        frame = sources.read_csv("correlation_model_fields", index_col=0)
        decisions = sources.read_csv("correlation_decisions")
    except SourceMissingError as error:
        raise _missing(error) from error

    return {
        "features": [str(name) for name in frame.index],
        "columns": [str(name) for name in frame.columns],
        "matrix": sources.matrix(frame),
        "decisions": records(decisions),
        "sources": provenance("correlation_model_fields", "correlation_decisions"),
    }


@app.get("/api/figures", tags=["metrics"])
def list_figures() -> dict[str, Any]:
    """The rendered PNGs, for anywhere the live charts do not replace them."""
    return {"figures": sources.figure_files()}


# --- refresh ---------------------------------------------------------------


def _pipeline_python() -> str:
    """The interpreter that has the gridiron package installed.

    The API may run in its own environment with only FastAPI and pandas, so
    ``sys.executable`` is not necessarily the one that can import gridiron.
    Order: an explicit override, then the project's own virtualenv, then this
    interpreter.
    """
    override = os.environ.get("GRIDIRON_PYTHON")
    if override:
        return override

    for candidate in (
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",
        PROJECT_ROOT / ".venv" / "bin" / "python",
    ):
        if Path(candidate).exists():
            return str(candidate)

    return sys.executable


@app.post("/api/refresh/{week}", tags=["refresh"])
def refresh_week(
    week: int = PathParam(ge=1, le=22),
    season: int = PREDICTION_SEASON,
    refresh_data: bool = False,
) -> dict[str, Any]:
    """Run the existing prediction command for one week.

    This is a shell-out, nothing more: the same command a person would type.
    It adds no prediction logic, and it is the only place the service causes a
    write. A completed week is refused by the command itself, and that refusal
    is passed back with its exit code rather than being hidden.
    """
    command = [
        _pipeline_python(),
        "-m",
        "gridiron.cli",
        "predict",
        "--season",
        str(season),
        "--week",
        str(week),
    ]
    if refresh_data:
        command.append("--refresh-data")

    try:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        raise HTTPException(
            status_code=504,
            detail=f"'{' '.join(command[1:])}' did not finish within 900 seconds.",
        ) from error
    except OSError as error:
        raise HTTPException(
            status_code=500,
            detail=f"Could not start the prediction command: {error}",
        ) from error

    written = sources.prediction_path(week)
    return {
        "season": season,
        "week": week,
        "command": " ".join(command),
        "returncode": completed.returncode,
        "succeeded": completed.returncode == 0,
        "stdout": completed.stdout[-8000:],
        "stderr": completed.stderr[-8000:],
        "predictions_written": written.exists(),
        "predictions_path": (
            written.relative_to(PROJECT_ROOT).as_posix() if written.exists() else None
        ),
    }


# --- service ---------------------------------------------------------------


@app.get("/api/health", tags=["service"])
def health() -> dict[str, Any]:
    """Which artifacts the service can see, and which are missing."""
    available = {}
    for name, path in sources.SOURCES.items():
        available[name] = {
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "present": path.exists(),
            "remedy": None if path.exists() else sources.REMEDIES.get(name),
        }

    weeks_on_disk = sorted(
        int(path.stem.split("_")[1])
        for path in sources.PREDICTION_DIR.glob("week_*_predictions.csv")
    )
    return {
        "status": "ok" if all(v["present"] for v in available.values()) else "degraded",
        "project_root": str(PROJECT_ROOT),
        "prediction_season": PREDICTION_SEASON,
        "prediction_weeks_on_disk": weeks_on_disk,
        "artifacts": available,
    }
