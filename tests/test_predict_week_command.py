"""End-to-end smoke test for the weekly prediction command.

This runs ``scripts/predict_week.py`` the way a person would -- argument
parsing, data loading, feature rebuild, scoring, and both output files -- on a
synthetic universe of four teams. Every other prediction test checks one
function; this one checks that they still join up.

The fixtures are deliberately tiny and fixed: four teams, fifteen played weeks,
and one unplayed week to predict. That is enough to exercise the rolling
features (which need prior games), both play-by-play sources (finished seasons
and the one in progress), and the completed-game exclusion, while staying fast
enough to run on every commit.

Nothing here reaches the network or reads the repository's own data files, so
this passes on a fresh clone with no downloads and no fitted model.
"""

from __future__ import annotations

import importlib.util
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import (
    HISTORY_SEASON,
    PREDICTION_SEASON,
    TARGET_WEEK,
    synthetic_schedule,
    write_raw_data,
)
from gridiron.config import PROJECT_ROOT
from gridiron.modeling.deploy import DeploymentMetadata, save_deployment
from gridiron.modeling.pipeline import build_pipeline, get_feature_columns
from gridiron.prediction.weekly import PREDICTION_COLUMNS, STATUS_PREDICTED

FEATURES = get_feature_columns()


def _load_command():
    """Import scripts/predict_week.py as a module, by path.

    The scripts directory is not a package and is not on the path, so this
    loads the file the command line actually runs rather than a copy of it.
    """
    path = PROJECT_ROOT / "scripts" / "predict_week.py"
    spec = importlib.util.spec_from_file_location("predict_week_cli", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _deployment(directory):
    """A fitted model on disk, built here rather than read from the repo."""
    rng = np.random.default_rng(5)
    features = pd.DataFrame(rng.normal(0, 1, (400, len(FEATURES))), columns=FEATURES)
    labels = pd.Series((features["spread_line"] > 0).astype(int))
    pipeline = build_pipeline().fit(features, labels)

    schema = {
        "feature_names": FEATURES,
        "n_features": len(FEATURES),
        "target": "home_cover",
    }
    metadata = DeploymentMetadata(
        model_type="LogisticRegression",
        training_seasons=[HISTORY_SEASON],
        prediction_season=PREDICTION_SEASON,
        target="home_cover",
        feature_names=FEATURES,
        created_at="2026-09-01T00:00:00+00:00",
        n_training_games=len(features),
        excluded_pushes=0,
        selected_C=1.0,
        class_weight=None,
    )
    save_deployment(pipeline, schema, metadata, directory)
    return directory


@pytest.fixture
def world(tmp_path, monkeypatch):
    """A complete, self-contained installation: data, model, output directory."""
    raw = tmp_path / "raw"
    paths = write_raw_data(raw)

    from gridiron.prediction import weekly

    monkeypatch.setattr(weekly, "RAW_DIR", raw)
    monkeypatch.setattr(weekly, "SCHEDULE_PATH", paths["schedule"])
    monkeypatch.setattr(weekly, "HISTORICAL_PBP_PATH", paths["historical_pbp"])

    return {
        "command": _load_command(),
        "models": _deployment(tmp_path / "models"),
        "output": tmp_path / "predictions",
        "raw": raw,
        "schedule": synthetic_schedule(),
    }


def _run(world, week: int = TARGET_WEEK) -> int:
    return world["command"].main(
        [
            "--season",
            str(PREDICTION_SEASON),
            "--week",
            str(week),
            "--model-dir",
            str(world["models"]),
            "--output-dir",
            str(world["output"]),
        ]
    )


def _saved(world) -> pd.DataFrame:
    return pd.read_csv(world["output"] / f"week_{TARGET_WEEK:02d}_predictions.csv")


# --- the happy path --------------------------------------------------------


def test_the_command_runs_end_to_end_and_succeeds(world, capsys):
    assert _run(world) == 0
    assert "Wrote" in capsys.readouterr().out


def test_both_output_files_are_written(world):
    _run(world)

    directory = world["output"]
    assert (directory / f"week_{TARGET_WEEK:02d}_predictions.csv").exists()
    assert (directory / f"week_{TARGET_WEEK:02d}_report.txt").exists()


def test_the_saved_csv_has_every_agreed_column(world):
    _run(world)

    assert list(_saved(world).columns) == PREDICTION_COLUMNS


def test_one_row_per_game_in_the_requested_week(world):
    _run(world)
    table = _saved(world)

    schedule = world["schedule"]
    expected = schedule.loc[
        schedule["season"].eq(PREDICTION_SEASON) & schedule["week"].eq(TARGET_WEEK)
    ]
    assert len(table) == len(expected)
    assert set(table["game_id"]) == set(expected["game_id"])


def test_every_game_is_scored_and_the_probabilities_are_coherent(world):
    _run(world)
    table = _saved(world)

    assert (table["status"] == STATUS_PREDICTED).all()
    assert table["home_cover_probability"].between(0, 1).all()
    assert np.allclose(
        table["home_cover_probability"] + table["away_cover_probability"], 1.0
    )


def test_no_completed_game_reaches_the_output(world):
    """Weeks 1 and 2 of the prediction season are played; they have results."""
    _run(world)

    assert (_saved(world)["week"] == TARGET_WEEK).all()


def test_the_printed_report_is_the_report_that_was_saved(world, capsys):
    _run(world)

    saved = (world["output"] / f"week_{TARGET_WEEK:02d}_report.txt").read_text(
        encoding="utf-8"
    )
    printed = capsys.readouterr().out
    for line in saved.splitlines():
        if line.strip():
            assert line in printed


def test_the_command_reaches_no_network_without_refresh_data(world, monkeypatch):
    """A default run must read the disk only; downloading is opt-in."""
    import gridiron.data.download as download

    def _forbidden(*args, **kwargs):
        raise AssertionError("the command downloaded data without --refresh-data")

    monkeypatch.setattr(download, "load_schedules", _forbidden)
    monkeypatch.setattr(download, "load_historical_pbp", _forbidden)

    assert _run(world) == 0


def test_rerunning_the_command_reproduces_the_same_predictions(world):
    path = world["output"] / f"week_{TARGET_WEEK:02d}_predictions.csv"

    _run(world)
    first = path.read_text(encoding="utf-8")
    _run(world)

    assert path.read_text(encoding="utf-8") == first


# --- the failure paths -----------------------------------------------------


def test_a_missing_model_is_reported_with_the_remedy(world, tmp_path, caplog):
    world["models"] = tmp_path / "no-model-here"

    with caplog.at_level("ERROR"):
        assert _run(world) == 1

    assert "gridiron.cli deploy" in caplog.text


def test_an_unscheduled_week_is_refused_rather_than_invented(world, caplog):
    with caplog.at_level("ERROR"):
        assert _run(world, week=18) == 1

    assert "week 18" in caplog.text.lower()
    assert not world["output"].exists()


def test_a_fully_played_week_is_refused(world, caplog):
    """Week 1 has results. Results are not recommendations."""
    with caplog.at_level("ERROR"):
        assert _run(world, week=1) == 1

    assert "already been played" in caplog.text
