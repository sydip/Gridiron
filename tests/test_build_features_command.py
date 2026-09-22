"""End-to-end tests for the build-features and backtest commands.

``build-features`` is run for real against the synthetic universe, because it
is the stage everything downstream depends on and it is cheap to run. The
backtest is not: it fits a model per fold over every season, which is minutes
rather than seconds, so what is checked here is its guard rails -- the
validation, the refusal, and the skip -- rather than its arithmetic, which
``test_walk_forward.py`` and ``test_policies.py`` already cover.
"""

from __future__ import annotations

import json
import os
import time

import pandas as pd
import pytest

from conftest import combined_pbp_path, write_raw_data
from gridiron.features.build import feature_output_paths
from gridiron.workflow import load_script


@pytest.fixture
def raw(tmp_path):
    """Raw parquet on disk, plus a single combined play-by-play file."""
    paths = write_raw_data(tmp_path / "raw")
    paths["pbp"] = combined_pbp_path(tmp_path / "raw")
    return paths


@pytest.fixture
def build(raw, tmp_path):
    """The build-features command, wired to the synthetic data."""
    module = load_script("build_features")
    output = tmp_path / "reports"
    runs = tmp_path / "runs"

    def _run(*extra):
        return module.main(
            [
                "--pbp-path",
                str(raw["pbp"]),
                "--schedule-path",
                str(raw["schedule"]),
                "--output-path",
                str(output),
                "--run-record-dir",
                str(runs),
                *extra,
            ]
        )

    _run.output = output
    _run.runs = runs
    return _run


def _latest_record(directory):
    records = sorted(directory.glob("*.json"))
    return json.loads(records[-1].read_text(encoding="utf-8"))


# --- build-features, the happy path ----------------------------------------


def test_the_command_succeeds_on_the_synthetic_universe(build):
    assert build() == 0


def test_every_expected_table_is_written(build):
    build()

    for path in feature_output_paths(build.output):
        assert path.exists(), f"{path.name} was not written"


def test_the_model_table_has_one_row_per_game(build, raw):
    build()

    table = pd.read_csv(build.output / "model_table.csv")
    schedule = pd.read_parquet(raw["schedule"])

    assert len(table) == len(schedule)
    assert not table["game_id"].duplicated().any()


def test_the_artifacts_land_under_output_path(build, tmp_path):
    build()

    assert (build.output / "matchups_wide.csv").exists()
    assert not (tmp_path / "matchups_wide.csv").exists()


def test_a_season_filter_narrows_the_table(build):
    build("--seasons", "2025")

    table = pd.read_csv(build.output / "model_table.csv")

    assert set(table["season"]) == {2025}


# --- build-features, idempotency -------------------------------------------


def test_running_twice_produces_identical_tables(build):
    path = build.output / "model_table.csv"

    build()
    first = path.read_text(encoding="utf-8")
    build("--force-refresh")

    assert path.read_text(encoding="utf-8") == first


def test_a_second_run_skips_the_work_and_says_so(build, capsys):
    build()
    capsys.readouterr()

    assert build() == 0
    output = capsys.readouterr().out
    assert "already up to date" in output
    assert "--force-refresh" in output


def test_force_refresh_rebuilds_anyway(build, capsys):
    build()
    capsys.readouterr()

    assert build("--force-refresh") == 0
    assert "already up to date" not in capsys.readouterr().out


def test_a_changed_input_makes_the_tables_stale_again(build, raw, capsys):
    build()
    # Touch the schedule forward, as a fresh download would.
    stamp = time.time() + 60
    os.utime(raw["schedule"], (stamp, stamp))
    capsys.readouterr()

    assert build() == 0
    assert "already up to date" not in capsys.readouterr().out


def test_a_season_filter_never_takes_the_skip_path(build, capsys):
    """The filtered table differs from the full one built at the same time."""
    build()
    capsys.readouterr()

    build("--seasons", "2025")

    assert "already up to date" not in capsys.readouterr().out


# --- build-features, failures ----------------------------------------------


def test_missing_raw_data_exits_nonzero_and_names_the_remedy(tmp_path, caplog):
    module = load_script("build_features")

    with caplog.at_level("ERROR"):
        code = module.main(
            [
                "--pbp-path",
                str(tmp_path / "absent_pbp.parquet"),
                "--schedule-path",
                str(tmp_path / "absent_schedule.parquet"),
                "--output-path",
                str(tmp_path / "reports"),
                "--run-record-dir",
                str(tmp_path / "runs"),
            ]
        )

    assert code == 1
    assert "absent_pbp.parquet" in caplog.text
    assert "absent_schedule.parquet" in caplog.text
    assert "download" in caplog.text


def test_a_failed_run_still_leaves_a_record(tmp_path):
    module = load_script("build_features")
    runs = tmp_path / "runs"
    module.main(
        [
            "--pbp-path",
            str(tmp_path / "absent.parquet"),
            "--schedule-path",
            str(tmp_path / "absent2.parquet"),
            "--output-path",
            str(tmp_path / "reports"),
            "--run-record-dir",
            str(runs),
        ]
    )

    assert _latest_record(runs)["exit_code"] == 1


# --- build-features, the run record ----------------------------------------


def test_the_run_records_the_configuration_it_used(build):
    build()
    record = _latest_record(build.runs)

    assert record["command"] == "build-features"
    assert record["exit_code"] == 0
    assert record["arguments"]["output_path"] == str(build.output)
    assert record["environment"]["packages"]["pandas"] != "not installed"


def test_the_run_record_digests_the_tables_it_wrote(build):
    build()
    record = _latest_record(build.runs)

    written = {item["path"]: item for item in record["artifacts"]}
    entry = written[str(build.output / "model_table.csv")]

    assert entry["sha256"]
    assert entry["bytes"] > 0


def test_a_skipped_run_records_why_it_did_nothing(build):
    build()
    build()

    assert "Skipped" in " ".join(_latest_record(build.runs)["notes"])


def test_the_two_runs_record_the_same_digest(build):
    """Idempotency, stated in a form that can be checked later."""
    build()
    first = _latest_record(build.runs)
    build("--force-refresh")
    second = _latest_record(build.runs)

    def _digests(record):
        return {item["path"]: item["sha256"] for item in record["artifacts"]}

    assert _digests(first) == _digests(second)


# --- backtest, guard rails -------------------------------------------------


def test_the_backtest_refuses_without_a_matchup_table(tmp_path, caplog):
    module = load_script("backtest_model")

    with caplog.at_level("ERROR"):
        code = module.main(
            [
                "--matchup-path",
                str(tmp_path / "absent.csv"),
                "--output-path",
                str(tmp_path / "reports"),
                "--run-record-dir",
                str(tmp_path / "runs"),
            ]
        )

    assert code == 1
    assert "absent.csv" in caplog.text
    assert "build-features" in caplog.text


def test_the_backtest_skips_when_its_reports_are_current(build, tmp_path, capsys):
    """Its outputs are expensive, so an unchanged input must not rerun them."""
    build()
    module = load_script("backtest_model")

    # Stand in for a completed backtest: the files exist and post-date the
    # matchup table, which is exactly the condition for skipping.
    reports = tmp_path / "backtest"
    reports.mkdir()
    names = module.WALK_FORWARD_OUTPUTS + module.POLICY_OUTPUTS
    for name in names:
        (reports / name).write_text("placeholder\n", encoding="utf-8")

    capsys.readouterr()
    code = module.main(
        [
            "--matchup-path",
            str(build.output / "matchups_wide.csv"),
            "--output-path",
            str(reports),
            "--run-record-dir",
            str(tmp_path / "runs"),
        ]
    )

    assert code == 0
    assert "already up to date" in capsys.readouterr().out
