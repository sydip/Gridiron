"""Tests for the central command line.

Two things matter here. The dispatch table must stay honest -- every command
must name a script that exists and exposes a ``main`` that takes an argument
list -- and failures must be visible as nonzero exit codes rather than as a
cheerful zero and a message nobody reads.

Nothing in this file runs a real pipeline stage; the stages have their own
tests. What is tested is the wiring.
"""

from __future__ import annotations

import pytest

from gridiron import cli
from gridiron.workflow import SCRIPT_DIR, load_script

# --- the dispatch table ----------------------------------------------------


@pytest.mark.parametrize("name", sorted(cli.COMMANDS))
def test_every_command_names_a_script_that_exists(name):
    assert (SCRIPT_DIR / f"{cli.COMMANDS[name].script}.py").exists()


@pytest.mark.parametrize("name", sorted(cli.COMMANDS))
def test_every_command_exposes_a_main_that_takes_an_argument_list(name):
    """``main(argv)`` is what lets the CLI pass flags straight through."""
    import inspect

    module = load_script(cli.COMMANDS[name].script)
    parameters = inspect.signature(module.main).parameters

    assert callable(module.main)
    assert "argv" in parameters


@pytest.mark.parametrize("name", sorted(cli.COMMANDS))
def test_every_command_has_a_summary(name):
    summary = cli.COMMANDS[name].summary

    assert summary.endswith(".")
    assert len(summary) > 20


def test_the_five_documented_commands_are_all_present():
    """The command set the project documents, and promises to keep."""
    for name in ("download", "build-features", "train", "backtest", "predict"):
        assert name in cli.COMMANDS


def test_the_rebuild_chain_is_in_pipeline_order():
    assert cli.FULL_REBUILD == [
        "download",
        "build-features",
        "train",
        "backtest",
        "deploy",
    ]


def test_predict_is_not_in_the_rebuild_chain():
    """It needs a --week, so there is no sensible default run."""
    assert "predict" not in cli.FULL_REBUILD


def test_tune_is_not_in_the_rebuild_chain():
    """It rewrites version-controlled configuration; that stays deliberate."""
    assert "tune" not in cli.FULL_REBUILD


# --- usage and unknown commands --------------------------------------------


def test_no_arguments_prints_usage_and_exits_nonzero(capsys):
    assert cli.main([]) == 2
    assert "Usage: python -m gridiron.cli" in capsys.readouterr().out


def test_help_prints_usage_and_exits_zero(capsys):
    assert cli.main(["--help"]) == 0
    assert "Commands:" in capsys.readouterr().out


def test_usage_lists_every_command():
    text = cli.usage()

    for name in cli.COMMANDS:
        assert name in text
    assert "all" in text


def test_an_unknown_command_exits_two_and_says_so(caplog, capsys):
    with caplog.at_level("ERROR"):
        assert cli.main(["frobnicate"]) == 2

    assert "Unknown command 'frobnicate'" in caplog.text
    assert "Usage:" in capsys.readouterr().out


def test_a_near_miss_is_offered_a_suggestion(caplog):
    with caplog.at_level("ERROR"):
        cli.main(["prediction"])

    assert "predict" in caplog.text


# --- dispatch --------------------------------------------------------------


def test_arguments_are_passed_through_untouched(monkeypatch):
    seen = {}

    class _Stub:
        @staticmethod
        def main(argv):
            seen["argv"] = argv
            return 0

    monkeypatch.setattr(cli, "load_script", lambda name: _Stub)

    assert cli.main(["predict", "--season", "2026", "--week", "3"]) == 0
    assert seen["argv"] == ["--season", "2026", "--week", "3"]


def test_a_failing_command_propagates_its_exit_code(monkeypatch):
    class _Stub:
        @staticmethod
        def main(argv):
            return 1

    monkeypatch.setattr(cli, "load_script", lambda name: _Stub)

    assert cli.main(["train"]) == 1


def test_a_command_returning_none_is_treated_as_success(monkeypatch):
    class _Stub:
        @staticmethod
        def main(argv):
            return None

    monkeypatch.setattr(cli, "load_script", lambda name: _Stub)

    assert cli.main(["train"]) == 0


# --- the full rebuild ------------------------------------------------------


def test_the_rebuild_runs_every_stage_in_order(monkeypatch):
    ran = []

    def _dispatch(name, argv):
        ran.append(name)
        return 0

    monkeypatch.setattr(cli, "dispatch", _dispatch)

    assert cli.main(["all"]) == 0
    assert ran == cli.FULL_REBUILD


def test_the_rebuild_stops_at_the_first_failure(monkeypatch, caplog):
    ran = []

    def _dispatch(name, argv):
        ran.append(name)
        return 1 if name == "build-features" else 0

    monkeypatch.setattr(cli, "dispatch", _dispatch)

    with caplog.at_level("ERROR"):
        assert cli.main(["all"]) == 1

    assert ran == ["download", "build-features"]
    assert "build-features" in caplog.text


def test_force_refresh_reaches_every_stage(monkeypatch):
    forwarded = []

    def _dispatch(name, argv):
        forwarded.append(argv)
        return 0

    monkeypatch.setattr(cli, "dispatch", _dispatch)
    cli.main(["all", "--force-refresh"])

    assert all(argv == ["--force-refresh"] for argv in forwarded)


def test_a_stage_specific_flag_is_refused_rather_than_misapplied(monkeypatch, caplog):
    """--week means nothing to 'download'; silently dropping it would mislead."""
    monkeypatch.setattr(cli, "dispatch", lambda name, argv: 0)

    with caplog.at_level("ERROR"):
        assert cli.main(["all", "--week", "3"]) == 2

    assert "--week" in caplog.text
