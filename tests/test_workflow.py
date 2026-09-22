"""Tests for the shared command-line support.

The behaviours checked here are the ones a user notices when something goes
wrong: does the error say what to do, does the command warn before replacing
work, does a re-run skip what it would only reproduce, and is there a record
afterwards of what actually ran.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from gridiron.workflow import (
    OPTIONAL_PACKAGES,
    REQUIRED_PACKAGES,
    RunRecord,
    WorkflowError,
    add_common_arguments,
    check_dependencies,
    file_digest,
    is_up_to_date,
    load_script,
    missing_packages,
    require_inputs,
    warn_on_overwrite,
)


def _write(path, text="content"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _age(path, seconds):
    """Backdate a file, so staleness can be tested without sleeping."""
    stamp = time.time() + seconds
    os.utime(path, (stamp, stamp))
    return path


# --- dependency validation -------------------------------------------------


def test_the_real_environment_has_every_required_package():
    assert missing_packages() == []


def test_a_missing_package_is_named_with_the_install_command():
    with pytest.raises(WorkflowError) as error:
        check_dependencies(required={"not-a-real-package": "not_a_real_module"})

    message = str(error.value)
    assert "not-a-real-package" in message
    assert "pip install -r requirements.txt" in message


def test_every_optional_package_is_requestable_by_name():
    """A command asking for an optional package must not get a typo past us."""
    for name in OPTIONAL_PACKAGES:
        check_dependencies((name,))


def test_an_unknown_optional_package_is_rejected():
    with pytest.raises(WorkflowError, match="Unknown optional dependency"):
        check_dependencies(("scikit-lurn",))


def test_the_required_and_optional_sets_do_not_overlap():
    assert set(REQUIRED_PACKAGES) & set(OPTIONAL_PACKAGES) == set()


# --- input validation ------------------------------------------------------


def test_a_missing_input_is_named_along_with_the_remedy(tmp_path):
    with pytest.raises(WorkflowError) as error:
        require_inputs(
            {"matchup table": tmp_path / "matchups_wide.csv"},
            remedy="Run 'python -m gridiron.cli build-features' first.",
        )

    message = str(error.value)
    assert "matchup table" in message
    assert "matchups_wide.csv" in message
    assert "build-features" in message


def test_all_missing_inputs_are_reported_at_once(tmp_path):
    """One command should not have to be run once per missing file."""
    with pytest.raises(WorkflowError) as error:
        require_inputs(
            {
                "schedule": tmp_path / "schedule.parquet",
                "play-by-play": tmp_path / "pbp.parquet",
            },
            remedy="Run download first.",
        )

    message = str(error.value)
    assert "schedule" in message
    assert "play-by-play" in message


def test_present_inputs_raise_nothing(tmp_path):
    require_inputs({"schedule": _write(tmp_path / "s.csv")}, remedy="n/a")


# --- overwrite warnings ----------------------------------------------------


def test_existing_artifacts_are_counted_and_named(tmp_path, caplog):
    existing = _write(tmp_path / "model_table.csv")
    absent = tmp_path / "not_written_yet.csv"

    with caplog.at_level("WARNING"):
        count = warn_on_overwrite([existing, absent])

    assert count == 1
    assert "model_table.csv" in caplog.text
    assert "not_written_yet.csv" not in caplog.text


def test_nothing_is_said_when_nothing_would_be_replaced(tmp_path, caplog):
    with caplog.at_level("WARNING"):
        assert warn_on_overwrite([tmp_path / "absent.csv"]) == 0

    assert caplog.text == ""


def test_the_warning_reports_size_and_age(tmp_path, caplog):
    path = _age(_write(tmp_path / "old.csv", "x" * 500), -3 * 24 * 3600)

    with caplog.at_level("WARNING"):
        warn_on_overwrite([path])

    assert "500 bytes" in caplog.text
    assert "days old" in caplog.text


# --- staleness -------------------------------------------------------------


def test_outputs_newer_than_inputs_are_up_to_date(tmp_path):
    source = _age(_write(tmp_path / "in.parquet"), -60)
    built = _write(tmp_path / "out.csv")

    assert is_up_to_date([source], [built])


def test_a_touched_input_makes_the_outputs_stale(tmp_path):
    built = _age(_write(tmp_path / "out.csv"), -60)
    source = _write(tmp_path / "in.parquet")

    assert not is_up_to_date([source], [built])


def test_one_missing_output_is_enough_to_be_stale(tmp_path):
    source = _age(_write(tmp_path / "in.parquet"), -60)
    built = _write(tmp_path / "out.csv")

    assert not is_up_to_date([source], [built, tmp_path / "second.csv"])


def test_a_missing_input_is_not_reported_as_up_to_date(tmp_path):
    """Absent inputs are a validation failure, not a reason to skip work."""
    built = _write(tmp_path / "out.csv")

    assert not is_up_to_date([tmp_path / "gone.parquet"], [built])


def test_empty_sides_are_not_up_to_date(tmp_path):
    assert not is_up_to_date([], [_write(tmp_path / "out.csv")])
    assert not is_up_to_date([_write(tmp_path / "in.csv")], [])


# --- digests ---------------------------------------------------------------


def test_identical_content_digests_identically(tmp_path):
    first = _write(tmp_path / "a.csv", "season,week\n2026,1\n")
    second = _write(tmp_path / "b.csv", "season,week\n2026,1\n")

    assert file_digest(first) == file_digest(second)


def test_changed_content_changes_the_digest(tmp_path):
    path = _write(tmp_path / "a.csv", "one")
    before = file_digest(path)
    _write(path, "two")

    assert file_digest(path) != before


def test_an_oversized_file_is_skipped_rather_than_read(tmp_path):
    path = _write(tmp_path / "big.csv", "x" * 100)

    assert file_digest(path, limit_bytes=10) is None


def test_a_missing_file_has_no_digest(tmp_path):
    assert file_digest(tmp_path / "absent.csv") is None


# --- the run record --------------------------------------------------------


def _record(tmp_path, **kwargs):
    return RunRecord("test-command", ["--flag"], tmp_path / "runs", **kwargs)


def test_the_record_is_written_where_it_was_asked_to_be(tmp_path):
    path = _record(tmp_path).finish(0)

    assert path.parent == tmp_path / "runs"
    assert path.name.startswith("test-command-")
    assert path.suffix == ".json"


def test_the_record_captures_the_command_and_its_exit_code(tmp_path):
    payload = json.loads(_record(tmp_path).finish(1).read_text(encoding="utf-8"))

    assert payload["command"] == "test-command"
    assert payload["argv"] == ["--flag"]
    assert payload["exit_code"] == 1


def test_the_record_captures_the_resolved_arguments(tmp_path):
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2026)
    add_common_arguments(parser, tmp_path / "out")
    args = parser.parse_args([])

    record = _record(tmp_path)
    record.configure(args)
    payload = json.loads(record.finish(0).read_text(encoding="utf-8"))

    assert payload["arguments"]["season"] == 2026
    assert payload["arguments"]["force_refresh"] is False
    # Paths are recorded as text, so the record stays JSON-serialisable.
    assert payload["arguments"]["output_path"] == str(tmp_path / "out")


def test_the_record_digests_every_artifact_it_wrote(tmp_path):
    artifact = _write(tmp_path / "out.csv", "season\n2026\n")

    record = _record(tmp_path)
    record.add_artifact(artifact)
    payload = json.loads(record.finish(0).read_text(encoding="utf-8"))

    entry = payload["artifacts"][0]
    assert entry["path"] == str(artifact)
    assert entry["bytes"] == artifact.stat().st_size
    assert entry["sha256"] == file_digest(artifact)


def test_two_runs_over_the_same_content_record_the_same_digest(tmp_path):
    """This is what makes the idempotency claim checkable after the fact."""
    artifact = _write(tmp_path / "out.csv", "season\n2026\n")

    digests = []
    for _ in range(2):
        record = _record(tmp_path)
        record.add_artifact(artifact)
        payload = json.loads(record.finish(0).read_text(encoding="utf-8"))
        digests.append(payload["artifacts"][0]["sha256"])

    assert digests[0] == digests[1]


def test_inputs_are_recorded_without_being_digested(tmp_path):
    """Raw play-by-play runs to hundreds of megabytes; hashing it is waste."""
    source = _write(tmp_path / "in.parquet")

    record = _record(tmp_path)
    record.add_input(source, "play-by-play")
    payload = json.loads(record.finish(0).read_text(encoding="utf-8"))

    entry = payload["inputs"][0]
    assert entry["label"] == "play-by-play"
    assert "sha256" not in entry


def test_the_record_captures_the_code_and_package_versions(tmp_path):
    payload = json.loads(_record(tmp_path).finish(0).read_text(encoding="utf-8"))

    assert payload["environment"]["python"]
    assert payload["environment"]["packages"]["pandas"] != "not installed"
    assert set(payload["git"]) == {"commit", "branch", "dirty"}


def test_an_artifact_that_was_never_written_is_recorded_as_absent(tmp_path):
    record = _record(tmp_path)
    record.add_artifact(tmp_path / "never.csv")
    payload = json.loads(record.finish(0).read_text(encoding="utf-8"))

    assert payload["artifacts"][0]["exists"] is False


def test_notes_survive_into_the_record(tmp_path):
    record = _record(tmp_path)
    record.note("Skipped: outputs were newer than inputs.")
    payload = json.loads(record.finish(0).read_text(encoding="utf-8"))

    assert payload["notes"] == ["Skipped: outputs were newer than inputs."]


# --- common arguments ------------------------------------------------------


def test_output_dir_is_accepted_as_an_alias_of_output_path(tmp_path):
    import argparse

    parser = argparse.ArgumentParser()
    add_common_arguments(parser, tmp_path / "default")

    assert parser.parse_args([]).output_path == tmp_path / "default"
    assert parser.parse_args(["--output-path", "a"]).output_path.name == "a"
    assert parser.parse_args(["--output-dir", "b"]).output_path.name == "b"


def test_force_refresh_defaults_to_off(tmp_path):
    import argparse

    parser = argparse.ArgumentParser()
    add_common_arguments(parser, tmp_path)

    assert parser.parse_args([]).force_refresh is False
    assert parser.parse_args(["--force-refresh"]).force_refresh is True


# --- loading the phase scripts ---------------------------------------------


def test_a_script_loads_and_exposes_a_main():
    module = load_script("build_features")

    assert callable(module.main)


def test_an_unknown_script_is_named_in_the_error():
    with pytest.raises(WorkflowError, match="no_such_command"):
        load_script("no_such_command")
