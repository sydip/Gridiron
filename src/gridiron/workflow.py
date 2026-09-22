"""Shared support for the command-line workflow.

Every command in this project has the same obligations: check that what it
needs is installed and present, refuse clearly when it is not, say what it is
about to overwrite, record the configuration it ran under, and exit nonzero on
failure. Doing that once here keeps the commands themselves short and, more
importantly, keeps them consistent -- a user who learns what ``--force-refresh``
means for one command has learned what it means for all of them.

Two conventions are worth stating outright.

**Overwriting warns rather than refuses.** Re-running a command with the same
inputs should produce the same outputs, so blocking on an existing file would
break the idempotency the workflow depends on. Instead, every command lists
what it is about to replace, with the age of each file, before it does so.

**``--force-refresh`` skips the up-to-date check, never the work.** A command
whose outputs are newer than its inputs has nothing to do, and says so rather
than spending a minute reproducing a file byte for byte. ``--force-refresh``
is how you insist anyway.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import subprocess
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

from gridiron.config import DEFAULT_RANDOM_SEED, PROJECT_ROOT

LOGGER = logging.getLogger("gridiron.workflow")

RUN_RECORD_DIR = PROJECT_ROOT / "outputs" / "runs"

# Distribution name -> import name, for the packages a command cannot run
# without. The two differ often enough that guessing is not safe.
REQUIRED_PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "scikit-learn": "sklearn",
    "pyarrow": "pyarrow",
    "joblib": "joblib",
}

# Packages only some commands need. Named separately so a command that draws
# no charts does not fail because seaborn is missing.
OPTIONAL_PACKAGES = {
    "nflreadpy": "nflreadpy",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "PyYAML": "yaml",
}


class WorkflowError(RuntimeError):
    """Raised when a command cannot run, with a message that says what to do.

    Every message this carries names both the problem and the command that
    fixes it. "File not found" tells a user they have a problem; it does not
    tell them they need to run ``download`` first.
    """


# --- dependency and input validation ---------------------------------------


def missing_packages(required: dict[str, str] | None = None) -> list[str]:
    """Distribution names of the packages that cannot be imported."""
    import importlib.util

    required = REQUIRED_PACKAGES if required is None else required
    missing = []
    for distribution, module in sorted(required.items()):
        if importlib.util.find_spec(module) is None:
            missing.append(distribution)
    return missing


def check_dependencies(
    extra: Sequence[str] = (),
    required: dict[str, str] | None = None,
) -> None:
    """Confirm every package this command imports is installed.

    ``extra`` names optional packages the calling command actually needs, by
    distribution name as it appears in ``requirements.txt``.
    """
    wanted = dict(REQUIRED_PACKAGES if required is None else required)
    for name in extra:
        if name not in OPTIONAL_PACKAGES:
            raise WorkflowError(f"Unknown optional dependency requested: {name}.")
        wanted[name] = OPTIONAL_PACKAGES[name]

    missing = missing_packages(wanted)
    if missing:
        raise WorkflowError(
            "Missing required package(s): "
            + ", ".join(missing)
            + ". Install them with 'pip install -r requirements.txt'."
        )


def require_inputs(paths: dict[str, Path], remedy: str) -> None:
    """Confirm every named input exists, naming all that do not.

    Reporting one missing file at a time makes a user run the command once per
    problem, so this reports the whole set.
    """
    missing = {name: path for name, path in paths.items() if not Path(path).exists()}
    if not missing:
        return

    detail = "; ".join(f"{name} ({path})" for name, path in sorted(missing.items()))
    raise WorkflowError(f"Missing required input(s): {detail}. {remedy}")


# --- artifacts -------------------------------------------------------------


def _age(path: Path) -> str:
    """How long ago a file was written, in words."""
    delta = datetime.now(tz=UTC) - datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
    hours = delta.total_seconds() / 3600
    if hours < 1:
        return f"{delta.total_seconds() / 60:.0f} minutes old"
    if hours < 48:
        return f"{hours:.0f} hours old"
    return f"{hours / 24:.0f} days old"


def warn_on_overwrite(paths: Iterable[Path], logger: logging.Logger = LOGGER) -> int:
    """Say which existing files a command is about to replace.

    Returns the count, so a caller can mention it in its summary. This warns
    and proceeds: refusing would mean a command could not be run twice, and
    these commands are meant to be re-runnable.
    """
    existing = [Path(path) for path in paths if Path(path).exists()]
    if existing:
        logger.warning("Overwriting %d existing artifact(s):", len(existing))
        for path in existing:
            logger.warning(
                "  %s  (%s, %s)", path, f"{path.stat().st_size:,} bytes", _age(path)
            )
    return len(existing)


def is_up_to_date(inputs: Iterable[Path], outputs: Iterable[Path]) -> bool:
    """True when every output exists and is newer than every input.

    Used to skip work that would reproduce a file exactly. A missing input is
    not treated as "up to date"; validation catches that first.
    """
    output_paths = [Path(path) for path in outputs]
    input_paths = [Path(path) for path in inputs]
    if not output_paths or not all(path.exists() for path in output_paths):
        return False
    if not input_paths or not all(path.exists() for path in input_paths):
        return False

    newest_input = max(path.stat().st_mtime for path in input_paths)
    oldest_output = min(path.stat().st_mtime for path in output_paths)
    return oldest_output >= newest_input


def file_digest(path: Path, limit_bytes: int = 64 * 1024 * 1024) -> str | None:
    """A SHA-256 of a written artifact, or None if it is too large to bother.

    The digest is what makes the idempotency claim checkable: run a command
    twice and compare the run records.
    """
    path = Path(path)
    if not path.exists() or path.stat().st_size > limit_bytes:
        return None

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


# --- the run record --------------------------------------------------------


def _package_versions() -> dict[str, str]:
    versions = {}
    for distribution in sorted({*REQUIRED_PACKAGES, *OPTIONAL_PACKAGES}):
        try:
            versions[distribution] = metadata.version(distribution)
        except metadata.PackageNotFoundError:
            versions[distribution] = "not installed"
    return versions


def _git_state() -> dict[str, str]:
    """The commit the code was at, and whether it had been edited.

    A run record that does not say this cannot be tied back to the code that
    produced it. Absent git is recorded as unknown rather than raising.
    """

    def _run(*command: str) -> str:
        try:
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                cwd=PROJECT_ROOT,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            return "unknown"
        return result.stdout.strip() if result.returncode == 0 else "unknown"

    status = _run("git", "status", "--porcelain")
    return {
        "commit": _run("git", "rev-parse", "HEAD"),
        "branch": _run("git", "rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": "unknown" if status == "unknown" else str(bool(status)),
    }


@dataclass
class _Artifact:
    path: Path
    label: str


class RunRecord:
    """Collects what a command ran with, and writes it beside the outputs.

    The point is reproducibility after the fact: six months later, a CSV in
    ``outputs/reports`` is only as trustworthy as the record of what produced
    it. Each record names the code version, the package versions, the resolved
    arguments, and a digest of every artifact written.
    """

    def __init__(
        self,
        command: str,
        argv: Sequence[str] | None = None,
        directory: Path | None = None,
    ) -> None:
        self.command = command
        self.argv = list(argv or sys.argv[1:])
        self.directory = Path(directory) if directory else RUN_RECORD_DIR
        self.started_at = datetime.now(tz=UTC)
        self.arguments: dict[str, object] = {}
        self.notes: list[str] = []
        self._inputs: list[_Artifact] = []
        self._artifacts: list[_Artifact] = []

    def configure(self, args: argparse.Namespace) -> None:
        """Record the resolved arguments, defaults included."""
        self.arguments = {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        }

    def add_input(self, path: Path, label: str = "input") -> None:
        self._inputs.append(_Artifact(Path(path), label))

    def add_artifact(self, path: Path, label: str = "output") -> None:
        self._artifacts.append(_Artifact(Path(path), label))

    def add_artifacts(self, paths: Iterable[Path], label: str = "output") -> None:
        for path in paths:
            self.add_artifact(path, label)

    def note(self, message: str) -> None:
        self.notes.append(message)

    @property
    def artifact_paths(self) -> list[Path]:
        return [item.path for item in self._artifacts]

    def _describe(self, item: _Artifact, digest: bool) -> dict[str, object]:
        path = item.path
        entry: dict[str, object] = {
            "label": item.label,
            "path": str(path),
            "exists": path.exists(),
        }
        if path.exists():
            stat = path.stat()
            entry["bytes"] = stat.st_size
            entry["modified_at"] = datetime.fromtimestamp(
                stat.st_mtime, tz=UTC
            ).isoformat(timespec="seconds")
            if digest:
                entry["sha256"] = file_digest(path)
        return entry

    def finish(self, exit_code: int) -> Path:
        """Write the record and return where it went."""
        finished_at = datetime.now(tz=UTC)
        payload = {
            "command": self.command,
            "argv": self.argv,
            "exit_code": exit_code,
            "started_at": self.started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_seconds": round(
                (finished_at - self.started_at).total_seconds(), 3
            ),
            "arguments": self.arguments,
            "random_seed": DEFAULT_RANDOM_SEED,
            "notes": self.notes,
            "environment": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "packages": _package_versions(),
            },
            "git": _git_state(),
            "inputs": [self._describe(item, digest=False) for item in self._inputs],
            "artifacts": [
                self._describe(item, digest=True) for item in self._artifacts
            ],
        }

        self.directory.mkdir(parents=True, exist_ok=True)
        stamp = self.started_at.strftime("%Y%m%dT%H%M%SZ")
        path = self.directory / f"{self.command}-{stamp}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return path


# --- argument parsing ------------------------------------------------------


def add_common_arguments(
    parser: argparse.ArgumentParser,
    output_default: Path,
    output_help: str = "Directory for this command's artifacts.",
) -> argparse.ArgumentParser:
    """Add the flags every command in the workflow accepts.

    ``--output-dir`` is kept as an alias because the phase scripts already used
    it; ``--output-path`` is the documented name.
    """
    parser.add_argument(
        "--output-path",
        "--output-dir",
        dest="output_path",
        type=Path,
        default=output_default,
        help=output_help,
    )
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Rebuild even when the existing artifacts are already up to date.",
    )
    parser.add_argument(
        "--run-record-dir",
        type=Path,
        default=RUN_RECORD_DIR,
        help="Where to write the JSON record of this run.",
    )
    return parser


# --- output ----------------------------------------------------------------


def print_artifacts(paths: Iterable[Path], title: str = "Artifacts") -> None:
    """List what a command produced, with sizes."""
    listed = [Path(path) for path in paths]
    if not listed:
        return
    print(f"\n=== {title} ===")
    for path in listed:
        size = f"{path.stat().st_size:,} bytes" if path.exists() else "MISSING"
        print(f"  {path}  ({size})")


def print_up_to_date(outputs: Iterable[Path], command: str) -> None:
    """Explain a no-op run, and how to override it."""
    print(f"\n=== {command}: already up to date ===")
    for path in outputs:
        print(f"  {path}")
    print("\nInputs have not changed since these were written.")
    print("Re-run with --force-refresh to rebuild them anyway.")


# --- dispatching to the phase scripts --------------------------------------

SCRIPT_DIR = PROJECT_ROOT / "scripts"


def load_script(name: str):
    """Import a file from ``scripts/`` and return the module.

    ``scripts/`` is a directory of entry points, not a package, and is not on
    the import path. Loading by path is what lets the CLI and the composite
    commands call exactly the code ``python scripts/<name>.py`` runs, instead
    of a second copy of it that can drift.
    """
    import importlib.util

    path = SCRIPT_DIR / f"{name}.py"
    if not path.exists():
        raise WorkflowError(f"No such command script: {path}.")

    spec = importlib.util.spec_from_file_location(f"gridiron_script_{name}", path)
    if spec is None or spec.loader is None:
        raise WorkflowError(f"Could not load {path}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
