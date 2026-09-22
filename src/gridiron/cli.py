"""The central command line for the project.

Every stage of the pipeline is reachable two ways -- as a script, or as a
subcommand here::

    python scripts/build_features.py --force-refresh
    python -m gridiron.cli build-features --force-refresh

Both run the same code. This module does not reimplement any stage; it loads
the script and calls its ``main``, so the two entry points cannot drift apart.

Arguments after the command name are passed through untouched, which means
``python -m gridiron.cli predict --help`` shows the prediction command's own
options rather than a summary of them written here and left to go stale.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass

from gridiron.workflow import WorkflowError, load_script

LOGGER = logging.getLogger("gridiron.cli")


@dataclass(frozen=True)
class Command:
    """One subcommand, and the script that implements it."""

    script: str
    summary: str
    in_full_rebuild: bool = False


# Declaration order is pipeline order, and is what ``--help`` prints.
COMMANDS: dict[str, Command] = {
    "download": Command(
        "download_data",
        "Fetch play-by-play and schedules from nflverse into data/raw.",
        in_full_rebuild=True,
    ),
    "build-features": Command(
        "build_features",
        "Build the matchup and model tables from the raw data.",
        in_full_rebuild=True,
    ),
    "train": Command(
        "train_model",
        "Fit the model on the training seasons and score it on held-out ones.",
        in_full_rebuild=True,
    ),
    "backtest": Command(
        "backtest_model",
        "Walk-forward validation and the betting-policy backtest.",
        in_full_rebuild=True,
    ),
    "deploy": Command(
        "deploy_model",
        "Fit the final model on every training season and persist it.",
        in_full_rebuild=True,
    ),
    "predict": Command(
        "predict_week",
        "Predict one week of the upcoming season. Needs --season and --week.",
    ),
    "tune": Command(
        "tune_model",
        "Search regularisation and class weighting, and record the choice.",
    ),
    "report": Command(
        "build_reports",
        "Draw the historical and weekly visual reports.",
    ),
}

# Not in the rebuild chain on purpose:
#   predict  needs a --week, so there is no sensible default run.
#   tune     rewrites config/model_params.json, which is version-controlled
#            configuration. Re-deriving it should be a deliberate act, not a
#            side effect of rebuilding.
#   report   is presentation, and needs predictions for a specific week.
FULL_REBUILD = [name for name, command in COMMANDS.items() if command.in_full_rebuild]


def usage() -> str:
    width = max(len(name) for name in COMMANDS) + 2
    lines = [
        "Usage: python -m gridiron.cli <command> [options]",
        "",
        "Commands:",
    ]
    lines += [
        f"  {name:<{width}}{command.summary}" for name, command in COMMANDS.items()
    ]
    lines += [
        f"  {'all':<{width}}Run the full rebuild: " + " -> ".join(FULL_REBUILD) + ".",
        "",
        "Options after the command are passed to it unchanged, so",
        "'python -m gridiron.cli predict --help' shows that command's own flags.",
        "",
        "Every command accepts --output-path and --force-refresh, exits nonzero",
        "on failure, and writes a JSON record of the run to outputs/runs.",
    ]
    return "\n".join(lines)


def run_full_rebuild(argv: list[str]) -> int:
    """Run each stage in order, stopping at the first failure.

    Shared flags are forwarded to every stage. Anything stage-specific belongs
    on that stage's own invocation, not here.
    """
    forwarded = [flag for flag in argv if flag == "--force-refresh"]
    unexpected = [flag for flag in argv if flag not in forwarded]
    if unexpected:
        LOGGER.error(
            "'all' accepts only --force-refresh; got: %s. Run the stages "
            "individually to pass stage-specific options.",
            " ".join(unexpected),
        )
        return 2

    for position, name in enumerate(FULL_REBUILD, start=1):
        print(f"\n{'=' * 72}")
        print(f"[{position}/{len(FULL_REBUILD)}] {name}")
        print(f"{'=' * 72}")

        code = dispatch(name, forwarded)
        if code != 0:
            LOGGER.error(
                "Stage '%s' exited %d; stopping. Fix it and re-run, or run the "
                "remaining stages individually.",
                name,
                code,
            )
            return code

    print(f"\n{'=' * 72}")
    print("Rebuild complete.")
    print("Next: python -m gridiron.cli predict --season 2026 --week 1")
    print(f"{'=' * 72}")
    return 0


def dispatch(name: str, argv: list[str]) -> int:
    """Run one command and return its exit code."""
    module = load_script(COMMANDS[name].script)
    code = module.main(argv)
    # A script that returns None succeeded; treat anything else as its code.
    return 0 if code is None else int(code)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    arguments = list(sys.argv[1:] if argv is None else argv)

    if not arguments:
        print(usage())
        return 2

    name, rest = arguments[0], arguments[1:]

    if name in ("-h", "--help", "help"):
        print(usage())
        return 0

    if name == "all":
        return run_full_rebuild(rest)

    if name not in COMMANDS:
        close = [known for known in COMMANDS if known.startswith(name[:3])]
        suggestion = f" Did you mean: {', '.join(close)}?" if close else ""
        LOGGER.error("Unknown command '%s'.%s", name, suggestion)
        print()
        print(usage())
        return 2

    try:
        return dispatch(name, rest)
    except WorkflowError as error:
        LOGGER.error("%s", error)
        return 1


if __name__ == "__main__":
    sys.exit(main())
