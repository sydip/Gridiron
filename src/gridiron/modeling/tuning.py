"""Hyperparameter search over the walk-forward folds.

Settings are chosen on the validation seasons and never on the deployment
season. The search is scored by a ranked decision process, not by return:
optimising a twelve-cell grid for ROI on 1,800-odd near-coinflip games selects
the luckiest cell, not the best one. Probabilistic scores come first because
they respond to whether the model is right *and* how confidently, which is far
harder to get right by chance than a win rate.

The ranking, in order:

1. lowest mean log loss,
2. lowest mean Brier score,
3. most stable accuracy across seasons,
4. competitive mean accuracy,
5. smallest coefficients, where everything above is equivalent.

Steps 1 and 2 narrow by a tolerance rather than picking an outright winner. At
this sample size a log-loss difference of 0.0002 is noise, and treating it as a
decision would be false precision.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from gridiron.config import PROJECT_ROOT
from gridiron.modeling.baselines import eligible_games
from gridiron.modeling.evaluate import (
    UNINFORMATIVE_BRIER,
    UNINFORMATIVE_LOG_LOSS,
    run_walk_forward,
    season_metrics,
)
from gridiron.modeling.splits import (
    FINAL_VALIDATION_SEASON,
    FIRST_VALIDATION_SEASON,
    SeasonSplit,
    season_walk_forward_splits,
)

C_VALUES = [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]
CLASS_WEIGHTS: list[str | None] = [None, "balanced"]

# The season the model would be deployed into. It is never scored here, and a
# row from it reaching the search is an error rather than something to filter.
DEPLOYMENT_SEASON = 2026

# Differences below these are noise at roughly 1,800 games and are treated as
# ties rather than as decisions.
LOG_LOSS_TOLERANCE = 0.001
BRIER_TOLERANCE = 0.001
ACCURACY_TOLERANCE = 0.005

# On the standardised scale a coefficient is log-odds per standard deviation,
# so 1.0 is an odds ratio of about 2.7 per SD. For a near-coinflip target that
# would be implausibly strong and points at overfitting rather than signal.
LARGE_COEFFICIENT = 1.0

RESULT_COLUMNS = [
    "C",
    "class_weight",
    "mean_accuracy",
    "mean_log_loss",
    "mean_brier",
    "mean_roi",
    "accuracy_std",
    "worst_season_accuracy",
]

DIAGNOSTIC_COLUMNS = [
    "best_season_accuracy",
    "max_abs_coefficient",
    "mean_abs_coefficient",
    "unstable_features",
    "beats_uninformative",
]

SELECTED_PARAMS_PATH = PROJECT_ROOT / "config" / "model_params.json"


class TuningError(ValueError):
    """Raised when a hyperparameter search cannot be run or resolved."""


@dataclass(frozen=True)
class SelectedParameters:
    """The chosen settings and the reasoning that chose them."""

    c: float
    class_weight: str | None
    rationale: list[str] = field(default_factory=list)
    selected_on: str = ""
    folds: int = 0
    validation_seasons: list[int] = field(default_factory=list)
    mean_log_loss: float = float("nan")
    mean_brier: float = float("nan")
    mean_accuracy: float = float("nan")
    accuracy_std: float = float("nan")
    warnings: list[str] = field(default_factory=list)

    def as_summary(self) -> str:
        return "; ".join(self.rationale)


def _coefficient_diagnostics(coefficients: pd.DataFrame) -> dict[str, object]:
    """Magnitude and cross-fold stability of the fitted coefficients.

    A feature whose coefficient changes sign between folds is not measuring
    anything durable, whatever its size. That is reported separately from
    magnitude because the two problems are different: a large coefficient may
    be real, a sign-flipping one is noise being fitted.
    """
    if coefficients is None or coefficients.empty:
        return {
            "max_abs_coefficient": float("nan"),
            "mean_abs_coefficient": float("nan"),
            "unstable_features": 0,
            "unstable_feature_names": [],
        }

    by_feature = coefficients.groupby("feature")["coefficient"]
    flips = by_feature.apply(lambda values: np.sign(values).nunique() > 1)
    unstable = sorted(flips.loc[flips].index)

    return {
        "max_abs_coefficient": float(coefficients["abs_coefficient"].max()),
        "mean_abs_coefficient": float(coefficients["abs_coefficient"].mean()),
        "unstable_features": len(unstable),
        "unstable_feature_names": unstable,
    }


def run_grid_search(
    matchups: pd.DataFrame,
    c_values: list[float] | None = None,
    class_weights: list[str | None] | None = None,
    first_validation_season: int = FIRST_VALIDATION_SEASON,
    final_validation_season: int = FINAL_VALIDATION_SEASON,
) -> pd.DataFrame:
    """Score every parameter set over one identical set of folds.

    The folds are computed once and handed to every parameter set, so a
    difference between two rows of the result is a difference in the parameters
    and can never be a difference in the data they saw.
    """
    if DEPLOYMENT_SEASON in set(
        range(first_validation_season, final_validation_season + 1)
    ):
        raise TuningError(
            f"Refusing to tune on the deployment season {DEPLOYMENT_SEASON}."
        )

    eligible = eligible_games(matchups).sort_values(
        ["gameday", "game_id"], kind="stable"
    )
    deployment_rows = int((eligible["season"] == DEPLOYMENT_SEASON).sum())
    if deployment_rows:
        raise TuningError(
            f"{deployment_rows} settled row(s) from {DEPLOYMENT_SEASON} reached "
            "the search; the deployment season must not be scored."
        )

    splits: list[SeasonSplit] = list(
        season_walk_forward_splits(
            eligible, first_validation_season, final_validation_season
        )
    )
    if not splits:
        raise TuningError("No folds available to tune over.")

    rows = []
    for c in c_values or C_VALUES:
        for class_weight in (
            class_weights if class_weights is not None else CLASS_WEIGHTS
        ):
            predictions, folds = run_walk_forward(
                matchups,
                splits=splits,
                c=c,
                class_weight=class_weight,
                collect_coefficients=True,
            )
            seasons = season_metrics(predictions)
            diagnostics = _coefficient_diagnostics(folds.attrs.get("coefficients"))

            rows.append(
                {
                    "C": c,
                    "class_weight": "None" if class_weight is None else class_weight,
                    "mean_accuracy": float(seasons["accuracy"].mean()),
                    "mean_log_loss": float(seasons["log_loss"].mean()),
                    "mean_brier": float(seasons["brier_score"].mean()),
                    "mean_roi": float(seasons["roi"].mean()),
                    "accuracy_std": float(seasons["accuracy"].std(ddof=1)),
                    "worst_season_accuracy": float(seasons["accuracy"].min()),
                    "best_season_accuracy": float(seasons["accuracy"].max()),
                    "beats_uninformative": bool(
                        seasons["log_loss"].mean() < UNINFORMATIVE_LOG_LOSS
                        and seasons["brier_score"].mean() < UNINFORMATIVE_BRIER
                    ),
                    **{
                        key: value
                        for key, value in diagnostics.items()
                        if key != "unstable_feature_names"
                    },
                    "unstable_feature_names": ", ".join(
                        diagnostics["unstable_feature_names"]
                    ),
                }
            )

    results = pd.DataFrame(rows)
    results.attrs["folds"] = len(splits)
    results.attrs["validation_seasons"] = [split.validation_season for split in splits]
    return results.sort_values("mean_log_loss").reset_index(drop=True)


def select_parameters(results: pd.DataFrame) -> SelectedParameters:
    """Apply the ranked decision process and record why it landed where it did.

    Each step narrows the field and appends a line of reasoning, so the choice
    can be read back later without re-running anything.
    """
    if results.empty:
        raise TuningError("No results to select from.")

    rationale: list[str] = []
    candidates = results.copy()

    best_log_loss = candidates["mean_log_loss"].min()
    candidates = candidates.loc[
        candidates["mean_log_loss"] <= best_log_loss + LOG_LOSS_TOLERANCE
    ]
    rationale.append(
        f"Lowest mean log loss is {best_log_loss:.5f}; kept the "
        f"{len(candidates)} set(s) within {LOG_LOSS_TOLERANCE} of it, since a "
        "smaller gap than that is noise at this sample size"
    )

    best_brier = candidates["mean_brier"].min()
    candidates = candidates.loc[
        candidates["mean_brier"] <= best_brier + BRIER_TOLERANCE
    ]
    rationale.append(
        f"Lowest mean Brier among those is {best_brier:.5f}; kept the "
        f"{len(candidates)} set(s) within {BRIER_TOLERANCE}"
    )

    if len(candidates) > 1:
        most_stable = candidates["accuracy_std"].min()
        candidates = candidates.loc[candidates["accuracy_std"] <= most_stable + 1e-9]
        rationale.append(
            f"Broke the tie on stability: lowest accuracy standard deviation "
            f"across seasons is {most_stable:.5f}"
        )

    if len(candidates) > 1:
        best_accuracy = candidates["mean_accuracy"].max()
        candidates = candidates.loc[
            candidates["mean_accuracy"] >= best_accuracy - ACCURACY_TOLERANCE
        ]
        rationale.append(
            f"Kept the set(s) within {ACCURACY_TOLERANCE} of the best mean "
            f"accuracy ({best_accuracy:.5f})"
        )

    if len(candidates) > 1 and "mean_abs_coefficient" in candidates.columns:
        smallest = candidates["mean_abs_coefficient"].min()
        candidates = candidates.loc[
            candidates["mean_abs_coefficient"] <= smallest + 1e-9
        ]
        rationale.append(
            f"Preferred the smaller coefficients ({smallest:.5f} mean absolute) "
            "where everything above was equivalent"
        )

    chosen = candidates.sort_values(
        ["mean_log_loss", "accuracy_std", "mean_abs_coefficient"]
    ).iloc[0]
    rationale.append(f"Selected C={chosen['C']}, class_weight={chosen['class_weight']}")
    rationale.append(
        "ROI was not used to select; it is reported but optimising a small grid "
        "for return picks the luckiest cell"
    )

    warnings = coefficient_warnings(results, chosen)

    return SelectedParameters(
        selected_on=datetime.now(UTC).isoformat(timespec="seconds"),
        c=float(chosen["C"]),
        class_weight=(
            None if chosen["class_weight"] in ("None", None) else chosen["class_weight"]
        ),
        rationale=rationale,
        folds=int(results.attrs.get("folds", 0)),
        validation_seasons=list(results.attrs.get("validation_seasons", [])),
        mean_log_loss=float(chosen["mean_log_loss"]),
        mean_brier=float(chosen["mean_brier"]),
        mean_accuracy=float(chosen["mean_accuracy"]),
        accuracy_std=float(chosen["accuracy_std"]),
        warnings=warnings,
    )


def coefficient_warnings(results: pd.DataFrame, chosen: pd.Series) -> list[str]:
    """Flag coefficients that are too large or that will not hold still."""
    warnings: list[str] = []

    magnitude = chosen.get("max_abs_coefficient", float("nan"))
    if pd.notna(magnitude) and magnitude > LARGE_COEFFICIENT:
        warnings.append(
            f"Largest coefficient is {magnitude:.3f} on the standardised scale, "
            f"above the {LARGE_COEFFICIENT} threshold; that is implausibly "
            "strong for this target and suggests overfitting."
        )

    unstable = chosen.get("unstable_features", 0)
    if unstable:
        names = chosen.get("unstable_feature_names", "")
        warnings.append(
            f"{int(unstable)} feature(s) change coefficient sign between folds "
            f"({names}). Their direction is not stable across seasons, so they "
            "are fitting noise rather than measuring anything durable."
        )

    if not bool(chosen.get("beats_uninformative", False)):
        warnings.append(
            f"Mean log loss {chosen['mean_log_loss']:.5f} and Brier "
            f"{chosen['mean_brier']:.5f} are no better than predicting 0.5 for "
            f"every game ({UNINFORMATIVE_LOG_LOSS:.5f} and "
            f"{UNINFORMATIVE_BRIER}). The selected settings are the least bad "
            "of the grid, not good."
        )

    if not results["beats_uninformative"].any():
        warnings.append(
            "No parameter set in the grid produced informative probabilities."
        )

    return warnings


def save_selected_parameters(
    selection: SelectedParameters,
    path: Path | None = None,
) -> Path:
    """Write the chosen settings, and why, to the project configuration.

    This is configuration rather than a generated artefact: the choice is a
    project decision that later training runs read, so it belongs in version
    control next to the code and not in an ignored output directory.

    The default path is resolved at call time rather than bound as a default
    argument, so the location can be redirected -- by a test, or by a
    deployment that keeps its configuration elsewhere.
    """
    path = path or SELECTED_PARAMS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(selection), indent=2), encoding="utf-8")
    return path


def load_selected_parameters(
    path: Path | None = None,
) -> SelectedParameters:
    """Read the configured settings, falling back to the untuned defaults.

    A missing file is not an error: it means the search has not been run, and
    the untuned defaults are the documented starting point.

    As with saving, the default path is resolved at call time so it can be
    redirected.
    """
    path = path or SELECTED_PARAMS_PATH
    if not path.exists():
        return SelectedParameters(
            c=1.0,
            class_weight=None,
            rationale=["No tuning run found; using untuned defaults."],
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return SelectedParameters(**payload)
