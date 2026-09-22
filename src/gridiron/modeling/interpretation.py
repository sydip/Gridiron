"""Reading the fitted logistic regression.

A standardised logistic regression is about as interpretable as a model gets:
each coefficient is the change in log-odds of a home cover per one standard
deviation of that feature, and exponentiating gives an odds ratio. That makes
it tempting to read the table as a list of what drives covers. It is not one,
for three reasons this module measures rather than mentions.

* **These are associations, not causes.** The data is observational. Nothing was
  randomised, and a feature can carry a coefficient because it stands in for
  something else entirely.
* **Collinear features split credit arbitrarily.** When predictors move
  together, the fit can hand a large coefficient to one and a small one to
  another without either being more important. Variance inflation factors are
  reported alongside every coefficient so a magnitude is never read alone.
* **A coefficient that changes sign between folds is not measuring anything.**
  Stability across the walk-forward folds is reported for the same reason.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from gridiron.modeling.pipeline import (
    IMPUTE_STEP,
    MODEL_STEP,
    SCALE_STEP,
    fitted_feature_names,
    model_coefficients,
)

# A variance inflation factor above 5 means a feature is largely predictable
# from the others; above 10 is the conventional line past which an individual
# coefficient should not be interpreted on its own.
VIF_ELEVATED = 5.0
VIF_SEVERE = 10.0

# Below this, a one-standard-deviation move changes the odds by under 5% and
# the feature is doing almost nothing in the fitted model.
NEGLIGIBLE_ODDS_RATIO = 0.05

COEFFICIENT_COLUMNS = [
    "feature",
    "coefficient",
    "odds_ratio",
    "odds_change_pct",
    "abs_coefficient",
    "direction",
    "interpretation",
    "vif",
    "collinearity_flag",
]

# For each feature: how to describe an INCREASE in it, and any caveat that
# belongs with it. The sentence is composed from the coefficient's sign at read
# time, so it stays natural whichever way the fit lands.
FEATURE_DIRECTIONS: dict[str, tuple[str, str]] = {
    "spread_line": (
        "a larger home-team spread (nflverse prices the home side, so a "
        "positive value means the home team is favoured)",
        "",
    ),
    "off_epa_diff_last_5": (
        "a stronger home offence relative to the away offence over the last five games",
        "",
    ),
    "def_epa_strength_diff_last_5": (
        "a stronger home defence relative to the away defence over the last "
        "five games (the measure is sign-adjusted, so higher is better)",
        "",
    ),
    "pace_diff_last_5": (
        "a faster or higher-volume home offence relative to the away offence "
        "over the last five games",
        "",
    ),
    "rest_diff": (
        "more home rest than away rest, measured in days",
        "",
    ),
    "point_margin_diff_last_5": (
        "a better recent home scoring margin than the away team's",
        "",
    ),
    "win_pct_diff_last_5": (
        "a better recent home win rate than the away team's",
        "",
    ),
    "home_short_week": (
        "the home team playing on short rest",
        "A positive coefficient here would run against intuition and is a "
        "reason to distrust the magnitude rather than to believe the effect.",
    ),
    "away_short_week": (
        "the away team playing on short rest",
        "",
    ),
    "div_game": (
        "a divisional matchup",
        "",
    ),
    "week_1_flag": (
        "either team playing its season opener",
        "This flag is perfectly confounded with the rolling features being "
        "missing, so it partly measures the imputation rather than football.",
    ),
}


class InterpretationError(ValueError):
    """Raised when a fitted model cannot be interpreted."""


@dataclass(frozen=True)
class Limitation:
    """One measured reason to be careful with the table above it."""

    heading: str
    detail: str


def variance_inflation_factors(
    pipeline: Pipeline,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """How much each feature is explained by the others.

    Computed on the matrix the classifier actually sees -- after imputation and
    scaling -- because that is where the collinearity either does or does not
    bite. A feature's VIF is ``1 / (1 - R^2)`` from regressing it on all the
    rest, so 10 means 90% of its variation is already carried by the others.
    """
    names = fitted_feature_names(pipeline)
    missing = [name for name in names if name not in features.columns]
    if missing:
        raise InterpretationError(
            "Cannot compute VIF; missing feature(s): " + ", ".join(missing)
        )

    imputed = pipeline.named_steps[IMPUTE_STEP].transform(features[names])
    scaled = pd.DataFrame(
        pipeline.named_steps[SCALE_STEP].transform(imputed), columns=names
    )

    rows = []
    for name in names:
        others = scaled.drop(columns=name)
        r_squared = float(
            LinearRegression().fit(others, scaled[name]).score(others, scaled[name])
        )
        rows.append(
            {
                "feature": name,
                "r_squared": r_squared,
                "vif": float("inf") if r_squared >= 1.0 else 1.0 / (1.0 - r_squared),
            }
        )
    return pd.DataFrame(rows).sort_values("vif", ascending=False).reset_index(drop=True)


def _collinearity_flag(vif: float) -> str:
    if not np.isfinite(vif):
        return "perfectly collinear"
    if vif >= VIF_SEVERE:
        return "severe: do not read this coefficient alone"
    if vif >= VIF_ELEVATED:
        return "elevated: magnitude is shared with correlated features"
    return ""


def coefficient_table(
    pipeline: Pipeline,
    features: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Coefficients, odds ratios, direction, and collinearity in one table.

    ``odds_change_pct`` restates the odds ratio as a percentage change, which
    is the form most people read correctly: an odds ratio of 0.96 is a 4%
    *reduction* in the odds, not a 96% one.

    The direction text describes an association in the fitted model. It is
    phrased that way deliberately and should not be reworded into a causal
    claim.
    """
    table = model_coefficients(pipeline).copy()
    table["odds_change_pct"] = (table["odds_ratio"] - 1.0) * 100.0
    table["direction"] = np.where(table["coefficient"] >= 0, "increases", "decreases")
    table["interpretation"] = [
        _interpretation(row.feature, row.coefficient) for row in table.itertuples()
    ]

    if features is not None:
        inflation = variance_inflation_factors(pipeline, features)
        table = table.merge(inflation[["feature", "vif"]], on="feature", how="left")
        table["collinearity_flag"] = table["vif"].map(_collinearity_flag)
    else:
        table["vif"] = float("nan")
        table["collinearity_flag"] = ""

    table = table.sort_values("abs_coefficient", ascending=False).reset_index(drop=True)
    table.attrs["intercept"] = float(pipeline.named_steps[MODEL_STEP].intercept_[0])
    return table[COEFFICIENT_COLUMNS]


def _interpretation(feature: str, coefficient: float) -> str:
    """One sentence on what this feature's sign means, as an association.

    Deliberately phrased as an association in the fitted model. It should not
    be reworded into a claim about cause.
    """
    described = FEATURE_DIRECTIONS.get(feature)
    if described is None:
        return (
            f"No documented direction for {feature!r}. A positive coefficient "
            "would associate a higher value with a higher modelled probability "
            "of a home cover."
        )

    increase, caveat = described
    movement = "higher" if coefficient >= 0 else "lower"
    sentence = (
        f"{increase[0].upper()}{increase[1:]} is associated with a {movement} "
        "modelled probability of a home cover."
    )
    return f"{sentence} {caveat}".strip()


def coefficient_stability(
    fold_coefficients: pd.DataFrame,
) -> pd.DataFrame:
    """How steady each coefficient is across the walk-forward folds.

    A feature whose sign changes between folds is not measuring a durable
    relationship, whatever magnitude the final fit gives it.
    """
    required = {"feature", "coefficient"}
    missing = sorted(required.difference(fold_coefficients.columns))
    if missing:
        raise InterpretationError(
            "Cannot assess stability; missing: " + ", ".join(missing)
        )

    grouped = fold_coefficients.groupby("feature")["coefficient"]
    table = pd.DataFrame(
        {
            "folds": grouped.size(),
            "mean_coefficient": grouped.mean(),
            "std_coefficient": grouped.std(ddof=1),
            "min_coefficient": grouped.min(),
            "max_coefficient": grouped.max(),
        }
    ).reset_index()
    table["sign_changes"] = [
        bool(np.sign(values).nunique() > 1) for _, values in grouped
    ]
    table["stable"] = ~table["sign_changes"]
    return table.sort_values("std_coefficient", ascending=False).reset_index(drop=True)


def multicollinearity_warnings(table: pd.DataFrame) -> list[str]:
    """Plain-language warnings about what collinearity does to this table."""
    warnings: list[str] = []
    if "vif" not in table.columns or table["vif"].isna().all():
        return warnings

    severe = table.loc[table["vif"] >= VIF_SEVERE, "feature"].tolist()
    elevated = table.loc[
        table["vif"].between(VIF_ELEVATED, VIF_SEVERE, inclusive="left"), "feature"
    ].tolist()

    if severe:
        warnings.append(
            f"Severe collinearity (VIF >= {VIF_SEVERE:.0f}): "
            + ", ".join(severe)
            + ". At least 90% of each one's variation is already carried by the "
            "other features, so the fit can move size between them almost "
            "freely. Their individual coefficients should not be read as "
            "measures of importance."
        )
    if elevated:
        warnings.append(
            f"Elevated collinearity (VIF {VIF_ELEVATED:.0f} to "
            f"{VIF_SEVERE:.0f}): "
            + ", ".join(elevated)
            + ". These share explanatory work with correlated features, so their "
            "magnitudes are split rather than independent."
        )
    if not severe and not elevated:
        warnings.append(
            f"No feature reaches a VIF of {VIF_ELEVATED:.0f}; the coefficients "
            "are not obviously competing with one another."
        )
    return warnings


def model_limitations(
    table: pd.DataFrame,
    stability: pd.DataFrame | None = None,
    metrics: dict[str, float] | None = None,
) -> list[Limitation]:
    """The caveats that belong beside any coefficient table.

    Every one is derived from something measured, not asserted, so the list
    shrinks by itself if the model ever improves.
    """
    limitations = [
        Limitation(
            "These are associations, not causes",
            "The data is observational. No feature was manipulated, so a "
            "coefficient describes what moved together in the training seasons "
            "and not what would happen if a team changed that feature.",
        )
    ]

    largest = float(table["abs_coefficient"].max()) if len(table) else 0.0
    largest_odds = float(table["odds_change_pct"].abs().max()) if len(table) else 0.0
    if largest_odds < NEGLIGIBLE_ODDS_RATIO * 100:
        limitations.append(
            Limitation(
                "The coefficients are tiny",
                f"The largest is {largest:.4f}, a {largest_odds:.1f}% change in "
                "the odds per standard deviation. The selected regularisation "
                "shrank every feature close to zero because none of them "
                "improved the fit enough to resist it.",
            )
        )

    severe = table.loc[table["vif"] >= VIF_SEVERE, "feature"].tolist()
    if severe:
        verb = "reaches" if len(severe) == 1 else "reach"
        pronoun = "its coefficient is" if len(severe) == 1 else "their coefficients are"
        limitations.append(
            Limitation(
                "Some features are largely redundant",
                f"{', '.join(severe)} {verb} a variance inflation factor of "
                f"{VIF_SEVERE:.0f} or more, so {pronoun} not separable from "
                "those of the features they overlap with.",
            )
        )

    if stability is not None and len(stability):
        unstable = stability.loc[stability["sign_changes"], "feature"].tolist()
        if unstable:
            limitations.append(
                Limitation(
                    "Several coefficients change sign between folds",
                    f"{len(unstable)} of {len(stability)} features "
                    f"({', '.join(unstable)}) point one way in some validation "
                    "seasons and the other way in others. A direction that will "
                    "not hold still is not a finding.",
                )
            )

    if metrics:
        auc = metrics.get("roc_auc")
        if auc is not None and auc <= 0.5:
            limitations.append(
                Limitation(
                    "The model does not rank games",
                    f"Out-of-sample ROC-AUC is {auc:.4f}, at or below the 0.5 "
                    "of a coin flip. Interpreting the coefficients of a model "
                    "with no discrimination explains how it makes its mistakes, "
                    "not how football works.",
                )
            )
        log_loss_value = metrics.get("log_loss")
        if log_loss_value is not None and log_loss_value >= float(np.log(2)):
            limitations.append(
                Limitation(
                    "The probabilities carry no information",
                    f"Out-of-sample log loss is {log_loss_value:.5f} against "
                    f"{float(np.log(2)):.5f} for predicting 0.5 every time.",
                )
            )

    limitations.append(
        Limitation(
            "A linear model in the log-odds",
            "Logistic regression assumes each feature moves the log-odds at a "
            "constant rate. Real interactions -- a rest advantage mattering "
            "more in a divisional game, say -- cannot appear unless they are "
            "built as features.",
        )
    )
    return limitations
