"""Correlation analysis and the record of what was done about it.

The policy is a threshold plus an obligation: any feature pair at
``|r| >= 0.80`` is flagged, and nothing is dropped without a written reason.
:func:`correlation_decisions` is that written record -- every flagged pair
carries a decision and the reasoning behind it, so a later reader can see what
was considered even where the answer was "keep both".
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

CORRELATION_THRESHOLD = 0.80

KEEP_BOTH = "keep both"
DROP_SECOND = "drop the second"
AVOIDED_BY_CONSTRUCTION = "avoided by construction"


class CorrelationError(ValueError):
    """Raised when a frame cannot support a correlation report."""


@dataclass(frozen=True)
class CorrelationDecision:
    """One flagged pair and what was decided about it."""

    left: str
    right: str
    correlation: float
    decision: str
    reason: str


def numeric_fields(frame: pd.DataFrame, columns: list[str]) -> list[str]:
    """Return the columns that a correlation matrix can actually use.

    Booleans are excluded as well as non-numerics: a correlation between two
    indicator flags is a phi coefficient and does not belong on the same scale
    as the continuous features.
    """
    return [
        column
        for column in columns
        if column in frame.columns
        and pd.api.types.is_numeric_dtype(frame[column])
        and not pd.api.types.is_bool_dtype(frame[column])
    ]


def correlation_matrix(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Pearson correlation over the given numeric columns."""
    usable = numeric_fields(frame, columns)
    if len(usable) < 2:
        raise CorrelationError(
            f"Need at least two numeric columns to correlate; got {len(usable)}."
        )
    return frame[usable].corr()


def correlated_pairs(
    matrix: pd.DataFrame,
    threshold: float = CORRELATION_THRESHOLD,
) -> pd.DataFrame:
    """Every unique pair at or above ``threshold``, strongest first.

    Only the upper triangle is walked, so a pair appears once rather than twice,
    and a column is never reported against itself.
    """
    columns = list(matrix.columns)
    rows = [
        {
            "left": left,
            "right": right,
            "correlation": float(matrix.loc[left, right]),
            "abs_correlation": abs(float(matrix.loc[left, right])),
        }
        for index, left in enumerate(columns)
        for right in columns[index + 1 :]
        if pd.notna(matrix.loc[left, right])
        and abs(float(matrix.loc[left, right])) >= threshold
    ]
    report = pd.DataFrame(
        rows, columns=["left", "right", "correlation", "abs_correlation"]
    )
    return report.sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def _family(name: str) -> str:
    """The metric a differential belongs to, ignoring its window."""
    for suffix in ("_last_3", "_last_5", "_season"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def _explain(left: str, right: str, correlation: float) -> tuple[str, str]:
    """Decide what to do about one flagged pair, and say why."""
    if _family(left) == _family(right):
        return (
            KEEP_BOTH,
            "Two windows over the same metric. They are redundant by design and "
            "only one window per metric enters the model row, so the pair never "
            "reaches a model together. Kept in the wider table because the "
            "shorter window is the recency signal a later phase may prefer.",
        )
    if {"point_margin_diff_last_5", "ats_margin_diff_last_5"} == {left, right}:
        return (
            KEEP_BOTH,
            "Margin and margin-against-the-spread move together because one is "
            "the other shifted by the line. Only point_margin_diff_last_5 is a "
            "model field; ats_margin_diff_last_5 is diagnostic, so both are "
            "kept and neither is fed to a model alongside the other.",
        )
    return (
        KEEP_BOTH,
        f"Flagged at r={correlation:+.3f}. No removal was made because the pair "
        "does not duplicate a single underlying quantity; revisit if a model "
        "shows unstable coefficients.",
    )


def correlation_decisions(
    pairs: pd.DataFrame,
    extra: tuple[CorrelationDecision, ...] = (),
) -> pd.DataFrame:
    """Turn flagged pairs into a decision record, one row per pair.

    ``extra`` carries decisions that no pair in the data can show -- most
    importantly a redundancy that was designed out rather than measured, which
    would otherwise go unrecorded because the two columns never coexist.
    """
    rows = []
    for row in pairs.itertuples(index=False):
        decision, reason = _explain(row.left, row.right, row.correlation)
        rows.append(
            CorrelationDecision(
                left=row.left,
                right=row.right,
                correlation=row.correlation,
                decision=decision,
                reason=reason,
            )
        )
    rows.extend(extra)
    return pd.DataFrame(
        [
            {
                "left": item.left,
                "right": item.right,
                "correlation": item.correlation,
                "decision": item.decision,
                "reason": item.reason,
            }
            for item in rows
        ]
    )


# Recorded here because the two columns never appear together, so no measured
# pair can surface it: the EPA aggregation produces both the allowed and the
# sign-adjusted strength view of defensive EPA, and they are exact negatives
# (r = -1.00 by construction). The rolling layer consumes only the strength
# version, so the redundancy is designed out rather than filtered out.
DEFENSIVE_EPA_DECISION = CorrelationDecision(
    left="defensive_epa_allowed_per_play",
    right="defensive_epa_strength_per_play",
    correlation=-1.0,
    decision=AVOIDED_BY_CONSTRUCTION,
    reason=(
        "Exact negatives of one another. Only the strength view is carried into "
        "the rolling features, so the pair cannot reach a model. The strength "
        "view was chosen because it points the same way as the offensive "
        "measure: higher is a better unit."
    ),
)
