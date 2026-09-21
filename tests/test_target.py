"""Tests for against-the-spread target construction."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
import pytest

from gridiron.features.target import (
    AWAY_COVER,
    HOME_COVER,
    PUSH,
    UNRESOLVED,
    add_ats_target,
    calculate_adjusted_home_margin,
    classify_ats_result,
)


@pytest.mark.parametrize(
    ("home_score", "away_score", "spread", "adjusted", "result", "target"),
    [
        pytest.param(30, 20, 6.5, 3.5, HOME_COVER, 1, id="home-favorite-covers"),
        pytest.param(24, 20, 6.5, -2.5, AWAY_COVER, 0, id="home-favorite-fails"),
        pytest.param(20, 23, -4.5, 1.5, HOME_COVER, 1, id="home-underdog-covers-loss"),
        pytest.param(27, 20, -3.0, 10.0, HOME_COVER, 1, id="home-underdog-wins"),
        pytest.param(24, 21, 3.0, 0.0, PUSH, None, id="exact-push"),
        pytest.param(21, 17, 0.0, 4.0, HOME_COVER, 1, id="pickem-line"),
        pytest.param(17, 21, 0.0, -4.0, AWAY_COVER, 0, id="pickem-away"),
    ],
)
def test_completed_game_scenarios(
    home_score, away_score, spread, adjusted, result, target
):
    schedules = pd.DataFrame(
        {
            "home_score": [home_score],
            "away_score": [away_score],
            "spread_line": [spread],
        }
    )

    labeled = add_ats_target(schedules)

    assert calculate_adjusted_home_margin(home_score, away_score, spread) == adjusted
    assert classify_ats_result(adjusted) == result
    assert labeled.loc[0, "adjusted_home_margin"] == adjusted
    assert labeled.loc[0, "ats_result"] == result
    if target is None:
        assert pd.isna(labeled.loc[0, "home_cover"])
    else:
        assert labeled.loc[0, "home_cover"] == target


def test_overtime_game_uses_final_score_and_preserves_metadata():
    schedules = pd.DataFrame(
        {
            "home_score": [27],
            "away_score": [24],
            "spread_line": [2.5],
            "overtime": [True],
        }
    )

    labeled = add_ats_target(schedules)

    assert labeled.loc[0, "overtime"]
    assert labeled.loc[0, "adjusted_home_margin"] == 0.5
    assert labeled.loc[0, "ats_result"] == HOME_COVER


@pytest.mark.parametrize(
    ("home_score", "away_score", "spread"),
    [
        pytest.param(None, 20, 3.0, id="missing-home-score"),
        pytest.param(20, None, 3.0, id="missing-away-score"),
        pytest.param(20, 17, None, id="missing-spread"),
    ],
)
def test_missing_inputs_remain_unresolved(home_score, away_score, spread):
    labeled = add_ats_target(
        pd.DataFrame(
            {
                "home_score": [home_score],
                "away_score": [away_score],
                "spread_line": [spread],
            }
        )
    )

    assert math.isnan(calculate_adjusted_home_margin(home_score, away_score, spread))
    assert labeled.loc[0, "ats_result"] == UNRESOLVED
    assert pd.isna(labeled.loc[0, "home_cover"])
    assert not labeled.loc[0, "is_push"]


def test_push_is_null_not_away_cover():
    labeled = add_ats_target(
        pd.DataFrame({"home_score": [24], "away_score": [21], "spread_line": [3.0]})
    )

    assert labeled.loc[0, "ats_result"] == PUSH
    assert labeled.loc[0, "is_push"]
    assert pd.isna(labeled.loc[0, "home_cover"])
    assert set(labeled["home_cover"].dropna().unique()).issubset({0, 1})


def test_target_domain_is_binary_or_null():
    labeled = add_ats_target(
        pd.DataFrame(
            {
                "home_score": [30, 20, 24, None],
                "away_score": [20, 24, 21, None],
                "spread_line": [3.0, 3.0, 3.0, None],
            }
        )
    )

    assert set(labeled["home_cover"].dropna().astype(int)) == {0, 1}
    assert pd.isna(labeled.loc[labeled["ats_result"].eq(PUSH), "home_cover"]).all()
    assert pd.isna(
        labeled.loc[labeled["ats_result"].eq(UNRESOLVED), "home_cover"]
    ).all()


def test_manual_audit_matches_independent_arithmetic():
    audit_path = (
        Path(__file__).resolve().parents[1]
        / "outputs"
        / "reports"
        / "spread_validation_sample.csv"
    )
    audit = pd.read_csv(audit_path)
    scores = audit["score"].str.split("-", expand=True).astype(float)
    independently_adjusted = scores[0] - scores[1] - audit["spread_line"]
    independently_classified = independently_adjusted.map(
        lambda margin: (
            "HOME_COVER" if margin > 0 else ("AWAY_COVER" if margin < 0 else "PUSH")
        )
    )

    assert len(audit) >= 20
    assert (audit["spread_line"] > 0).any()
    assert (audit["spread_line"] < 0).any()
    assert (audit["spread_line"] == 0).any()
    pd.testing.assert_series_equal(
        audit["adjusted_home_margin"], independently_adjusted, check_names=False
    )
    pd.testing.assert_series_equal(
        audit["calculated_result"], independently_classified, check_names=False
    )
    pd.testing.assert_series_equal(
        audit["manually_verified_result"],
        independently_classified,
        check_names=False,
    )


def test_input_is_not_modified():
    schedules = pd.DataFrame(
        {"home_score": [24], "away_score": [21], "spread_line": [2.5]}
    )

    add_ats_target(schedules)

    assert list(schedules.columns) == ["home_score", "away_score", "spread_line"]
