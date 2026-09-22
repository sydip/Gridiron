"""Tests that pin the project to its written specification.

These exist so a later change cannot quietly drift away from an agreed
decision: the pipeline's shape, the baseline feature set, the odds assumption,
and above all the sign convention of the spread.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from gridiron.config import DEFAULT_RANDOM_SEED
from gridiron.features.matchup import (
    FEATURE_COLUMNS,
    MARKET_COLUMNS,
    add_matchup_target,
    build_matchups,
)
from gridiron.features.target import (
    HOME_FAVOURED_WHEN_POSITIVE,
    AtsTargetError,
    calculate_adjusted_home_margin,
    classify_ats_result,
    validate_spread_convention,
)
from gridiron.modeling.baselines import predict_no_bet
from gridiron.modeling.evaluate import CONFIDENCE_TIERS, confidence_tier
from gridiron.modeling.metrics import BREAK_EVEN_ACCURACY, accuracy_by_market_role
from gridiron.modeling.pipeline import (
    IMPUTE_STEP,
    MAX_ITER,
    MODEL_STEP,
    SCALE_STEP,
    SOLVER,
    build_pipeline,
)
from gridiron.modeling.tuning import C_VALUES, CLASS_WEIGHTS

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SCHEDULE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "raw" / "schedules_2016_2026.parquet"
)

# The baseline feature set, exactly as the specification lists it.
SPEC_FEATURES = [
    "spread_line",
    "off_epa_diff_last_5",
    "def_epa_strength_diff_last_5",
    "pace_diff_last_5",
    "rest_diff",
    "point_margin_diff_last_5",
    "win_pct_diff_last_5",
    "home_short_week",
    "away_short_week",
    "div_game",
    "week_1_flag",
]


# --- the spread sign convention --------------------------------------------


def _schedule(home_score: float, away_score: float, spread_line: float) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "game_id": "g",
                "home_score": home_score,
                "away_score": away_score,
                "spread_line": spread_line,
            }
        ]
    )


def test_a_positive_spread_means_the_home_team_is_favoured():
    """The convention this whole project depends on.

    nflverse prices the home side, so a positive spread_line is a home
    favourite -- the opposite of the sportsbook notation where a favourite is
    quoted negative. Getting this backwards inverts every label.
    """
    # Home favoured by 3 and wins by 7: covers.
    assert calculate_adjusted_home_margin(27, 20, 3.0) == pytest.approx(4.0)
    assert classify_ats_result(4.0) == "HOME_COVER"

    # Home favoured by 7 but wins by only 3: does not cover.
    assert calculate_adjusted_home_margin(24, 21, 7.0) == pytest.approx(-4.0)
    assert classify_ats_result(-4.0) == "AWAY_COVER"

    # Home underdog by 3 and loses by 1: covers.
    assert calculate_adjusted_home_margin(20, 21, -3.0) == pytest.approx(2.0)
    assert classify_ats_result(2.0) == "HOME_COVER"


def test_landing_exactly_on_the_number_is_a_push():
    assert classify_ats_result(calculate_adjusted_home_margin(27, 24, 3.0)) == "PUSH"


def test_the_convention_check_detects_the_real_one():
    """Built so a reversed feed would be caught rather than silently used."""
    rng = np.random.default_rng(1)
    spread = rng.normal(0, 6, 500)
    # Margins that follow the spread: positive spread, home wins by more.
    margin = spread + rng.normal(0, 10, 500)
    schedules = pd.DataFrame(
        {
            "home_score": 21 + margin,
            "away_score": np.full(500, 21.0),
            "spread_line": spread,
        }
    )

    report = validate_spread_convention(schedules)

    assert report["convention"] == HOME_FAVOURED_WHEN_POSITIVE
    assert report["correlation"] > 0
    assert report["agrees_with_implementation"]


def test_the_convention_check_would_catch_a_reversed_feed():
    rng = np.random.default_rng(2)
    spread = rng.normal(0, 6, 500)
    margin = -spread + rng.normal(0, 10, 500)
    schedules = pd.DataFrame(
        {
            "home_score": 21 + margin,
            "away_score": np.full(500, 21.0),
            "spread_line": spread,
        }
    )

    report = validate_spread_convention(schedules)

    assert report["correlation"] < 0
    assert not report["agrees_with_implementation"]


def test_the_convention_check_refuses_a_tiny_sample():
    with pytest.raises(AtsTargetError, match="at least 100"):
        validate_spread_convention(_schedule(27, 24, 3.0))


@pytest.mark.skipif(
    not SCHEDULE_PATH.exists(), reason="raw schedule parquet is not available"
)
def test_the_real_data_uses_the_convention_the_code_assumes():
    """The check that matters: run against the actual downloaded schedule."""
    from gridiron.data.clean import clean_schedules

    report = validate_spread_convention(clean_schedules(pd.read_parquet(SCHEDULE_PATH)))

    assert report["agrees_with_implementation"], (
        "The spread convention in the data does not match the one the target "
        "calculation assumes; every label would be inverted."
    )
    assert report["convention"] == HOME_FAVOURED_WHEN_POSITIVE
    assert report["games"] > 2000


# --- the model pipeline ----------------------------------------------------


def test_the_pipeline_has_the_specified_steps_in_order():
    pipeline = build_pipeline()

    assert [name for name, _ in pipeline.steps] == ["imputer", "scaler", "classifier"]
    assert isinstance(pipeline.named_steps[IMPUTE_STEP], SimpleImputer)
    assert isinstance(pipeline.named_steps[SCALE_STEP], StandardScaler)
    assert isinstance(pipeline.named_steps[MODEL_STEP], LogisticRegression)


def test_the_classifier_matches_the_specification():
    classifier = build_pipeline().named_steps[MODEL_STEP]

    assert SOLVER == "liblinear"
    assert MAX_ITER == 2000
    assert classifier.solver == "liblinear"
    assert classifier.max_iter == 2000
    assert classifier.random_state == DEFAULT_RANDOM_SEED == 42
    assert classifier.C == 1.0


def test_regularisation_is_l2():
    """L2 is required, and is scikit-learn's default.

    It is not passed explicitly because the argument is deprecated and due for
    removal, so this asserts the behaviour rather than the argument.
    """
    classifier = build_pipeline(c=0.5).named_steps[MODEL_STEP]

    assert classifier.C == 0.5
    # An L2 fit shrinks coefficients toward zero without driving them to it,
    # which an L1 fit on the same data would.
    rng = np.random.default_rng(4)
    features = pd.DataFrame(rng.normal(0, 1, (300, 6)), columns=list("abcdef"))
    labels = pd.Series((features["a"] + rng.normal(0, 1, 300) > 0).astype(int))
    fitted = build_pipeline(c=0.01).fit(features, labels)

    coefficients = fitted.named_steps[MODEL_STEP].coef_.ravel()
    assert np.all(coefficients != 0.0)
    assert np.abs(coefficients).max() < 1.0


def test_the_hyperparameter_grid_matches_the_specification():
    assert C_VALUES == [0.01, 0.1, 0.5, 1.0, 2.0, 10.0]
    assert CLASS_WEIGHTS == [None, "balanced"]


# --- features --------------------------------------------------------------


def test_the_baseline_feature_set_is_exactly_the_specified_one():
    assert FEATURE_COLUMNS == SPEC_FEATURES


def test_market_descriptors_exist_but_stay_out_of_the_baseline():
    assert MARKET_COLUMNS == [
        "absolute_spread",
        "home_favorite",
        "close_game_line",
        "large_favorite",
    ]
    assert not set(MARKET_COLUMNS) & set(FEATURE_COLUMNS)


def _matchup_frame(spread: float) -> pd.DataFrame:
    schedule = pd.DataFrame(
        [
            {
                "game_id": "2023_01_BBB_AAA",
                "season": 2023,
                "week": 1,
                "gameday": pd.Timestamp("2023-09-10"),
                "home_team": "AAA",
                "away_team": "BBB",
                "spread_line": spread,
                "home_score": 24.0,
                "away_score": 17.0,
                "is_completed": True,
            }
        ]
    )
    features = pd.DataFrame(
        [
            {"season": 2023, "week": 1, "game_id": "2023_01_BBB_AAA", "team": team}
            for team in ("AAA", "BBB")
        ]
    )
    return add_matchup_target(build_matchups(features, schedule))


def test_market_descriptors_follow_the_verified_convention():
    """A positive spread means the home team is favoured."""
    favoured = _matchup_frame(7.0).iloc[0]
    underdog = _matchup_frame(-7.0).iloc[0]

    assert favoured["home_favorite"] == 1
    assert underdog["home_favorite"] == 0
    assert favoured["absolute_spread"] == 7.0
    assert underdog["absolute_spread"] == 7.0


def test_close_and_large_line_flags():
    close = _matchup_frame(2.5).iloc[0]
    large = _matchup_frame(9.0).iloc[0]

    assert close["close_game_line"] == 1
    assert close["large_favorite"] == 0
    assert large["close_game_line"] == 0
    assert large["large_favorite"] == 1


# --- prediction policy -----------------------------------------------------


def test_confidence_tiers_are_the_specified_bands():
    labels = [label for _, label in CONFIDENCE_TIERS]

    assert labels == [
        "lean only",
        "low confidence",
        "medium confidence",
        "high model confidence",
    ]


def test_each_probability_lands_in_the_right_tier():
    tiers = confidence_tier(np.array([0.505, 0.53, 0.56, 0.62, 0.38]))

    assert list(tiers) == [
        "lean only",
        "low confidence",
        "medium confidence",
        "high model confidence",
        # 0.38 is a strong away call: 0.62 of confidence in the away side.
        "high model confidence",
    ]


def test_the_no_bet_baseline_places_no_bets():
    frame = pd.DataFrame({"game_id": ["a", "b", "c"]})

    picks = predict_no_bet(frame)

    assert len(picks) == 3
    assert picks.isna().all()


def test_break_even_is_the_documented_rate():
    assert pytest.approx(110 / 210) == BREAK_EVEN_ACCURACY
    assert pytest.approx(0.5238, abs=1e-4) == BREAK_EVEN_ACCURACY


def test_accuracy_splits_by_favourite_and_underdog():
    matchups = pd.DataFrame(
        {
            "game_id": ["a", "b", "c", "d"],
            "season": [2023] * 4,
            "spread_line": [7.0, 7.0, -7.0, -7.0],
            "home_cover": [1.0, 0.0, 1.0, 0.0],
        }
    )
    # Back the home side every time: the favourite when the spread is
    # positive, the underdog when it is negative.
    picks = pd.Series([1, 1, 1, 1], index=matchups.index, dtype="int8")

    table = accuracy_by_market_role(matchups, picks)

    assert set(table["role"]) == {"backed the favourite", "backed the underdog"}
    assert table["bets"].sum() == 4


# --- configuration ---------------------------------------------------------


@pytest.mark.parametrize("name", ["base", "model", "features"])
def test_each_config_file_parses(name):
    payload = yaml.safe_load((CONFIGS / f"{name}.yaml").read_text(encoding="utf-8"))

    assert isinstance(payload, dict)
    assert payload


def test_base_config_matches_the_specification():
    payload = yaml.safe_load((CONFIGS / "base.yaml").read_text(encoding="utf-8"))

    assert payload["project"]["random_seed"] == 42
    assert payload["data"]["training_seasons"] == list(range(2016, 2026))
    assert payload["data"]["prediction_season"] == 2026
    assert payload["data"]["game_type"] == "REG"
    assert payload["betting"]["american_odds"] == -110
    assert payload["betting"]["exclude_pushes"] is True
    assert payload["features"]["week_one_rest_days"] == 7


def test_the_config_seed_matches_the_code():
    payload = yaml.safe_load((CONFIGS / "base.yaml").read_text(encoding="utf-8"))

    assert payload["project"]["random_seed"] == DEFAULT_RANDOM_SEED


def test_the_config_feature_list_matches_the_code():
    payload = yaml.safe_load((CONFIGS / "features.yaml").read_text(encoding="utf-8"))

    assert payload["baseline"] == FEATURE_COLUMNS


def test_the_config_grid_matches_the_code():
    payload = yaml.safe_load((CONFIGS / "model.yaml").read_text(encoding="utf-8"))

    assert payload["search"]["C"] == C_VALUES
    assert payload["search"]["class_weight"] == CLASS_WEIGHTS
    assert payload["validation"]["first_validation_season"] == 2019
    assert payload["validation"]["final_validation_season"] == 2025


def test_the_config_model_settings_match_the_pipeline():
    payload = yaml.safe_load((CONFIGS / "base.yaml").read_text(encoding="utf-8"))
    classifier = build_pipeline().named_steps[MODEL_STEP]

    assert payload["model"]["solver"] == classifier.solver
    assert payload["model"]["max_iter"] == classifier.max_iter
    assert payload["model"]["penalty"] == "l2"
