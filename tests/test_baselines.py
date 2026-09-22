"""Tests for the baseline strategies and their metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gridiron.modeling.baselines import (
    BASELINES,
    DEFAULT_SEED,
    BaselineError,
    eligible_games,
    epa_fallback_count,
    predict_away_every_game,
    predict_better_epa,
    predict_favorite,
    predict_home_every_game,
    predict_random,
    predict_underdog,
)
from gridiron.modeling.metrics import (
    BREAK_EVEN_ACCURACY,
    WIN_PROFIT,
    MetricError,
    accuracy,
    comparison_table,
    evaluate,
    max_drawdown,
    profit_curve,
    roi,
    season_accuracy,
    season_table,
    wins_and_losses,
)


def _games(
    *,
    home_cover: list[float],
    spread_line: list[float] | None = None,
    epa: list[float] | None = None,
    seasons: list[int] | None = None,
    is_push: list[bool] | None = None,
) -> pd.DataFrame:
    count = len(home_cover)
    return pd.DataFrame(
        {
            "game_id": [f"g{index:03d}" for index in range(count)],
            "season": seasons or [2023] * count,
            "week": list(range(1, count + 1)),
            "gameday": [
                pd.Timestamp("2023-09-10") + pd.Timedelta(days=7 * index)
                for index in range(count)
            ],
            "spread_line": spread_line if spread_line is not None else [3.0] * count,
            "off_epa_diff_last_5": epa if epa is not None else [0.1] * count,
            "home_cover": home_cover,
            "is_push": is_push if is_push is not None else [False] * count,
        }
    )


# --- eligibility -----------------------------------------------------------


def test_eligibility_excludes_pushes_and_unsettled_games():
    games = _games(home_cover=[1.0, 0.0, float("nan"), float("nan")])
    games.loc[2, "is_push"] = True

    eligible = eligible_games(games)

    assert len(eligible) == 2
    assert not eligible["is_push"].any()
    assert eligible["home_cover"].notna().all()


def test_every_baseline_scores_the_same_games():
    """The comparison is only meaningful on one identical sample."""
    games = _games(
        home_cover=[1.0, 0.0, 1.0, 0.0],
        epa=[0.2, -0.2, float("nan"), 0.0],
        spread_line=[3.0, -3.0, 0.0, 7.0],
    )
    eligible = eligible_games(games)

    sizes = {name: len(function(eligible)) for name, function in BASELINES.items()}

    assert set(sizes.values()) == {len(eligible)}
    for function in BASELINES.values():
        predictions = function(eligible)
        assert predictions.index.equals(eligible.index)
        assert predictions.notna().all()


def test_require_epa_narrows_the_set_when_asked():
    games = _games(home_cover=[1.0, 0.0, 1.0], epa=[0.2, float("nan"), -0.1])

    assert len(eligible_games(games)) == 3
    assert len(eligible_games(games, require_epa=True)) == 2


def test_eligibility_needs_the_target_column():
    with pytest.raises(BaselineError, match="home_cover"):
        eligible_games(pd.DataFrame({"game_id": ["a"]}))


# --- individual baselines --------------------------------------------------


def test_home_and_away_baselines_are_opposites():
    games = _games(home_cover=[1.0, 0.0, 1.0])

    home = predict_home_every_game(games)
    away = predict_away_every_game(games)

    assert set(home) == {1}
    assert set(away) == {0}
    assert ((home + away) == 1).all()


def test_favorite_follows_the_spread_sign():
    """A positive spread means the home team is favoured."""
    games = _games(home_cover=[1.0, 1.0], spread_line=[6.5, -6.5])

    predictions = predict_favorite(games)

    assert list(predictions) == [1, 0]


def test_pick_em_resolves_to_the_home_side():
    games = _games(home_cover=[1.0], spread_line=[0.0])

    assert list(predict_favorite(games)) == [1]
    assert list(predict_underdog(games)) == [0]


def test_underdog_is_the_exact_inverse_of_favorite():
    games = _games(home_cover=[1.0] * 4, spread_line=[7.0, -7.0, 0.0, 2.5])

    favourite = predict_favorite(games)
    underdog = predict_underdog(games)

    assert ((favourite + underdog) == 1).all()


def test_better_epa_follows_the_differential():
    games = _games(home_cover=[1.0, 1.0], epa=[0.3, -0.3])

    assert list(predict_better_epa(games)) == [1, 0]


def test_better_epa_falls_back_when_the_signal_is_missing():
    """Openers have no EPA history, so the fallback decides them."""
    games = _games(home_cover=[1.0, 1.0], epa=[float("nan"), 0.2])

    assert list(predict_better_epa(games, fallback=1)) == [1, 1]
    assert list(predict_better_epa(games, fallback=0)) == [0, 1]
    assert epa_fallback_count(games) == 1


def test_better_epa_breaks_an_exact_tie_with_the_fallback():
    games = _games(home_cover=[1.0], epa=[0.0])

    assert list(predict_better_epa(games, fallback=0)) == [0]


def test_baseline_reports_a_missing_column():
    games = _games(home_cover=[1.0]).drop(columns="spread_line")

    with pytest.raises(BaselineError, match="spread_line"):
        predict_favorite(games)


# --- the random baseline ---------------------------------------------------


def test_random_baseline_is_reproducible():
    games = _games(home_cover=[1.0] * 50)

    first = predict_random(games, seed=DEFAULT_SEED)
    second = predict_random(games, seed=DEFAULT_SEED)

    pd.testing.assert_series_equal(first, second)


def test_random_baseline_is_unaffected_by_global_state():
    """Seeding numpy's legacy global RNG must not change the picks."""
    games = _games(home_cover=[1.0] * 50)

    expected = predict_random(games, seed=DEFAULT_SEED)
    np.random.seed(1234)
    [np.random.random() for _ in range(10)]

    pd.testing.assert_series_equal(predict_random(games, seed=DEFAULT_SEED), expected)


def test_different_seeds_give_different_picks():
    games = _games(home_cover=[1.0] * 200)

    assert not predict_random(games, seed=1).equals(predict_random(games, seed=2))


def test_random_baseline_only_predicts_valid_classes():
    games = _games(home_cover=[1.0] * 100)

    assert set(predict_random(games)).issubset({0, 1})


# --- metrics ---------------------------------------------------------------


def test_wins_and_losses_count_correct_picks():
    games = _games(home_cover=[1.0, 1.0, 0.0, 0.0])

    wins, losses = wins_and_losses(games, predict_home_every_game(games))

    assert (wins, losses) == (2, 2)


def test_accuracy_is_the_share_correct():
    games = _games(home_cover=[1.0, 1.0, 1.0, 0.0])

    assert accuracy(games, predict_home_every_game(games)) == pytest.approx(0.75)


def test_roi_uses_the_standard_price():
    """Ten wins and ten losses lose money at -110, not break even."""
    assert roi(10, 10) == pytest.approx((10 * WIN_PROFIT - 10) / 20)
    assert roi(10, 10) < 0


def test_break_even_accuracy_is_above_half():
    assert pytest.approx(0.5238, abs=1e-4) == BREAK_EVEN_ACCURACY
    assert roi(5238, 4762) == pytest.approx(0.0, abs=1e-3)


def test_roi_of_a_perfect_record_is_the_win_profit():
    assert roi(10, 0) == pytest.approx(WIN_PROFIT)


def test_profit_curve_follows_the_order_of_play():
    games = _games(home_cover=[1.0, 0.0, 1.0])
    shuffled = games.iloc[::-1]

    curve = profit_curve(shuffled, predict_home_every_game(shuffled))

    assert list(curve.index) == list(games.index)
    assert curve.iloc[-1] == pytest.approx(2 * WIN_PROFIT - 1)


def test_max_drawdown_measures_the_worst_fall():
    curve = pd.Series([1.0, 3.0, -2.0, 0.0])

    assert max_drawdown(curve) == pytest.approx(5.0)


def test_max_drawdown_counts_a_fall_from_the_starting_bankroll():
    """A strategy that only ever loses still has a real drawdown."""
    curve = pd.Series([-1.0, -2.0, -3.0])

    assert max_drawdown(curve) == pytest.approx(3.0)


def test_season_accuracy_breaks_results_down_by_year():
    games = _games(home_cover=[1.0, 1.0, 0.0, 0.0], seasons=[2022, 2022, 2023, 2023])

    seasons = season_accuracy(games, predict_home_every_game(games))

    assert list(seasons["season"]) == [2022, 2023]
    assert list(seasons["accuracy"]) == [1.0, 0.0]
    assert list(seasons["games"]) == [2, 2]


def test_scoring_rejects_unsettled_games():
    games = _games(home_cover=[1.0, float("nan")])

    with pytest.raises(MetricError, match="Unsettled"):
        accuracy(games, predict_home_every_game(games))


def test_scoring_rejects_misaligned_predictions():
    games = _games(home_cover=[1.0, 0.0])
    predictions = pd.Series([1, 0], index=[99, 100], dtype="int8")

    with pytest.raises(MetricError, match="aligned"):
        accuracy(games, predictions)


def test_scoring_rejects_a_length_mismatch():
    games = _games(home_cover=[1.0, 0.0])

    with pytest.raises(MetricError, match="differ in length"):
        accuracy(games, pd.Series([1], index=[0], dtype="int8"))


# --- comparison table ------------------------------------------------------


def test_comparison_table_scores_every_strategy_on_one_sample():
    games = _games(
        home_cover=[1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        spread_line=[3.0, -3.0, 0.0, 7.0, -1.0, 2.0],
        epa=[0.2, -0.2, float("nan"), 0.1, -0.1, 0.0],
    )
    eligible = eligible_games(games)
    predictions = {name: function(eligible) for name, function in BASELINES.items()}

    table = comparison_table(eligible, predictions)

    assert len(table) == len(BASELINES)
    assert set(table["games"]) == {len(eligible)}
    assert (table["wins"] + table["losses"] == table["games"]).all()


def test_comparison_table_is_sorted_by_roi():
    games = _games(home_cover=[1.0, 1.0, 1.0, 0.0])
    eligible = eligible_games(games)
    predictions = {name: function(eligible) for name, function in BASELINES.items()}

    table = comparison_table(eligible, predictions)

    assert table["roi"].is_monotonic_decreasing


def test_comparison_table_rejects_an_empty_strategy_set():
    with pytest.raises(MetricError, match="No strategies"):
        comparison_table(_games(home_cover=[1.0]), {})


def test_evaluate_reports_the_record_as_a_string():
    games = _games(home_cover=[1.0, 1.0, 0.0])

    row = evaluate(games, predict_home_every_game(games), "home")

    assert row["ats_record"] == "2-1"
    assert row["strategy"] == "home"


def test_season_table_covers_every_strategy_and_season():
    games = _games(home_cover=[1.0, 0.0, 1.0, 0.0], seasons=[2022, 2022, 2023, 2023])
    eligible = eligible_games(games)
    predictions = {name: function(eligible) for name, function in BASELINES.items()}

    table = season_table(eligible, predictions)

    assert set(table["strategy"]) == set(BASELINES)
    assert set(table["season"]) == {2022, 2023}
    assert len(table) == len(BASELINES) * 2


def test_opposite_baselines_have_complementary_records():
    """Backing home and backing away must split every settled game."""
    games = _games(home_cover=[1.0, 0.0, 1.0, 1.0, 0.0])
    eligible = eligible_games(games)

    home_wins, home_losses = wins_and_losses(
        eligible, predict_home_every_game(eligible)
    )
    away_wins, away_losses = wins_and_losses(
        eligible, predict_away_every_game(eligible)
    )

    assert home_wins == away_losses
    assert home_losses == away_wins
    assert home_wins + away_wins == len(eligible)
