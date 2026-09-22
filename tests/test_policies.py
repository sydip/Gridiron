"""Tests for the betting-policy backtest."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from gridiron.eda import policy_charts  # noqa: E402
from gridiron.modeling.policies import (  # noqa: E402
    DEFAULT_ODDS,
    LOSS,
    MIN_BETS_FOR_A_CLAIM,
    POLICY_THRESHOLDS,
    PUSH,
    RESULT_COLUMNS,
    TINY_SAMPLE,
    WIN,
    PolicyError,
    apply_policy,
    backtest_policies,
    backtest_policy,
    cumulative_units,
    edge_percentage_points,
    flag_unreliable_results,
    longest_losing_streak,
    max_drawdown,
    season_units,
    settle,
    wilson_interval,
    win_rate_by_confidence,
)


def _predictions(
    probability: list[float],
    actual: list[float],
    pushes: list[bool] | None = None,
    seasons: list[int] | None = None,
) -> pd.DataFrame:
    count = len(probability)
    return pd.DataFrame(
        {
            "game_id": [f"g{index:04d}" for index in range(count)],
            "season": seasons or [2020] * count,
            "week": [(index % 17) + 1 for index in range(count)],
            "actual_home_cover": actual,
            "predicted_home_cover": [1 if value >= 0.5 else 0 for value in probability],
            "home_cover_probability": probability,
            "is_push": pushes if pushes is not None else [False] * count,
        }
    )


def _coinflip(count: int = 600, seed: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    probability = rng.uniform(0.40, 0.60, count)
    actual = rng.binomial(1, 0.5, count).astype(float)
    seasons = list(np.repeat([2019, 2020, 2021], count // 3))[:count]
    return _predictions(list(probability), list(actual), seasons=seasons)


# --- the edge and policy selection -----------------------------------------


def test_edge_is_distance_from_a_coin_flip_in_points():
    edges = edge_percentage_points(pd.Series([0.50, 0.55, 0.45, 0.60]))

    assert list(edges) == pytest.approx([0.0, 5.0, 5.0, 10.0])


def test_policies_cover_the_required_thresholds():
    assert POLICY_THRESHOLDS == [0.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0]


def test_a_stricter_threshold_never_selects_more_bets():
    predictions = _coinflip()

    counts = [
        len(apply_policy(predictions, threshold)) for threshold in POLICY_THRESHOLDS
    ]

    assert counts == sorted(counts, reverse=True)


def test_betting_every_prediction_selects_everything():
    predictions = _coinflip()

    assert len(apply_policy(predictions, 0.0)) == len(predictions)


def test_a_policy_selects_both_sides_of_the_threshold():
    """An edge of 5 points means 0.55 or 0.45, not only the high side."""
    predictions = _predictions([0.55, 0.45, 0.52], [1.0, 0.0, 1.0])

    selected = apply_policy(predictions, 5.0)

    assert len(selected) == 2
    assert set(selected["home_cover_probability"]) == {0.55, 0.45}


def test_applying_a_policy_needs_probabilities():
    with pytest.raises(PolicyError, match="probabilities"):
        apply_policy(pd.DataFrame({"game_id": ["a"]}), 0.0)


# --- settlement, and pushes ------------------------------------------------


def test_settlement_labels_wins_losses_and_pushes():
    predictions = _predictions(
        [0.6, 0.6, 0.6],
        [1.0, 0.0, float("nan")],
        pushes=[False, False, True],
    )

    settlements = settle(predictions)

    assert list(settlements) == [WIN, LOSS, PUSH]


def test_a_push_is_neither_a_win_nor_a_loss():
    """The central requirement: a refunded stake must not move any rate."""
    predictions = _predictions(
        [0.6] * 4,
        [1.0, 0.0, float("nan"), float("nan")],
        pushes=[False] * 2 + [True] * 2,
    )

    row = backtest_policy(predictions, 0.0)

    assert row["bets"] == 4
    assert row["wins"] == 1
    assert row["losses"] == 1
    assert row["pushes"] == 2
    assert row["win_rate"] == pytest.approx(0.5)


def test_pushes_contribute_nothing_to_profit():
    only_pushes = _predictions([0.6] * 3, [float("nan")] * 3, pushes=[True] * 3)

    row = backtest_policy(only_pushes, 0.0)

    assert row["units_won"] == pytest.approx(0.0)
    assert row["pushes"] == 3
    assert row["wins"] == 0
    assert row["losses"] == 0


def test_a_push_does_not_break_a_losing_streak():
    settlements = pd.Series([LOSS, LOSS, PUSH, LOSS, WIN, LOSS])

    assert longest_losing_streak(settlements) == 3


def test_a_win_breaks_a_losing_streak():
    settlements = pd.Series([LOSS, LOSS, WIN, LOSS])

    assert longest_losing_streak(settlements) == 2


# --- the odds assumption ---------------------------------------------------


def test_the_odds_assumption_is_documented_and_used():
    assert DEFAULT_ODDS.name == "American -110"
    assert DEFAULT_ODDS.win_profit == pytest.approx(100 / 110)
    assert DEFAULT_ODDS.break_even_rate == pytest.approx(0.5238, abs=1e-4)


def test_an_even_record_loses_money_at_the_documented_price():
    predictions = _predictions([0.6] * 20, [1.0] * 10 + [0.0] * 10)

    row = backtest_policy(predictions, 0.0)

    assert row["win_rate"] == pytest.approx(0.5)
    assert row["units_won"] < 0
    assert row["roi"] < 0


def test_breaking_even_requires_more_than_half():
    wins, losses = 5238, 4762
    predictions = _predictions([0.6] * (wins + losses), [1.0] * wins + [0.0] * losses)

    row = backtest_policy(predictions, 0.0)

    assert row["roi"] == pytest.approx(0.0, abs=1e-3)


# --- results ---------------------------------------------------------------


def test_results_have_every_required_column():
    results = backtest_policies(_coinflip())

    for column in RESULT_COLUMNS:
        assert column in results.columns


def test_every_policy_is_reported_even_when_it_loses():
    results = backtest_policies(_coinflip())

    assert len(results) == len(POLICY_THRESHOLDS)


def test_bets_equal_wins_plus_losses_plus_pushes():
    predictions = _predictions(
        [0.6] * 10,
        [1.0, 1.0, 0.0, 0.0, float("nan")] + [1.0] * 5,
        pushes=[False] * 4 + [True] + [False] * 5,
    )

    row = backtest_policy(predictions, 0.0)

    assert row["bets"] == row["wins"] + row["losses"] + row["pushes"]


def test_season_profitability_counts_winning_seasons():
    predictions = _predictions(
        [0.6] * 6,
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        seasons=[2019] * 3 + [2020] * 3,
    )

    row = backtest_policy(predictions, 0.0)

    assert row["seasons"] == 2
    assert row["profitable_seasons"] == 1


def test_max_drawdown_is_reported_as_a_positive_number():
    profit = pd.Series([1.0, 1.0, -1.0, -1.0, -1.0])

    assert max_drawdown(profit) == pytest.approx(3.0)


def test_max_drawdown_of_a_pure_loser_is_its_whole_decline():
    assert max_drawdown(pd.Series([-1.0, -1.0, -1.0])) == pytest.approx(3.0)


def test_an_empty_policy_reports_zero_bets_not_an_error():
    predictions = _predictions([0.50] * 5, [1.0] * 5)

    row = backtest_policy(predictions, 10.0)

    assert row["bets"] == 0
    assert row["sample_warning"] == "no bets placed"


# --- statistical caution ---------------------------------------------------


def test_the_confidence_interval_widens_as_the_sample_shrinks():
    wide = wilson_interval(4, 7)
    narrow = wilson_interval(400, 700)

    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_the_interval_stays_inside_zero_and_one():
    low, high = wilson_interval(7, 7)

    assert low >= 0.0
    assert high <= 1.0


def test_an_interval_needs_resolved_bets():
    low, high = wilson_interval(0, 0)

    assert np.isnan(low) and np.isnan(high)


def test_a_tiny_sample_is_flagged_even_with_a_great_return():
    """The case this exists for: four wins from seven bets is not a strategy."""
    predictions = _predictions([0.65] * 7, [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0])

    row = backtest_policy(predictions, 10.0)

    assert row["bets"] == 7
    assert row["roi"] > 0
    assert row["sample_warning"]
    assert "noise" in row["sample_warning"]


def test_a_thin_sample_is_flagged_below_the_claim_threshold():
    predictions = _predictions([0.65] * 50, [1.0] * 30 + [0.0] * 20)

    row = backtest_policy(predictions, 10.0)

    assert TINY_SAMPLE <= row["bets"] < MIN_BETS_FOR_A_CLAIM
    assert str(MIN_BETS_FOR_A_CLAIM) in row["sample_warning"]


def test_a_large_sample_is_not_flagged():
    predictions = _predictions([0.65] * 500, [1.0] * 250 + [0.0] * 250)

    row = backtest_policy(predictions, 10.0)

    assert row["sample_warning"] == ""


def test_unreliable_results_are_collected_with_their_interval_width():
    predictions = pd.concat(
        [
            _predictions([0.51] * 500, [1.0] * 250 + [0.0] * 250),
            _predictions([0.68] * 6, [1.0] * 4 + [0.0] * 2),
        ],
        ignore_index=True,
    )

    flagged = flag_unreliable_results(backtest_policies(predictions))

    assert not flagged.empty
    assert "interval_width" in flagged.columns
    assert (flagged["interval_width"] > 0).all()


def test_win_rate_by_confidence_carries_counts_and_intervals():
    table = win_rate_by_confidence(_coinflip())

    assert "bets" in table.columns
    assert "ci_low" in table.columns
    assert "sparse" in table.columns
    assert (table["bets"] > 0).all()


# --- curves ----------------------------------------------------------------


def test_cumulative_units_follows_the_order_of_play():
    predictions = _predictions(
        [0.6] * 4, [1.0, 1.0, 0.0, 0.0], seasons=[2019, 2019, 2020, 2020]
    )

    curve = cumulative_units(predictions, 0.0)

    assert list(curve["season"]) == [2019, 2019, 2020, 2020]
    assert curve["cumulative_units"].iloc[-1] == pytest.approx(
        2 * DEFAULT_ODDS.win_profit - 2
    )


def test_season_units_splits_by_year():
    predictions = _predictions(
        [0.6] * 4, [1.0, 1.0, 0.0, 0.0], seasons=[2019, 2019, 2020, 2020]
    )

    table = season_units(predictions, 0.0)

    assert list(table["season"]) == [2019, 2020]
    assert table.loc[0, "units_won"] > 0
    assert table.loc[1, "units_won"] < 0


# --- charts ----------------------------------------------------------------


def test_all_six_policy_charts_are_written(tmp_path):
    predictions = _coinflip()
    results = backtest_policies(predictions)

    paths = policy_charts.build_policy_charts(predictions, results, tmp_path)

    assert len(paths) == 6
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 5_000


def test_policy_charts_have_titles_and_labelled_axes(tmp_path, monkeypatch):
    figures = []
    original = policy_charts._save

    def _capture(figure, output_dir, name):
        figures.append((name, figure))
        return original(figure, output_dir, name)

    monkeypatch.setattr(policy_charts, "_save", _capture)
    predictions = _coinflip()
    policy_charts.build_policy_charts(
        predictions, backtest_policies(predictions), tmp_path
    )

    assert len(figures) == 6
    for name, figure in figures:
        titles = [axis.get_title() for axis in figure.axes if axis.get_title()]
        suptitle = figure._suptitle.get_text() if figure._suptitle else ""
        assert titles or suptitle, f"{name} has no title"

        labelled = [
            axis
            for axis in figure.axes
            if axis.get_visible() and axis.get_xlabel() and axis.get_ylabel()
        ]
        assert labelled, f"{name} has no fully labelled axis"


def test_bet_count_chart_shows_the_minimum_claim_line(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(
        policy_charts,
        "_save",
        lambda figure, output_dir, name: figures.append(figure) or tmp_path / name,
    )
    predictions = _coinflip()
    policy_charts.bets_by_threshold_chart(backtest_policies(predictions), tmp_path)

    labels = [text.get_text() for text in figures[0].axes[0].get_legend().get_texts()]

    assert any(str(MIN_BETS_FOR_A_CLAIM) in label for label in labels)


def test_confidence_chart_labels_every_bucket_with_its_count(tmp_path, monkeypatch):
    figures = []
    monkeypatch.setattr(
        policy_charts,
        "_save",
        lambda figure, output_dir, name: figures.append(figure) or tmp_path / name,
    )
    policy_charts.win_rate_by_confidence_chart(_coinflip(), tmp_path)

    labels = [text.get_text() for text in figures[0].axes[0].get_xticklabels()]

    assert all("n=" in label for label in labels)
