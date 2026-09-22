"""Tests for the exploratory charts and the correlation review."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from gridiron.eda import charts  # noqa: E402
from gridiron.eda.correlation import (  # noqa: E402
    AVOIDED_BY_CONSTRUCTION,
    CORRELATION_THRESHOLD,
    DEFENSIVE_EPA_DECISION,
    CorrelationError,
    correlated_pairs,
    correlation_decisions,
    correlation_matrix,
    numeric_fields,
)
from gridiron.eda.theme import (  # noqa: E402
    CATEGORICAL,
)

FEATURES = [
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


@pytest.fixture
def matchups() -> pd.DataFrame:
    """A small but realistic game-level frame."""
    rng = np.random.default_rng(11)
    count = 240
    seasons = rng.choice([2022, 2023], count)
    weeks = rng.integers(1, 15, count)
    margin = rng.normal(0, 8, count)
    return pd.DataFrame(
        {
            "game_id": [f"g{index:04d}" for index in range(count)],
            "season": seasons,
            "week": weeks,
            "gameday": pd.Timestamp("2023-09-10") + pd.to_timedelta(weeks * 7, "D"),
            "home_team": rng.choice(["AAA", "BBB", "CCC", "DDD"], count),
            "away_team": rng.choice(["EEE", "FFF", "GGG", "HHH"], count),
            "spread_line": rng.normal(0, 5, count).round(1),
            "off_epa_diff_last_5": rng.normal(0, 0.2, count),
            "def_epa_strength_diff_last_5": rng.normal(0, 0.2, count),
            "pace_diff_last_5": rng.normal(0, 5, count),
            "rest_diff": rng.integers(-7, 8, count).astype(float),
            "point_margin_diff_last_5": margin,
            # Deliberately near-duplicate of the margin, to exercise flagging.
            "win_pct_diff_last_5": margin / 20 + rng.normal(0, 0.01, count),
            "home_short_week": rng.random(count) < 0.15,
            "away_short_week": rng.random(count) < 0.15,
            "div_game": rng.integers(0, 2, count).astype("int8"),
            "week_1_flag": (weeks == 1).astype("int8"),
            "home_cover": rng.integers(0, 2, count).astype("float"),
            "is_push": rng.random(count) < 0.02,
        }
    )


@pytest.fixture
def team_games() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    rows = []
    for team in ("AAA", "BBB", "CCC"):
        for season in (2022, 2023):
            for week in range(1, 11):
                rows.append(
                    {
                        "team": team,
                        "season": season,
                        "week": week,
                        "offensive_epa_per_play": rng.normal(0, 0.15),
                        "defensive_epa_strength_per_play": rng.normal(0, 0.15),
                        "pace_plays_per_game": rng.normal(62, 8),
                        "seconds_per_play": rng.normal(30, 3),
                    }
                )
    return pd.DataFrame(rows)


# --- correlation -----------------------------------------------------------


def test_numeric_fields_excludes_booleans_and_missing_columns(matchups):
    usable = numeric_fields(matchups, [*FEATURES, "not_a_column"])

    assert "home_short_week" not in usable
    assert "not_a_column" not in usable
    assert "off_epa_diff_last_5" in usable


def test_correlation_matrix_is_square_and_unit_diagonal(matchups):
    matrix = correlation_matrix(matchups, FEATURES)

    assert matrix.shape[0] == matrix.shape[1]
    assert np.allclose(np.diag(matrix), 1.0)


def test_correlation_matrix_needs_two_numeric_columns():
    with pytest.raises(CorrelationError, match="at least two"):
        correlation_matrix(pd.DataFrame({"a": [1.0, 2.0]}), ["a"])


def test_correlated_pairs_reports_each_pair_once(matchups):
    matrix = correlation_matrix(matchups, FEATURES)
    pairs = correlated_pairs(matrix, 0.0)

    seen = {frozenset({row.left, row.right}) for row in pairs.itertuples()}
    assert len(seen) == len(pairs)
    assert not (pairs["left"] == pairs["right"]).any()


def test_correlated_pairs_honours_the_threshold(matchups):
    matrix = correlation_matrix(matchups, FEATURES)
    pairs = correlated_pairs(matrix, CORRELATION_THRESHOLD)

    assert (pairs["abs_correlation"] >= CORRELATION_THRESHOLD).all()
    # The fixture builds win_pct as a scaled copy of the margin, so that pair
    # must be caught.
    caught = {frozenset({row.left, row.right}) for row in pairs.itertuples()}
    assert frozenset({"point_margin_diff_last_5", "win_pct_diff_last_5"}) in caught


def test_correlated_pairs_are_sorted_strongest_first(matchups):
    matrix = correlation_matrix(matchups, FEATURES)
    pairs = correlated_pairs(matrix, 0.0)

    assert pairs["abs_correlation"].is_monotonic_decreasing


def test_every_flagged_pair_carries_a_recorded_reason(matchups):
    matrix = correlation_matrix(matchups, FEATURES)
    pairs = correlated_pairs(matrix, CORRELATION_THRESHOLD)
    decisions = correlation_decisions(pairs)

    assert len(decisions) == len(pairs)
    assert decisions["reason"].str.len().gt(20).all()
    assert decisions["decision"].notna().all()


def test_window_pairs_are_explained_as_same_metric():
    pairs = pd.DataFrame(
        {
            "left": ["off_epa_diff_last_5"],
            "right": ["off_epa_diff_last_3"],
            "correlation": [0.9],
            "abs_correlation": [0.9],
        }
    )
    decisions = correlation_decisions(pairs)

    assert "same metric" in decisions.iloc[0]["reason"].lower()


def test_defensive_epa_redundancy_is_recorded_even_though_unmeasurable():
    """The pair never coexists, so only an explicit record can capture it."""
    decisions = correlation_decisions(
        pd.DataFrame(columns=["left", "right", "correlation", "abs_correlation"]),
        extra=(DEFENSIVE_EPA_DECISION,),
    )

    assert len(decisions) == 1
    row = decisions.iloc[0]
    assert row["decision"] == AVOIDED_BY_CONSTRUCTION
    assert row["correlation"] == -1.0
    assert "strength" in row["reason"]


# --- charts ----------------------------------------------------------------


@pytest.fixture
def captured(monkeypatch):
    """Capture each figure before the chart function closes it."""
    figures: list[tuple[str, plt.Figure]] = []
    original = charts._save

    def _capture(figure, output_dir, name):
        figures.append((name, figure))
        return original(figure, output_dir, name)

    monkeypatch.setattr(charts, "_save", _capture)
    return figures


def _titled_axes(figure: plt.Figure) -> list[plt.Axes]:
    """Visible axes that carry data, ignoring colourbars and hidden panels."""
    return [
        axis
        for axis in figure.axes
        if axis.get_visible() and axis.get_xlabel() and axis.get_ylabel()
    ]


def test_every_chart_writes_a_file(matchups, team_games, tmp_path):
    paths = charts.build_all_charts(matchups, team_games, FEATURES, tmp_path)

    assert len(paths) == 10
    for path in paths:
        assert path.exists()
        assert path.stat().st_size > 5_000


def test_every_chart_has_a_title_and_labelled_axes(
    matchups, team_games, tmp_path, captured
):
    charts.build_all_charts(matchups, team_games, FEATURES, tmp_path)

    assert len(captured) == 10
    for name, figure in captured:
        titles = [axis.get_title() for axis in figure.axes if axis.get_title()]
        suptitle = figure._suptitle.get_text() if figure._suptitle else ""
        assert titles or suptitle, f"{name} has no title"

        labelled = _titled_axes(figure)
        assert labelled, f"{name} has no axis with both labels"
        for axis in labelled:
            assert axis.get_xlabel().strip(), f"{name} x label empty"
            assert axis.get_ylabel().strip(), f"{name} y label empty"


def test_correlation_heatmap_uses_only_numeric_model_fields(
    matchups, tmp_path, captured
):
    numeric = numeric_fields(matchups, FEATURES)
    charts.correlation_heatmap(matchups, numeric, tmp_path)

    _, figure = captured[0]
    labels = [text.get_text() for text in figure.axes[0].get_xticklabels()]

    assert "home_short_week" not in labels
    assert set(labels).issubset(set(numeric))


def test_correlation_heatmap_is_diverging_and_centred(matchups, tmp_path, captured):
    """Two opposed hues with a neutral midpoint, symmetric about zero.

    Checked by colour rather than by colormap name: seaborn copies the map and
    the copy loses its name, and the colours are what the rule is actually
    about -- a hue at the midpoint would mean "no correlation" reads as
    something.
    """
    charts.correlation_heatmap(matchups, numeric_fields(matchups, FEATURES), tmp_path)

    _, figure = captured[0]
    mesh = figure.axes[0].collections[0]
    cmap = mesh.get_cmap()

    assert mesh.get_clim() == (-1.0, 1.0)

    low = cmap(0.0)
    mid = cmap(0.5)
    high = cmap(1.0)
    # Cool at one end, warm at the other.
    assert low[2] > low[0]
    assert high[0] > high[2]
    # The midpoint is neutral: no channel dominates.
    assert max(mid[:3]) - min(mid[:3]) < 0.06


def test_missing_heatmap_is_sequential(matchups, tmp_path, captured):
    """One hue running light to dark, never a rainbow.

    Missingness is a magnitude with no sign, so the ramp must darken
    monotonically and must not introduce a second hue along the way.
    """
    charts.missing_value_heatmap(matchups, FEATURES, tmp_path)

    _, figure = captured[0]
    mesh = figure.axes[0].collections[0]
    cmap = mesh.get_cmap()

    assert mesh.get_clim() == (0.0, 100.0)

    samples = [cmap(step / 8) for step in range(9)]
    luminance = [
        0.2126 * red + 0.7152 * green + 0.0722 * blue for red, green, blue, _ in samples
    ]
    assert all(
        later <= earlier + 1e-6
        for earlier, later in zip(luminance[:-1], luminance[1:], strict=True)
    )
    # Blue stays the dominant channel throughout: one hue, not a rainbow.
    assert all(blue >= red for red, _, blue, _ in samples)


def test_cover_rate_charts_draw_the_break_even_line(matchups, tmp_path, captured):
    charts.spread_bins_vs_cover(matchups, tmp_path)

    _, figure = captured[0]
    axis = figure.axes[0]
    heights = [line.get_ydata()[0] for line in axis.lines if len(line.get_ydata())]

    assert charts.BREAK_EVEN in heights


def test_cover_rate_bins_are_labelled_with_their_sample_size(
    matchups, tmp_path, captured
):
    charts.rest_advantage_vs_cover(matchups, tmp_path)

    _, figure = captured[0]
    labels = [text.get_text() for text in figure.axes[0].get_xticklabels()]

    assert all("n=" in label for label in labels)


def test_class_balance_legend_names_every_outcome(matchups, tmp_path, captured):
    charts.ats_class_balance(matchups, tmp_path)

    _, figure = captured[0]
    entries = {
        text.get_text()
        for axis in figure.axes
        if axis.get_legend()
        for text in axis.get_legend().get_texts()
    }

    assert {"Home cover", "Away cover", "Push"}.issubset(entries)


def test_categorical_slots_are_used_in_fixed_order(matchups, tmp_path, captured):
    charts.ats_class_balance(matchups, tmp_path)

    _, figure = captured[0]
    containers = figure.axes[0].containers
    colours = [container.patches[0].get_facecolor() for container in containers]

    from matplotlib.colors import to_rgba

    assert colours[0] == to_rgba(CATEGORICAL[0])
    assert colours[1] == to_rgba(CATEGORICAL[1])


def test_missing_column_raises_a_chart_error(matchups, tmp_path):
    with pytest.raises(charts.ChartError, match="rest_diff"):
        charts.rest_advantage_vs_cover(matchups.drop(columns="rest_diff"), tmp_path)


def test_team_season_chart_requires_its_value_column(team_games, tmp_path):
    with pytest.raises(charts.ChartError, match="offensive_epa_per_play"):
        charts.offensive_epa_by_team_season(
            team_games.drop(columns="offensive_epa_per_play"), tmp_path
        )


def test_distribution_chart_requires_at_least_one_feature(tmp_path):
    with pytest.raises(charts.ChartError, match="no features"):
        charts.feature_distributions(pd.DataFrame({"x": [1]}), tmp_path)
