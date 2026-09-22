"""Tests for the visual reports."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

from gridiron.eda import reports  # noqa: E402
from gridiron.eda.reports import (  # noqa: E402
    BASELINE_NAMES,
    FRIENDLY_NAMES,
    HISTORICAL_ACCENT,
    UPCOMING_ACCENT,
    ReportError,
    ReportInputs,
    as_percent,
    friendly,
    historical_report,
    load_report_inputs,
    prediction_contributions,
    weekly_report,
)
from gridiron.modeling.pipeline import build_pipeline, get_feature_columns  # noqa: E402

FEATURES = get_feature_columns()


def _inputs() -> ReportInputs:
    rng = np.random.default_rng(17)
    seasons = pd.DataFrame(
        {
            "season": [2019, 2020, 2021, 2022, 2023, 2024, 2025],
            "games": [246, 256, 268, 261, 258, 268, 271],
            "wins": [135, 130, 133, 130, 127, 122, 138],
            "losses": [111, 126, 135, 131, 131, 146, 133],
            "accuracy": [0.549, 0.508, 0.496, 0.498, 0.492, 0.455, 0.509],
            "roi": [0.048, -0.031, -0.053, -0.049, -0.060, -0.131, -0.028],
            "log_loss": [0.699] * 7,
            "brier_score": [0.253] * 7,
        }
    )
    aggregate = pd.DataFrame(
        {
            "metric": ["log_loss", "brier_score", "roi", "roc_auc"],
            "value": [0.6964, 0.2516, -0.034, 0.4817],
        }
    )
    baselines = pd.DataFrame(
        {
            "strategy": [
                "random",
                "underdog",
                "away_every_game",
                "better_epa",
                "MODEL walk_forward",
                "home_every_game",
                "favorite",
            ],
            "roi": [0.0005, -0.011, -0.021, -0.044, -0.034, -0.069, -0.080],
            "ats_record": [
                "958-870",
                "947-881",
                "937-891",
                "915-913",
                "925-903",
                "891-937",
                "881-947",
            ],
        }
    )
    policies = pd.DataFrame(
        {
            "threshold_pp": [0.0, 2.0, 3.0, 5.0],
            "bets": [1871, 938, 600, 188],
            "units_won": [-62.0, -74.0, -59.0, -25.0],
            "max_drawdown_units": [84.0, 83.0, 68.0, 30.0],
        }
    )
    reliability = pd.DataFrame(
        {
            "bucket": ["0.45-0.50", "0.50-0.55", "0.55-0.60"],
            "games": [1134, 481, 33],
            "mean_predicted": [0.478, 0.516, 0.572],
            "observed_rate": [0.482, 0.480, 0.485],
            "standard_error": [0.015, 0.023, 0.087],
            "sparse": [False, False, False],
        }
    )
    coefficients = pd.DataFrame(
        {
            "feature": FEATURES,
            "coefficient": rng.normal(0, 0.03, len(FEATURES)),
            "abs_coefficient": np.abs(rng.normal(0, 0.03, len(FEATURES))),
            "odds_ratio": np.exp(rng.normal(0, 0.03, len(FEATURES))),
            "odds_change_pct": rng.normal(0, 3, len(FEATURES)),
            "vif": rng.uniform(1, 11, len(FEATURES)),
        }
    )
    correlations = pd.DataFrame(
        rng.uniform(-1, 1, (len(FEATURES), len(FEATURES))),
        index=FEATURES,
        columns=FEATURES,
    )
    buckets = pd.DataFrame(
        {
            "edge_bucket": ["[0.0, 2.0)", "[2.0, 3.0)", "[3.0, 4.0)"],
            "bets": [933, 338, 255],
            "win_rate": [0.526, 0.488, 0.498],
        }
    )
    walk_forward = pd.DataFrame({"game_id": [f"g{i}" for i in range(1828)]})

    return ReportInputs(
        walk_forward=walk_forward,
        by_season=seasons,
        aggregate=aggregate,
        baselines=baselines,
        policies=policies,
        reliability=reliability,
        coefficients=coefficients,
        correlations=correlations,
        confidence_buckets=buckets,
    )


def _prediction_table(games: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(23)
    probability = rng.uniform(0.46, 0.56, games)
    return pd.DataFrame(
        {
            "season": 2026,
            "week": 3,
            "game_id": [f"2026_03_A{i}_H{i}" for i in range(games)],
            "gameday": "2026-09-27",
            "away_team": [f"A{i}" for i in range(games)],
            "home_team": [f"H{i}" for i in range(games)],
            "spread_line": rng.normal(0, 4, games).round(1),
            "home_cover_probability": probability,
            "away_cover_probability": 1 - probability,
            "predicted_side": [
                f"H{i}" if value >= 0.5 else f"A{i}"
                for i, value in enumerate(probability)
            ],
            "confidence": np.abs(probability - 0.5),
            "recommendation_tier": "lean only",
            "status": "predicted",
            "spread_line_used": rng.normal(0, 4, games).round(1),
        }
    )


def _matchups(table: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(29)
    frame = table[["game_id", "home_team", "away_team"]].copy()
    for name in FEATURES:
        frame[name] = rng.normal(0, 1, len(frame))
    return frame


def _pipeline():
    rng = np.random.default_rng(5)
    features = pd.DataFrame(rng.normal(0, 1, (200, len(FEATURES))), columns=FEATURES)
    labels = pd.Series((features["spread_line"] > 0).astype(int))
    return build_pipeline().fit(features, labels)


# --- capturing what a report drew ------------------------------------------


def _snapshot(monkeypatch) -> dict:
    """Record a figure's titles and text while it is still alive.

    The report functions call plt.close() once they have saved, and closing a
    figure clears its axes -- so inspecting the object afterwards finds
    nothing. The snapshot is taken inside savefig instead.
    """
    captured: dict = {"titles": [], "texts": []}
    original = plt.Figure.savefig

    def _capture(self, *args, **kwargs):
        # Titles are set with loc="left"; get_title() reads the centre slot by
        # default and would report every panel as untitled.
        captured["titles"] = [
            title
            for axis in self.axes
            for location in ("left", "center", "right")
            if (title := axis.get_title(loc=location))
        ]
        captured["texts"] = [item.get_text() for item in self.texts]
        if self._suptitle is not None:
            captured["texts"].append(self._suptitle.get_text())
        return original(self, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", _capture)
    return captured


# --- formatting ------------------------------------------------------------


def test_percentages_use_one_rule():
    assert as_percent(0.5238) == "52.4%"
    assert as_percent(0.5238, 2) == "52.38%"
    assert as_percent(-0.034) == "-3.4%"
    assert as_percent(float("nan")) == "n/a"


def test_every_model_feature_has_a_football_friendly_name():
    for name in FEATURES:
        assert name in FRIENDLY_NAMES
        label = friendly(name)
        assert "_" not in label
        assert label != name


def test_an_unknown_column_still_gets_a_readable_label():
    assert friendly("some_new_thing") == "Some new thing"


def test_baselines_have_plain_english_names():
    for key, label in BASELINE_NAMES.items():
        assert "_" not in label
        assert label != key


# --- loading ---------------------------------------------------------------


def test_missing_saved_tables_are_named(tmp_path):
    with pytest.raises(ReportError, match="walk_forward_predictions.csv"):
        load_report_inputs(tmp_path)


def test_inputs_load_from_saved_csvs(tmp_path):
    """Reports are drawn from saved tables, not recomputed."""
    inputs = _inputs()
    (tmp_path / "walk_forward_predictions.csv").write_text(
        inputs.walk_forward.to_csv(index=False)
    )
    inputs.by_season.to_csv(tmp_path / "walk_forward_by_season.csv", index=False)
    inputs.aggregate.to_csv(tmp_path / "walk_forward_aggregate.csv", index=False)
    inputs.baselines.to_csv(tmp_path / "walk_forward_vs_baselines.csv", index=False)
    inputs.policies.to_csv(tmp_path / "policy_backtest.csv", index=False)
    inputs.reliability.to_csv(tmp_path / "calibration_reliability.csv", index=False)
    inputs.coefficients.to_csv(tmp_path / "model_coefficients.csv", index=False)
    inputs.correlations.to_csv(tmp_path / "eda_correlation_model_fields.csv")
    inputs.confidence_buckets.to_csv(
        tmp_path / "policy_win_rate_by_confidence.csv", index=False
    )

    loaded = load_report_inputs(tmp_path)

    assert len(loaded.by_season) == 7
    assert list(loaded.correlations.index) == FEATURES


# --- the historical report -------------------------------------------------


def test_the_historical_report_is_written(tmp_path):
    path = historical_report(_inputs(), tmp_path)

    assert path.exists()
    assert path.stat().st_size > 20_000


def test_every_historical_panel_has_a_title(tmp_path, monkeypatch):
    captured = _snapshot(monkeypatch)

    historical_report(_inputs(), tmp_path)

    assert len(captured["titles"]) >= 8


def test_the_historical_report_says_it_is_historical(tmp_path, monkeypatch):
    """Historical and upcoming must be separable at a glance."""
    captured = _snapshot(monkeypatch)

    historical_report(_inputs(), tmp_path)

    text = " ".join(captured["texts"])

    assert "never seen" in text
    assert "settled result" in text
    assert "not a prediction" in text


# --- the weekly report -----------------------------------------------------


def test_contributions_reproduce_the_model_arithmetic():
    """The reasoning panel must be the model's own sum, not an approximation."""
    pipeline = _pipeline()
    table = _prediction_table()
    features = _matchups(table)[FEATURES]

    contributions = prediction_contributions(features, pipeline)
    intercept = float(pipeline.named_steps["classifier"].intercept_[0])
    logit = contributions.sum(axis=1) + intercept
    rebuilt = 1 / (1 + np.exp(-logit))

    assert rebuilt.to_numpy() == pytest.approx(pipeline.predict_proba(features)[:, 1])


def test_the_weekly_report_is_written(tmp_path):
    table = _prediction_table()
    matchups = _matchups(table)
    contributions = prediction_contributions(matchups[FEATURES], _pipeline())

    path = weekly_report(table, matchups, contributions, tmp_path, 2026, 3)

    assert path.exists()
    assert path.name == "23_week_03_report.png"


def test_the_weekly_report_says_these_are_predictions(tmp_path, monkeypatch):
    captured = _snapshot(monkeypatch)
    table = _prediction_table()
    matchups = _matchups(table)
    weekly_report(
        table,
        matchups,
        prediction_contributions(matchups[FEATURES], _pipeline()),
        tmp_path,
        2026,
        3,
    )

    text = " ".join(captured["texts"])

    assert "predictions, not results" in text
    assert "no demonstrated edge" in text


def test_the_weekly_report_explains_why(tmp_path, monkeypatch):
    """A reader must be able to see which inputs moved each pick."""
    captured = _snapshot(monkeypatch)
    table = _prediction_table()
    matchups = _matchups(table)
    weekly_report(
        table,
        matchups,
        prediction_contributions(matchups[FEATURES], _pipeline()),
        tmp_path,
        2026,
        3,
    )

    titles = captured["titles"]

    assert any("Why" in title for title in titles)
    assert any("comparison" in title.lower() for title in titles)


def test_a_week_with_no_priced_games_is_refused(tmp_path):
    table = _prediction_table()
    table["home_cover_probability"] = np.nan
    matchups = _matchups(table)

    with pytest.raises(ReportError, match="No priced games"):
        weekly_report(
            table,
            matchups,
            prediction_contributions(matchups[FEATURES], _pipeline()),
            tmp_path,
            2026,
            3,
        )


# --- the two reports are visually distinct ---------------------------------


def test_the_two_reports_use_different_accents():
    assert HISTORICAL_ACCENT != UPCOMING_ACCENT


def test_the_accent_colours_are_from_the_validated_palette():
    from gridiron.eda.theme import CATEGORICAL

    assert HISTORICAL_ACCENT in CATEGORICAL
    assert UPCOMING_ACCENT in CATEGORICAL


def test_seaborn_is_used_for_the_required_chart_types():
    """The brief names five seaborn functions; all five appear."""
    import inspect

    source = inspect.getsource(reports)
    for function in (
        "sns.heatmap",
        "sns.barplot",
        "sns.lineplot",
        "sns.histplot",
        "sns.scatterplot",
    ):
        assert function in source, function
