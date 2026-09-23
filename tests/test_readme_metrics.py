"""Every metric the README reports must match the artifact it came from.

A results section is a claim about what the pipeline produced. Left unchecked
it decays: a rerun shifts a number, the prose does not follow, and the document
slowly becomes fiction. These tests re-derive each headline figure from the
CSV the pipeline wrote and assert the README states that figure.

They skip when the artifacts are absent, because a fresh clone has not run the
pipeline yet. Run ``python -m gridiron.cli all`` and they become live.
"""

from __future__ import annotations

import pandas as pd
import pytest

from gridiron.config import PROJECT_ROOT
from gridiron.modeling.metrics import BREAK_EVEN_ACCURACY

README_PATH = PROJECT_ROOT / "README.md"
REPORT_DIR = PROJECT_ROOT / "outputs" / "reports"
IMAGE_DIR = PROJECT_ROOT / "docs" / "images"


def _normalised(text: str) -> str:
    """Fold typographic dashes to ASCII, so assertions can be written plainly."""
    for dash in ("−", "–", "—"):
        text = text.replace(dash, "-")
    return text


@pytest.fixture(scope="module")
def readme() -> str:
    return _normalised(README_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def results(readme) -> str:
    """Just the Results section.

    Scoped deliberately: these figures also appear in the phase write-ups
    further down, so searching the whole document would let a wrong number in
    the results table pass because the right one exists elsewhere.
    """
    start = readme.index("\n## Results")
    return readme[start : readme.index("\n## ", start + 1)]


def _table(name: str) -> pd.DataFrame:
    path = REPORT_DIR / name
    if not path.exists():
        pytest.skip(f"{name} is not available; run the pipeline first")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def aggregate() -> dict[str, float]:
    frame = _table("walk_forward_aggregate.csv")
    return frame.set_index("metric")["value"].to_dict()


@pytest.fixture(scope="module")
def seasons() -> pd.DataFrame:
    return _table("walk_forward_by_season.csv")


@pytest.fixture(scope="module")
def policies() -> pd.DataFrame:
    return _table("policy_backtest.csv")


@pytest.fixture(scope="module")
def baselines() -> pd.DataFrame:
    return _table("walk_forward_vs_baselines.csv")


# --- headline out-of-sample metrics ----------------------------------------


def test_the_out_of_sample_game_count_is_stated(results, seasons):
    assert f"{int(seasons['games'].sum()):,} games" in results


def test_the_ats_record_is_stated(results, seasons):
    record = f"{int(seasons['wins'].sum())}-{int(seasons['losses'].sum())}"

    assert record in results


def test_the_accuracy_is_stated(results, aggregate):
    assert f"{aggregate['pooled_accuracy']:.2%}" in results


def test_the_mean_season_accuracy_is_stated(results, aggregate):
    assert f"{aggregate['mean_season_accuracy']:.2%}" in results


def test_the_best_validation_season_is_stated_with_its_year(results, seasons):
    best = seasons.loc[seasons["accuracy"].idxmax()]

    assert f"{best['accuracy']:.2%}" in results
    assert f"({int(best['season'])})" in results


def test_the_worst_validation_season_is_stated_with_its_year(results, seasons):
    worst = seasons.loc[seasons["accuracy"].idxmin()]

    assert f"{worst['accuracy']:.2%}" in results
    assert f"({int(worst['season'])})" in results


def test_the_roc_auc_is_stated(results, aggregate):
    assert f"{aggregate['roc_auc']:.4f}" in results


def test_the_log_loss_is_stated(results, aggregate):
    assert f"{aggregate['log_loss']:.5f}" in results


def test_the_brier_score_is_stated(results, aggregate):
    assert f"{aggregate['brier_score']:.5f}" in results


def test_the_roi_is_stated(results, aggregate):
    assert f"{aggregate['roi']:.2%}" in results


def test_the_break_even_rate_is_stated(results):
    assert f"{BREAK_EVEN_ACCURACY:.2%}" in results


def test_the_season_spread_is_stated(results, aggregate):
    assert f"{aggregate['accuracy_std'] * 100:.2f}pp" in results


def test_precision_recall_and_f1_are_stated(results, aggregate):
    for name in ("precision", "recall", "f1"):
        assert f"{aggregate[name]:.4f}" in results


def test_balanced_accuracy_is_stated(results, aggregate):
    assert f"{aggregate['balanced_accuracy']:.2%}" in results


# --- the reference values the metrics are judged against -------------------


def test_the_uninformative_references_are_stated(results):
    """0.5 every time scores these; the model must be compared with them."""
    from gridiron.modeling.evaluate import UNINFORMATIVE_BRIER, UNINFORMATIVE_LOG_LOSS

    assert f"{UNINFORMATIVE_LOG_LOSS:.5f}" in results
    assert f"{UNINFORMATIVE_BRIER:.5f}" in results


# --- betting policies ------------------------------------------------------


def test_every_policy_row_is_reported(results, policies):
    for row in policies.itertuples():
        assert f"{int(row.bets):,}" in results, f"{row.policy}: bet count missing"
        assert f"{row.win_rate:.2%}" in results, f"{row.policy}: win rate missing"
        assert f"{row.roi:.2%}" in results, f"{row.policy}: ROI missing"


def test_no_policy_is_described_as_profitable_on_a_credible_sample(policies):
    """The README's central claim, checked against the data rather than trusted."""
    from gridiron.modeling.policies import MIN_BETS_FOR_A_CLAIM

    credible = policies.loc[policies["bets"] >= MIN_BETS_FOR_A_CLAIM]

    assert not credible.empty
    assert (credible["roi"] <= 0).all()


def test_the_confidence_reversal_is_stated(results):
    buckets = _table("policy_win_rate_by_confidence.csv")
    lowest = buckets.iloc[0]

    assert f"{lowest['win_rate']:.1%}" in results


# --- baselines -------------------------------------------------------------


def test_every_baseline_is_reported_with_its_accuracy_and_roi(results, baselines):
    for row in baselines.itertuples():
        assert f"{row.accuracy:.2%}" in results, f"{row.strategy}: accuracy missing"
        assert f"{row.roi:.2%}" in results, f"{row.strategy}: ROI missing"


def test_the_models_rank_among_the_baselines_is_stated_correctly(readme, baselines):
    """The README says fourth of seven. That must follow from the table."""
    ordered = baselines.sort_values("roi", ascending=False).reset_index(drop=True)
    position = int(ordered.index[ordered["strategy"].str.startswith("MODEL")][0] + 1)

    assert position == 4
    assert len(ordered) == 7
    assert "fourth of seven" in readme


# --- dataset counts --------------------------------------------------------


def test_the_dataset_counts_are_stated(results):
    table = _table("model_table.csv")
    historical = int((table["season"] <= 2025).sum())
    upcoming = int((table["season"] == 2026).sum())

    assert f"{historical:,}" in results
    assert f"{upcoming:,}" in results


def test_the_training_set_counts_match_the_deployment_metadata(results):
    import json

    path = PROJECT_ROOT / "models" / "model_metadata.json"
    if not path.exists():
        pytest.skip("no deployment metadata; run 'python -m gridiron.cli deploy'")

    metadata = json.loads(path.read_text(encoding="utf-8"))
    exclusions = metadata["exclusions"]

    assert f"{metadata['n_training_games']:,}" in results
    assert f"{exclusions['excluded_pushes']:,}" in results
    assert f"{exclusions['rows_with_imputed_features']:,}" in results


def test_the_feature_count_is_stated(results):
    from gridiron.modeling.pipeline import get_feature_columns

    assert f"| Features | {len(get_feature_columns())} |" in results


# --- the non-negotiable disclosures ----------------------------------------


def test_the_readme_states_the_model_has_no_edge(readme):
    assert "no demonstrated edge" in readme or "no edge" in readme


def test_backtested_results_are_not_presented_as_a_forecast(readme):
    assert "Backtested results are not a forecast" in readme
    assert "not betting advice" in readme


def test_the_limitations_section_exists_and_is_substantive(readme):
    start = readme.index("## Limitations")
    section = readme[start : readme.index("\n## ", start + 1)]

    assert section.count("\n- ") >= 5


def test_the_lines_are_not_described_as_live_quotes(readme):
    assert "not live quotes" in readme or "not a live sportsbook quote" in readme


def test_installation_and_execution_instructions_are_present(readme):
    assert "pip install -r requirements.txt" in readme
    assert "python -m gridiron.cli all" in readme
    assert "python -m gridiron.cli predict" in readme


# --- screenshots -----------------------------------------------------------


def test_every_referenced_image_exists(readme):
    import re

    referenced = re.findall(r"!\[[^\]]*\]\(([^)]+)\)", readme)

    assert referenced, "the README references no images"
    for relative in referenced:
        assert (PROJECT_ROOT / relative).exists(), f"{relative} is missing"


def test_the_screenshots_are_real_rendered_figures():
    """PNG magic bytes, so a placeholder cannot pass as a screenshot."""
    images = sorted(IMAGE_DIR.glob("*.png"))

    assert images
    for path in images:
        assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
        assert path.stat().st_size > 10_000
