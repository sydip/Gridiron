# Gridiron API

A read-only HTTP layer over what the pipeline has already produced. It computes
no statistics: every number it returns was read from a file in `outputs/` or
`models/`, and the only work done here is selecting columns, joining a week's
predictions to that week's results, and shaping the result into JSON.

Nothing outside this directory was changed to add it.

## Run

```powershell
pip install -r api/requirements.txt
uvicorn api.main:app --reload --port 8000
```

Interactive docs at <http://localhost:8000/docs>. Start with
<http://localhost:8000/api/health>, which lists every artifact the service can
see and names the command that produces any that are missing.

The API does not import `gridiron`, so it can run in its own environment with
just the four packages in `requirements.txt`. The one exception is
`POST /api/refresh/{week}`, which needs an interpreter that *can* import
`gridiron`; it looks for `$GRIDIRON_PYTHON`, then the project's `.venv`, then
falls back to its own interpreter.

## Endpoints

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/api/weeks` | Every 2026 week with its status and counts |
| GET | `/api/weeks/{week}` | One week's matchups, predictions and results |
| GET | `/api/metrics/summary` | Headline metrics, by season, baselines, policies |
| GET | `/api/metrics/calibration` | Reliability buckets, calibrator trial, confidence buckets |
| GET | `/api/metrics/coefficients` | Coefficients, odds ratios, VIF, cross-fold stability |
| GET | `/api/metrics/correlation` | The correlation matrix behind the heatmap |
| GET | `/api/figures` | The rendered PNGs on disk |
| GET | `/api/health` | Which artifacts exist, and the remedy for any that do not |
| POST | `/api/refresh/{week}` | Runs the existing CLI for that week |

## Where every field comes from

Several names in the original brief do not exist in this project. The real
files are these, and `api/sources.py` is the only place they are named.

| Logical source | File on disk | Produced by |
| --- | --- | --- |
| `matchups` | `outputs/reports/matchups_wide.csv` | `cli build-features` |
| `predictions` | `outputs/predictions/week_XX_predictions.csv` | `cli predict` |
| `walk_forward_aggregate` | `outputs/reports/walk_forward_aggregate.csv` | `cli backtest` |
| `walk_forward_by_season` | `outputs/reports/walk_forward_by_season.csv` | `cli backtest` |
| `walk_forward_vs_baselines` | `outputs/reports/walk_forward_vs_baselines.csv` | `cli backtest` |
| `policy_backtest` | `outputs/reports/policy_backtest.csv` | `cli backtest` |
| `policy_win_rate_by_confidence` | `outputs/reports/policy_win_rate_by_confidence.csv` | `cli backtest` |
| `calibration_reliability` | `outputs/reports/calibration_reliability.csv` | `scripts/calibration_report.py` |
| `calibration_comparison` | `outputs/reports/calibration_comparison.csv` | `scripts/calibration_report.py` |
| `coefficients` | `outputs/reports/model_coefficients.csv` | `cli train` |
| `odds_ratios` | `outputs/reports/model_odds_ratios.csv` | `scripts/interpret_model.py` |
| `collinearity` | `outputs/reports/model_collinearity.csv` | `scripts/interpret_model.py` |
| `coefficient_stability` | `outputs/reports/model_coefficient_stability.csv` | `scripts/interpret_model.py` |
| `limitations` | `outputs/reports/model_limitations.json` | `scripts/interpret_model.py` |
| `correlation_model_fields` | `outputs/reports/eda_correlation_model_fields.csv` | `scripts/eda_report.py` |
| `correlation_decisions` | `outputs/reports/eda_correlation_decisions.csv` | `scripts/eda_report.py` |
| `model_metadata` | `models/model_metadata.json` | `cli deploy` |

There is no `data/processed/2026_weekly_predictions.parquet`, no
`outputs/reports/model_metrics.json` and no `season_metrics.csv`. The first two
columns of this table are what exists.

Every response carries a `sources` block naming the files it was built from and
when each was last written, so a figure on a page can be traced to a file
without reading this code.

## Two things worth knowing

**Floats are served at full precision.** `DataFrame.to_json` defaults to
`double_precision=10` and silently rounds — a probability of
`0.5002683352080302` comes out as `0.5002683352`. Rounding a served value is
exactly what this layer must not do, so records are built through
`numpy.generic.item()` and Python's own JSON encoder instead.

**Completed weeks have no predictions and never will.** `predict_week` refuses
games that have already been played, because a finished game has a result
rather than a recommendation. 2026 weeks 1 and 2 finished before the model was
pointed at them, so they are served with their results and `prediction: null`.
`POST /api/refresh/1` will return `returncode: 1` and pass the command's own
refusal back to you. A frontend should say "no prediction was recorded" for
those weeks rather than implying the model called them.

## The one derived field

`pick_correct` is the only value in the service that is not read directly from
a file. It compares two fields that are returned beside it — `predicted_side`
from the prediction CSV and `home_cover` from the matchup table — and is null
when either is absent, including on a push, where `home_cover` is itself null
because no side covered.
