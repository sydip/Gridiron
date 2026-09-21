# Gridiron

A reproducible Python project for NFL data analysis and modeling.

## Setup

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest
```

For an exact replica of the environment verified for this repository, install
`requirements.lock` instead of `requirements.txt`.

### macOS/Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
pytest
```

Copy `.env.example` to `.env` if your shell or IDE supports dotenv files. The
project also supplies safe defaults for nflreadpy's filesystem cache. Existing
environment variables always take precedence.

```python
from gridiron.config import DEFAULT_RANDOM_SEED, configure_environment, seed_everything

configure_environment()
seed_everything(DEFAULT_RANDOM_SEED)
```

Pass `random_state=DEFAULT_RANDOM_SEED` to scikit-learn estimators and splitters
that expose a `random_state` parameter.

## Download NFL data

Download the default 2016–2025 historical datasets and the 2016–2026 schedule:

```powershell
python scripts/download_data.py
```

Subsequent runs load the ignored Parquet files from `data/raw` without calling
nflreadpy. Use `--force-refresh` to replace them from the upstream source, or
pass `--seasons` and `--prediction-season` to select different years.

Create and validate the canonical regular-season schedule:

```python
import pandas as pd

from gridiron.data import clean_schedule, validate_schedule

raw_schedule = pd.read_parquet("data/raw/schedules_2016_2026.parquet")
schedule = clean_schedule(raw_schedule)
report = validate_schedule(schedule, historical_seasons=list(range(2016, 2026)))
```

Unplayed games remain in the table. `is_prediction_ready` is true only when an
unplayed game has an available spread line.

Build nullable against-the-spread labels with:

```python
from gridiron.features import add_ats_target

labeled_schedule = add_ats_target(schedule)
```

`home_cover` is `1` for a home cover, `0` for an away cover, and null for both
pushes and unresolved games.

Aggregate play-by-play into one offensive-efficiency row per team per game:

```python
from gridiron.features import aggregate_offensive_epa, low_volume_team_games

pbp = pd.read_parquet("data/raw/pbp_2016_2025.parquet")
offense = aggregate_offensive_epa(pbp)
review = low_volume_team_games(offense)
```

Only `pass` and `run` plays with an EPA value and a known `posteam`/`defteam`
count, so administrative rows, special teams, kneel-downs and spikes are
excluded. EPA is reported as a game total (`offensive_epa`) and per play
(`offensive_epa_per_play`), alongside `offensive_plays`,
`offensive_success_rate`, and the pass/rush split.

`valid_game` is false when a team-game has fewer than 30 eligible plays. Those
rows are flagged for review, never dropped.

The defensive mirror groups the same eligible plays by `defteam`, so a defense's
totals are what it allowed:

```python
from gridiron.features import (
    aggregate_defensive_epa,
    epa_reconciliation_failures,
    reconcile_offense_and_defense,
)

defense = aggregate_defensive_epa(pbp)
report = reconcile_offense_and_defense(offense, defense)
failures = epa_reconciliation_failures(report)
```

`defensive_epa_strength` negates `defensive_epa_allowed` so it points the same
way as the offensive measure: higher is a stronger unit.

Because both aggregations start from the same eligible play set, a game's home
offense EPA *is* its away defense EPA allowed. `reconcile_offense_and_defense`
pairs every offense with the defense that faced it and compares play counts,
EPA, and success rate; anything beyond tolerance means the two sides were built
from different play sets, and `epa_reconciliation_failures` surfaces it. Neither
aggregation reads the score, spread, or result of the game being described.

## Pace

Pace describes how many snaps an offense ran and how quickly it ran them. One
row per team per game:

```python
from gridiron.features import aggregate_pace, flag_pace_outliers, pace_distribution_report

pace = aggregate_pace(pbp)
distribution = pace_distribution_report(pace)
flagged = flag_pace_outliers(pace)
```

`scripts/pace_report.py` runs the same three steps over the raw play-by-play and
writes `pace_team_game.csv`, `pace_distribution.csv`, `pace_outliers.csv`, and
`pace_coverage.csv` to `outputs/reports/`.

### Definition

The baseline statistic is a volume count:

```text
pace_plays_per_game = number of eligible offensive plays the team ran
```

Eligibility is not redefined here. `aggregate_pace` calls the same
`filter_offensive_plays` the EPA aggregation uses, so pace counts pass and run
scrimmage plays only -- never kickoffs, punts, field goals, extra points,
penalties that wiped out a snap, kneel-downs, or spikes -- and a team-game's
`pace_plays_per_game` is guaranteed to equal its `offensive_plays` from the EPA
table. `offensive_play_count` is carried alongside as the explicit raw count.

### Timing statistics

These are optional and gated on the clock fields being usable, which
`timing_fields_available` checks. When they are not, the columns are omitted
rather than filled with a placeholder.

| Column | Meaning |
| --- | --- |
| `seconds_per_play` | Mean game-clock seconds between consecutive snaps of the same drive |
| `pace_intervals` | How many snap intervals that mean rests on |
| `neutral_seconds_per_play` | The same mean, restricted to situation-neutral snaps |
| `neutral_pace_intervals` | How many neutral intervals that mean rests on |
| `no_huddle_rate` | Share of eligible plays run without a huddle |

An interval is the game clock difference between consecutive snaps within one
drive. The first snap of a drive has no predecessor. Intervals that are
non-positive (source ordering artefacts) or longer than 60 seconds are dropped:
the play clock is 40 seconds, so a longer gap means the clock stopped for a
timeout, injury, review, two-minute warning, or quarter change, and would
measure the stoppage instead of the tempo.

A snap counts as situation-neutral when the score margin is within 8 points and
more than 120 seconds remain in the half -- the states where the scoreboard is
not itself dictating tempo.

### Validity flags and missingness

Nothing is dropped and nothing is fabricated. Rows that fail a quality check are
flagged and kept so they can be imputed downstream:

- `valid_pace` is false below 30 eligible plays.
- `valid_neutral_pace` is false below 10 neutral intervals.
- A team-game with no measurable interval keeps a null `seconds_per_play`.

`flag_pace_outliers` returns the rows worth inspecting with a `flag_reasons`
column. Its bounds (30-100 plays, 18-45 seconds per play) come from what
football allows, not from the observed range, so they remain a real test.

### What the 2016-2025 data shows

Both teams have pace in **100% of the 2,639 completed games** in every season.

| Statistic | Mean | Median | SD | Min | p01 | p99 | Max |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `pace_plays_per_game` | 62.44 | 62.00 | 8.47 | 30 | 43 | 84 | 95 |
| `seconds_per_play` | 29.83 | 29.89 | 3.16 | 19.00 | 21.92 | 36.81 | 39.26 |
| `neutral_seconds_per_play` | 32.58 | 32.73 | 3.49 | 12.00 | 22.19 | 40.15 | 49.00 |
| `no_huddle_rate` | 0.096 | 0.061 | 0.114 | 0.00 | 0.00 | 0.60 | 0.81 |

No team-game has an implausible play count or seconds-per-play value. The
extremes were checked individually and are real football:

- **30 plays** (LV at KC, 2025 week 7) -- a 31-0 loss with six punts, not lost data.
- **95 plays** (SF vs ARI, 2018 week 5) -- SF trailed and threw 60 times.

The only flagged category is thin neutral samples: **456 of 5,522 team-games
(8.3%)** have fewer than 10 neutral intervals, and dispersion roughly doubles
below that threshold (SD 6.36 versus 3.09). Four team-games have no neutral
interval at all and keep a null. For this reason `neutral_seconds_per_play` is
**not** recommended as a required model input; `pace_plays_per_game` is the
statistic the baseline model should use.

## Layout

```text
src/gridiron/       Reusable project code
tests/              Automated tests
data/raw/           Immutable source data (ignored by Git)
data/interim/       Intermediate data (ignored by Git)
data/processed/     Analysis-ready data (ignored by Git)
models/             Serialized model artifacts (ignored by Git)
reports/figures/    Generated figures (ignored by Git)
outputs/reports/    Generated CSV reports (ignored by Git, except fixtures)
```
