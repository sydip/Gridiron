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

## Layout

```text
src/gridiron/       Reusable project code
tests/              Automated tests
data/raw/           Immutable source data (ignored by Git)
data/interim/       Intermediate data (ignored by Git)
data/processed/     Analysis-ready data (ignored by Git)
models/             Serialized model artifacts (ignored by Git)
reports/figures/    Generated figures (ignored by Git)
```
