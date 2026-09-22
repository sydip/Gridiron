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

## Team-game history and rest

A schedule stores one row per game. Most team-level questions want one row per
team per game, in time order:

```python
from gridiron.features import team_game_history, team_game_reconciliation

history = team_game_history(schedule)
report = team_game_reconciliation(history)
```

`scripts/team_game_report.py` writes `team_game_history.csv`,
`team_game_rest_distribution.csv`, `team_game_rest_by_weekday.csv`, and
`team_game_reconciliation_failures.csv` to `outputs/reports/`.

### Shape

Each game becomes a home row and an away row carrying `team`, `opponent`,
`is_home`, `points_for`, `points_against`, `point_margin`, `team_spread`,
`covered`, `gameday`, `season`, and `week`. Unplayed games are kept with null
scores and marked by `is_completed`, so one history serves training and
prediction alike.

`team_spread` keeps the nflverse sign convention: **a positive number means this
team is favoured**. The home row takes `spread_line` and the away row its
negation, so a game's two spreads always sum to zero. Note this is the opposite
of betting-sheet notation, where a favourite is quoted negative.

`covered` is nullable: 1 when the team beat its own spread, 0 when it did not,
and null for a push or a game with no score or line. A push is a real outcome,
not a loss against the spread, so it is never recorded as 0.

### Rest

Rest is measured within a team and season from that team's own previous game.
Rows are sorted by date before the shift, so a rest value can never be derived
from a future date.

| Column | Meaning |
| --- | --- |
| `previous_game_date` | That team's previous game in the same season |
| `rest_days` | Days since it; 7 for a season opener |
| `short_week` | Fewer than 7 days |
| `extra_rest` | More than 7 days |
| `bye_week_rest` | The team's week counter advanced by 2 or more |
| `week_1_flag` | The team's first game of the season |
| `rest_advantage_placeholder` | Reserved, always null (see below) |

**Season openers.** Rest is not carried across a season boundary: the gap to the
previous season's final game is an offseason, not rest, and varies by months
between teams. Openers take 7 days and are marked by `week_1_flag`.

**`week_1_flag` means "first game of the season", not `week == 1`.** These
usually coincide, but not always: Miami and Tampa Bay opened 2017 in week 2
after Hurricane Irma cancelled their week 1 game. Defining the flag by first
appearance keeps those two openers from falling through with undefined rest.

**Byes are detected from the week counter, not from elapsed days.** A bye is a
scheduled week with no game, so `bye_week_rest` triggers on a week gap of 2 or
more. Days alone cannot separate a true bye from the 10-to-12 day gap a team
gets after a Thursday game, and in this data a one-week gap never exceeds 12
days while 349 of 352 two-week gaps reach 13 or more.

**`rest_advantage_placeholder` is deliberately empty.** Rest advantage compares
a team against its opponent, which a later phase will fill. The column is
reserved now so downstream schemas stay stable; an empty column is honest where
an invented one would not be.

### What the 2016-2026 data shows

5,822 team-game rows from 2,911 games. All 2,911 reconcile: exactly two rows per
game, margins cancelling, spreads cancelling. Every one of the 2,639 completed
games produces exactly two rows.

Mean rest is 7.43 days (median 7, range 4 to 17). Rest by the weekday played:

| Weekday | Games | Mean rest | Short week | Extra rest |
| --- | --- | --- | --- | --- |
| Sunday | 4,542 | 7.64 | 7.9% | 16.6% |
| Monday | 376 | 8.87 | 0.8% | 98.7% |
| Thursday | 374 | 4.17 | 94.7% | 0.3% |
| Saturday | 146 | 6.49 | 80.8% | 11.6% |
| Friday | 16 | 5.19 | 100% | 0% |

Thursday games are short weeks 94.7% of the time. The 20 that are not are the
Thanksgiving pattern: 19 of them followed a game the *previous* Thursday, which
is an ordinary 7-day turnaround. Bye-week games show extra rest 100% of the
time, at a median of 14 days.

There is exactly one opener per team per season (352), and exactly one bye per
team-season for all but four, each a real event the data represents correctly:

- **BUF and CIN, 2022** carry two gameless weeks. Their week 17 meeting was
  cancelled after Damar Hamlin's cardiac arrest and never replayed, so both
  played 16 games.
- **MIA and TB, 2017** carry none. Losing week 1 to Hurricane Irma left them
  playing weeks 2 through 17 consecutively, with no in-season bye.

## Rolling team form

Every feature here describes what a team had done **before** the game it is
attached to:

```python
from gridiron.features import build_rolling_features

features = build_rolling_features(history, offense, defense, pace)
```

`scripts/rolling_report.py` writes `rolling_features.csv` and
`rolling_feature_coverage.csv` to `outputs/reports/`.

### Order of operations

This order is the whole guarantee, and it is not interchangeable:

1. sort each team's games chronologically,
2. **shift by one game**, excluding the current game,
3. take the rolling or expanding mean of what remains,
4. attach the result to the upcoming game.

```python
grouped.transform(lambda s: s.shift(1).rolling(5, min_periods=1).mean())  # correct
grouped.transform(lambda s: s.rolling(5).mean())                          # leaks
```

Without the shift, a game's own outcome sits inside the window that is supposed
to predict it.

### Features, windows, and minimum periods

All windows use **`min_periods=1`**: one prior game is enough to produce a
value. `last_3` and `last_5` are fixed-length trailing windows; `_season` is an
expanding window over every prior game in that season.

| Feature | Source | Window |
| --- | --- | --- |
| `off_epa_last_3` / `_last_5` / `_season` | `offensive_epa_per_play` | 3 / 5 / season |
| `def_epa_strength_last_3` / `_last_5` / `_season` | `defensive_epa_strength_per_play` | 3 / 5 / season |
| `pace_last_3` / `_last_5` / `_season` | `pace_plays_per_game` | 3 / 5 / season |
| `point_margin_last_3` / `_last_5` | `point_margin` | 3 / 5 |
| `win_pct_last_5` | `win_value` (tie = 0.5) | 5 |
| `ats_margin_last_5` | `point_margin - team_spread` | 5 |
| `cover_rate_last_5` | `covered` (pushes null) | 5 |

EPA features use the **per-play** rates rather than game totals, so they measure
efficiency rather than volume; pace already carries the volume signal.

### Season resets and the early season

`season` is part of the grouping key, so **both** the fixed windows and the
season-to-date values restart in week 1. Last season's closing form cannot reach
this season, in either direction. No prior-season prior is implemented here; if
one is wanted it should be a separate, separately documented feature.

Week 1 has no prior game, so all fourteen features are null by construction.
They are left that way for an imputer rather than being filled at this stage, so
the absence stays visible. Three context columns describe how much history backs
each row:

| Column | Meaning |
| --- | --- |
| `week_number` | The week the game is in |
| `is_early_season` | Week 4 or earlier |
| `prior_games_this_season` | Completed prior games behind the value |

### Leakage tests

`tests/test_leakage.py` runs the contract directly: build features from all
data, remove the target game's own values and every later game, rebuild, and
confirm the target's features are unchanged. The suite also tampers with future
results, tampers with the current game, and rebuilds a single season in
isolation.

One test matters more than the rest. `test_the_leak_detector_actually_detects_a_leak`
runs the same comparison against a deliberately unshifted window and asserts it
**fails**. Without it, every other leakage test could be passing vacuously.

### A join fault this phase had to fix

The play-by-play and the cleaned schedule disagree about franchise codes:
nflverse play-by-play calls the Rams `LA` in all ten seasons while the cleaned
schedule normalises to `LAR`. Joining without normalising silently dropped EPA
and pace for **every Rams team-game** (181 rows). `assemble_team_game_inputs`
normalises both sides before the join, and the join is validated `one_to_one`.

### Coverage over 2016-2026

5,822 rows, 14 features, and after the normalisation fix **no completed game is
missing its EPA or pace inputs**. 864 rows have null rolling features, all
explained:

- **350** week 1 openers,
- **2** the 2017 Miami and Tampa Bay openers, which fall in week 2,
- **512** rows in the unplayed 2026 season, which has no current-season history
  behind it yet.

`cover_rate_last_5` has 6 more, from teams whose only prior games were pushes.

## Matchup records and the model table

Phase 9 gives one row per team per game. A model needs one row per *game*:

```python
from gridiron.features import build_model_table

rows = build_model_table(features, schedule)
```

`scripts/matchup_report.py` writes `model_table.csv`, `matchups_wide.csv`, and
four missing-value breakdowns to `outputs/reports/`.

### The merge

The schedule is the spine. The home side joins on
`(season, week, game_id, home_team)` and the away side on
`(season, week, game_id, away_team)`, both `validate="one_to_one"` so a
duplicated team-game raises instead of silently fanning the join out. Matching
on the team code means a row can only be built when the side being attached is
the side the schedule says it is — a transposition cannot pass silently.

Each side's columns are prefixed (`home_off_epa_last_5`, `away_off_epa_last_5`),
and **every differential is home minus away**, without exception.

### The model row

| Group | Columns |
| --- | --- |
| Identifiers | `game_id`, `season`, `week`, `gameday`, `home_team`, `away_team` |
| Features | `spread_line`, `off_epa_diff_last_5`, `def_epa_strength_diff_last_5`, `pace_diff_last_5`, `rest_diff`, `point_margin_diff_last_5`, `win_pct_diff_last_5`, `home_short_week`, `away_short_week`, `div_game`, `week_1_flag` |
| Targets | `home_cover`, `is_push` |

`div_game` is derived from the two teams' divisions rather than read from the
source column, because `clean_schedule` does not carry that column through. The
derivation was checked against the source for all 2,911 regular-season games and
agrees on every one, at 96 division games per season.

`week_1_flag` is set when **either** team is playing its season opener. That is
the condition under which a matchup has a side with no current-season history,
which is what the flag exists to warn about.

### No postgame field in the feature matrix

`assert_no_postgame_fields` enforces the separation rather than trusting it.
Scores, margins, realised EPA, and the targets themselves are all named in
`POSTGAME_COLUMNS` and rejected if offered as features.

Matching is by **exact name, never by substring**: `point_margin_diff_last_5` is
a pregame rolling feature and must not be caught by a rule aimed at
`point_margin`. A test asserts the guard catches an injected `home_score`, so it
cannot pass vacuously.

### Missing-value report

`missing_value_report` returns four frames — by feature, season, week, and team.
Nothing is dropped; the report exists so early-season sparsity is visible to
whoever decides what to do about it.

Over 2016-2026 (2,911 games), 433 rows are missing rolling differentials, and
every one is accounted for:

- **175** week 1 games, which have no prior game by construction,
- **256** non-week-1 games in 2026, which has no played games in this snapshot,
- **2** the 2017 Miami and Tampa Bay week 2 openers.

Excluding 2026, week 2 is 1.25% missing and **week 3 onward is 0%**. The 211
missing `spread_line` values are all 2026 fixtures whose lines are not yet
posted.

Of 2,639 completed games, 2,574 carry a label. The 65 that do not are all
pushes — no completed game with a line is unlabelled for any other reason.

### A note on the 2026 season

2026 rows exist with their schedule-derived features populated (`rest_diff`,
`div_game`, `week_1_flag`, both `short_week` flags at 100%) and no target, as
intended. Their **rolling** differentials are null, because the play-by-play in
this snapshot ends with 2025 and no 2026 game has been played into it. Once
2026 play-by-play is downloaded, those fill in week by week with no code change.

## Exploratory analysis

```python
python scripts/eda_report.py
```

Ten charts are written to `outputs/figures/`; the correlation matrices, the
flagged pairs, and the decision record go to `outputs/reports/`. Both are
generated artefacts and are gitignored — re-run the script to rebuild them.

| # | Chart | Colour role |
| --- | --- | --- |
| 01 | Correlation heatmap, numeric model fields | Diverging |
| 02 | Feature distributions | Single hue |
| 03 | Offensive EPA by team and season | Diverging |
| 04 | Defensive EPA strength by team and season | Diverging |
| 05 | Pace distribution | Single hue |
| 06 | Rest advantage vs cover rate | Single hue |
| 07 | Spread bins vs home cover rate | Single hue |
| 08 | Rolling EPA differential vs cover rate | Single hue |
| 09 | Missing-value heatmap | Sequential |
| 10 | ATS class balance by season | Categorical |

Colour is assigned by the job it does. Anything with a sign (a correlation, an
EPA either side of zero) gets a **diverging** scale with a neutral grey
midpoint, so "no relationship" reads as nothing rather than as a colour.
Missingness is a magnitude with no sign, so it gets a **sequential** one-hue
ramp. Only chart 10 encodes identity, so only it uses categorical slots. The
categorical set was checked with a colour-vision validator and clears the
all-pairs separation floors.

Every chart carries a title and labelled axes, and a test asserts it rather
than leaving it to inspection.

### Correlation policy

Pairs at `|r| >= 0.80` are flagged. **No feature was removed**, and every
flagged pair carries a written reason in
`outputs/reports/eda_correlation_decisions.csv`.

**No pair among the nine numeric model fields reaches the threshold.** The
strongest is `point_margin_diff_last_5 ~ win_pct_diff_last_5` at **r = +0.795**,
just under it — close enough to watch if a model shows unstable coefficients on
those two, but not a breach.

All nine flagged pairs are in the wider candidate set of every differential,
and they fall into the shapes the policy anticipated:

| Pair | r | Decision |
| --- | --- | --- |
| `off_epa_diff_last_5` ~ `off_epa_diff_season` | +0.923 | keep both |
| `off_epa_diff_last_5` ~ `off_epa_diff_last_3` | +0.899 | keep both |
| `pace_diff_last_5` ~ `pace_diff_season` | +0.897 | keep both |
| `point_margin_diff_last_5` ~ `point_margin_diff_last_3` | +0.890 | keep both |
| `def_epa_strength_diff_last_5` ~ `def_epa_strength_diff_season` | +0.889 | keep both |
| `def_epa_strength_diff_last_5` ~ `def_epa_strength_diff_last_3` | +0.880 | keep both |
| `pace_diff_last_5` ~ `pace_diff_last_3` | +0.879 | keep both |
| `point_margin_diff_last_5` ~ `ats_margin_diff_last_5` | +0.868 | keep both |
| `off_epa_diff_last_3` ~ `off_epa_diff_season` | +0.833 | keep both |

Seven of the nine are two windows over the same metric. They are redundant by
design, and **only one window per metric reaches the model row**, so no model
ever sees the pair together. They stay in the wider table because a later phase
may prefer the shorter window's recency.

`point_margin` and `ats_margin` move together because one is the other shifted
by the line; only `point_margin_diff_last_5` is a model field.

One redundancy is recorded that **no correlation could have surfaced**:
`defensive_epa_allowed_per_play` and `defensive_epa_strength_per_play` are exact
negatives (`r = -1.00`), but the two never coexist in a frame, so no measured
pair can flag them. The rolling layer consumes only the strength view, so the
redundancy is designed out rather than filtered out. It is recorded explicitly
so the decision is not invisible.

### What the charts show

The features have **weak univariate signal against the spread**, which is the
expected result for a market that already prices team quality. The base home
cover rate is 49.1%; cover rate by rolling EPA differential quintile stays
within 47.8%–50.4%, and no spread bin departs far from break-even. Season ATS
balance sits between 43.1% (2019) and 52.4% (2017).

This is worth stating plainly before modelling: nothing here promises a
profitable edge, and a model that reports one on these features should be
treated as suspect until it survives the leakage checks and a held-out season.

The correlation heatmap also shows `week_1_flag` with **blank, not zero**, cells
against every rolling feature. The flag is constant wherever a rolling feature
is present, so the pair has no overlapping observations — the flag is perfectly
confounded with the missingness it marks.

## Baselines

A model is only worth something if it beats the strategies that cost nothing to
invent. These are those strategies:

```python
python scripts/baseline_report.py
python scripts/baseline_report.py --require-epa   # complete cases only
```

| Function | Strategy |
| --- | --- |
| `predict_home_every_game` | Always back the home side |
| `predict_away_every_game` | Always back the away side |
| `predict_favorite` | Always back the favourite |
| `predict_underdog` | Always back the underdog |
| `predict_better_epa` | Back the better rolling offensive EPA |
| `predict_random` | Pick a side at random, reproducibly |

### One eligible set, shared by everything

`eligible_games` is the single definition, and the model must use it too. A game
is eligible when `home_cover` is non-null — which is null for a push, for an
unplayed game, and for a game with no line, so **one condition excludes all
three identically** for every strategy. A test asserts every baseline returns a
prediction for every eligible game, because a strategy that quietly declined the
games it found hard would be scored on a different sample.

Two rules are stated rather than left to chance:

- **Pick'em games have no favourite.** The 4 such games resolve to the home
  side in `predict_favorite`, and inversely in `predict_underdog`.
- **Season openers have no EPA history.** `predict_better_epa` falls back to the
  home side for those 158 games, and `epa_fallback_count` reports how many it
  decided that way. `--require-epa` re-runs the comparison without them so you
  can see whether the fallback drove the result. It does not: the ranking is
  unchanged.

`predict_random` uses `default_rng(seed)`, so it is unaffected by global numpy
state and reproduces across processes — verified by a test that seeds the legacy
global RNG in between and gets identical picks.

### Metrics

Accuracy alone is misleading at a bookmaker's price. At the standard -110, a
winning unit returns 100/110, so **break-even is 52.38% accuracy, not 50%**.
Each strategy reports accuracy, ATS record, ROI, profit in units, season-by-
season accuracy, and maximum drawdown (worst peak-to-trough fall on the profit
curve, ordered by date).

### Results over 2016-2025 (2,574 settled games)

| Strategy | Record | Accuracy | ROI | Max drawdown |
| --- | --- | --- | --- | --- |
| underdog | 1316-1258 | 0.5113 | −0.0239 | 85.5 u |
| away_every_game | 1311-1263 | 0.5093 | −0.0277 | 81.4 u |
| better_epa | 1299-1275 | 0.5047 | −0.0366 | 127.6 u |
| random | 1295-1279 | 0.5031 | −0.0395 | 126.0 u |
| home_every_game | 1263-1311 | 0.4907 | −0.0633 | 168.7 u |
| favorite | 1258-1316 | 0.4887 | −0.0670 | 181.5 u |

**Every baseline loses money.** None reaches the 52.38% break-even; the best,
backing underdogs, still returns −2.39%. This is what an efficient market looks
like, and it is the honest starting point.

Note that `better_epa` — the only baseline that uses a model feature — lands
mid-table, barely ahead of random and behind simply backing every underdog. On
complete cases it ties `away_every_game` exactly. The rolling EPA differential
carries little standalone against-the-spread signal, which matches the weak
univariate relationships the exploratory charts showed.

### The bar for any model

**No model in this project may be described as valuable without appearing in a
table beside these numbers.** Specifically:

- beating 50% accuracy is **not** evidence of anything — it is below break-even,
- beating break-even (52.38%) is the minimum for a positive return,
- beating the best baseline's −2.39% ROI is the minimum for having added
  anything over a strategy requiring no model at all.

A model that reports a large edge on these features should be treated as
suspect until it has survived the Phase 9 leakage tests and a held-out season.

## The model

```python
python scripts/train_model.py                      # holds out 2024 and 2025
python scripts/train_model.py --test-seasons 2025
```

```python
from gridiron.modeling import build_pipeline, get_feature_columns, train_model
```

### Everything happens inside the pipeline

```text
SimpleImputer(strategy="median")  ->  StandardScaler()  ->  LogisticRegression()
```

That is not a stylistic choice, it is what makes the model leakage-safe. An
imputer fitted outside the pipeline learns its medians from whatever frame it
is handed; a scaler fitted outside learns its means the same way. Both would
absorb the test fold. Inside a pipeline, `fit` sees only the rows it is given
and `transform` reuses those statistics at predict time.

The order is fixed: impute first, because a scaler cannot average over missing
values; standardise second, because the L2 penalty would otherwise punish a
feature measured in points far more than one measured in EPA per play.

Three tests enforce this rather than trusting it. Two assert the learned
medians and scaling means equal the **training** statistics and differ from the
full-data statistics; a third confirms fitting on more rows *does* change them,
so the first two could have failed.

### Splits are chronological, never random

A random split would put January games in training and the previous September's
games in test — a subtler leak than anything the pipeline guards against, since
the model would be tested on a season it had partly seen through its own rolling
features. `chronological_split` holds out whole seasons by default.

### Feature order travels with the model

`get_feature_columns()` is the contract. scikit-learn stores it on the fitted
pipeline as `feature_names_in_` and **checks it on every predict call**, so a
frame whose columns arrive reordered raises instead of being scored against the
wrong coefficients. It is also written to
`models/logistic_regression.metadata.json`, readable without unpickling
anything, alongside the row count, class balance, date range, and seed.

`model_coefficients()` maps the fitted coefficients back to feature names on the
standardised scale — log-odds per standard deviation, directly comparable across
features in different units.

### Held-out results: the model does not add value

Trained on 2016-2023 (2,035 games), held out on 2024-2025 (539 games). The model
is scored through the same metric code as the baselines, on the same games:

| Strategy | Record | Accuracy | ROI |
| --- | --- | --- | --- |
| better_epa | 284-255 | 0.5269 | **+0.0059** |
| favorite | 274-265 | 0.5083 | −0.0295 |
| home_every_game | 272-267 | 0.5046 | −0.0366 |
| away_every_game | 267-272 | 0.4954 | −0.0543 |
| underdog | 265-274 | 0.4917 | −0.0614 |
| random | 261-278 | 0.4842 | −0.0756 |
| **MODEL logistic_regression** | **251-288** | **0.4657** | **−0.1110** |

**The model finishes last.** It is beaten by every trivial strategy including
random, and it is the only entry below 47% accuracy.

This is not an orientation bug, which was checked: in training, actual cover
rate rises monotonically across predicted-probability quintiles (0.442 to
0.543), so the model learned a real in-sample relationship in the right
direction. On the held-out seasons that relationship **inverts** (0.565 in the
lowest quintile, 0.472 in the highest). Training accuracy is 0.5224 against a
held-out 0.4657.

The honest reading is that the model overfitted a pattern that does not
generalise. The largest coefficients say better recent offensive and defensive
EPA make a team *less* likely to cover — a plausible market-overreaction story
in sample, and one that simply did not hold in 2024-2025.

No tuning was done against the held-out seasons, because choosing a model by its
test score is how a test set stops being one. The result stands as measured.

`scripts/train_model.py` prints this verdict itself and states plainly when the
model has not been shown to add value, so the comparison cannot be omitted by
whoever reads the output next.

## Walk-forward validation

One held-out season is 250-odd near-coinflip games — far too few to tell a real
edge from a lucky year. Seven expanding-window folds give seven readings, and a
spread:

```python
python scripts/walk_forward_report.py
python scripts/walk_forward_report.py --first 2020 --final 2025
```

### How the folds work

`season_walk_forward_splits` yields one fold per validation season. Each trains
on **every season strictly before** it and validates on that season alone, so
the training window expands one season at a time — which is how the model would
actually have been used: at the start of 2022 you have 2016–2021 and nothing
else.

| Validation | Trained on | Train games |
| --- | --- | --- |
| 2019 | 2016-2018 | 746 |
| 2020 | 2016-2019 | 992 |
| 2021 | 2016-2020 | 1,248 |
| 2022 | 2016-2021 | 1,516 |
| 2023 | 2016-2022 | 1,777 |
| 2024 | 2016-2023 | 2,035 |
| 2025 | 2016-2024 | 2,303 |

A **fresh pipeline is fitted inside every fold**. Reusing one would carry an
earlier fold's imputation medians and scaling statistics into a later one — the
same leak the pipeline exists to prevent, moved up a level.

`_check_chronology` asserts at runtime that no training game was played at or
after the fold's first validation game. Season boundaries make this true (a
season ends in early January, the next starts in September, verified across all
nine boundaries in the data) but it is the property the whole backtest rests on,
so it is checked rather than assumed.

### Out-of-sample results, 1,828 games

Every game from 2019 onward, predicted exactly once by a model that saw nothing
from its season or later:

| Season | Record | Accuracy | ROI |
| --- | --- | --- | --- |
| 2019 | 135-111 | 0.5488 | +0.0477 |
| 2020 | 130-126 | 0.5078 | −0.0305 |
| 2021 | 133-135 | 0.4963 | −0.0526 |
| 2022 | 130-131 | 0.4981 | −0.0491 |
| 2023 | 127-131 | 0.4922 | −0.0603 |
| 2024 | 122-146 | 0.4552 | −0.1309 |
| 2025 | 138-133 | 0.5092 | −0.0278 |

| Aggregate | |
| --- | --- |
| Mean season accuracy | 0.5011 |
| Median season accuracy | 0.4981 |
| Worst season | 0.4552 (2024) |
| Best season | 0.5488 (2019) |
| Accuracy std dev | 0.0277 |
| Pooled accuracy | 0.5006 |
| Log loss | 0.6989 |
| Brier score | 0.2528 |
| ROI | −0.0444 |

### The probabilities are worse than useless

A model that knows nothing and predicts 0.5 for every game scores a log loss of
**ln(2) = 0.6931** and a Brier score of **0.2500**. This model scores **0.6989**
and **0.2528** — *above* both.

That is a stronger statement than the accuracy figures. The model's
probabilities are not merely uninformative; replacing every one of them with a
flat 0.5 would make the model better. `probability_quality()` reports this
comparison directly, and the script prints it in plain words.

### Against the baselines, same 1,828 games

| Strategy | Accuracy | ROI |
| --- | --- | --- |
| random | 0.5241 | **+0.0005** |
| underdog | 0.5181 | −0.0110 |
| away_every_game | 0.5126 | −0.0214 |
| better_epa | 0.5005 | −0.0444 |
| **MODEL walk_forward** | **0.5005** | **−0.0444** |
| home_every_game | 0.4874 | −0.0695 |
| favorite | 0.4819 | −0.0799 |

The model ties `better_epa` and is beaten by four strategies including random.

**Do not read `random` finishing first as meaning random is good.** At 1,828
games the standard error on an accuracy near 0.5 is about 1.2 points, so random
landing 2.4 points above chance is roughly a two-sigma outcome — and seven
strategies were drawn, so one of them landing there is unremarkable. What the
table really shows is that the spread between all these strategies is within
noise. That is the point: none of them, the model included, has demonstrated an
edge.

### What this changes about Phase 13

The single 2024–2025 holdout put the model at 0.4657. Walk-forward shows 2024
was its **worst of seven seasons** (0.4552) and that the fuller picture is
0.5011 mean — still no edge, but the single-holdout figure was pessimistic
rather than representative. This is exactly why one split is not enough to judge
a model either way.

## Hyperparameter tuning

```python
python scripts/tune_model.py
```

Writes `outputs/reports/hyperparameter_results.csv` and saves the chosen
settings, with their reasoning, to **`config/model_params.json`** — version
controlled, because the choice is a project decision that later training runs
read, not a generated artefact. `train_from_matchups` loads it by default.

Grid: `C ∈ {0.01, 0.1, 0.5, 1.0, 2.0, 10.0}` × `class_weight ∈ {None,
"balanced"}` — twelve sets, scored over **one identical list of folds** computed
once and handed to every set, so a difference between rows is a difference in
the parameters and never in the data they saw.

The deployment season (2026) is never scored. A settled 2026 row reaching the
search raises rather than being silently filtered.

### Selection is ranked, and never ROI

1. lowest mean log loss,
2. lowest mean Brier score,
3. most stable accuracy across seasons,
4. competitive mean accuracy,
5. smallest coefficients where everything above is equivalent.

Steps 1 and 2 narrow by a **tolerance** (0.001) rather than picking an outright
winner: at 1,800 games a log-loss gap of 0.0002 is noise, and treating it as a
decision would be false precision. A test confirms that two sets inside the
tolerance go to the next criterion instead of the nominal winner.

ROI is reported but never decides. Optimising a twelve-cell grid for return on
near-coinflip games selects the luckiest cell, not the best one — a test
verifies that a set with by far the best ROI does not win on that basis.

### What it selected: C = 0.01, class_weight = None

| C | weight | accuracy | log loss | Brier | ROI | acc std | worst season |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.01 | balanced | 0.4945 | **0.69629** | **0.25155** | −0.0560 | 0.0235 | 0.4590 |
| **0.01** | **None** | **0.5025** | 0.69644 | 0.25163 | −0.0406 | **0.01934** | **0.4813** |
| 0.10 | balanced | 0.4875 | 0.69828 | 0.25251 | −0.0693 | 0.0245 | 0.4627 |
| 0.10 | None | 0.5049 | 0.69847 | 0.25260 | −0.0361 | 0.0267 | 0.4627 |
| … | | | | | | | |
| 10.0 | None | 0.5011 | 0.69897 | 0.25284 | −0.0434 | 0.0266 | 0.4590 |

Log loss narrowed the field to the two `C=0.01` sets; Brier kept both; **stability
broke the tie** (accuracy std 0.0193 against 0.0235). The selected set also has
the best worst-season accuracy in the entire grid, 0.4813.

The heaviest regularisation winning is exactly what you expect when there is no
signal: shrinking coefficients toward zero moves the model toward the
uninformative predictor, which beats fitting noise.

### Three warnings, all firing

The search flags its own results, and all three flags are raised:

1. **Six of eleven features change coefficient sign between folds** —
   `away_short_week`, `div_game`, `home_short_week`, `off_epa_diff_last_5`,
   `rest_diff`, `win_pct_diff_last_5`. Their direction is not stable across
   seasons, so they are fitting noise rather than measuring anything durable.
   That `off_epa_diff_last_5` is among them is notable: it is the feature the
   `better_epa` baseline is built on.
2. **The selected settings are no better than predicting 0.5** (log loss 0.69644
   against ln 2 = 0.69315; Brier 0.25163 against 0.25). They are the least bad
   of the grid, not good.
3. **No parameter set in the grid produced informative probabilities** — 0 of 12.

No coefficient exceeded the magnitude threshold (largest is 0.093 on the
standardised scale, against a limit of 1.0), which is unsurprising given the
regularisation that was selected.

### What tuning did and did not achieve

It picked the most defensible cell in the grid and recorded why. It did not
produce a useful model, and the warnings say so in the configuration file
itself, so anyone loading these parameters sees the caveats alongside them.

## Calibration

```python
python scripts/calibration_report.py
```

Three charts to `outputs/figures/` (reliability diagram, prediction histogram,
cover rate by bucket) and the reliability table plus calibration trial to
`outputs/reports/`. **Assessed on the walk-forward predictions only** — no
in-sample prediction is scored here.

### Reliability, with counts on every row

| Bucket | Games | Predicted | Observed | Gap | Sparse |
| --- | --- | --- | --- | --- | --- |
| <0.40 | 2 | 0.373 | 0.500 | +0.127 | **yes** |
| 0.40-0.45 | 173 | 0.438 | 0.538 | **+0.099** | no |
| 0.45-0.50 | 1,134 | 0.478 | 0.482 | +0.005 | no |
| 0.50-0.55 | 481 | 0.516 | 0.480 | −0.035 | no |
| 0.55-0.60 | 33 | 0.572 | 0.485 | **−0.087** | no |
| 0.60-0.65 | 4 | 0.620 | 0.750 | +0.130 | **yes** |
| 0.65+ | 1 | 0.686 | 0.000 | −0.686 | **yes** |

Every figure carries its sample count and a standard error, because a
three-game bucket looks identical to a thousand-game one otherwise. **Three
buckets are sparse** (7 games total): their error bars are ±0.22 to ±0.35, far
wider than any miscalibration worth acting on, and they are excluded from the
bias test for that reason.

The gap runs monotonically from **+0.099 to −0.087** across the populated
buckets. That is a real systematic bias: predictions are spread wider than the
outcomes justify.

### Calibration was measured, and declined

Systematic bias is the phase brief's trigger for adding `CalibratedClassifierCV`.
It was not added, for reasons that were measured rather than assumed:

- **AUC is 0.4805**, against a chance threshold of 0.5270 at this sample size.
  The scores do not rank games — they rank them slightly *worse* than random.
- **The calibration slope is −0.54.** Higher predicted probability goes with a
  *lower* observed rate.

The distinction that decides it is between **calibration** and
**discrimination**. Calibration fixes whether the numbers mean what they say;
discrimination is whether they rank one game above another at all. A calibrator
can supply the first and never the second.

A temporally-valid trial confirms it. Within each fold the model was fitted on
the earlier training seasons and the calibrator on the latest training season
alone — disjoint, both entirely before the validation season:

| Probabilities | Log loss | Brier |
| --- | --- | --- |
| raw | 0.69611 | 0.25147 |
| platt (temporal) | 0.69383 | 0.25033 |
| isotonic (temporal) | 0.75977 | 0.25484 |
| **constant base rate** | **0.69283** | **0.24984** |

Platt "improves" log loss to 0.69383 — which is the base-rate figure to three
decimals. It improves the model by deleting it. And its fitted coefficient
flips sign across folds (`+0.15, −0.06, −0.21, −0.19, −0.01, +0.05, −0.24`), so
it cannot even agree which direction to map scores in. Isotonic is markedly
worse than raw, overfitting its single calibration season.

### The threshold that a test corrected

`MIN_USEFUL_AUC` was originally a flat 0.5. A test built from pure noise
produced AUC 0.5122 — numerically above 0.5, and enough to authorise
calibration on nothing at all. The floor is now **0.5 plus two null standard
errors**, `sqrt((n1 + n0 + 1) / (12 · n1 · n0))`, which scales with the sample
instead of being a fixed guess. On this data that threshold is 0.5270.

### If the model ever does discriminate

The machinery is built, tested, and temporally valid — `temporal_calibration_split`,
`fit_calibrator`, `apply_calibrator`, and a `decide_calibration` rule that
*recommends* calibration when scores rank but are biased (a test covers that
case). It is waiting on a model whose scores rank games. Calibration is not the
blocker here; discrimination is.

## Betting policies

```python
python scripts/policy_backtest.py
```

Seven policies, all scored on walk-forward fold predictions. The thresholds are
fixed in advance and every one is reported, including the losers — **no
threshold was chosen by looking at these results**.

### Odds assumption

**American -110**: stake 110 to win 100. A winning unit returns 0.9091 in
profit, a loser costs the full unit, and a push returns the stake. Break-even
is **52.38% of resolved bets, not 50%**. ROI is profit per *resolved* bet;
a push neither earns nor risks anything, so it is excluded from that
denominator but counted in `bets`.

### Pushes are bet on, not dropped

The walk-forward evaluation excludes pushes because they have no label to
train or score against. A betting backtest cannot: a push is a game the policy
*did* stake money on and got refunded. `bettable_predictions` therefore trains
each fold on the settled games of its earlier seasons and then predicts every
completed, priced game of its validation season — pushes included. That takes
the out-of-sample set from 1,828 to **1,871 games, 43 of them pushes**.

### Results

| Policy | Bets | W-L-P | Win rate | Units | ROI | Max DD | Streak | Prof. seasons |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bet every prediction | 1,871 | 918-910-43 | 0.5022 | −75.5 | −0.0413 | 93.2 | 14 | 1/7 |
| >= 2pp | 995 | 468-505-22 | 0.4810 | −79.5 | −0.0818 | 93.0 | 9 | 1/7 |
| >= 3pp | 641 | 301-330-10 | 0.4770 | −56.4 | −0.0893 | 63.2 | 5 | 0/7 |
| >= 4pp | 375 | 170-198-7 | 0.4620 | −43.5 | −0.1181 | 50.5 | 9 | 0/7 |
| >= 5pp | 218 | 100-113-5 | 0.4695 | −22.1 | −0.1037 | 30.5 | 6 | 2/7 |
| >= 7.5pp | 45 | 21-22-2 | 0.4884 | −2.9 | −0.0677 | 5.9 | 4 | 1/6 |
| >= 10pp | 7 | 4-3-0 | 0.5714 | **+0.6** | **+0.0909** | 2.1 | 2 | 1/2 |

**Confidence filtering did not help.** It made results worse through 4pp
(−4.13% to −11.81%), then wandered. Every policy with enough bets to measure
lost money.

### The one profitable row is the reason for the caution section

`>= 10pp` returns **+9.09% on seven bets**. Its 95% Wilson interval on the win
rate runs **0.2505 to 0.8418** — 59 percentage points wide. It covers two
seasons out of seven. The backtest flags it automatically:

> only 7 bets: far below the 100 needed for a win rate to mean anything; treat
> every figure on this row as noise

Intervals are Wilson rather than the textbook normal approximation, which at
n=7 produces bounds outside [0, 1] and far too narrow.

### Win rate does not rise with claimed confidence

| Claimed edge | Bets | Win rate | 95% interval |
| --- | --- | --- | --- |
| 0-2pp | 876 | **0.5263** | 0.4928 – 0.5596 |
| 2-3pp | 354 | 0.4883 | 0.4358 – 0.5411 |
| 3-4pp | 266 | 0.4981 | 0.4381 – 0.5581 |
| 4-5pp | 157 | 0.4516 | 0.3754 – 0.5302 |
| 5-7.5pp | 173 | 0.4647 | 0.3913 – 0.5396 |
| 7.5pp+ | 45 | 0.4884 | 0.3462 – 0.6325 |

The relationship is **inverted**: the games the model was least sure about won
most often. That is consistent with the −0.54 calibration slope measured in the
previous phase, and it is why filtering on confidence degrades rather than
improves the results.

### No claim of profitability

Nothing here establishes a profitable strategy. Every policy with a measurable
sample lost money, the only positive return rests on seven bets, and the
model's confidence is inversely related to its accuracy. A backtest that found
an edge in these numbers would be reporting noise.

### A bug the tests caught

`abs(0.45 - 0.5) * 100` is `4.999999999999999` in binary floating point, while
`abs(0.55 - 0.5) * 100` is `5.000000000000004`. A bare `>=` admitted a
prediction of 0.55 to the 5-point policy and turned away its mirror image at
0.45 — the two sides of the threshold were being treated differently.
Comparisons now carry a tolerance.

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
outputs/figures/    Generated charts (ignored by Git)
```
