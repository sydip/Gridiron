# Gridiron web

React + TypeScript + Vite, Tailwind v4, Recharts, React Router. Reads the
read-only API in `api/` and renders nothing it did not receive from there.

```powershell
npm install
npm run dev          # http://localhost:3000
npm run build        # typecheck + production bundle
npm run typecheck
```

Usually you want both halves at once — see [LOCALHOST.md](../LOCALHOST.md) and
`run_local.ps1`.

## Design system

Broadcast-sports styling, not a copy of any network's branding: no borrowed
logos, wordmarks or typefaces. All tokens live in `src/index.css` under
`@theme`, so Tailwind generates the utilities from them.

| Token | Use |
| --- | --- |
| `pitch-900` `#13151a` | page charcoal |
| `navy-900` `#0d1b2a` | raised card |
| `chalk` / `fog` / `slate-ink` | ink, stepping down |
| `accent-600` `#e4002b` | state: live, upcoming, selected |
| `gold-500` `#f5b301` | the high-confidence tier, and break-even lines |
| `covered-500` / `nocover-500` | outcome, always paired with a glyph |

Barlow Condensed for headlines and scores, Inter for anything that has to be
read rather than glanced at.

**Colour is never the only signal.** Covered and no-cover always ship with a
word and a ✓/✕; week pills carry ✓ / ✕ / ◷ / –; sparse samples carry ⚠ and the
word "sparse". A red/green colourblind reader loses decoration, not meaning.

## Layout

```text
src/
  api/          client.ts (fetch + in-memory cache), types.ts (mirrors the API)
  components/   TickerNav, WeekPicker, MatchupCard, SplitProbabilityBar,
                ConfidenceBar, ChartPanel, StatTable, States, Layout
  hooks/        useApi.ts — loading/error/data, season shell, week grades
  lib/          teams.ts (32 codes + colours), format.ts, season.ts,
                charts.ts, backtest.ts (the derived curves + their check)
  pages/        ThisWeekPage, WeekPage, OverviewPage, BacktestPage,
                CalibrationPage, CoefficientsPage, CorrelationPage,
                NotFoundPage
```

## Routes

| Path | Shows |
| --- | --- |
| `/` | Redirects to the auto-detected current week |
| `/week/:week` | That week's matchups, predictions and results |
| `/model` | Scoreboard: season-to-date, out-of-sample tiles, every baseline |
| `/model/backtest` | Cumulative units, drawdown, ROI by threshold |
| `/model/calibration` | Reliability, calibrator trial, confidence buckets |
| `/model/coefficients` | Coefficients with odds ratios, stability, VIF, limitations |
| `/model/correlation` | Correlation heatmap and the kept/dropped decisions |

## Two things worth knowing

**New weeks appear on their own.** Nothing is keyed to a week number. The week
strip is built from `/api/weeks`, and a week page renders whatever
`/api/weeks/{n}` returns, so a week appears the moment
`outputs/predictions/week_XX_predictions.csv` lands and the page is reloaded.

The default week advances on the calendar: `detectCurrentWeek` picks the first
week whose last game has not been played yet. Simulated across the 2026
schedule it steps forward every Tuesday from week 1 through week 18,
monotonically, landing on week 1 before the season and week 18 after it. The
data state is only the fallback, for a schedule with no usable dates.

**A week has four states, and says which.** They are genuinely different
facts, and `WeekEmptyState` names each rather than leaving a blank:

| State | What the page says |
| --- | --- |
| Lines posted, predictions run | the picks, plus "Last refreshed" from `data_updated_at` |
| Lines posted, never generated | "Not predicted yet", with the exact `predict_week.py` command |
| No lines posted yet | "Lines not posted" — waiting on nflverse |
| Finished before the model ran | "No predictions recorded", and that it will not change |

A line and a pick are separate facts: a game can have a posted spread and no
prediction, so the card header reads "Not predicted" rather than "No line yet".
Conflating the two made week 4 claim its games had no lines while printing
them.

## Dates

`gameday` is `"2026-09-27"` — a calendar day, not an instant. `new Date()`
reads that as UTC midnight, which renders as the 26th anywhere west of
Greenwich, so every card showed its game a day early. `parseGameday` in
`lib/format.ts` constructs from the parts instead. Any new date formatting
should go through it.

## The spread convention

nflverse publishes `spread_line` with a **positive value meaning the home team
is favoured** — the opposite of a betting sheet. `spreadForTeam` flips it for
display only; the stored value is never touched and the API still serves the
original.

## The one place numbers are derived

Everything on the dashboard is read from a file the pipeline wrote, with one
exception: the cumulative-units and drawdown curves on `/model/backtest`. The
pipeline writes every bet but not a running total, so `lib/backtest.ts` walks
the bets in order at the documented flat −110.

That derivation checks itself. `reconcile` compares the derived season totals
against `policy_by_season.csv` and the derived deepest drawdown against
`policy_backtest.csv`, and the page prints the result — currently both match to
0.000000 units. If they ever disagree, the banner turns gold and tells the
reader to trust the tables instead of the curves.

Drawdown is tracked **per bet**, not per week, because that is how
`max_drawdown` in `policies.py` does it. Tracking it weekly missed intra-week
troughs and reported 80.2u where the pipeline's own table said 83.6u.
