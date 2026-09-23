/**
 * The scoreboard: where the model stands, at a glance.
 *
 * Two records sit side by side on purpose, because they answer different
 * questions. Season-to-date is how the model has done in 2026 so far, which
 * right now is nothing — the weeks that have finished were played before the
 * model was pointed at them. The backtest is seven seasons of out-of-sample
 * games, and is the only record with enough behind it to mean anything.
 *
 * Every tile here reports a number the pipeline wrote. None of them flatters
 * it: accuracy is shown against the 52.38% a bettor needs rather than against
 * 50%, and the baseline tiles are ordered by return, which puts the model
 * fourth of seven.
 */

import { useOutletContext } from "react-router-dom";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChartPanel } from "../components/ChartPanel";
import { StatTable, type Column } from "../components/StatTable";
import { ErrorState, Loading } from "../components/States";
import { useSummary, type useSeason } from "../hooks/useApi";
import type { BaselineRow, SeasonMetrics } from "../api/types";
import { count, decimal, percent, signedPercent } from "../lib/format";
import { TOOLTIP_STYLE, valueFormatter } from "../lib/charts";
import { ODDS } from "../lib/backtest";

type SeasonContext = ReturnType<typeof useSeason>;

type Tone = "neutral" | "good" | "bad" | "muted";

const TONE_CLASS: Record<Tone, string> = {
  neutral: "text-chalk",
  good: "text-covered-300",
  bad: "text-nocover-300",
  muted: "text-slate-ink",
};

function Tile({
  value,
  label,
  sub,
  tone = "neutral",
}: {
  value: string;
  label: string;
  sub?: string;
  tone?: Tone;
}) {
  return (
    <div className="bcard px-5 py-4">
      <div className={`callout-value ${TONE_CLASS[tone]}`}>{value}</div>
      <div className="callout-label">{label}</div>
      {sub && <div className="mt-1 text-[0.7rem] text-slate-ink">{sub}</div>}
    </div>
  );
}

/** A compact tile for one baseline strategy. */
function BaselineTile({
  row,
  isModel,
}: {
  row: BaselineRow;
  isModel: boolean;
}) {
  const name = isModel ? "The model" : row.strategy.replace(/_/g, " ");
  return (
    <div
      className={
        "rounded-card border px-4 py-3 " +
        (isModel
          ? "border-accent-600/60 bg-accent-600/10"
          : "border-line-800 bg-navy-900/60")
      }
    >
      <div className="flex items-baseline justify-between gap-2">
        <span
          className={
            "font-display text-sm font-semibold uppercase tracking-wide " +
            (isModel ? "text-chalk" : "text-fog")
          }
        >
          {name}
        </span>
        <span
          className={
            "font-mono text-xs " +
            (row.roi > 0 ? "text-covered-300" : "text-nocover-300")
          }
        >
          {signedPercent(row.roi)}
        </span>
      </div>
      <div className="mt-1.5 flex items-baseline gap-2">
        <span
          className={
            "font-display text-2xl font-bold tabular-nums " +
            (row.accuracy >= ODDS.breakEven ? "text-covered-300" : "text-fog")
          }
        >
          {percent(row.accuracy, 2)}
        </span>
        <span className="font-mono text-[0.7rem] text-slate-ink">
          {row.ats_record}
        </span>
      </div>
      <div className="mt-1 text-[0.65rem] uppercase tracking-[0.1em] text-slate-ink">
        {row.beats_break_even ? "above break-even" : "below break-even"}
      </div>
    </div>
  );
}

export function OverviewPage() {
  const season = useOutletContext<SeasonContext>();
  const { data, loading, error } = useSummary();

  if (loading) return <Loading label="Loading model metrics" />;
  if (error) return <ErrorState message={error} />;
  if (!data) return null;

  const overall = data.overall;
  const seasonRows = [...data.by_season].sort((a, b) => a.season - b.season);
  const baselineRows = [...data.baselines].sort((a, b) => b.roi - a.roi);

  // Season-to-date, tallied from the per-week grades the shell already holds.
  let stdCorrect = 0;
  let stdGraded = 0;
  for (const grade of season.grades.values()) {
    stdCorrect += grade.correct;
    stdGraded += grade.graded;
  }
  const predictedWeeks = season.weeks.filter((w) => w.has_predictions).length;

  const outOfSampleGames = seasonRows.reduce((sum, r) => sum + r.games, 0);
  const wins = seasonRows.reduce((sum, r) => sum + r.wins, 0);
  const losses = seasonRows.reduce((sum, r) => sum + r.losses, 0);

  const seasonColumns: Column<SeasonMetrics>[] = [
    { key: "season", header: "Season", render: (r) => r.season },
    { key: "games", header: "Games", numeric: true, render: (r) => count(r.games) },
    {
      key: "record",
      header: "Record",
      numeric: true,
      render: (r) => `${r.wins}-${r.losses}`,
    },
    {
      key: "accuracy",
      header: "Accuracy",
      numeric: true,
      render: (r) => (
        <span
          className={r.accuracy >= ODDS.breakEven ? "text-covered-300" : "text-fog"}
        >
          {percent(r.accuracy, 2)}
        </span>
      ),
    },
    {
      key: "roi",
      header: "ROI",
      numeric: true,
      render: (r) => (
        <span className={r.roi > 0 ? "text-covered-300" : "text-nocover-300"}>
          {signedPercent(r.roi)}
        </span>
      ),
    },
    { key: "ll", header: "Log loss", numeric: true, render: (r) => decimal(r.log_loss, 5) },
    {
      key: "brier",
      header: "Brier",
      numeric: true,
      render: (r) => decimal(r.brier_score, 5),
    },
  ];

  return (
    <div className="space-y-6">
      {/* 2026 so far. */}
      <section>
        <h2 className="mb-2 font-display text-sm font-bold uppercase tracking-[0.14em] text-slate-ink">
          {season.season ?? 2026} season to date
        </h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <Tile
            value={stdGraded > 0 ? `${stdCorrect}-${stdGraded - stdCorrect}` : "—"}
            label="ATS record"
            sub={stdGraded > 0 ? `${stdGraded} graded` : "no graded picks yet"}
            tone={
              stdGraded === 0
                ? "muted"
                : stdCorrect / stdGraded >= ODDS.breakEven
                  ? "good"
                  : "bad"
            }
          />
          <Tile
            value={stdGraded > 0 ? percent(stdCorrect / stdGraded, 1) : "—"}
            label="Hit rate"
            sub={`${percent(ODDS.breakEven, 2)} needed at −110`}
            tone={stdGraded > 0 ? "neutral" : "muted"}
          />
          <Tile
            value={String(predictedWeeks)}
            label="Weeks predicted"
            sub={`of ${season.weeks.length} scheduled`}
            tone={predictedWeeks > 0 ? "neutral" : "muted"}
          />
          <Tile
            value={String(
              season.weeks.filter((w) => w.status === "completed").length,
            )}
            label="Weeks completed"
            sub="results available"
            tone="muted"
          />
        </div>
        {stdGraded === 0 && (
          <p className="mt-2 text-xs leading-relaxed text-slate-ink">
            Nothing is graded yet this season. The weeks that have finished were
            played before the model was pointed at them, and the prediction
            command refuses completed games, so there are no picks to score
            against them.
          </p>
        )}
      </section>

      {/* The backtest: the record that actually has weight behind it. */}
      <section>
        <h2 className="mb-2 font-display text-sm font-bold uppercase tracking-[0.14em] text-slate-ink">
          Out of sample · {count(outOfSampleGames)} games · 2019–2025
        </h2>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-6">
          {/* Judged against break-even, not against .500. A record of 925-903
              is above half and still loses money at -110; colouring it green
              would contradict the ROI tile sitting beside it. */}
          <Tile
            value={`${wins}-${losses}`}
            label="ATS record"
            sub={`${percent(ODDS.breakEven, 2)} of bets needed`}
            tone={
              wins + losses > 0 && wins / (wins + losses) >= ODDS.breakEven
                ? "good"
                : "bad"
            }
          />
          <Tile
            value={percent(overall.pooled_accuracy, 2)}
            label="Accuracy"
            sub={`${percent(ODDS.breakEven, 2)} to break even`}
            tone="bad"
          />
          <Tile
            value={signedPercent(overall.roi)}
            label="ROI at −110"
            tone="bad"
          />
          <Tile
            value={decimal(overall.log_loss, 5)}
            label="Log loss"
            sub="0.69315 for a flat 0.50"
            tone="bad"
          />
          <Tile
            value={decimal(overall.brier_score, 5)}
            label="Brier score"
            sub="0.25000 for a flat 0.50"
            tone="bad"
          />
          <Tile
            value={decimal(overall.roc_auc, 4)}
            label="ROC-AUC"
            sub="0.5 is a coin flip"
            tone="bad"
          />
        </div>
        <p className="mt-2 text-xs leading-relaxed text-slate-ink">
          Accuracy above 50% while ROC-AUC sits below it is not a contradiction:
          the model calls slightly more than half of games correctly, but its
          probabilities do not rank them. Log loss and Brier score are both
          worse than a flat 0.50 forecast, which is the same statement in a
          second form.
        </p>
      </section>

      {/* Every baseline as its own tile, ordered by return. */}
      <ChartPanel
        title="Accuracy against every baseline"
        subtitle="The same 1,828 games, through the same metric code"
        note="The model places fourth of seven, behind three strategies that need no model at all. The coin flip finishing narrowly above break-even on this sample is noise, and is exactly the kind of result that gets mistaken for a system."
        sources={data.sources.filter((s) => s.name.includes("baselines"))}
      >
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {baselineRows.map((row) => (
            <BaselineTile
              key={row.strategy}
              row={row}
              isModel={row.strategy.startsWith("MODEL")}
            />
          ))}
        </div>
      </ChartPanel>

      <ChartPanel
        title="Accuracy by validation season"
        subtitle="Expanding-window walk-forward: each season predicted by a model that saw nothing from it or later"
        callout={{ value: count(outOfSampleGames), label: "Games out of sample" }}
        note="The line to beat is 52.38%, not 50%. One season of seven finished above it, and that season is 2019 — the fold with the least training data behind it."
        sources={data.sources.filter((s) => s.name.includes("season"))}
      >
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={seasonRows} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="2 4" vertical={false} />
            <XAxis dataKey="season" tickLine={false} axisLine={false} />
            <YAxis
              domain={[0.44, 0.58]}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              tickLine={false}
              axisLine={false}
              width={44}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={valueFormatter((v) => percent(v, 2), "Accuracy")}
            />
            <ReferenceLine
              y={ODDS.breakEven}
              stroke="#f5b301"
              strokeDasharray="4 4"
              label={{
                value: "break even 52.38%",
                position: "insideTopRight",
                fill: "#f5b301",
                fontSize: 10,
              }}
            />
            <Line
              type="monotone"
              dataKey="accuracy"
              stroke="#e4002b"
              strokeWidth={2}
              dot={{ r: 4, fill: "#e4002b", strokeWidth: 0 }}
              activeDot={{ r: 6 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </ChartPanel>

      <ChartPanel
        title="Season by season"
        sources={data.sources.filter((s) => s.name.includes("season"))}
      >
        <StatTable
          columns={seasonColumns}
          rows={seasonRows}
          rowKey={(r) => String(r.season)}
          highlight={(r) => r.accuracy >= ODDS.breakEven}
        />
      </ChartPanel>

      <ChartPanel title="What is deployed" subtitle="From the model metadata">
        <dl className="grid gap-x-8 gap-y-3 sm:grid-cols-2 lg:grid-cols-3">
          {[
            ["Model", data.model.model_type],
            [
              "Trained on",
              `${data.model.training_seasons[0]}–${data.model.training_seasons.at(-1)} · ${count(data.model.n_training_games)} games`,
            ],
            ["For season", String(data.model.prediction_season)],
            ["Regularisation C", String(data.model.selected_C)],
            ["Class weight", data.model.class_weight ?? "none"],
            ["Pushes excluded", count(data.model.excluded_pushes)],
            ["Features", String(data.model.feature_names.length)],
            ["Random seed", String(data.model.random_seed)],
          ].map(([label, value]) => (
            <div key={label}>
              <dt className="callout-label mt-0">{label}</dt>
              <dd className="font-display text-base font-semibold text-chalk">
                {value}
              </dd>
            </div>
          ))}
        </dl>
      </ChartPanel>
    </div>
  );
}
