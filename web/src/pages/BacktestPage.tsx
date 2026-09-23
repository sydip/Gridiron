/**
 * Betting simulation: cumulative units, drawdown, and ROI by threshold.
 *
 * The curves on this page are the only numbers in the dashboard the pipeline
 * did not write itself. The per-bet table exists; a running total of it does
 * not, so it is built here from the same flat -110 arithmetic the backtest
 * documents. That is stated on the page, and checked: the derived season
 * totals are compared against `policy_by_season.csv`, and if they disagree the
 * page says so instead of drawing a curve nobody can vouch for.
 *
 * Every caveat is the project's own. The sample warnings come from the
 * `sample_warning` column, the sparse flags from `sparse`, and the closing
 * language from the same sentence the pipeline prints under every weekly
 * report.
 */

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChartPanel } from "../components/ChartPanel";
import { StatTable, type Column } from "../components/StatTable";
import { ErrorState, Loading } from "../components/States";
import { useBacktest } from "../hooks/useApi";
import type { PolicyRow, PolicySeasonRow } from "../api/types";
import { count, decimal, percent, signedPercent } from "../lib/format";
import { TOOLTIP_STYLE, labelFormatter, valueFormatter } from "../lib/charts";
import {
  MIN_BETS_FOR_A_CLAIM,
  ODDS,
  cumulativeByWeek,
  reconcile,
  summarise,
} from "../lib/backtest";

export function BacktestPage() {
  const { data, loading, error } = useBacktest();

  if (loading) return <Loading label="Loading backtest" />;
  if (error) return <ErrorState message={error} />;
  if (!data) return null;

  const points = cumulativeByWeek(data.bets);
  const totals = summarise(points);
  const everyBetRow = data.policies.find((p) => p.threshold_pp === 0);
  const check = reconcile(
    data.bets,
    data.by_season,
    everyBetRow?.max_drawdown_units,
  );

  const everyBet = everyBetRow;
  const credible = data.policies.filter((p) => p.bets >= MIN_BETS_FOR_A_CLAIM);
  const profitable = credible.filter((p) => p.roi > 0);

  const seasonRows = data.by_season
    .filter((row) => row.threshold_pp === 0)
    .sort((a, b) => a.season - b.season);

  const seasonColumns: Column<PolicySeasonRow>[] = [
    { key: "season", header: "Season", render: (r) => r.season },
    { key: "bets", header: "Bets", numeric: true, render: (r) => count(r.bets) },
    {
      key: "resolved",
      header: "Resolved",
      numeric: true,
      render: (r) => count(r.resolved),
    },
    {
      key: "units",
      header: "Units",
      numeric: true,
      render: (r) => (
        <span className={r.units_won > 0 ? "text-covered-300" : "text-nocover-300"}>
          {r.units_won > 0 ? "+" : ""}
          {decimal(r.units_won, 2)}
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
  ];

  const policyColumns: Column<PolicyRow>[] = [
    { key: "policy", header: "Policy", render: (r) => r.policy },
    { key: "bets", header: "Bets", numeric: true, render: (r) => count(r.bets) },
    {
      key: "win",
      header: "Win rate",
      numeric: true,
      render: (r) => percent(r.win_rate, 2),
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
    {
      key: "units",
      header: "Units",
      numeric: true,
      render: (r) => decimal(r.units_won, 2),
    },
    {
      key: "dd",
      header: "Max drawdown",
      numeric: true,
      render: (r) => `${decimal(r.max_drawdown_units, 1)}u`,
    },
    {
      key: "streak",
      header: "Worst run",
      numeric: true,
      render: (r) => `${r.longest_losing_streak} L`,
    },
    {
      key: "warn",
      // The caveat is the pipeline's own sentence, shown rather than
      // paraphrased.
      header: "",
      render: (r) =>
        r.sample_warning ? (
          <span className="text-gold-300" title={r.sample_warning}>
            ⚠ {r.bets} bets
          </span>
        ) : null,
    },
  ];

  return (
    <div className="space-y-6">
      {/* Scoreboard strap. */}
      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {[
          {
            value: `${decimal(totals.finalUnits, 1)}u`,
            label: `Units after ${count(totals.bets)} bets`,
          },
          {
            value: signedPercent(everyBet?.roi ?? 0),
            label: "ROI, bet every prediction",
          },
          {
            value: `${decimal(totals.maxDrawdown, 1)}u`,
            label: "Deepest drawdown",
          },
          {
            value: `${profitable.length} of ${credible.length}`,
            label: `Profitable policies over ${MIN_BETS_FOR_A_CLAIM} bets`,
          },
        ].map((stat) => (
          <div key={stat.label} className="bcard px-5 py-4">
            <div className="callout-value text-nocover-300">{stat.value}</div>
            <div className="callout-label">{stat.label}</div>
          </div>
        ))}
      </div>

      {/* The one place the dashboard derives rather than reads, so it says so
          and shows the check. */}
      <div
        className={
          "bcard px-5 py-3 text-sm leading-relaxed " +
          (check.agrees ? "text-slate-ink" : "text-gold-300")
        }
      >
        {check.agrees ? (
          <>
            <strong className="font-semibold text-fog">
              Curves derived in the browser, and reconciled.
            </strong>{" "}
            The pipeline writes every bet but not a running total, so the two
            curves below are built by walking the bets in order at{" "}
            {ODDS.name} — a win returns {decimal(ODDS.winProfit, 4)} in profit,
            a loss costs one unit, a push returns the stake. The derived season
            totals match <code className="font-mono">policy_by_season.csv</code>{" "}
            exactly (largest difference {decimal(check.worstGap, 6)} units
            across {check.seasons.length} seasons), and the deepest drawdown
            matches <code className="font-mono">policy_backtest.csv</code> to{" "}
            {decimal(check.drawdown?.gap ?? 0, 6)} units.
          </>
        ) : (
          <>
            <strong className="font-semibold">
              These curves do not reconcile with the pipeline.
            </strong>{" "}
            Derived season totals differ from{" "}
            <code className="font-mono">policy_by_season.csv</code> by up to{" "}
            {decimal(check.worstGap, 4)} units. Treat the two charts below as
            unverified and trust the tables instead.
          </>
        )}
      </div>

      <ChartPanel
        title="Cumulative units, week by week"
        subtitle="Every out-of-sample bet, one unit each, in the order they settled"
        callout={{
          value: `${decimal(totals.finalUnits, 1)}u`,
          label: `over ${count(totals.weeks)} weeks`,
          tone: totals.finalUnits > 0 ? "good" : "bad",
        }}
        note="Backtested results describe games that have already been played. They are not a forecast, and nothing here is betting advice."
        sources={data.sources.filter((s) => s.name.includes("bettable"))}
      >
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart data={points} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="label"
              tickLine={false}
              axisLine={false}
              interval={Math.max(1, Math.floor(points.length / 12))}
              minTickGap={24}
            />
            <YAxis
              tickFormatter={(v: number) => `${v.toFixed(0)}u`}
              tickLine={false}
              axisLine={false}
              width={52}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={valueFormatter((v) => `${decimal(v, 2)}u`, "Cumulative")}
              labelFormatter={labelFormatter((l) => l)}
            />
            <ReferenceLine y={0} stroke="#3a4356" />
            <Line
              type="monotone"
              dataKey="units"
              stroke="#e4002b"
              strokeWidth={2}
              dot={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </ChartPanel>

      <ChartPanel
        title="Drawdown"
        subtitle="Units below the running peak — how far underwater a flat bettor would have been"
        callout={{
          value: `${decimal(totals.maxDrawdown, 1)}u`,
          label: "Deepest",
          tone: "bad",
        }}
        note="Drawdown is plotted downward from zero; deeper is worse. The depth here is on the same order as the entire stake at risk, which is the part a return figure alone does not convey."
        sources={data.sources.filter((s) => s.name.includes("bettable"))}
      >
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart
            data={points.map((p) => ({ ...p, negative: -p.drawdown }))}
            margin={{ top: 8, right: 16, bottom: 4, left: 0 }}
          >
            <defs>
              <linearGradient id="ddFill" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#9e4a52" stopOpacity={0.1} />
                <stop offset="100%" stopColor="#9e4a52" stopOpacity={0.6} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="label"
              tickLine={false}
              axisLine={false}
              interval={Math.max(1, Math.floor(points.length / 12))}
              minTickGap={24}
            />
            <YAxis
              tickFormatter={(v: number) => `${Math.abs(v).toFixed(0)}u`}
              tickLine={false}
              axisLine={false}
              width={52}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={valueFormatter(
                (v) => `${decimal(Math.abs(v), 2)}u below peak`,
                "Drawdown",
              )}
              labelFormatter={labelFormatter((l) => l)}
            />
            <Area
              type="monotone"
              dataKey="negative"
              stroke="#c27f86"
              strokeWidth={1.5}
              fill="url(#ddFill)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </ChartPanel>

      <ChartPanel
        title="ROI by confidence threshold"
        subtitle="Seven thresholds, all fixed in advance and all reported"
        note={
          <>
            Grey bars are samples below {MIN_BETS_FOR_A_CLAIM} bets, where a win
            rate carries no information. Hover any bar for the pipeline's own
            warning text. Selectivity does not help here: filtering to the
            model's more confident picks makes the return worse, not better.
          </>
        }
        sources={data.sources.filter((s) => s.name === "policy_backtest")}
      >
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={data.policies} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
            <CartesianGrid strokeDasharray="2 4" vertical={false} />
            <XAxis dataKey="policy" tickLine={false} axisLine={false} />
            <YAxis
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              tickLine={false}
              axisLine={false}
              width={48}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={valueFormatter((v) => signedPercent(v), "ROI")}
            />
            <ReferenceLine y={0} stroke="#3a4356" />
            <Bar dataKey="roi" radius={[3, 3, 0, 0]}>
              {data.policies.map((policy) => (
                <Cell
                  key={policy.policy}
                  fill={
                    policy.bets < MIN_BETS_FOR_A_CLAIM ? "#3a4356" : "#e4002b"
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartPanel>

      <div className="grid gap-4 xl:grid-cols-2">
        <ChartPanel
          title="By season"
          subtitle="Betting every prediction"
          sources={data.sources.filter((s) => s.name.includes("by_season"))}
        >
          <StatTable
            columns={seasonColumns}
            rows={seasonRows}
            rowKey={(r) => String(r.season)}
            highlight={(r) => r.units_won > 0}
          />
        </ChartPanel>

        <ChartPanel
          title="Every policy"
          subtitle="Including the ones that lose"
          note={
            profitable.length === 0
              ? `No policy with at least ${MIN_BETS_FOR_A_CLAIM} bets is profitable.`
              : undefined
          }
          sources={data.sources.filter((s) => s.name === "policy_backtest")}
        >
          <StatTable
            columns={policyColumns}
            rows={data.policies}
            rowKey={(r) => r.policy}
            highlight={(r) => r.bets >= MIN_BETS_FOR_A_CLAIM && r.roi > 0}
          />
        </ChartPanel>
      </div>
    </div>
  );
}
