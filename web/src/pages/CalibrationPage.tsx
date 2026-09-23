/**
 * Calibration: do the stated chances come true, and does confidence mean
 * anything?
 *
 * Both answers here are no, and both panels are built to make that legible
 * rather than to flatter the model. The reliability chart carries the perfect
 * line so the scatter can be read against it, and the confidence panel is
 * ordered by claimed edge so the reversal is visible as a shape.
 */

import {
  CartesianGrid,
  Cell,
  ComposedChart,
  ErrorBar,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { ChartPanel } from "../components/ChartPanel";
import { StatTable, type Column } from "../components/StatTable";
import { ErrorState, Loading } from "../components/States";
import { useCalibration } from "../hooks/useApi";
import type { CalibrationComparison, ConfidenceBucket } from "../api/types";
import { count, decimal, percent } from "../lib/format";
import type { ReliabilityBucket } from "../api/types";
import { TOOLTIP_STYLE, valueFormatter } from "../lib/charts";

const BREAK_EVEN = 0.5238095238095238;

/**
 * The reliability tooltip.
 *
 * Sample size is the point of it. A bucket holding one game and a bucket
 * holding 1,096 are drawn on the same axes, and without the count beside them
 * the extremes read as findings rather than as noise.
 */
function ReliabilityTooltip({
  active,
  payload,
}: {
  active?: boolean;
  payload?: { payload: ReliabilityBucket }[];
}) {
  if (!active || !payload?.length) return null;
  const bucket = payload[0].payload;

  return (
    <div className="rounded border border-line-700 bg-pitch-950 px-3 py-2 text-xs">
      <div className="font-display text-sm font-semibold uppercase tracking-wide text-chalk">
        {bucket.bucket}
      </div>
      <dl className="mt-1.5 space-y-0.5">
        <div className="flex justify-between gap-6">
          <dt className="text-slate-ink">Games</dt>
          <dd className="font-mono text-fog">{count(bucket.games)}</dd>
        </div>
        <div className="flex justify-between gap-6">
          <dt className="text-slate-ink">Model said</dt>
          <dd className="font-mono text-fog">{percent(bucket.mean_predicted, 1)}</dd>
        </div>
        <div className="flex justify-between gap-6">
          <dt className="text-slate-ink">Actually covered</dt>
          <dd className="font-mono text-fog">{percent(bucket.observed_rate, 1)}</dd>
        </div>
        <div className="flex justify-between gap-6">
          <dt className="text-slate-ink">Gap</dt>
          <dd className="font-mono text-fog">
            {bucket.gap > 0 ? "+" : ""}
            {decimal(bucket.gap, 3)}
          </dd>
        </div>
      </dl>
      {bucket.sparse && (
        <p className="mt-1.5 max-w-[16rem] text-gold-300">
          Too few games for this point to mean anything on its own.
        </p>
      )}
      {bucket.biased && !bucket.sparse && (
        <p className="mt-1.5 max-w-[16rem] text-gold-300">
          Observed rate sits more than two standard errors from the stated
          chance.
        </p>
      )}
    </div>
  );
}

export function CalibrationPage() {
  const { data, loading, error } = useCalibration();

  if (loading) return <Loading label="Loading calibration" />;
  if (error) return <ErrorState message={error} />;
  if (!data) return null;

  const reliability = data.reliability.map((bucket) => ({
    ...bucket,
    perfect: bucket.mean_predicted,
  }));

  // Recharts wants the interval as distances from the point, not as bounds.
  const buckets = data.win_rate_by_confidence.map((bucket) => ({
    ...bucket,
    interval: [
      bucket.win_rate - bucket.ci_low,
      bucket.ci_high - bucket.win_rate,
    ] as [number, number],
  }));

  const comparisonColumns: Column<CalibrationComparison>[] = [
    {
      key: "probabilities",
      header: "Probabilities",
      render: (r) => (
        <span className={r.probabilities.includes("constant") ? "text-gold-300" : ""}>
          {r.probabilities}
        </span>
      ),
    },
    { key: "ll", header: "Log loss", numeric: true, render: (r) => decimal(r.log_loss, 5) },
    { key: "brier", header: "Brier", numeric: true, render: (r) => decimal(r.brier, 5) },
  ];

  const bucketColumns: Column<ConfidenceBucket>[] = [
    { key: "bucket", header: "Claimed edge", render: (r) => r.edge_bucket },
    { key: "bets", header: "Bets", numeric: true, render: (r) => count(r.bets) },
    {
      key: "win",
      header: "Win rate",
      numeric: true,
      render: (r) => (
        <span className={r.win_rate >= BREAK_EVEN ? "text-covered-300" : "text-nocover-300"}>
          {percent(r.win_rate, 2)}
        </span>
      ),
    },
    {
      key: "ci",
      header: "95% interval",
      numeric: true,
      render: (r) => `${percent(r.ci_low, 1)} – ${percent(r.ci_high, 1)}`,
    },
    {
      key: "sparse",
      header: "",
      render: (r) =>
        r.sparse ? (
          <span className="text-gold-300" title="Too few bets to interpret">
            ⚠ sparse
          </span>
        ) : null,
    },
  ];

  return (
    <div className="space-y-6">
      <ChartPanel
        title="Do the stated chances come true?"
        subtitle="Predicted probability against the rate actually observed, bucketed"
        note="A perfectly calibrated model sits on the diagonal. The fitted slope is −0.53: higher stated probability goes with a lower observed rate, which is the opposite of calibration. Grey points are buckets of five games or fewer — the one at 0% is a single game, and means nothing on its own."
        sources={data.sources.filter((s) => s.name.includes("reliability"))}
      >
        <ResponsiveContainer width="100%" height={300}>
          <ComposedChart
            data={reliability}
            margin={{ top: 8, right: 20, bottom: 8, left: 0 }}
          >
            <CartesianGrid strokeDasharray="2 4" />
            <XAxis
              dataKey="mean_predicted"
              type="number"
              domain={[0.3, 0.75]}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              tickLine={false}
              axisLine={false}
              label={{
                value: "Chance the model gave the home team",
                position: "insideBottom",
                offset: -4,
                fill: "#76829a",
                fontSize: 11,
              }}
            />
            <YAxis
              domain={[0, 1]}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              tickLine={false}
              axisLine={false}
              width={44}
            />
            <ZAxis dataKey="games" range={[40, 420]} />
            <Tooltip content={<ReliabilityTooltip />} />
            <Line
              type="linear"
              dataKey="perfect"
              stroke="#76829a"
              strokeWidth={1}
              strokeDasharray="4 4"
              dot={false}
              name="Perfectly calibrated"
            />
            <Scatter dataKey="observed_rate" name="Observed">
              {reliability.map((bucket) => (
                <Cell
                  key={bucket.bucket}
                  fill={bucket.sparse ? "#3a4356" : "#e4002b"}
                />
              ))}
            </Scatter>
          </ComposedChart>
        </ResponsiveContainer>
      </ChartPanel>

      <ChartPanel
        title="Does the model win more when it is surer?"
        subtitle="Win rate by the edge the model claimed"
        note="Win rate falls as claimed confidence rises, which is the opposite of what a working confidence signal does. Bars would have to start at zero and would hide a nine-point spread, so these are points with their 95% intervals — and every interval crosses break-even, which is the more honest reading."
        sources={data.sources.filter((s) => s.name.includes("policy"))}
      >
        <ResponsiveContainer width="100%" height={280}>
          <ComposedChart
            data={buckets}
            margin={{ top: 8, right: 16, bottom: 4, left: 0 }}
          >
            <CartesianGrid strokeDasharray="2 4" vertical={false} />
            <XAxis
              dataKey="edge_bucket"
              type="category"
              tickLine={false}
              axisLine={false}
            />
            <YAxis
              domain={[0.3, 0.7]}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
              tickLine={false}
              axisLine={false}
              width={44}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={valueFormatter((v) => percent(v, 2), "Win rate")}
            />
            <ReferenceLine
              y={BREAK_EVEN}
              stroke="#f5b301"
              strokeDasharray="4 4"
              label={{
                value: "break even 52.38%",
                position: "insideTopRight",
                fill: "#f5b301",
                fontSize: 10,
              }}
            />
            {/* Dots rather than bars: the axis does not start at zero, which
                would be dishonest under a bar but is the only way to see a
                nine-point spread. The interval is the point of the panel --
                every bucket's range crosses break-even. */}
            <Scatter dataKey="win_rate" name="Win rate">
              {buckets.map((bucket) => (
                <Cell
                  key={bucket.edge_bucket}
                  fill={bucket.sparse ? "#3a4356" : "#e4002b"}
                />
              ))}
              <ErrorBar
                dataKey="interval"
                width={6}
                strokeWidth={1.5}
                stroke="#76829a"
              />
            </Scatter>
          </ComposedChart>
        </ResponsiveContainer>
      </ChartPanel>

      <div className="grid gap-4 xl:grid-cols-2">
        <ChartPanel
          title="Calibrators, tried and declined"
          subtitle="Fitted on earlier seasons only, so the trial stays temporally valid"
          note="Platt 'improves' log loss to the base-rate figure — it improves the model by deleting it. Isotonic is markedly worse. Neither was adopted."
          sources={data.sources.filter((s) => s.name.includes("comparison"))}
        >
          <StatTable
            columns={comparisonColumns}
            rows={data.comparison}
            rowKey={(r) => r.probabilities}
            highlight={(r) => r.probabilities.includes("constant")}
          />
        </ChartPanel>

        <ChartPanel
          title="Confidence buckets"
          sources={data.sources.filter((s) => s.name.includes("policy"))}
        >
          <StatTable
            columns={bucketColumns}
            rows={data.win_rate_by_confidence}
            rowKey={(r) => r.edge_bucket}
          />
        </ChartPanel>
      </div>
    </div>
  );
}
