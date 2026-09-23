/**
 * What the model leans on, with the odds ratio and the direction in words.
 *
 * The direction explanations are not written here. They are the
 * `interpretation` column of `model_odds_ratios.csv`, printed as the pipeline
 * phrased them, so the page cannot quietly soften a claim the analysis made
 * more carefully.
 *
 * The chart is diverging -- blue toward an away cover, red toward home --
 * because the values are signed and a single-hue ramp would hide the sign.
 * Stability and VIF sit directly beneath, because a coefficient that flips
 * sign across folds or carries a VIF above 10 is not a finding, and the layout
 * should make that hard to miss.
 */

import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { ChartPanel } from "../components/ChartPanel";
import { StatTable, type Column } from "../components/StatTable";
import { ErrorState, Loading } from "../components/States";
import { useCoefficients } from "../hooks/useApi";
import type { Collinearity, OddsRatio, Stability } from "../api/types";
import { decimal, featureLabel } from "../lib/format";
import { TOOLTIP_STYLE, labelFormatter, valueFormatter } from "../lib/charts";

/** Blue → grey → red, with grey at zero. Never a hue at the midpoint. */
export function divergingColor(value: number, maxAbs: number): string {
  const t = maxAbs === 0 ? 0 : Math.min(1, Math.abs(value) / maxAbs);
  const strength = 0.25 + 0.75 * t;
  return value >= 0
    ? `rgba(228, 0, 43, ${strength})`
    : `rgba(45, 125, 210, ${strength})`;
}

export function CoefficientsPage() {
  const { data, loading, error } = useCoefficients();

  if (loading) return <Loading label="Loading coefficients" />;
  if (error) return <ErrorState message={error} />;
  if (!data) return null;

  const oddsByFeature = new Map(data.odds_ratios.map((o) => [o.feature, o]));

  // Odds ratio travels with the coefficient so it can be labelled on the bar,
  // which is what turns a log-odds number into something readable.
  const coefficients = [...data.coefficients]
    .sort((a, b) => b.coefficient - a.coefficient)
    .map((c) => ({
      ...c,
      oddsLabel: `×${decimal(c.odds_ratio, 3)}`,
      changePct: oddsByFeature.get(c.feature)?.odds_change_pct ?? null,
    }));

  const maxAbs = Math.max(...coefficients.map((c) => Math.abs(c.coefficient)));

  const oddsColumns: Column<OddsRatio>[] = [
    { key: "feature", header: "Feature", render: (r) => featureLabel(r.feature) },
    {
      key: "ratio",
      header: "Odds ratio",
      numeric: true,
      render: (r) => decimal(r.odds_ratio, 4),
    },
    {
      key: "pct",
      header: "Change in odds",
      numeric: true,
      render: (r) => (
        <span className={r.odds_change_pct >= 0 ? "text-accent-300" : "text-fog"}>
          {r.odds_change_pct > 0 ? "+" : ""}
          {decimal(r.odds_change_pct, 2)}%
        </span>
      ),
    },
    {
      key: "interpretation",
      // Straight from the CSV; the dashboard does not rephrase it.
      header: "As the pipeline puts it",
      render: (r) => (
        <span className="text-fog">{r.interpretation}</span>
      ),
      className: "max-w-md",
    },
  ];

  const vifColumns: Column<Collinearity>[] = [
    { key: "feature", header: "Feature", render: (r) => featureLabel(r.feature) },
    { key: "r2", header: "R²", numeric: true, render: (r) => decimal(r.r_squared, 3) },
    {
      key: "vif",
      header: "VIF",
      numeric: true,
      render: (r) => (
        <span className={r.vif >= 10 ? "font-semibold text-nocover-300" : ""}>
          {decimal(r.vif, 2)}
          {r.vif >= 10 && <span className="ml-1 text-[0.65rem]">⚠</span>}
        </span>
      ),
    },
  ];

  const stabilityColumns: Column<Stability>[] = [
    { key: "feature", header: "Feature", render: (r) => featureLabel(r.feature) },
    {
      key: "mean",
      header: "Mean",
      numeric: true,
      render: (r) => decimal(r.mean_coefficient, 4),
    },
    {
      key: "range",
      header: "Range across folds",
      numeric: true,
      render: (r) => `${decimal(r.min_coefficient, 3)} … ${decimal(r.max_coefficient, 3)}`,
    },
    {
      key: "flips",
      header: "Sign changes",
      numeric: true,
      render: (r) => (
        <span className={r.sign_changes > 0 ? "text-nocover-300" : "text-fog"}>
          {r.sign_changes}
        </span>
      ),
    },
    {
      key: "stable",
      header: "",
      render: (r) =>
        r.stable ? (
          <span className="text-covered-300">✓ stable</span>
        ) : (
          <span className="text-nocover-300">✕ unstable</span>
        ),
    },
  ];

  return (
    <div className="space-y-6">
      <ChartPanel
        title="What the model leans on"
        subtitle="Standardised coefficients, labelled with their odds ratio"
        note="Red pushes toward a home cover, blue toward away. Every coefficient is tiny — the selected regularisation (C = 0.01) shrank them all toward zero, because none improved the fit enough to resist it. An odds ratio of 1.000 would mean no effect at all."
        sources={data.sources.filter((s) => s.name === "coefficients")}
      >
        <ResponsiveContainer width="100%" height={360}>
          <BarChart
            data={coefficients}
            layout="vertical"
            margin={{ top: 4, right: 70, bottom: 4, left: 8 }}
          >
            <CartesianGrid strokeDasharray="2 4" horizontal={false} />
            <XAxis type="number" tickLine={false} axisLine={false} />
            <YAxis
              type="category"
              dataKey="feature"
              width={175}
              tickLine={false}
              axisLine={false}
              tickFormatter={featureLabel}
            />
            <Tooltip
              contentStyle={TOOLTIP_STYLE}
              formatter={valueFormatter((v) => decimal(v, 5), "Coefficient")}
              labelFormatter={labelFormatter(featureLabel)}
            />
            <ReferenceLine x={0} stroke="#3a4356" />
            <Bar dataKey="coefficient" radius={[0, 3, 3, 0]}>
              {coefficients.map((row) => (
                <Cell key={row.feature} fill={divergingColor(row.coefficient, maxAbs)} />
              ))}
              {/* The odds ratio beside each bar: the log-odds value alone is
                  not something most readers can interpret. */}
              <LabelList
                dataKey="oddsLabel"
                position="right"
                fill="#76829a"
                fontSize={10}
              />
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </ChartPanel>

      <ChartPanel
        title="Direction, in the pipeline's own words"
        subtitle="The interpretation column of model_odds_ratios.csv, unedited"
        note="These describe associations in the training seasons, not causes. No feature was manipulated, so none of this says what would happen if a team changed one."
        sources={data.sources.filter((s) => s.name === "odds_ratios")}
      >
        <StatTable
          columns={oddsColumns}
          rows={[...data.odds_ratios].sort(
            (a, b) => Math.abs(b.odds_change_pct) - Math.abs(a.odds_change_pct),
          )}
          rowKey={(r) => r.feature}
        />
      </ChartPanel>

      <div className="grid gap-4 xl:grid-cols-2">
        <ChartPanel
          title="Coefficients that will not hold still"
          subtitle="Refitted per walk-forward fold"
          note="A direction that changes between folds is not a finding."
          sources={data.sources.filter((s) => s.name.includes("stability"))}
        >
          <StatTable
            columns={stabilityColumns}
            rows={[...data.stability].sort((a, b) => b.sign_changes - a.sign_changes)}
            rowKey={(r) => r.feature}
            highlight={(r) => !r.stable}
          />
        </ChartPanel>

        <ChartPanel
          title="Multicollinearity"
          subtitle="Variance inflation factor per feature"
          note="A VIF at or above 10 means that feature's coefficient is not separable from the ones it overlaps with."
          sources={data.sources.filter((s) => s.name.includes("collinearity"))}
        >
          <StatTable
            columns={vifColumns}
            rows={[...data.collinearity].sort((a, b) => b.vif - a.vif)}
            rowKey={(r) => r.feature}
            highlight={(r) => r.vif >= 10}
          />
        </ChartPanel>
      </div>

      <ChartPanel
        title="Limitations"
        subtitle="Printed with the coefficients, by the pipeline itself"
        note="Written by interpret_model.py, which fits its own model. Figures quoted here can differ from the coefficient table above when the two were produced by different runs."
        sources={data.sources.filter((s) => s.name === "limitations")}
      >
        <ul className="space-y-3">
          {data.limitations.map((limitation) => (
            <li key={limitation.heading} className="border-l-2 border-gold-500/50 pl-3">
              <p className="font-display text-sm font-semibold uppercase tracking-wide text-gold-300">
                {limitation.heading}
              </p>
              <p className="mt-0.5 text-sm leading-relaxed text-fog">
                {limitation.detail}
              </p>
            </li>
          ))}
        </ul>
      </ChartPanel>
    </div>
  );
}
