/**
 * The correlation matrix, drawn client-side from the numbers rather than shown
 * as the static PNG.
 *
 * Diverging blue–grey–red with grey at zero, because correlation is signed.
 * The kept/dropped decisions below come from `eda_correlation_decisions.csv`
 * with the pipeline's own reason text, so the page shows which pairs were
 * judged redundant and why, not merely that some were.
 */

import { ChartPanel } from "../components/ChartPanel";
import { StatTable, type Column } from "../components/StatTable";
import { ErrorState, Loading } from "../components/States";
import { useCorrelation } from "../hooks/useApi";
import type { CorrelationDecision } from "../api/types";
import { decimal, featureLabel } from "../lib/format";
import { divergingColor } from "./CoefficientsPage";

export function CorrelationPage() {
  const { data, loading, error } = useCorrelation();

  if (loading) return <Loading label="Loading correlations" />;
  if (error) return <ErrorState message={error} />;
  if (!data) return null;

  const decisionColumns: Column<CorrelationDecision>[] = [
    {
      key: "pair",
      header: "Pair",
      render: (r) => (
        <span>
          {featureLabel(r.left)}{" "}
          <span className="text-slate-ink">vs</span> {featureLabel(r.right)}
        </span>
      ),
    },
    {
      key: "r",
      header: "Correlation",
      numeric: true,
      render: (r) => (
        <span className={Math.abs(r.correlation) >= 0.8 ? "text-nocover-300" : ""}>
          {decimal(r.correlation, 4)}
        </span>
      ),
    },
    {
      key: "decision",
      header: "Decision",
      render: (r) => (
        <span
          className={
            r.decision.toLowerCase().includes("drop")
              ? "text-nocover-300"
              : "text-covered-300"
          }
        >
          {r.decision}
        </span>
      ),
    },
    {
      key: "reason",
      header: "Reason",
      render: (r) => <span className="text-fog">{r.reason}</span>,
      className: "max-w-lg",
    },
  ];

  return (
    <div className="space-y-6">
      <ChartPanel
        title="Which inputs move together"
        subtitle="Correlation among the model fields, rendered from the matrix"
        note={
          <>
            Pairwise correlation is not the whole story: point_margin_diff_last_5
            has a VIF of 10.1, which no single pair in this grid reveals. Blank
            cells are undefined rather than missing — in week 1 every rolling
            feature is empty, so its correlation with the season-opener flag has
            nothing to be computed from.
          </>
        }
        sources={data.sources.filter((s) => s.name.includes("correlation_model"))}
      >
        <div className="overflow-x-auto">
          <table className="border-collapse text-[0.65rem]">
            <thead>
              <tr>
                <th className="sticky left-0 z-10 bg-navy-900 p-1" />
                {data.columns.map((column) => (
                  <th
                    key={column}
                    className="h-28 p-1 align-bottom text-slate-ink"
                    title={featureLabel(column)}
                  >
                    <div className="origin-bottom-left translate-x-3 -rotate-45 whitespace-nowrap text-left">
                      {featureLabel(column)}
                    </div>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {data.features.map((feature, row) => (
                <tr key={feature}>
                  <th className="sticky left-0 z-10 whitespace-nowrap bg-navy-900 py-1 pr-3 text-right font-normal text-slate-ink">
                    {featureLabel(feature)}
                  </th>
                  {data.columns.map((column, col) => {
                    const value = data.matrix[row]?.[col];
                    const missing = value === null || value === undefined;
                    return (
                      <td
                        key={column}
                        className="p-0"
                        title={
                          missing
                            ? `${featureLabel(feature)} vs ${featureLabel(column)}: undefined — no overlapping games to compute from`
                            : `${featureLabel(feature)} vs ${featureLabel(column)}: ${decimal(value, 3)}`
                        }
                      >
                        <div
                          className="m-px flex h-9 w-14 items-center justify-center rounded-sm tabular-nums"
                          style={{
                            backgroundColor: missing
                              ? "transparent"
                              : divergingColor(value, 1),
                            color:
                              !missing && Math.abs(value) > 0.6
                                ? "#f4f7fb"
                                : "#aeb8c7",
                          }}
                        >
                          {missing ? "" : value.toFixed(2)}
                        </div>
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </ChartPanel>

      <ChartPanel
        title="What was kept, and why"
        subtitle="Correlation decisions, with the pipeline's own reasoning"
        note="A high correlation was not automatically a reason to drop a feature. The reason column records the judgement that was actually made."
        sources={data.sources.filter((s) => s.name.includes("decisions"))}
      >
        <StatTable
          columns={decisionColumns}
          rows={[...data.decisions].sort(
            (a, b) => Math.abs(b.correlation) - Math.abs(a.correlation),
          )}
          rowKey={(r) => `${r.left}-${r.right}`}
          highlight={(r) => r.decision.toLowerCase().includes("drop")}
        />
      </ChartPanel>
    </div>
  );
}
