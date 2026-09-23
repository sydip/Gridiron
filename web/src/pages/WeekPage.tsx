/**
 * One week of matchups.
 *
 * Nothing on this page is keyed to a particular week number. It renders
 * whatever `/api/weeks/{n}` returns, so a week that did not exist yesterday
 * appears the moment `week_XX_predictions.csv` lands in `outputs/predictions`
 * and the page is reloaded. No frontend change is needed per week, for the
 * whole season.
 *
 * The header states which of the week's four states applies, because they are
 * genuinely different facts and a reader should not have to infer them from
 * whether a card looks empty.
 */

import { useParams, useNavigate } from "react-router-dom";
import { MatchupCard } from "../components/MatchupCard";
import { WeekEmptyState } from "../components/WeekEmptyState";
import { ChartPanel } from "../components/ChartPanel";
import { EmptyState, ErrorState, Loading } from "../components/States";
import { useWeek, useWeekTotals } from "../hooks/useApi";
import { percent, timestamp } from "../lib/format";
import { WEEK_STATUS_LABEL } from "../lib/season";
import { ODDS } from "../lib/backtest";

// Regular season only; playoffs are out of scope for the model and the data.
const FIRST_WEEK = 1;
const LAST_WEEK = 18;

function StatBlock({
  value,
  label,
  tone = "neutral",
}: {
  value: string;
  label: string;
  tone?: "neutral" | "good" | "bad" | "muted";
}) {
  const colour = {
    neutral: "text-chalk",
    good: "text-covered-300",
    bad: "text-nocover-300",
    muted: "text-slate-ink",
  }[tone];

  return (
    <div>
      <div className={`callout-value ${colour}`}>{value}</div>
      <div className="callout-label">{label}</div>
    </div>
  );
}

export function WeekPage() {
  const params = useParams();
  const navigate = useNavigate();
  const week = Number(params.week);
  const { data, loading, error } = useWeek(Number.isFinite(week) ? week : null);
  const totals = useWeekTotals(data?.matchups);


  if (loading) return <Loading label={`Loading week ${week}`} />;
  if (error) return <ErrorState message={error} />;
  if (!data || !totals) return <ErrorState message="No data for this week." />;

  const { grade } = totals;
  const played = data.status === "completed";
  const withSpread = data.matchups.filter((m) => m.spread_line !== null).length;

  // From the prediction CSV's own data_updated_at, not the file's mtime: the
  // column records when the source data was pulled, which is the thing worth
  // knowing.
  const refreshedAt =
    data.matchups.find((m) => m.prediction?.data_updated_at)?.prediction
      ?.data_updated_at ?? null;

  // Which absence, if any, this week is in.
  const emptyReason =
    totals.predicted > 0
      ? null
      : played
        ? ("already-played" as const)
        : withSpread === 0
          ? ("no-lines" as const)
          : ("not-run" as const);

  return (
    <div className="space-y-6">
      {/* Week header: the scoreboard strap. */}
      <div className="bcard">
        <div className="flex flex-wrap items-center justify-between gap-4 px-5 py-4">
          <div>
            <div className="flex items-baseline gap-3">
              <h1 className="font-display text-3xl font-bold uppercase tracking-tight text-chalk">
                Week {data.week}
              </h1>
              <span
                className={
                  "rounded-full px-2.5 py-0.5 text-[0.65rem] font-bold uppercase tracking-[0.1em] ring-1 " +
                  (data.status === "upcoming"
                    ? "bg-accent-600/15 text-accent-300 ring-accent-600/40"
                    : data.status === "completed"
                      ? "bg-navy-700 text-fog ring-line-600"
                      : "bg-navy-800 text-slate-ink ring-line-700")
                }
              >
                {WEEK_STATUS_LABEL[data.status]}
              </span>
            </div>
            <p className="mt-1 text-sm text-slate-ink">
              {data.season} season · {totals.games} games
              {totals.predicted > 0 && ` · ${totals.predicted} predicted`}
              {totals.pending > 0 && ` · ${totals.pending} awaiting a line`}
            </p>
          </div>

          <div className="flex items-center gap-8">
            {grade.graded > 0 ? (
              <>
                <StatBlock
                  value={`${grade.correct}-${grade.graded - grade.correct}`}
                  label="Model record"
                  tone={
                    grade.correct / grade.graded >= ODDS.breakEven
                      ? "good"
                      : "bad"
                  }
                />
                <StatBlock
                  value={percent(grade.accuracy, 1)}
                  label="Hit rate"
                  tone={
                    (grade.accuracy ?? 0) >= ODDS.breakEven ? "good" : "bad"
                  }
                />
                {grade.pushes > 0 && (
                  <StatBlock value={String(grade.pushes)} label="Pushes" tone="muted" />
                )}
              </>
            ) : played ? (
              <StatBlock value="—" label="Not predicted" tone="muted" />
            ) : (
              <StatBlock
                value={String(totals.predicted)}
                label="Picks posted"
                tone={totals.predicted > 0 ? "neutral" : "muted"}
              />
            )}
          </div>
        </div>

        {/* When a week was last generated, from its own rows. */}
        {refreshedAt && (
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-line-800 px-5 py-2 text-xs text-slate-ink">
            <span>
              <span className="font-semibold uppercase tracking-[0.1em]">
                Last refreshed
              </span>{" "}
              {timestamp(refreshedAt)}
            </span>
            <span className="text-slate-ink/70">
              Source data pulled at this time; lines are nflverse schedule
              lines, not live sportsbook quotes.
            </span>
          </div>
        )}
      </div>

      {emptyReason && (
        <WeekEmptyState
          reason={emptyReason}
          week={data.week}
          season={data.season}
          games={totals.games}
          gamesWithSpread={withSpread}
        />
      )}

      {/* The matchups. */}
      {data.matchups.length === 0 ? (
        <EmptyState
          title="No games scheduled"
          detail="The schedule has no games for this week."
        />
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
          {data.matchups.map((matchup) => (
            <MatchupCard key={matchup.game_id} matchup={matchup} />
          ))}
        </div>
      )}

      {/* The text report the pipeline itself wrote, verbatim. */}
      {data.report_text && (
        <ChartPanel
          title="Pipeline report"
          subtitle="Written by predict_week, shown unedited"
          sources={data.sources}
        >
          <pre className="overflow-x-auto whitespace-pre-wrap font-mono text-[0.72rem] leading-relaxed text-fog">
            {data.report_text}
          </pre>
        </ChartPanel>
      )}

      <p className="text-center text-xs text-slate-ink">
        {data.sources.map((source) => (
          <span key={source.name} className="mx-2 font-mono">
            {source.path}
            {source.modified_at && ` · ${timestamp(source.modified_at)}`}
          </span>
        ))}
      </p>

      {/* Rendered only when the target week exists, rather than shown disabled:
          a greyed "Week 0" button names a week that is not a thing. */}
      <div className="flex justify-center gap-2">
        {week > FIRST_WEEK && (
          <button
            type="button"
            onClick={() => navigate(`/week/${week - 1}`)}
            className="rounded border border-line-700 px-4 py-1.5 text-sm font-semibold
                       text-fog transition-colors hover:bg-navy-800"
          >
            ← Week {week - 1}
          </button>
        )}
        {week < LAST_WEEK && (
          <button
            type="button"
            onClick={() => navigate(`/week/${week + 1}`)}
            className="rounded border border-line-700 px-4 py-1.5 text-sm font-semibold
                       text-fog transition-colors hover:bg-navy-800"
          >
            Week {week + 1} →
          </button>
        )}
      </div>
    </div>
  );
}
