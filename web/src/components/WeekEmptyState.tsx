/**
 * What a week says when it has nothing to show, and why.
 *
 * Three different absences, which a reader should not have to tell apart from
 * a blank space:
 *
 *   no-lines     nflverse has not posted a spread for these games yet, so no
 *                probability exists to show. Waiting is the only action.
 *   not-run      the lines are there but the week has never been generated.
 *                There is a command for that, and it is printed.
 *   already-played  the week finished before the model was pointed at it.
 *                Nothing can be generated, now or later.
 *
 * The third is the one that most needs saying, because it is the only one that
 * will never resolve, and an interface that leaves it blank implies it might.
 */

import { useRefresh } from "../hooks/useApi";

type Reason = "no-lines" | "not-run" | "already-played";

interface Props {
  reason: Reason;
  week: number;
  season: number;
  games: number;
  gamesWithSpread: number;
}

export function WeekEmptyState({
  reason,
  week,
  season,
  games,
  gamesWithSpread,
}: Props) {
  const { refresh, running, message } = useRefresh();

  if (reason === "no-lines") {
    return (
      <div className="bcard px-5 py-6 text-center">
        <p className="font-display text-xl font-bold uppercase tracking-wide text-fog">
          Lines not posted
        </p>
        <p className="mx-auto mt-2 max-w-xl text-sm leading-relaxed text-slate-ink">
          nflverse has not published a spread for any of week {week}&rsquo;s{" "}
          {games} games yet. The model prices a game against its line, so there
          is nothing to predict until the lines appear — usually a week or two
          ahead. The matchups below are the schedule only.
        </p>
      </div>
    );
  }

  if (reason === "already-played") {
    return (
      <div className="bcard px-5 py-6 text-center">
        <p className="font-display text-xl font-bold uppercase tracking-wide text-fog">
          No predictions recorded
        </p>
        <p className="mx-auto mt-2 max-w-xl text-sm leading-relaxed text-slate-ink">
          Week {week} finished before the model was pointed at the {season}{" "}
          season, and the prediction command refuses games that have already
          been played — a finished game has a result, not a recommendation. The
          results below are shown without picks against them, and that will not
          change.
        </p>
      </div>
    );
  }

  return (
    <div className="bcard px-5 py-6 text-center">
      <p className="font-display text-xl font-bold uppercase tracking-wide text-gold-300">
        Not predicted yet
      </p>
      <p className="mx-auto mt-2 max-w-xl text-sm leading-relaxed text-slate-ink">
        {gamesWithSpread} of week {week}&rsquo;s {games} games have a line, but
        the week has never been generated. Run it from a terminal:
      </p>

      <p className="mt-3">
        <code className="inline-block rounded bg-pitch-950 px-3 py-2 font-mono text-xs text-gold-300">
          python scripts/predict_week.py --season {season} --week {week}
        </code>
      </p>

      <p className="mt-3 text-xs text-slate-ink">or do it from here:</p>
      <button
        type="button"
        onClick={() => refresh(week)}
        disabled={running}
        className="mt-2 rounded bg-accent-600 px-4 py-2 font-display text-sm font-semibold
                   uppercase tracking-wide text-chalk transition-colors
                   hover:bg-accent-500 disabled:opacity-50"
      >
        {running ? "Running…" : `Generate week ${week}`}
      </button>

      {message && <p className="mt-3 text-sm text-gold-300">{message}</p>}
    </div>
  );
}
