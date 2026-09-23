/**
 * The page shell: ticker nav on top, routed page below, disclaimer at the foot.
 *
 * The footer is not decoration. This model has no demonstrated edge, and a
 * dashboard that renders probabilities in broadcast styling has an obligation
 * to say so on every page rather than only where it is convenient.
 */

import { Outlet, useParams } from "react-router-dom";
import { useSeason } from "../hooks/useApi";
import { TickerNav } from "./TickerNav";
import { ErrorState, Loading } from "./States";

export function Layout() {
  const season = useSeason();
  const params = useParams();
  const activeWeek = params.week ? Number(params.week) : null;

  return (
    <div className="flex min-h-screen flex-col">
      <TickerNav
        season={season.season ?? 2026}
        weeks={season.weeks}
        activeWeek={activeWeek}
        currentWeek={season.currentWeek}
        grades={season.grades}
        degraded={season.degraded}
      />

      <main className="mx-auto w-full max-w-[92rem] flex-1 px-4 py-6">
        {season.loading ? (
          <Loading label="Loading season" />
        ) : season.error ? (
          <ErrorState message={season.error} />
        ) : (
          <Outlet context={season} />
        )}
      </main>

      <footer className="border-t border-line-800 bg-pitch-950/80">
        <div className="mx-auto max-w-[92rem] space-y-1 px-4 py-4 text-xs leading-relaxed text-slate-ink">
          <p>
            <strong className="font-semibold text-fog">
              This model has no demonstrated edge.
            </strong>{" "}
            Out-of-sample ROC-AUC is 0.4817, log loss and Brier score are both
            worse than a flat 0.50 forecast, and no betting policy with a
            credible sample was profitable. These are model outputs, not betting
            advice.
          </p>
          <p>
            Lines are nflverse schedule lines, not live sportsbook quotes. Every
            number shown is read from a file the pipeline wrote; nothing is
            recomputed in the browser.
          </p>
        </div>
      </footer>
    </div>
  );
}
