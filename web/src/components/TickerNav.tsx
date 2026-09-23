/**
 * The scoreboard ticker across the top: wordmark, season, live pill, sections,
 * and the week strip.
 *
 * The week strip is the primary navigation, so it behaves like a scoreboard
 * rather than a dropdown: every week visible at once, the current one marked
 * with a pulsing dot, each carrying its own state glyph.
 */

import { NavLink } from "react-router-dom";
import type { WeekSummary } from "../api/types";
import type { WeekGrade } from "../lib/season";
import { WeekPicker } from "./WeekPicker";

interface Props {
  season: number;
  weeks: WeekSummary[];
  activeWeek: number | null;
  currentWeek: number | null;
  grades: Map<number, WeekGrade>;
  degraded: boolean;
}

const SECTIONS = [
  { to: "/model", label: "Overview", end: true },
  { to: "/model/backtest", label: "Backtest", end: false },
  { to: "/model/calibration", label: "Calibration", end: false },
  { to: "/model/coefficients", label: "Coefficients", end: false },
  { to: "/model/correlation", label: "Correlation", end: false },
];

export function TickerNav({
  season,
  weeks,
  activeWeek,
  currentWeek,
  grades,
  degraded,
}: Props) {
  return (
    <header className="sticky top-0 z-30 border-b border-line-700 bg-pitch-950/95 backdrop-blur">
      <div className="mx-auto flex max-w-[92rem] items-center gap-4 px-4 py-2.5">
        <NavLink to="/" className="flex items-baseline gap-2">
          <span className="font-display text-2xl font-bold uppercase tracking-tight text-chalk">
            Gridiron
          </span>
          <span className="hidden text-[0.65rem] font-semibold uppercase tracking-[0.18em] text-slate-ink sm:inline">
            ATS Model
          </span>
        </NavLink>

        <span className="rounded border border-line-700 px-2 py-0.5 font-display text-sm font-semibold text-fog">
          {season}
        </span>

        {currentWeek !== null && (
          <NavLink
            to={`/week/${currentWeek}`}
            className="flex items-center gap-1.5 rounded-full bg-accent-600/15 px-2.5 py-1
                       text-[0.68rem] font-bold uppercase tracking-[0.1em] text-accent-300
                       ring-1 ring-accent-600/40 transition-colors hover:bg-accent-600/25"
          >
            <span className="relative flex h-1.5 w-1.5">
              <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-accent-500 opacity-75" />
              <span className="relative inline-flex h-1.5 w-1.5 rounded-full bg-accent-500" />
            </span>
            Week {currentWeek}
          </NavLink>
        )}

        <nav className="ml-auto flex items-center gap-1">
          {SECTIONS.map((section) => (
            <NavLink
              key={section.to}
              to={section.to}
              end={section.end}
              className={({ isActive }) =>
                "rounded px-3 py-1.5 text-sm font-semibold transition-colors " +
                (isActive
                  ? "bg-navy-700 text-chalk"
                  : "text-slate-ink hover:bg-navy-800 hover:text-fog")
              }
            >
              {section.label}
            </NavLink>
          ))}
        </nav>
      </div>

      <div className="border-t border-line-800 bg-navy-900/60">
        <div className="mx-auto flex max-w-[92rem] items-center gap-3 px-4 py-1">
          <span className="shrink-0 text-[0.62rem] font-bold uppercase tracking-[0.16em] text-slate-ink">
            Week
          </span>
          <WeekPicker
            weeks={weeks}
            activeWeek={activeWeek}
            currentWeek={currentWeek}
            grades={grades}
            dense
          />
        </div>
      </div>

      {degraded && (
        <div className="border-t border-gold-500/30 bg-gold-500/10 px-4 py-1.5 text-center text-xs text-gold-300">
          Some pipeline artifacts are missing — parts of this dashboard will be
          empty. Run <code className="font-mono">python -m gridiron.cli all</code>.
        </div>
      )}
    </header>
  );
}
