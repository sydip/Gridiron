/**
 * The week strip: 18 regular-season pills, scrollable, current week marked.
 *
 * Each pill carries its own state as a glyph, so status never depends on
 * colour alone:
 *
 *   ✓ / ✕   a finished week that was predicted, and whether it finished above
 *           half. Grey when it could not be graded at all.
 *   ◷       still to come, or waiting on a line.
 *
 * A finished week that the model never predicted gets a dash rather than a
 * tick or a cross. Those weeks ended before the model was pointed at them, so
 * there is no record to grade, and inventing a neutral tick would imply one.
 */

import { useNavigate } from "react-router-dom";
import type { WeekSummary } from "../api/types";
import type { WeekGrade } from "../lib/season";
import { timestamp } from "../lib/format";

interface Props {
  weeks: WeekSummary[];
  activeWeek: number | null;
  currentWeek: number | null;
  /** Hit rates for finished weeks, keyed by week number. */
  grades: Map<number, WeekGrade>;
  /** Compact form for the sticky nav; the full form has labels. */
  dense?: boolean;
}

function Badge({ week, grade }: { week: WeekSummary; grade?: WeekGrade }) {
  if (week.status !== "completed") {
    return (
      <span className="text-[0.62rem] leading-none text-slate-ink" title="Not yet played">
        ◷
      </span>
    );
  }

  if (!grade || grade.accuracy === null) {
    return (
      <span
        className="text-[0.62rem] leading-none text-slate-ink/60"
        title="Played, but no prediction was recorded for this week"
      >
        –
      </span>
    );
  }

  const hit = grade.accuracy >= 0.5;
  return (
    <span
      className={`text-[0.62rem] leading-none ${hit ? "text-covered-300" : "text-nocover-300"}`}
      title={`${grade.correct} of ${grade.graded} picks covered`}
    >
      {hit ? "✓" : "✕"}
    </span>
  );
}

export function WeekPicker({
  weeks,
  activeWeek,
  currentWeek,
  grades,
  dense = false,
}: Props) {
  const navigate = useNavigate();

  return (
    <div
      className="flex gap-0.5 overflow-x-auto py-0.5"
      role="tablist"
      aria-label="Season week"
    >
      {weeks.map((week) => {
        const active = week.week === activeWeek;
        const isCurrent = week.week === currentWeek;
        const grade = grades.get(week.week);

        const tone = active
          ? "bg-accent-600 text-chalk"
          : week.status === "completed"
            ? "text-slate-ink hover:bg-navy-800 hover:text-fog"
            : week.status === "pending_spread"
              ? "text-slate-ink/70 hover:bg-navy-800 hover:text-fog"
              : "text-fog hover:bg-navy-800 hover:text-chalk";

        return (
          <button
            key={week.week}
            type="button"
            role="tab"
            aria-selected={active}
            onClick={() => navigate(`/week/${week.week}`)}
            title={
              `Week ${week.week} — ${week.status.replace("_", " ")}, ${week.games} games` +
              (grade && grade.accuracy !== null
                ? ` · ${grade.correct}/${grade.graded} correct`
                : "") +
              (week.data_updated_at
                ? ` · refreshed ${timestamp(week.data_updated_at)}`
                : week.status === "pending_spread"
                  ? " · lines not posted"
                  : "")
            }
            className={
              "relative flex shrink-0 flex-col items-center rounded transition-colors " +
              "focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-gold-500 " +
              (dense ? "px-2.5 py-1" : "px-3 py-1.5 min-w-[3rem]") +
              ` ${tone}`
            }
          >
            <span className="font-display text-sm font-semibold uppercase leading-none tracking-wide">
              {dense ? week.week : `Wk ${week.week}`}
            </span>
            <span className="mt-0.5 flex h-2 items-center">
              <Badge week={week} grade={grade} />
            </span>

            {isCurrent && !active && (
              <span
                className="absolute -right-0.5 -top-0.5 h-1.5 w-1.5 rounded-full bg-accent-500"
                aria-hidden="true"
              />
            )}
          </button>
        );
      })}
    </div>
  );
}
