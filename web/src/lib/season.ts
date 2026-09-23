/**
 * Working out where the season is, from what the API reports.
 *
 * Nothing here reads a calendar. The "current" week is derived from the data's
 * own state -- which games have scores, which have lines -- so the dashboard
 * follows the pipeline rather than the clock. A week appears, becomes current,
 * and grades itself as files land in `outputs/`, with no code change per week.
 */

import type { Matchup, WeekSummary } from "../api/types";

/**
 * The week to show by default.
 *
 * Primarily the calendar: the current week is the first one whose last game
 * has not yet been played, which is how a reader thinks about "this week" and
 * what makes the dashboard advance on its own every Tuesday for the whole
 * season, with no code change and no dependency on when the pipeline last ran.
 *
 * The data state is the fallback, for the two cases the calendar cannot
 * settle. Before the season opens and after it closes there is no week whose
 * games are still ahead, so it lands on the first or last week respectively.
 * And if the schedule carries no usable dates at all, it falls back to the
 * first week still holding unplayed games.
 *
 * `today` is injectable so the behaviour can be reasoned about at any point in
 * the season rather than only on the day it happens to be.
 */
export function detectCurrentWeek(
  weeks: WeekSummary[],
  today: Date = new Date(),
): number | null {
  if (weeks.length === 0) return null;

  const ordered = [...weeks].sort((a, b) => a.week - b.week);

  // Midnight local, so a game today still counts as ahead of us.
  const cutoff = new Date(
    today.getFullYear(),
    today.getMonth(),
    today.getDate(),
  ).getTime();

  const stillAhead = ordered.find((week) => {
    const last = parseWeekDate(week.last_gameday);
    return last !== null && last.getTime() >= cutoff;
  });
  if (stillAhead) return stillAhead.week;

  // Every dated week is behind us. If any date parsed at all, the season is
  // over and the last week is the useful one.
  const anyDated = ordered.some((week) => parseWeekDate(week.last_gameday));
  if (anyDated) return ordered[ordered.length - 1].week;

  const unplayed = ordered.find((week) => week.status !== "completed");
  return (unplayed ?? ordered[ordered.length - 1]).week;
}

/**
 * A `YYYY-MM-DD` from the schedule, as a local calendar date.
 *
 * Constructed from the parts rather than parsed: `new Date("2026-09-27")` is
 * UTC midnight, which is the previous day west of Greenwich, and that is
 * enough to make the dashboard advance a day early.
 */
function parseWeekDate(value: string | null): Date | null {
  if (!value) return null;
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(value.trim());
  if (!match) return null;
  return new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
}

export interface WeekGrade {
  /** Games that were both predicted and settled. Pushes are excluded. */
  graded: number;
  correct: number;
  pushes: number;
  /** Null when nothing in the week can be graded. */
  accuracy: number | null;
}

/**
 * How the model did in one week.
 *
 * Counts only -- it tallies `pick_correct`, which the API already decided by
 * comparing the predicted side with the covering side. Pushes are counted
 * separately and kept out of the rate, because a refunded stake is neither a
 * hit nor a miss.
 */
export function gradeWeek(matchups: Matchup[]): WeekGrade {
  let graded = 0;
  let correct = 0;
  let pushes = 0;

  for (const matchup of matchups) {
    if (!matchup.prediction || !matchup.result) continue;
    if (matchup.result.home_cover === null) {
      pushes += 1;
      continue;
    }
    graded += 1;
    if (matchup.pick_correct === true) correct += 1;
  }

  return {
    graded,
    correct,
    pushes,
    accuracy: graded > 0 ? correct / graded : null,
  };
}

/** Weeks that are finished and were predicted, so a badge can be computed. */
export function gradableWeeks(weeks: WeekSummary[]): number[] {
  return weeks
    .filter((week) => week.status === "completed" && week.has_predictions)
    .map((week) => week.week);
}

export const WEEK_STATUS_LABEL: Record<WeekSummary["status"], string> = {
  completed: "Final",
  upcoming: "Upcoming",
  pending_spread: "No line yet",
};
