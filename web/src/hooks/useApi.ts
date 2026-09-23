/**
 * Small data hooks over the API client.
 *
 * Deliberately not a data-fetching library: three states (loading, error,
 * data) is all this app needs, and the client already caches, so a cache
 * layer on top would be two caches disagreeing with each other.
 *
 * Errors are kept as the API's own message, which names the command that
 * produces the missing artifact. That sentence is the most useful thing the
 * screen can show when a panel is empty.
 */

import { useCallback, useEffect, useMemo, useState } from "react";
import { api } from "../api/client";
import type { Matchup, WeekSummary } from "../api/types";
import { detectCurrentWeek, gradableWeeks, gradeWeek } from "../lib/season";
import type { WeekGrade } from "../lib/season";

export interface AsyncState<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

export function useAsync<T>(
  load: () => Promise<T>,
  deps: React.DependencyList,
): AsyncState<T> {
  const [state, setState] = useState<AsyncState<T>>({
    data: null,
    loading: true,
    error: null,
  });

  useEffect(() => {
    let live = true;
    setState({ data: null, loading: true, error: null });

    load()
      .then((data) => {
        if (live) setState({ data, loading: false, error: null });
      })
      .catch((error: unknown) => {
        if (!live) return;
        const message =
          error instanceof Error ? error.message : "Something went wrong.";
        setState({ data: null, loading: false, error: message });
      });

    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return state;
}

export function useHealth() {
  return useAsync(() => api.health(), []);
}

export function useWeeks() {
  return useAsync(() => api.weeks(), []);
}

export function useWeek(week: number | null) {
  return useAsync(
    () => (week === null ? Promise.resolve(null) : api.week(week)),
    [week],
  );
}

export function useSummary() {
  return useAsync(() => api.summary(), []);
}

export function useCalibration() {
  return useAsync(() => api.calibration(), []);
}

export function useCoefficients() {
  return useAsync(() => api.coefficients(), []);
}

export function useCorrelation() {
  return useAsync(() => api.correlation(), []);
}

export function useBacktest() {
  return useAsync(() => api.backtest(), []);
}

/**
 * Hit rates for every finished week that was predicted.
 *
 * Computed here rather than served, so the API keeps its promise that every
 * field it returns is read straight from a file. The per-week responses are
 * cached by the client, so a week fetched for a badge costs nothing when the
 * reader then opens it.
 *
 * With no finished-and-predicted weeks this fetches nothing at all, which is
 * the current state of the 2026 season: weeks 1 and 2 were played before the
 * model was pointed at them.
 */
export function useWeekGrades(weeks: WeekSummary[] | null) {
  const [grades, setGrades] = useState<Map<number, WeekGrade>>(new Map());

  const targets = useMemo(
    () => (weeks ? gradableWeeks(weeks) : []),
    [weeks],
  );
  const key = targets.join(",");

  useEffect(() => {
    if (targets.length === 0) {
      setGrades(new Map());
      return;
    }

    let live = true;
    Promise.all(
      targets.map(async (week) => {
        try {
          const response = await api.week(week);
          return [week, gradeWeek(response.matchups)] as const;
        } catch {
          // One unreadable week must not blank every other badge.
          return null;
        }
      }),
    ).then((results) => {
      if (!live) return;
      const next = new Map<number, WeekGrade>();
      for (const entry of results) {
        if (entry) next.set(entry[0], entry[1]);
      }
      setGrades(next);
    });

    return () => {
      live = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  return grades;
}

/** The season shell: weeks, the auto-detected current week, and health. */
export function useSeason() {
  const weeks = useWeeks();
  const health = useHealth();
  const grades = useWeekGrades(weeks.data?.weeks ?? null);

  const currentWeek = useMemo(
    () => (weeks.data ? detectCurrentWeek(weeks.data.weeks) : null),
    [weeks.data],
  );

  return {
    season: weeks.data?.season ?? null,
    weeks: weeks.data?.weeks ?? [],
    currentWeek,
    grades,
    degraded: health.data?.status === "degraded",
    loading: weeks.loading,
    error: weeks.error,
  };
}

/** Week-level totals a header can show without recomputing them in the view. */
export function useWeekTotals(matchups: Matchup[] | undefined) {
  return useMemo(() => {
    if (!matchups) return null;
    const predicted = matchups.filter(
      (m) => m.prediction?.home_cover_probability != null,
    );
    return {
      games: matchups.length,
      predicted: predicted.length,
      pending: matchups.filter((m) => m.spread_line === null).length,
      played: matchups.filter((m) => m.result !== null).length,
      grade: gradeWeek(matchups),
    };
  }, [matchups]);
}

/** Kick off a refresh and report what the CLI said. */
export function useRefresh() {
  const [running, setRunning] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const refresh = useCallback(async (week: number) => {
    setRunning(true);
    setMessage(null);
    try {
      const result = await api.refresh(week);
      if (result.succeeded) {
        setMessage(`Week ${week} regenerated. Reload to see it.`);
      } else {
        // The command's own refusal is more useful than "it failed".
        const line =
          result.stderr.split("\n").find((l) => l.includes("ERROR")) ??
          result.stdout.split("\n").find((l) => l.includes("ERROR")) ??
          `exit code ${result.returncode}`;
        setMessage(line.replace(/^ERROR\s*/, ""));
      }
    } catch (error) {
      setMessage(error instanceof Error ? error.message : "Refresh failed.");
    } finally {
      setRunning(false);
    }
  }, []);

  return { refresh, running, message };
}
