/**
 * The running totals behind the ROI page.
 *
 * These are derived in the browser, and that deserves stating plainly. The
 * pipeline writes the per-bet table and the per-season totals, but not a
 * cumulative curve, so the curve is built here by walking the bets in order
 * and applying the same flat -110 arithmetic the backtest already documents.
 *
 * Nothing new is claimed. The arithmetic is fixed by the odds assumption, not
 * chosen here, and `reconcile` checks the result against the pipeline's own
 * `units_won` per season. If those disagree, the page says so rather than
 * drawing a curve nobody can vouch for -- which is the only honest way to show
 * a number the pipeline did not write itself.
 */

import type { BacktestBet, PolicySeasonRow } from "../api/types";

/**
 * American -110: stake one unit, win 100/110 in profit, lose the unit, get the
 * stake back on a push. Break-even is 110/210, not one half.
 */
export const ODDS = {
  name: "American −110",
  winProfit: 100 / 110,
  lossCost: 1,
  pushProfit: 0,
  breakEven: 110 / 210,
} as const;

/** Under the 100-bet floor, a win rate carries no information. */
export const MIN_BETS_FOR_A_CLAIM = 100;

export interface CumulativePoint {
  index: number;
  season: number;
  week: number;
  /** "2019 W1" — the x-axis label. */
  label: string;
  bets: number;
  units: number;
  /** Units below the running peak, as a positive number. */
  drawdown: number;
}

function unitsFor(bet: BacktestBet): number {
  if (bet.is_push) return ODDS.pushProfit;
  const correct = bet.predicted_home_cover === bet.actual_home_cover;
  return correct ? ODDS.winProfit : -ODDS.lossCost;
}

/**
 * Cumulative units and drawdown, one point per season-week.
 *
 * The running peak is tracked per *bet*, not per week, because that is how
 * `max_drawdown` in the pipeline does it -- `profit.cumsum()` over individual
 * bets with the peak seeded at zero. Tracking it weekly would miss a trough
 * that opens and closes inside one Sunday, and would report a shallower worst
 * case than the backtest tables show on the same page.
 *
 * The points themselves stay weekly: 1,871 of them is more than a line chart
 * can usefully draw, and the week is the unit a reader thinks in. Each point
 * carries the week's closing balance and the deepest drawdown reached during
 * it.
 */
export function cumulativeByWeek(bets: BacktestBet[]): CumulativePoint[] {
  const ordered = [...bets].sort(
    (a, b) => a.season - b.season || a.week - b.week,
  );

  const points: CumulativePoint[] = [];
  let running = 0;
  let peak = 0;
  let index = 0;

  let currentKey = "";
  let weekBets = 0;
  let weekWorstDrawdown = 0;
  let season = 0;
  let week = 0;

  const flush = () => {
    if (currentKey === "") return;
    points.push({
      index: index++,
      season,
      week,
      label: `${season} W${week}`,
      bets: weekBets,
      units: running,
      drawdown: weekWorstDrawdown,
    });
  };

  for (const bet of ordered) {
    const key = `${bet.season}-${bet.week}`;
    if (key !== currentKey) {
      flush();
      currentKey = key;
      weekBets = 0;
      weekWorstDrawdown = 0;
      season = bet.season;
      week = bet.week;
    }

    running += unitsFor(bet);
    peak = Math.max(peak, running);
    weekWorstDrawdown = Math.max(weekWorstDrawdown, peak - running);
    weekBets += 1;
  }
  flush();

  return points;
}

export interface Reconciliation {
  agrees: boolean;
  /** Largest absolute difference against the pipeline's season totals. */
  worstGap: number;
  seasons: { season: number; derived: number; published: number }[];
  /** Derived deepest drawdown against the published one, when available. */
  drawdown: { derived: number; published: number; gap: number } | null;
}

/**
 * Check the derived units against the totals the pipeline published.
 *
 * `policy_by_season.csv` holds several thresholds; only the 0.0pp rows cover
 * every bet, which is what the cumulative curve draws.
 */
export function reconcile(
  bets: BacktestBet[],
  published: PolicySeasonRow[],
  publishedDrawdown?: number,
): Reconciliation {
  const derivedBySeason = new Map<number, number>();
  for (const bet of bets) {
    derivedBySeason.set(
      bet.season,
      (derivedBySeason.get(bet.season) ?? 0) + unitsFor(bet),
    );
  }

  const seasons: Reconciliation["seasons"] = [];
  let worstGap = 0;

  for (const row of published) {
    if (row.threshold_pp !== 0) continue;
    const derived = derivedBySeason.get(row.season);
    if (derived === undefined) continue;
    const gap = Math.abs(derived - row.units_won);
    worstGap = Math.max(worstGap, gap);
    seasons.push({ season: row.season, derived, published: row.units_won });
  }

  // The drawdown is walked per bet here exactly as the pipeline walks it, so
  // it should match to floating-point noise rather than merely be close.
  let drawdown: Reconciliation["drawdown"] = null;
  if (publishedDrawdown !== undefined) {
    const derived = Math.max(
      0,
      ...cumulativeByWeek(bets).map((point) => point.drawdown),
    );
    drawdown = {
      derived,
      published: publishedDrawdown,
      gap: Math.abs(derived - publishedDrawdown),
    };
  }

  // A hundredth of a unit over 1,871 bets is rounding, not disagreement.
  const agrees =
    seasons.length > 0 &&
    worstGap < 0.01 &&
    (drawdown === null || drawdown.gap < 0.01);

  return { agrees, worstGap, seasons, drawdown };
}

/** Totals for the page header, from the same walk. */
export function summarise(points: CumulativePoint[]) {
  if (points.length === 0) {
    return { finalUnits: 0, maxDrawdown: 0, weeks: 0, bets: 0 };
  }
  return {
    finalUnits: points[points.length - 1].units,
    maxDrawdown: Math.max(...points.map((p) => p.drawdown)),
    weeks: points.length,
    bets: points.reduce((sum, p) => sum + p.bets, 0),
  };
}
