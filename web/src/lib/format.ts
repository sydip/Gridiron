/**
 * Display helpers.
 *
 * Formatting only: nothing here derives a statistic. Every number these
 * functions take came from a file the pipeline wrote, and the job is to round
 * it for reading, not to change what it says. Null is rendered as an em dash
 * rather than as zero, because "no line yet" and "a line of zero" are
 * different facts.
 */

const DASH = "—";

export function percent(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  return `${(value * 100).toFixed(digits)}%`;
}

export function signedPercent(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  const formatted = (value * 100).toFixed(digits);
  return value > 0 ? `+${formatted}%` : `${formatted}%`;
}

export function decimal(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  return value.toFixed(digits);
}

export function signed(value: number | null | undefined, digits = 1): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  const formatted = value.toFixed(digits);
  return value > 0 ? `+${formatted}` : formatted;
}

export function count(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return DASH;
  return value.toLocaleString("en-US");
}

/**
 * The spread as a bettor reads it, from nflverse's convention.
 *
 * nflverse publishes `spread_line` with a positive value meaning the HOME team
 * is favoured -- the opposite of a betting sheet, where the favourite carries
 * the minus. This flips it for display only; the stored value is untouched,
 * and the API still serves the original.
 */
export function spreadForTeam(
  spreadLine: number | null | undefined,
  side: "home" | "away",
): string {
  if (spreadLine === null || spreadLine === undefined || Number.isNaN(spreadLine)) {
    return DASH;
  }
  const line = side === "home" ? -spreadLine : spreadLine;
  if (line === 0) return "PK";
  return line > 0 ? `+${line}` : `${line}`;
}

/**
 * A calendar date, parsed as a calendar date.
 *
 * `gameday` is "2026-09-27" -- a day, not an instant. `new Date()` reads that
 * as UTC midnight, which then renders as the 26th anywhere west of Greenwich,
 * so every card showed its game a day early. Constructing from the parts keeps
 * it in local time and on the right day.
 */
export function parseGameday(value: string): Date {
  const match = /^(\d{4})-(\d{2})-(\d{2})/.exec(value.trim());
  if (!match) return new Date(value);
  return new Date(Number(match[1]), Number(match[2]) - 1, Number(match[3]));
}

/** "Sun, Sep 27" — short enough for a card header. */
export function gameDate(value: string | null | undefined): string {
  if (!value) return DASH;
  const parsed = parseGameday(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleDateString("en-US", {
    weekday: "short",
    month: "short",
    day: "numeric",
  });
}

export function timestamp(value: string | null | undefined): string {
  if (!value) return DASH;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString("en-US", {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

/** Feature names as football rather than as column headings. */
export const FRIENDLY_FEATURES: Record<string, string> = {
  spread_line: "Betting line",
  off_epa_diff_last_5: "Offence edge, last 5",
  def_epa_strength_diff_last_5: "Defence edge, last 5",
  pace_diff_last_5: "Pace edge, last 5",
  rest_diff: "Rest advantage",
  point_margin_diff_last_5: "Scoring margin, last 5",
  win_pct_diff_last_5: "Win rate edge, last 5",
  home_short_week: "Home on a short week",
  away_short_week: "Away on a short week",
  div_game: "Divisional matchup",
  week_1_flag: "Season opener",
};

export function featureLabel(name: string): string {
  return FRIENDLY_FEATURES[name] ?? name.replace(/_/g, " ");
}

/**
 * The confidence tier, as the pipeline itself names it.
 *
 * The strings come from `recommendation_tier` in the prediction CSV; this maps
 * them to a visual weight and never invents a tier of its own.
 */
export type TierKey = "high" | "medium" | "low" | "lean";

export function tierKey(tier: string | null | undefined): TierKey {
  const text = (tier ?? "").toLowerCase();
  if (text.includes("high")) return "high";
  if (text.includes("medium")) return "medium";
  if (text.includes("low")) return "low";
  return "lean";
}
