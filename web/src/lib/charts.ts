/**
 * Shared Recharts plumbing.
 *
 * Two jobs. One, the tooltip chrome lives here so every chart's tooltip looks
 * the same rather than each page declaring its own. Two, Recharts types a
 * tooltip value as `ValueType | undefined`, so formatters have to accept that
 * and narrow it themselves; doing it once here keeps the pages readable and
 * stops a chart from crashing on a value that arrives as a string.
 */

export const TOOLTIP_STYLE = {
  backgroundColor: "#0a0c10",
  border: "1px solid #2c3342",
  borderRadius: 6,
  fontSize: 12,
  color: "#f4f7fb",
} as const;

export const TOOLTIP_CURSOR = { fill: "rgba(255,255,255,0.04)" } as const;

/** Coerce whatever Recharts hands a formatter into a number, or null. */
function asNumber(value: unknown): number | null {
  const parsed = typeof value === "number" ? value : Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

/**
 * A tooltip formatter that renders one numeric series under a fixed label.
 *
 * Params are `unknown` on purpose: Recharts' own signature is wider than the
 * values it actually passes, and accepting the wider type is what makes the
 * function assignable without a cast.
 */
export function valueFormatter(
  format: (value: number) => string,
  label: string,
): (value: unknown) => [string, string] {
  return (value: unknown) => {
    const parsed = asNumber(value);
    return [parsed === null ? "—" : format(parsed), label];
  };
}

/** A tooltip formatter whose label depends on which series was hovered. */
export function seriesFormatter(
  format: (value: number, name: string) => [string, string],
): (value: unknown, name: unknown) => [string, string] {
  return (value: unknown, name: unknown) => {
    const parsed = asNumber(value);
    if (parsed === null) return ["—", String(name ?? "")];
    return format(parsed, String(name ?? ""));
  };
}

/** A label formatter for the tooltip header. */
export function labelFormatter(
  format: (label: string) => string,
): (label: unknown) => string {
  return (label: unknown) => format(String(label ?? ""));
}
