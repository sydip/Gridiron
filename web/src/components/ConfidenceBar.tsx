/**
 * How far past a coin flip the model went, as a horizontal bar.
 *
 * Scaled to 10 percentage points, which is the practical ceiling: the widest
 * confidence in the backtest was under 8pp, and a bar scaled to 50 would leave
 * every game as an invisible sliver. The number is printed beside the bar, so
 * the scale is a visual aid rather than the only way to read the value.
 */

import type { TierKey } from "../lib/format";

const SCALE_MAX_PP = 10;

const TIER_FILL: Record<TierKey, string> = {
  high: "bg-gold-500",
  medium: "bg-accent-500",
  low: "bg-accent-600/60",
  lean: "bg-slate-ink/60",
};

export function ConfidenceBar({
  confidence,
  tier,
}: {
  confidence: number;
  tier: TierKey;
}) {
  const points = confidence * 100;
  const width = Math.min(100, (points / SCALE_MAX_PP) * 100);

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between">
        <span className="text-[0.6rem] font-semibold uppercase tracking-[0.12em] text-slate-ink">
          Edge over even
        </span>
        <span className="font-mono text-[0.7rem] font-semibold text-fog">
          {points.toFixed(2)}pp
        </span>
      </div>
      <div
        className="h-1.5 w-full overflow-hidden rounded-full bg-pitch-950"
        role="meter"
        aria-valuenow={Number(points.toFixed(2))}
        aria-valuemin={0}
        aria-valuemax={SCALE_MAX_PP}
        aria-label="Percentage points above an even call"
      >
        <div
          className={`h-full rounded-full transition-all ${TIER_FILL[tier]}`}
          style={{ width: `${Math.max(width, 1.5)}%` }}
        />
      </div>
    </div>
  );
}
