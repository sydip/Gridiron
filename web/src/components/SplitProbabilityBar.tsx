/**
 * Away and home cover probability as one bar split at the midpoint.
 *
 * The two numbers sum to one, so a split bar is the honest form: it shows a
 * share of a whole rather than two independent magnitudes. Each side is tinted
 * with its own team colour and labelled with its code, so the split is
 * readable without relying on being able to tell the two colours apart.
 *
 * The tick at 50% is what the eye actually needs here. This model's
 * probabilities sit within a few points of even, so without a fixed reference
 * every game looks like a coin flip -- which is true, and the mark is how the
 * bar says so rather than hiding it.
 */

import { teamColor } from "../lib/teams";

interface Props {
  awayTeam: string;
  homeTeam: string;
  awayProbability: number;
  homeProbability: number;
  /** The side the model picked, drawn at full strength. */
  predictedSide: string | null;
}

export function SplitProbabilityBar({
  awayTeam,
  homeTeam,
  awayProbability,
  homeProbability,
  predictedSide,
}: Props) {
  const awayPercent = awayProbability * 100;
  const homePercent = homeProbability * 100;
  const awayPicked = predictedSide === awayTeam;
  const homePicked = predictedSide === homeTeam;

  return (
    <div>
      <div className="mb-1 flex items-baseline justify-between text-[0.6rem] font-semibold uppercase tracking-[0.12em]">
        <span className={awayPicked ? "text-chalk" : "text-slate-ink"}>
          {awayTeam} {awayPercent.toFixed(1)}%
        </span>
        <span className="text-slate-ink/60">cover chance</span>
        <span className={homePicked ? "text-chalk" : "text-slate-ink"}>
          {homePercent.toFixed(1)}% {homeTeam}
        </span>
      </div>

      <div
        className="relative flex h-2.5 w-full overflow-hidden rounded-full bg-pitch-950"
        role="img"
        aria-label={
          `${awayTeam} ${awayPercent.toFixed(1)} percent to cover, ` +
          `${homeTeam} ${homePercent.toFixed(1)} percent`
        }
      >
        <div
          className="h-full transition-all"
          style={{
            width: `${awayPercent}%`,
            backgroundColor: teamColor(awayTeam),
            opacity: awayPicked ? 1 : 0.45,
          }}
        />
        <div
          className="h-full transition-all"
          style={{
            width: `${homePercent}%`,
            backgroundColor: teamColor(homeTeam),
            opacity: homePicked ? 1 : 0.45,
          }}
        />

        {/* The even mark. Without it, a 50.3/49.7 split reads as a decision. */}
        <span
          className="absolute inset-y-0 left-1/2 w-px -translate-x-1/2 bg-chalk/70"
          aria-hidden="true"
        />
      </div>
    </div>
  );
}
