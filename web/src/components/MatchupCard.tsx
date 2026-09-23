/**
 * One game, as a scoreboard tile.
 *
 * Four states, and the card has to be honest about which one it is in:
 *
 *   predicted    a line, a probability, a pick
 *   graded       played, with the pick marked right or wrong
 *   unpredicted  played before the model ever saw it -- result, but no pick
 *   pending      no line published yet, so no probability exists
 *
 * The third is the one worth care. Those weeks finished before the model was
 * pointed at them, and `predict_week` refuses completed games by design, so
 * there is no pick and never will be. The card says that in words rather than
 * leaving a blank that reads as a missing number.
 */

import type { Matchup } from "../api/types";
import { gameDate, percent, spreadForTeam, tierKey } from "../lib/format";
import { teamColor, teamName } from "../lib/teams";
import { ConfidenceBar } from "./ConfidenceBar";
import { SplitProbabilityBar } from "./SplitProbabilityBar";

const TIER_STYLE: Record<string, { chip: string; label: string }> = {
  high: { chip: "bg-gold-500/20 text-gold-300 ring-gold-500/40", label: "High confidence" },
  medium: { chip: "bg-accent-600/15 text-accent-300 ring-accent-600/40", label: "Medium" },
  low: { chip: "bg-navy-700 text-fog ring-line-600", label: "Low" },
  lean: { chip: "bg-navy-800 text-slate-ink ring-line-700", label: "Lean only" },
};

function TeamRow({
  code,
  spread,
  score,
  probability,
  picked,
}: {
  code: string;
  spread: string;
  score: number | null | undefined;
  probability: number | null | undefined;
  picked: boolean;
}) {
  return (
    <div className="flex items-center gap-3">
      <span
        className="h-8 w-1 shrink-0 rounded-full"
        style={{ backgroundColor: teamColor(code) }}
        aria-hidden="true"
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-baseline gap-2">
          <span
            className={
              "font-display text-2xl font-bold uppercase leading-none tracking-tight " +
              (picked ? "text-chalk" : "text-fog")
            }
          >
            {code}
          </span>
          {picked && (
            <span className="rounded-sm bg-accent-600 px-1 py-px text-[0.6rem] font-bold uppercase tracking-wider text-chalk">
              Pick
            </span>
          )}
        </div>
        <span className="block truncate text-[0.7rem] text-slate-ink">
          {teamName(code)}
        </span>
      </div>

      <span className="shrink-0 font-mono text-xs text-slate-ink">{spread}</span>

      {score !== null && score !== undefined ? (
        <span className="w-10 shrink-0 text-right font-display text-2xl font-bold tabular-nums text-chalk">
          {score}
        </span>
      ) : (
        <span className="w-14 shrink-0 text-right font-display text-xl font-semibold tabular-nums text-fog">
          {percent(probability, 1)}
        </span>
      )}
    </div>
  );
}

export function MatchupCard({ matchup }: { matchup: Matchup }) {
  const { prediction, result, pick_correct: pickCorrect } = matchup;
  const played = result !== null;
  const hasPrediction = prediction !== null && prediction.home_cover_probability !== null;
  // A line and a pick are different facts. A game can have a posted spread and
  // still have no prediction, because the week has not been generated yet.
  const hasLine = matchup.spread_line !== null && matchup.spread_line !== undefined;
  const tier = TIER_STYLE[tierKey(prediction?.recommendation_tier)];
  const isPush = played && result?.home_cover === null;

  const pickedHome = prediction?.predicted_side === matchup.home_team;
  const pickedAway = prediction?.predicted_side === matchup.away_team;

  return (
    <article className="bcard overflow-hidden transition-colors hover:border-line-600">
      {/* Header: date, divisional flag, and the game's state. */}
      <div className="flex items-center justify-between border-b border-line-800 px-4 py-2">
        <span className="text-[0.68rem] font-semibold uppercase tracking-[0.1em] text-slate-ink">
          {gameDate(matchup.gameday)}
          {matchup.div_game === 1 && (
            <span className="ml-2 text-slate-ink/70">· Division</span>
          )}
        </span>

        {played ? (
          <span className="text-[0.62rem] font-bold uppercase tracking-[0.12em] text-slate-ink">
            Final
          </span>
        ) : hasPrediction ? (
          <span className={`rounded-full px-2 py-0.5 text-[0.62rem] font-bold uppercase tracking-[0.1em] ring-1 ${tier.chip}`}>
            {tier.label}
          </span>
        ) : hasLine ? (
          <span className="text-[0.62rem] font-bold uppercase tracking-[0.12em] text-gold-300/80">
            Not predicted
          </span>
        ) : (
          <span className="text-[0.62rem] font-bold uppercase tracking-[0.12em] text-slate-ink/70">
            No line yet
          </span>
        )}
      </div>

      <div className="space-y-2.5 px-4 py-3">
        <TeamRow
          code={matchup.away_team}
          spread={spreadForTeam(matchup.spread_line, "away")}
          score={result?.away_score}
          probability={prediction?.away_cover_probability}
          picked={pickedAway}
        />
        <TeamRow
          code={matchup.home_team}
          spread={spreadForTeam(matchup.spread_line, "home")}
          score={result?.home_score}
          probability={prediction?.home_cover_probability}
          picked={pickedHome}
        />
      </div>

      {hasPrediction && (
        <div className="space-y-3 px-4 pb-3">
          <SplitProbabilityBar
            awayTeam={matchup.away_team}
            homeTeam={matchup.home_team}
            awayProbability={prediction.away_cover_probability ?? 0}
            homeProbability={prediction.home_cover_probability ?? 0}
            predictedSide={prediction.predicted_side}
          />
          {!played && (
            <ConfidenceBar
              confidence={prediction.confidence ?? 0}
              tier={tierKey(prediction.recommendation_tier)}
            />
          )}
        </div>
      )}

      {/* The footer is where the card states its outcome, in words. Colour is
          never the only carrier: each verdict ships with a glyph and a label. */}
      <Footer
        played={played}
        hasPrediction={hasPrediction}
        hasLine={hasLine}
        pickCorrect={pickCorrect}
        isPush={isPush}
        predictedSide={prediction?.predicted_side ?? null}
      />
    </article>
  );
}

function Footer({
  played,
  hasPrediction,
  hasLine,
  pickCorrect,
  isPush,
  predictedSide,
}: {
  played: boolean;
  hasPrediction: boolean;
  hasLine: boolean;
  pickCorrect: boolean | null;
  isPush: boolean;
  predictedSide: string | null;
}) {
  const shell =
    "flex items-center gap-2 border-t px-4 py-2 text-[0.7rem] font-semibold uppercase tracking-[0.08em]";

  if (played && !hasPrediction) {
    return (
      <div className={`${shell} border-line-800 bg-pitch-850/60 text-slate-ink`}>
        <span aria-hidden="true">—</span>
        <span className="normal-case tracking-normal">
          No prediction recorded; this game was already played.
        </span>
      </div>
    );
  }

  if (isPush) {
    return (
      <div className={`${shell} border-line-800 bg-pitch-850/60 text-push-500`}>
        <span aria-hidden="true">=</span>
        Push — stake returned
      </div>
    );
  }

  if (pickCorrect === true) {
    return (
      <div className={`${shell} border-covered-500/40 bg-covered-500/10 text-covered-300`}>
        <span aria-hidden="true">✓</span>
        {predictedSide} covered
      </div>
    );
  }

  if (pickCorrect === false) {
    return (
      <div className={`${shell} border-nocover-500/40 bg-nocover-500/10 text-nocover-300`}>
        <span aria-hidden="true">✕</span>
        {predictedSide} did not cover
      </div>
    );
  }

  if (!hasPrediction) {
    // Two different waits, and only one of them has an action attached.
    return hasLine ? (
      <div className={`${shell} border-line-800 bg-pitch-850/60 text-slate-ink`}>
        <span aria-hidden="true">·</span>
        <span className="normal-case tracking-normal">
          Line posted; this week has not been generated yet.
        </span>
      </div>
    ) : (
      <div className={`${shell} border-line-800 bg-pitch-850/60 text-slate-ink`}>
        <span aria-hidden="true">·</span>
        <span className="normal-case tracking-normal">
          Waiting on a spread line.
        </span>
      </div>
    );
  }

  return null;
}
