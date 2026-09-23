/**
 * The broadcast-graphic shell every chart and table sits in.
 *
 * One component so that panel chrome -- border, header, note, source line --
 * is identical everywhere, and a chart can never quietly invent its own.
 */

import type { ReactNode } from "react";
import type { SourceRef } from "../api/types";

interface Props {
  title: string;
  /** A short line under the title: what the panel is showing. */
  subtitle?: string;
  /** The single number worth reading first, printed top-right. */
  callout?: { value: string; label: string; tone?: "neutral" | "good" | "bad" };
  /** Caveat printed under the body, in muted text. */
  note?: ReactNode;
  /** Files this panel's numbers came from. */
  sources?: SourceRef[];
  className?: string;
  children: ReactNode;
}

const CALLOUT_TONE = {
  neutral: "text-chalk",
  good: "text-covered-300",
  bad: "text-nocover-300",
};

export function ChartPanel({
  title,
  subtitle,
  callout,
  note,
  sources,
  className = "",
  children,
}: Props) {
  return (
    <section className={`bcard ${className}`}>
      <div className="bcard-header">
        <div className="min-w-0">
          <h2 className="bcard-title">{title}</h2>
          {subtitle && (
            <p className="mt-0.5 text-xs text-slate-ink">{subtitle}</p>
          )}
        </div>
        {callout && (
          <div className="shrink-0 text-right">
            <div className={`callout-value ${CALLOUT_TONE[callout.tone ?? "neutral"]}`}>
              {callout.value}
            </div>
            <div className="callout-label">{callout.label}</div>
          </div>
        )}
      </div>

      <div className="p-4">{children}</div>

      {(note || sources?.length) && (
        <div className="space-y-1 border-t border-line-800 px-4 py-2.5">
          {note && <p className="text-xs leading-relaxed text-slate-ink">{note}</p>}
          {sources?.length ? (
            <p className="font-mono text-[0.62rem] text-slate-ink/70">
              source: {sources.map((s) => s.path).join(", ")}
            </p>
          ) : null}
        </div>
      )}
    </section>
  );
}
