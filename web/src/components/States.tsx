/**
 * Loading, error and empty states.
 *
 * The error state prints the API's own sentence, which names the command that
 * produces whatever is missing. "Failed to load" tells a reader they have a
 * problem; "run python -m gridiron.cli backtest" tells them what to do.
 */

export function Loading({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center justify-center gap-3 py-16 text-slate-ink">
      <span className="h-3 w-3 animate-pulse rounded-full bg-accent-600" />
      <span className="font-display text-sm uppercase tracking-[0.14em]">
        {label}
      </span>
    </div>
  );
}

export function ErrorState({ message }: { message: string }) {
  // The API puts a command in its 503 detail; pull it out for emphasis.
  const command = message.match(/'([^']+)'/)?.[1] ?? null;

  return (
    <div className="bcard mx-auto max-w-2xl p-6 text-center">
      <p className="font-display text-lg font-semibold uppercase tracking-wide text-accent-300">
        Nothing to show
      </p>
      <p className="mt-2 text-sm leading-relaxed text-fog">{message}</p>
      {command && (
        <p className="mt-4">
          <code className="rounded bg-pitch-950 px-3 py-1.5 font-mono text-xs text-gold-300">
            {command}
          </code>
        </p>
      )}
    </div>
  );
}

export function EmptyState({ title, detail }: { title: string; detail: string }) {
  return (
    <div className="bcard p-8 text-center">
      <p className="font-display text-base font-semibold uppercase tracking-wide text-fog">
        {title}
      </p>
      <p className="mx-auto mt-2 max-w-md text-sm leading-relaxed text-slate-ink">
        {detail}
      </p>
    </div>
  );
}
