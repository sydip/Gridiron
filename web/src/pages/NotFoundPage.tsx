import { Link } from "react-router-dom";

export function NotFoundPage() {
  return (
    <div className="bcard mx-auto max-w-lg p-10 text-center">
      <p className="font-display text-5xl font-bold uppercase tracking-tight text-accent-600">
        4th &amp; long
      </p>
      <p className="mt-3 text-sm text-fog">That page does not exist.</p>
      <Link
        to="/"
        className="mt-6 inline-block rounded bg-accent-600 px-4 py-2 font-display
                   text-sm font-semibold uppercase tracking-wide text-chalk
                   transition-colors hover:bg-accent-500"
      >
        This week
      </Link>
    </div>
  );
}
