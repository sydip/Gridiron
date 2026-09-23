/**
 * The landing view: redirect to whichever week the data says is current.
 *
 * "Current" is derived from the API's week statuses, not from a calendar, so
 * the dashboard follows the pipeline. If the season has not started, this
 * lands on week 1; if it has finished, on week 18.
 */

import { Navigate, useOutletContext } from "react-router-dom";
import { EmptyState } from "../components/States";
import type { useSeason } from "../hooks/useApi";

type SeasonContext = ReturnType<typeof useSeason>;

export function ThisWeekPage() {
  const season = useOutletContext<SeasonContext>();

  if (season.currentWeek === null) {
    return (
      <EmptyState
        title="No season loaded"
        detail="The schedule has no weeks for the prediction season. Run 'python -m gridiron.cli build-features' to generate the matchup table."
      />
    );
  }

  return <Navigate to={`/week/${season.currentWeek}`} replace />;
}
