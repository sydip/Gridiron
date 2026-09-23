/**
 * The only place this app talks to the API.
 *
 * Requests go to a relative `/api/...`, which Vite proxies to port 8000, so
 * the browser stays on one origin and nothing depends on CORS. Set
 * `VITE_API_BASE` to point at an absolute backend instead.
 *
 * Responses are cached in memory for the life of the page. The pipeline writes
 * these files once per run, so refetching the same metric table on every route
 * change would be pure waste; a reload picks up anything new.
 */

import type {
  BacktestResponse,
  CalibrationResponse,
  CoefficientsResponse,
  CorrelationResponse,
  HealthResponse,
  RefreshResponse,
  SummaryResponse,
  WeekResponse,
  WeeksResponse,
} from "./types";

const BASE = import.meta.env.VITE_API_BASE ?? "";

export class ApiError extends Error {
  readonly status: number;
  /** The API's own message, which names the command that fixes the problem. */
  readonly detail: string;

  constructor(status: number, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.status = status;
    this.detail = detail;
  }
}

const cache = new Map<string, Promise<unknown>>();

async function request<T>(path: string): Promise<T> {
  const response = await fetch(`${BASE}${path}`, {
    headers: { Accept: "application/json" },
  });

  if (!response.ok) {
    // The API puts an actionable sentence in `detail` -- which artifact is
    // missing and which command produces it. Surface that, not the status.
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body?.detail === "string") detail = body.detail;
    } catch {
      /* A non-JSON error body leaves the status text in place. */
    }
    throw new ApiError(response.status, detail);
  }

  return (await response.json()) as T;
}

function cached<T>(path: string): Promise<T> {
  const hit = cache.get(path);
  if (hit) return hit as Promise<T>;

  // A failed request must not be cached, or the page can never recover from a
  // backend that was briefly down.
  const pending = request<T>(path).catch((error) => {
    cache.delete(path);
    throw error;
  });
  cache.set(path, pending);
  return pending;
}

/** Drop cached responses, so the next read hits disk again. */
export function invalidate(prefix?: string): void {
  if (!prefix) {
    cache.clear();
    return;
  }
  for (const key of cache.keys()) {
    if (key.startsWith(prefix)) cache.delete(key);
  }
}

export const api = {
  health: () => cached<HealthResponse>("/api/health"),
  weeks: () => cached<WeeksResponse>("/api/weeks"),
  week: (week: number) => cached<WeekResponse>(`/api/weeks/${week}`),
  summary: () => cached<SummaryResponse>("/api/metrics/summary"),
  calibration: () => cached<CalibrationResponse>("/api/metrics/calibration"),
  coefficients: () => cached<CoefficientsResponse>("/api/metrics/coefficients"),
  correlation: () => cached<CorrelationResponse>("/api/metrics/correlation"),
  backtest: () => cached<BacktestResponse>("/api/metrics/backtest"),

  /**
   * Run the pipeline's own predict command for a week.
   *
   * Never cached, and deliberately not a GET: it is the one call that causes
   * the backend to write. The API refuses completed weeks by passing the
   * command's own nonzero exit straight back, so a caller should check
   * `succeeded` rather than assuming a 200 means a prediction was made.
   */
  async refresh(week: number, refreshData = false): Promise<RefreshResponse> {
    const query = refreshData ? "?refresh_data=true" : "";
    const response = await fetch(`${BASE}/api/refresh/${week}${query}`, {
      method: "POST",
      headers: { Accept: "application/json" },
    });
    if (!response.ok) {
      throw new ApiError(response.status, `Refresh failed: ${response.status}`);
    }
    const body = (await response.json()) as RefreshResponse;
    invalidate("/api/weeks");
    return body;
  },
};
