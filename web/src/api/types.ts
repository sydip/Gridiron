/**
 * The shapes the Gridiron API returns.
 *
 * These mirror `api/main.py` field for field. Where the API can return null --
 * an unplayed game, a week the model never predicted, a push with no covering
 * side -- the type says so, because those states are the ones the interface
 * most needs to render honestly rather than as a blank.
 */

export type WeekStatus = "completed" | "upcoming" | "pending_spread";

/** Every response names the files it was built from. */
export interface SourceRef {
  name: string;
  path: string;
  modified_at: string | null;
}

export interface WeekSummary {
  week: number;
  season: number;
  status: WeekStatus;
  games: number;
  games_played: number;
  games_with_spread: number;
  first_gameday: string | null;
  last_gameday: string | null;
  has_predictions: boolean;
  predictions_available: number;
  /** From the prediction CSV's own column; null if the week was never run. */
  data_updated_at: string | null;
}

export interface WeeksResponse {
  season: number;
  weeks: WeekSummary[];
  sources: SourceRef[];
}

/** Written by `predict_week`; absent for any week that was already played. */
export interface Prediction {
  away_cover_probability: number | null;
  home_cover_probability: number | null;
  predicted_side: string | null;
  confidence: number | null;
  recommendation_tier: string | null;
  status: string | null;
  spread_line_used: number | null;
  spread_source: string | null;
  spread_timestamp: string | null;
  data_updated_at: string | null;
  prior_games_this_season: number | null;
}

/** Present once a game has been played. `home_cover` is null on a push. */
export interface GameResult {
  home_score: number | null;
  away_score: number | null;
  home_margin: number | null;
  adjusted_home_margin: number | null;
  home_cover: number | null;
}

export interface Matchup {
  game_id: string;
  season: number;
  week: number;
  gameday: string | null;
  away_team: string;
  home_team: string;
  spread_line: number | null;
  div_game: number | null;
  prediction: Prediction | null;
  result: GameResult | null;
  /** Null when ungraded, unpredicted, or a push. */
  pick_correct: boolean | null;
}

export interface WeekResponse {
  season: number;
  week: number;
  status: WeekStatus;
  has_predictions: boolean;
  matchups: Matchup[];
  report_text?: string;
  sources: SourceRef[];
}

export interface SeasonMetrics {
  season: number;
  games: number;
  wins: number;
  losses: number;
  accuracy: number;
  roi: number;
  log_loss: number;
  brier_score: number;
  mean_confidence: number;
}

export interface BaselineRow {
  strategy: string;
  games: number;
  wins: number;
  losses: number;
  ats_record: string;
  accuracy: number;
  roi: number;
  profit_units: number;
  max_drawdown_units: number;
  beats_break_even: boolean;
}

export interface PolicyRow {
  policy: string;
  threshold_pp: number;
  bets: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number;
  win_rate_ci_low: number;
  win_rate_ci_high: number;
  units_won: number;
  roi: number;
  max_drawdown_units: number;
  longest_losing_streak: number;
  profitable_seasons: number;
  seasons: number;
  sample_warning: string | null;
}

export interface ModelInfo {
  model_type: string;
  training_seasons: number[];
  prediction_season: number;
  n_training_games: number;
  excluded_pushes: number;
  selected_C: number | string;
  class_weight: string | null;
  created_at: string;
  random_seed: number;
  feature_names: string[];
  exclusions: Record<string, number>;
  validation_summary: Record<string, unknown>;
}

/** Keys are metric names from `walk_forward_aggregate.csv`. */
export interface SummaryResponse {
  overall: Record<string, number>;
  by_season: SeasonMetrics[];
  baselines: BaselineRow[];
  policies: PolicyRow[];
  model: ModelInfo;
  sources: SourceRef[];
}

export interface ReliabilityBucket {
  bucket: string;
  games: number;
  mean_predicted: number;
  observed_rate: number;
  gap: number;
  sparse: boolean;
  standard_error: number;
  biased: boolean;
}

export interface CalibrationComparison {
  probabilities: string;
  log_loss: number;
  brier: number;
}

export interface ConfidenceBucket {
  edge_bucket: string;
  bets: number;
  wins: number;
  losses: number;
  pushes: number;
  win_rate: number;
  ci_low: number;
  ci_high: number;
  sparse: boolean;
}

export interface CalibrationResponse {
  reliability: ReliabilityBucket[];
  comparison: CalibrationComparison[];
  win_rate_by_confidence: ConfidenceBucket[];
  sources: SourceRef[];
}

export interface Coefficient {
  feature: string;
  coefficient: number;
  odds_ratio: number;
  abs_coefficient: number;
}

export interface OddsRatio {
  feature: string;
  odds_ratio: number;
  odds_change_pct: number;
  interpretation: string;
}

export interface Collinearity {
  feature: string;
  r_squared: number;
  vif: number;
}

export interface Stability {
  feature: string;
  folds: number;
  mean_coefficient: number;
  std_coefficient: number;
  min_coefficient: number;
  max_coefficient: number;
  sign_changes: number;
  stable: boolean;
}

export interface Limitation {
  heading: string;
  detail: string;
}

export interface CoefficientsResponse {
  coefficients: Coefficient[];
  odds_ratios: OddsRatio[];
  collinearity: Collinearity[];
  stability: Stability[];
  limitations: Limitation[];
  sources: SourceRef[];
}

export interface CorrelationDecision {
  left: string;
  right: string;
  correlation: number;
  decision: string;
  reason: string;
}

export interface CorrelationResponse {
  features: string[];
  columns: string[];
  /** Row-major, indexed to `features` then `columns`. */
  matrix: (number | null)[][];
  decisions: CorrelationDecision[];
  sources: SourceRef[];
}

/** One row of `policy_bettable_predictions.csv`: a single staked game. */
export interface BacktestBet {
  game_id: string;
  season: number;
  week: number;
  /** 1 if the home team covered, 0 if not. Null only if ungraded. */
  actual_home_cover: number | null;
  predicted_home_cover: number;
  home_cover_probability: number;
  is_push: boolean;
}

/** One season at one confidence threshold, from `policy_by_season.csv`. */
export interface PolicySeasonRow {
  season: number;
  bets: number;
  units_won: number;
  resolved: number;
  roi: number;
  threshold_pp: number;
}

export interface BacktestResponse {
  bets: BacktestBet[];
  by_season: PolicySeasonRow[];
  policies: PolicyRow[];
  win_rate_by_confidence: ConfidenceBucket[];
  sources: SourceRef[];
}

export interface HealthArtifact {
  path: string;
  present: boolean;
  remedy: string | null;
}

export interface HealthResponse {
  status: "ok" | "degraded";
  project_root: string;
  prediction_season: number;
  prediction_weeks_on_disk: number[];
  artifacts: Record<string, HealthArtifact>;
}

export interface RefreshResponse {
  season: number;
  week: number;
  command: string;
  returncode: number;
  succeeded: boolean;
  stdout: string;
  stderr: string;
  predictions_written: boolean;
  predictions_path: string | null;
}
