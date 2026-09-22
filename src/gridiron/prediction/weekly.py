"""Weekly predictions for the upcoming season.

The pipeline is rebuilt from source every run rather than reading a cached
feature table. That is deliberate: the rolling features for week *n* depend on
every completed game before it, so a run made after last week's results landed
must produce different numbers than one made before. Recomputing is the only
way that happens reliably.

Two things this module will not do:

* **Predict a game that has already been played.** A completed game has a
  result, not a recommendation, and including one would quietly inflate any
  apparent accuracy.
* **Present the schedule spread as a live market quote.** nflverse ships one
  line per game with no timestamp of its own. Every row records the line used,
  when the data carrying it was refreshed, and what that line actually is.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from gridiron.config import PROJECT_ROOT
from gridiron.data.clean import clean_schedules
from gridiron.features.epa import aggregate_defensive_epa, aggregate_offensive_epa
from gridiron.features.matchup import add_matchup_target, build_matchups
from gridiron.features.pace import aggregate_pace
from gridiron.features.rolling import build_rolling_features
from gridiron.features.team_games import team_game_history
from gridiron.modeling.evaluate import confidence_tier
from gridiron.modeling.pipeline import feature_matrix

LOGGER = logging.getLogger(__name__)

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PREDICTION_DIR = PROJECT_ROOT / "outputs" / "predictions"

HISTORICAL_SEASONS = list(range(2016, 2026))
PREDICTION_SEASON = 2026

SCHEDULE_PATH = RAW_DIR / "schedules_2016_2026.parquet"
HISTORICAL_PBP_PATH = RAW_DIR / "pbp_2016_2025.parquet"


def prediction_pbp_path(season: int = PREDICTION_SEASON) -> Path:
    """Where the in-progress season's play-by-play is cached.

    Kept separate from the historical file so a weekly refresh downloads only
    the games that have been added, rather than re-fetching ten seasons.
    """
    return RAW_DIR / f"pbp_{season}.parquet"


# The columns the feature pipeline reads. Named here so the two play-by-play
# sources are loaded to the same shape before being combined.
PBP_COLUMNS = [
    "season",
    "week",
    "game_id",
    "posteam",
    "defteam",
    "play_type",
    "epa",
    "success",
    "pass",
    "rush",
    "qb_kneel",
    "qb_spike",
    "game_seconds_remaining",
    "half_seconds_remaining",
    "no_huddle",
    "fixed_drive",
    "play_id",
    "score_differential",
]

PREDICTION_COLUMNS = [
    "season",
    "week",
    "game_id",
    "gameday",
    "away_team",
    "home_team",
    "spread_line",
    "away_cover_probability",
    "home_cover_probability",
    "predicted_side",
    "confidence",
    "recommendation_tier",
    "status",
    "spread_line_used",
    "spread_source",
    "spread_timestamp",
    "data_updated_at",
    "prior_games_this_season",
]

STATUS_PREDICTED = "predicted"
STATUS_PENDING_NO_SPREAD = "pending: no spread line available"
STATUS_COMPLETED = "completed"

# nflverse publishes a single spread per game on the schedule. It is not a
# live quote and carries no timestamp of its own, so every row says so.
SPREAD_SOURCE = "nflverse schedule line (not a live sportsbook quote)"


class WeeklyPredictionError(ValueError):
    """Raised when a week cannot be predicted."""


@dataclass(frozen=True)
class RefreshRecord:
    """What was refreshed, and when."""

    refreshed: bool
    data_updated_at: str
    schedule_path: str
    prediction_pbp_path: str | None = None
    schedule_games: int = 0
    prediction_season_plays: int = 0
    completed_games: int = 0
    notes: list[str] = field(default_factory=list)


def _timestamp(path: Path) -> str:
    """When a file was last written, as an ISO timestamp."""
    if not path.exists():
        return ""
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat(
        timespec="seconds"
    )


def refresh_source_data(
    season: int = PREDICTION_SEASON,
    schedule_path: Path | None = None,
    pbp_path: Path | None = None,
) -> RefreshRecord:
    """Download the schedule and the in-progress season's play-by-play.

    Only the prediction season's play-by-play is fetched. The historical file
    does not change once a season is over, so re-downloading it every week
    would move hundreds of megabytes to learn nothing.
    """
    from gridiron.config import configure_environment
    from gridiron.data.download import (
        load_historical_pbp,
        load_schedules,
        save_raw_data,
    )

    configure_environment()
    schedule_path = schedule_path or SCHEDULE_PATH
    pbp_path = pbp_path or prediction_pbp_path(season)
    notes: list[str] = []

    schedules = load_schedules(sorted({*HISTORICAL_SEASONS, season}))
    save_raw_data(schedules, schedule_path)

    plays = load_historical_pbp([season])
    if plays.empty:
        notes.append(
            f"No {season} play-by-play is published yet; rolling features for "
            "that season will be empty until games are played."
        )
    else:
        save_raw_data(plays, pbp_path)

    upcoming = schedules.loc[
        (schedules["season"] == season) & (schedules["game_type"] == "REG")
    ]
    return RefreshRecord(
        refreshed=True,
        data_updated_at=datetime.now(UTC).isoformat(timespec="seconds"),
        schedule_path=str(schedule_path),
        prediction_pbp_path=str(pbp_path) if not plays.empty else None,
        schedule_games=len(upcoming),
        prediction_season_plays=len(plays),
        completed_games=int(upcoming["home_score"].notna().sum()),
        notes=notes,
    )


def source_state(
    season: int = PREDICTION_SEASON,
    schedule_path: Path | None = None,
    pbp_path: Path | None = None,
) -> RefreshRecord:
    """Describe the data already on disk, without downloading anything."""
    schedule_path = schedule_path or SCHEDULE_PATH
    pbp_path = pbp_path or prediction_pbp_path(season)

    if not schedule_path.exists():
        raise WeeklyPredictionError(
            f"No schedule at {schedule_path}. Run with --refresh-data, or run "
            "scripts/download_data.py first."
        )

    schedules = pd.read_parquet(schedule_path)
    upcoming = schedules.loc[
        (schedules["season"] == season) & (schedules["game_type"] == "REG")
    ]
    notes = []
    if not pbp_path.exists():
        notes.append(
            f"No {season} play-by-play on disk, so rolling features for that "
            "season will be empty. Re-run with --refresh-data once games have "
            "been played."
        )
    return RefreshRecord(
        refreshed=False,
        data_updated_at=_timestamp(schedule_path),
        schedule_path=str(schedule_path),
        prediction_pbp_path=str(pbp_path) if pbp_path.exists() else None,
        schedule_games=len(upcoming),
        prediction_season_plays=0,
        completed_games=int(upcoming["home_score"].notna().sum()),
        notes=notes,
    )


def load_play_by_play(
    season: int = PREDICTION_SEASON,
    historical_path: Path | None = None,
    pbp_path: Path | None = None,
) -> pd.DataFrame:
    """Combine the historical play-by-play with the in-progress season's."""
    historical_path = historical_path or HISTORICAL_PBP_PATH
    pbp_path = pbp_path or prediction_pbp_path(season)

    frames = []
    if historical_path.exists():
        frames.append(pd.read_parquet(historical_path, columns=PBP_COLUMNS))
    if pbp_path.exists():
        frames.append(pd.read_parquet(pbp_path, columns=PBP_COLUMNS))
    if not frames:
        raise WeeklyPredictionError(
            "No play-by-play data found; run scripts/download_data.py first."
        )

    combined = pd.concat(frames, ignore_index=True)
    return combined.drop_duplicates(subset=["game_id", "play_id"], keep="last")


def build_prediction_features(
    season: int = PREDICTION_SEASON,
    schedule_path: Path | None = None,
    historical_path: Path | None = None,
    pbp_path: Path | None = None,
) -> pd.DataFrame:
    """Rebuild the matchup table from source, through the latest results.

    The rolling features shift by one game within a season, so a team's row for
    week *n* is built from its completed games up to week *n - 1* and nothing
    later. Rebuilding from source is what lets last week's results reach this
    week's numbers.
    """
    schedule_path = schedule_path or SCHEDULE_PATH
    if not schedule_path.exists():
        raise WeeklyPredictionError(f"No schedule at {schedule_path}.")

    schedule = clean_schedules(pd.read_parquet(schedule_path))
    plays = load_play_by_play(season, historical_path, pbp_path)

    features = build_rolling_features(
        team_game_history(schedule),
        aggregate_offensive_epa(plays),
        aggregate_defensive_epa(plays),
        aggregate_pace(plays),
    )
    return add_matchup_target(build_matchups(features, schedule))


def predict_week(
    matchups: pd.DataFrame,
    season: int,
    week: int,
    pipeline,
    schema: dict[str, object],
    data_updated_at: str,
) -> pd.DataFrame:
    """Produce one row per eligible game in the requested week.

    Completed games are dropped: they have results, not recommendations. Games
    without a spread are kept but marked pending, because a missing line means
    the target is undefined, not that the matchup does not exist.
    """
    selected = matchups.loc[
        matchups["season"].eq(season) & matchups["week"].eq(week)
    ].copy()
    if selected.empty:
        raise WeeklyPredictionError(f"No games found for season {season}, week {week}.")

    completed = selected["home_score"].notna() & selected["away_score"].notna()
    upcoming = selected.loc[~completed].copy()
    if upcoming.empty:
        raise WeeklyPredictionError(
            f"Every game in season {season}, week {week} has already been "
            "played; there is nothing to recommend."
        )

    validate_schema(upcoming, schema)

    has_spread = upcoming["spread_line"].notna()
    probabilities = np.full(len(upcoming), np.nan)
    if has_spread.any():
        features = feature_matrix(upcoming.loc[has_spread])
        probabilities[has_spread.to_numpy()] = pipeline.predict_proba(features)[:, 1]

    home_probability = pd.Series(probabilities, index=upcoming.index)
    away_probability = 1.0 - home_probability

    table = pd.DataFrame(
        {
            "season": upcoming["season"].to_numpy(),
            "week": upcoming["week"].to_numpy(),
            "game_id": upcoming["game_id"].to_numpy(),
            "gameday": pd.to_datetime(upcoming["gameday"]).dt.date.astype(str),
            "away_team": upcoming["away_team"].to_numpy(),
            "home_team": upcoming["home_team"].to_numpy(),
            "spread_line": upcoming["spread_line"].to_numpy(),
            "away_cover_probability": away_probability.to_numpy(),
            "home_cover_probability": home_probability.to_numpy(),
            "prior_games_this_season": upcoming.get(
                "home_prior_games_this_season", pd.Series(np.nan, index=upcoming.index)
            ).to_numpy(),
        },
        index=upcoming.index,
    )

    backs_home = table["home_cover_probability"] >= 0.5
    table["predicted_side"] = np.where(
        table["home_cover_probability"].isna(),
        "",
        np.where(backs_home, table["home_team"], table["away_team"]),
    )
    # The specification's definition: how far the call sits from a coin flip.
    table["confidence"] = (table["home_cover_probability"] - 0.5).abs()
    table["recommendation_tier"] = np.where(
        table["home_cover_probability"].isna(),
        "",
        confidence_tier(table["home_cover_probability"].fillna(0.5).to_numpy()),
    )
    table["status"] = np.where(
        table["home_cover_probability"].isna(),
        STATUS_PENDING_NO_SPREAD,
        STATUS_PREDICTED,
    )

    table["spread_line_used"] = table["spread_line"]
    table["spread_source"] = SPREAD_SOURCE
    table["spread_timestamp"] = data_updated_at
    table["data_updated_at"] = data_updated_at

    return (
        table[PREDICTION_COLUMNS]
        .sort_values(["gameday", "game_id"])
        .reset_index(drop=True)
    )


def validate_schema(frame: pd.DataFrame, schema: dict[str, object]) -> None:
    """Confirm the frame can supply exactly what the model was trained on."""
    expected = list(schema.get("feature_names", []))
    if not expected:
        raise WeeklyPredictionError("Feature schema carries no feature names.")

    missing = [name for name in expected if name not in frame.columns]
    if missing:
        raise WeeklyPredictionError(
            "Generated features do not match the training schema; missing: "
            + ", ".join(missing)
        )

    built = list(feature_matrix(frame.head(1)).columns)
    if built != expected:
        raise WeeklyPredictionError(
            f"Feature order does not match the training schema: {built} vs {expected}."
        )


def format_report(
    table: pd.DataFrame,
    record: RefreshRecord,
    metadata=None,
) -> str:
    """A readable report, with the caveats attached rather than appended."""
    lines: list[str] = []
    season = int(table["season"].iloc[0]) if len(table) else 0
    week = int(table["week"].iloc[0]) if len(table) else 0

    lines.append(f"Gridiron predictions - {season} week {week}")
    lines.append("=" * 60)
    lines.append(f"Data refreshed at : {record.data_updated_at or 'unknown'}")
    lines.append(f"Schedule source   : {record.schedule_path}")
    lines.append(f"Spread line       : {SPREAD_SOURCE}")
    lines.append(
        f"Games in week     : {len(table)} upcoming "
        f"({int(table['status'].eq(STATUS_PREDICTED).sum())} predicted, "
        f"{int(table['status'].ne(STATUS_PREDICTED).sum())} pending)"
    )
    lines.append("")

    for row in table.itertuples():
        matchup = f"{row.away_team} at {row.home_team}"
        if row.status != STATUS_PREDICTED:
            lines.append(f"  {row.gameday}  {matchup:<16} PENDING - {row.status}")
            continue
        lines.append(
            f"  {row.gameday}  {matchup:<16} "
            f"line {row.spread_line_used:+.1f}  "
            f"home {row.home_cover_probability:.4f} / "
            f"away {row.away_cover_probability:.4f}  "
            f"-> {row.predicted_side} ({row.confidence:.4f}, "
            f"{row.recommendation_tier})"
        )

    lines.append("")
    lines.append("-" * 60)
    if metadata is not None:
        summary = getattr(metadata, "validation_summary", {}) or {}
        if summary:
            lines.append("Model performance on unseen seasons:")
            for key in ("mean_season_accuracy", "roc_auc", "log_loss", "roi"):
                if key in summary:
                    lines.append(f"  {key:22s} {float(summary[key]):.5f}")
    lines.append(
        "This model has no demonstrated edge against the spread. Its "
        "out-of-sample accuracy is near chance, its probabilities score worse "
        "than a flat 0.50, and no confidence threshold produced a profit in "
        "backtesting. These are model outputs, not betting advice."
    )
    for note in record.notes:
        lines.append(f"Note: {note}")
    return "\n".join(lines)


def save_predictions(
    table: pd.DataFrame,
    report: str,
    season: int,
    week: int,
    directory: Path | None = None,
) -> dict[str, Path]:
    """Write the prediction CSV and the formatted report."""
    directory = directory or PREDICTION_DIR
    directory.mkdir(parents=True, exist_ok=True)

    paths = {
        "csv": directory / f"week_{week:02d}_predictions.csv",
        "report": directory / f"week_{week:02d}_report.txt",
    }
    table.to_csv(paths["csv"], index=False)
    paths["report"].write_text(report, encoding="utf-8")
    return paths
