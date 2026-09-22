"""A synthetic NFL universe, shared by the end-to-end command tests.

Four teams, a fixed schedule, and play-by-play generated from a seeded
generator. It is small enough to run in seconds and complete enough for the
whole feature pipeline: rolling form needs prior games, the matchup merge
needs both sides of every game, and the division flag needs real team codes.

Commands are tested against this rather than against the repository's own
``data/raw``, so they pass on a fresh clone with nothing downloaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

TEAMS = ["KC", "BUF", "MIA", "NYJ"]
HISTORY_SEASON = 2025
PREDICTION_SEASON = 2026
# The week left unplayed, so there is always something to predict.
TARGET_WEEK = 3

PLAYED_HISTORY_WEEKS = 14


def pairings(week: int) -> list[tuple[str, str]]:
    """Two games a week, with home advantage alternating by week."""
    if week % 2:
        return [("KC", "BUF"), ("MIA", "NYJ")]
    return [("BUF", "KC"), ("NYJ", "MIA")]


def synthetic_schedule() -> pd.DataFrame:
    """Fourteen played weeks in one season, then a part-played season."""
    rng = np.random.default_rng(11)
    rows = []
    weeks = [(HISTORY_SEASON, week) for week in range(1, PLAYED_HISTORY_WEEKS + 1)]
    weeks += [(PREDICTION_SEASON, week) for week in range(1, TARGET_WEEK + 1)]

    for season, week in weeks:
        played = not (season == PREDICTION_SEASON and week == TARGET_WEEK)
        for home, away in pairings(week):
            home_score = float(rng.integers(13, 35)) if played else np.nan
            away_score = float(rng.integers(13, 35)) if played else np.nan
            kickoff = pd.Timestamp(f"{season}-09-04") + pd.Timedelta(weeks=week - 1)
            rows.append(
                {
                    "game_id": f"{season}_{week:02d}_{away}_{home}",
                    "season": season,
                    "week": week,
                    "gameday": kickoff.strftime("%Y-%m-%d"),
                    "game_type": "REG",
                    "home_team": home,
                    "away_team": away,
                    "home_score": home_score,
                    "away_score": away_score,
                    "spread_line": float(rng.choice([-6.5, -3.0, 1.5, 3.0, 7.0])),
                }
            )
    return pd.DataFrame(rows)


def synthetic_plays(schedule: pd.DataFrame) -> pd.DataFrame:
    """Sixty scrimmage plays per completed game, thirty to a side."""
    rng = np.random.default_rng(23)
    played = schedule.loc[schedule["home_score"].notna()]
    rows = []
    play_id = 0

    for game in played.itertuples():
        for index in range(60):
            offense = game.home_team if index % 2 == 0 else game.away_team
            defense = game.away_team if index % 2 == 0 else game.home_team
            is_pass = index % 3 != 0
            play_id += 1
            rows.append(
                {
                    "season": game.season,
                    "week": game.week,
                    "game_id": game.game_id,
                    "posteam": offense,
                    "defteam": defense,
                    "play_type": "pass" if is_pass else "run",
                    "epa": float(rng.normal(0.02, 0.9)),
                    "success": int(rng.integers(0, 2)),
                    "pass": int(is_pass),
                    "rush": int(not is_pass),
                    "qb_kneel": 0,
                    "qb_spike": 0,
                    # Descending, so seconds-per-play is well defined.
                    "game_seconds_remaining": float(3600 - index * 55),
                    "half_seconds_remaining": float(1800 - (index % 30) * 55),
                    "no_huddle": 0,
                    "fixed_drive": index // 4 + 1,
                    "play_id": play_id,
                    "score_differential": 0.0,
                }
            )
    return pd.DataFrame(rows)


def write_raw_data(directory: Path) -> dict[str, Path]:
    """Lay the universe out on disk the way the pipeline stores it.

    Finished seasons go in one play-by-play file and the season in progress in
    its own, so a weekly refresh stays small.
    """
    directory.mkdir(parents=True, exist_ok=True)
    schedule = synthetic_schedule()
    plays = synthetic_plays(schedule)

    paths = {
        "schedule": directory / "schedules.parquet",
        "historical_pbp": directory / "pbp_historical.parquet",
        "prediction_pbp": directory / f"pbp_{PREDICTION_SEASON}.parquet",
    }
    schedule.to_parquet(paths["schedule"])
    plays.loc[plays["season"] != PREDICTION_SEASON].to_parquet(paths["historical_pbp"])
    plays.loc[plays["season"] == PREDICTION_SEASON].to_parquet(paths["prediction_pbp"])
    return paths


def combined_pbp_path(directory: Path) -> Path:
    """One play-by-play file covering every season, for the feature build."""
    schedule = synthetic_schedule()
    path = directory / "pbp_all.parquet"
    synthetic_plays(schedule).to_parquet(path)
    return path
