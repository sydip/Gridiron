"""Reusable nflverse ingestion with local Parquet caching."""

from __future__ import annotations

import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from gridiron.config import PROJECT_ROOT, configure_environment

# nflreadpy reads environment variables during import, so configure them first.
NFLREADPY_CACHE_DIR = configure_environment()

import nflreadpy as nfl  # noqa: E402
from nflreadpy.config import update_config  # noqa: E402

LOGGER = logging.getLogger(__name__)
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

PBP_REQUIRED_COLUMNS = frozenset({"season", "game_id", "game_date"})
TEAM_STATS_REQUIRED_COLUMNS = frozenset({"season", "team"})
SCHEDULE_REQUIRED_COLUMNS = frozenset(
    {"season", "game_id", "game_type", "gameday", "spread_line"}
)


class NflverseSchemaError(ValueError):
    """Raised when downloaded or cached nflverse data has an unexpected schema."""


def _configure_nflreadpy_cache() -> None:
    """Apply filesystem caching even when nflreadpy was imported previously."""
    update_config(cache_mode="filesystem", cache_dir=NFLREADPY_CACHE_DIR)
    LOGGER.debug("nflreadpy filesystem cache: %s", NFLREADPY_CACHE_DIR)


def _validate_seasons(seasons: list[int]) -> list[int]:
    """Return unique sorted seasons after validating the public API input."""
    if not seasons:
        raise ValueError("At least one season is required.")
    if any(type(season) is not int for season in seasons):
        raise TypeError("Seasons must be integers.")
    return sorted(set(seasons))


def _to_pandas(frame: Any, source: str) -> pd.DataFrame:
    """Convert nflreadpy's Polars result to pandas with a useful error."""
    if isinstance(frame, pd.DataFrame):
        return frame.copy()
    to_pandas = getattr(frame, "to_pandas", None)
    if not callable(to_pandas):
        raise TypeError(
            f"{source} returned {type(frame).__name__}; expected a Polars or "
            "pandas DataFrame."
        )
    converted = to_pandas()
    if not isinstance(converted, pd.DataFrame):
        raise TypeError(f"{source}.to_pandas() did not return a pandas DataFrame.")
    return converted


def _require_columns(df: pd.DataFrame, required: frozenset[str], source: str) -> None:
    missing = sorted(required.difference(df.columns))
    if missing:
        raise NflverseSchemaError(
            f"{source} is missing required column(s): {', '.join(missing)}. "
            "The nflverse schema may have changed."
        )


def _require_historical_seasons(
    df: pd.DataFrame, seasons: list[int], source: str
) -> None:
    available = set(pd.to_numeric(df["season"], errors="coerce").dropna().astype(int))
    missing = sorted(set(seasons).difference(available))
    if missing:
        raise ValueError(f"{source} has no rows for requested season(s): {missing}.")


def _extract(source: str, loader: Any, seasons: list[int]) -> pd.DataFrame:
    _configure_nflreadpy_cache()
    started = time.perf_counter()
    extracted_at = datetime.now(UTC)
    LOGGER.info(
        "Starting %s extraction at %s for seasons %s",
        source,
        extracted_at.isoformat(),
        seasons,
    )
    try:
        frame = loader(seasons)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to download {source} for seasons {seasons}: {exc}"
        ) from exc
    df = _to_pandas(frame, source)
    LOGGER.info(
        "Finished %s extraction: %d rows in %.2f seconds",
        source,
        len(df),
        time.perf_counter() - started,
    )
    return df


def load_historical_pbp(seasons: list[int]) -> pd.DataFrame:
    """Download play-by-play data and return it as a pandas DataFrame."""
    seasons = _validate_seasons(seasons)
    df = _extract("play-by-play", nfl.load_pbp, seasons)
    _require_columns(df, PBP_REQUIRED_COLUMNS, "play-by-play data")
    _require_historical_seasons(df, seasons, "play-by-play data")
    return df


def load_team_stats(seasons: list[int]) -> pd.DataFrame:
    """Download weekly team statistics and return a pandas DataFrame."""
    seasons = _validate_seasons(seasons)
    df = _extract("team stats", nfl.load_team_stats, seasons)
    _require_columns(df, TEAM_STATS_REQUIRED_COLUMNS, "team stats data")
    _require_historical_seasons(df, seasons, "team stats data")
    return df


def load_schedules(seasons: list[int]) -> pd.DataFrame:
    """Download schedules, allowing future seasons to be partially available."""
    seasons = _validate_seasons(seasons)
    df = _extract("schedules", nfl.load_schedules, seasons)
    _require_columns(df, SCHEDULE_REQUIRED_COLUMNS, "schedule data")
    available = set(pd.to_numeric(df["season"], errors="coerce").dropna().astype(int))
    missing = sorted(set(seasons).difference(available))
    if missing:
        LOGGER.warning(
            "Schedule data is not yet available for requested season(s): %s",
            missing,
        )
    return df


def save_raw_data(df: pd.DataFrame, path: Path) -> None:
    """Atomically save a pandas DataFrame as Parquet."""
    path = Path(path)
    if path.suffix.lower() != ".parquet":
        raise ValueError(f"Raw data path must end in .parquet: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(".tmp.parquet")
    df.to_parquet(temporary_path, index=False)
    temporary_path.replace(path)
    LOGGER.info("Saved %d rows to %s", len(df), path)


def cache_paths(
    seasons: list[int],
    prediction_season: int,
    directory: Path | None = None,
) -> dict[str, Path]:
    """Where each raw dataset is cached, for a given season window.

    Public because the download command needs to say what it is about to
    write before it writes it. ``directory`` defaults to the module-level raw
    data directory, resolved at call time so it stays overridable.
    """
    root = Path(directory) if directory is not None else RAW_DATA_DIR
    first_season, last_season = min(seasons), max(seasons)
    return {
        "pbp": root / f"pbp_{first_season}_{last_season}.parquet",
        "team_stats": root / f"team_stats_{first_season}_{last_season}.parquet",
        "schedules": root / f"schedules_{first_season}_{prediction_season}.parquet",
        "manifest": root / f"manifest_{first_season}_{prediction_season}.json",
    }


def _load_cached(path: Path, required: frozenset[str], source: str) -> pd.DataFrame:
    LOGGER.info("Loading cached %s from %s", source, path)
    try:
        df = pd.read_parquet(path)
    except Exception as exc:
        raise RuntimeError(f"Failed to read cached {source} at {path}: {exc}") from exc
    _require_columns(df, required, f"cached {source}")
    return df


def _date_range(df: pd.DataFrame, column: str | None) -> dict[str, str | None]:
    if column is None or column not in df.columns or df.empty:
        return {"earliest_date": None, "latest_date": None}
    dates = pd.to_datetime(df[column], errors="coerce").dropna()
    if dates.empty:
        return {"earliest_date": None, "latest_date": None}
    return {
        "earliest_date": dates.min().date().isoformat(),
        "latest_date": dates.max().date().isoformat(),
    }


def _write_manifest(data: dict[str, pd.DataFrame], paths: dict[str, Path]) -> None:
    date_columns = {"pbp": "game_date", "team_stats": None, "schedules": "gameday"}
    manifest: dict[str, Any] = {"recorded_at_utc": datetime.now(UTC).isoformat()}
    for name, df in data.items():
        seasons = sorted(
            set(pd.to_numeric(df["season"], errors="coerce").dropna().astype(int))
        )
        manifest[name] = {
            "path": str(paths[name]),
            "row_count": len(df),
            "seasons": seasons,
            **_date_range(df, date_columns[name]),
        }
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info("Recorded extraction metadata in %s", paths["manifest"])


def load_or_download_data(
    seasons: list[int],
    prediction_season: int,
    force_refresh: bool = False,
    directory: Path | None = None,
) -> dict[str, pd.DataFrame]:
    """Load cached Parquet datasets or download and persist missing datasets."""
    seasons = _validate_seasons(seasons)
    if type(prediction_season) is not int:
        raise TypeError("prediction_season must be an integer.")
    schedule_seasons = sorted(set([*seasons, prediction_season]))
    paths = cache_paths(seasons, prediction_season, directory)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    specifications = {
        "pbp": (PBP_REQUIRED_COLUMNS, load_historical_pbp, seasons),
        "team_stats": (TEAM_STATS_REQUIRED_COLUMNS, load_team_stats, seasons),
        "schedules": (SCHEDULE_REQUIRED_COLUMNS, load_schedules, schedule_seasons),
    }
    data: dict[str, pd.DataFrame] = {}
    downloaded = False
    for name, (required, loader, requested_seasons) in specifications.items():
        path = paths[name]
        if path.exists() and not force_refresh:
            data[name] = _load_cached(path, required, name)
        else:
            data[name] = loader(requested_seasons)
            save_raw_data(data[name], path)
            downloaded = True

    _require_historical_seasons(data["pbp"], seasons, "play-by-play data")
    _require_historical_seasons(data["team_stats"], seasons, "team stats data")
    if downloaded or not paths["manifest"].exists():
        _write_manifest(data, paths)
    return data
