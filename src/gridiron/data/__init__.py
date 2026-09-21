"""Data access utilities."""

from gridiron.data.clean import clean_schedule, clean_schedules
from gridiron.data.download import (
    load_historical_pbp,
    load_or_download_data,
    load_schedules,
    load_team_stats,
    save_raw_data,
)
from gridiron.data.team_names import CANONICAL_TEAMS, TEAM_MAP
from gridiron.data.validate import validate_schedule

__all__ = [
    "CANONICAL_TEAMS",
    "TEAM_MAP",
    "clean_schedule",
    "clean_schedules",
    "load_historical_pbp",
    "load_or_download_data",
    "load_schedules",
    "load_team_stats",
    "save_raw_data",
    "validate_schedule",
]
