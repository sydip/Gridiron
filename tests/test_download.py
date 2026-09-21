"""Tests for nflverse ingestion without network access."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from gridiron.data import download


def _pbp() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season": [2024, 2025],
            "game_id": ["2024_01_A_B", "2025_01_A_B"],
            "game_date": ["2024-09-01", "2025-09-01"],
        }
    )


def _team_stats() -> pd.DataFrame:
    return pd.DataFrame({"season": [2024, 2025], "team": ["A", "A"]})


def _schedules(include_prediction_season: bool = True) -> pd.DataFrame:
    seasons = [2024, 2025, 2026] if include_prediction_season else [2024, 2025]
    return pd.DataFrame(
        {
            "season": seasons,
            "game_id": [f"{season}_01_A_B" for season in seasons],
            "game_type": ["REG"] * len(seasons),
            "gameday": [f"{season}-09-01" for season in seasons],
            "spread_line": [1.5, None, None][: len(seasons)],
        }
    )


def test_load_or_download_uses_parquet_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "RAW_DATA_DIR", tmp_path)
    calls = {"pbp": 0, "team_stats": 0, "schedules": 0}

    def loader(name, frame):
        def load(_seasons):
            calls[name] += 1
            return frame

        return load

    monkeypatch.setattr(download, "load_historical_pbp", loader("pbp", _pbp()))
    monkeypatch.setattr(
        download, "load_team_stats", loader("team_stats", _team_stats())
    )
    monkeypatch.setattr(download, "load_schedules", loader("schedules", _schedules()))

    first = download.load_or_download_data([2024, 2025], 2026)
    second = download.load_or_download_data([2024, 2025], 2026)

    assert calls == {"pbp": 1, "team_stats": 1, "schedules": 1}
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])
    manifest = json.loads(next(tmp_path.glob("manifest_*.json")).read_text())
    assert manifest["pbp"]["row_count"] == 2
    assert manifest["schedules"]["latest_date"] == "2026-09-01"


def test_force_refresh_downloads_again(monkeypatch, tmp_path):
    monkeypatch.setattr(download, "RAW_DATA_DIR", tmp_path)
    calls = {"count": 0}

    def count(frame):
        def load(_seasons):
            calls["count"] += 1
            return frame

        return load

    monkeypatch.setattr(download, "load_historical_pbp", count(_pbp()))
    monkeypatch.setattr(download, "load_team_stats", count(_team_stats()))
    monkeypatch.setattr(download, "load_schedules", count(_schedules()))

    download.load_or_download_data([2024, 2025], 2026)
    download.load_or_download_data([2024, 2025], 2026, force_refresh=True)

    assert calls["count"] == 6


def test_partial_prediction_schedule_is_allowed(monkeypatch):
    monkeypatch.setattr(
        download.nfl,
        "load_schedules",
        lambda _seasons: _schedules(include_prediction_season=False),
    )

    result = download.load_schedules([2024, 2025, 2026])

    assert set(result["season"]) == {2024, 2025}


def test_missing_required_column_has_clear_error(monkeypatch):
    monkeypatch.setattr(
        download.nfl,
        "load_pbp",
        lambda _seasons: _pbp().drop(columns="game_id"),
    )

    with pytest.raises(download.NflverseSchemaError, match="game_id.*schema"):
        download.load_historical_pbp([2024, 2025])
