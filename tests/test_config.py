"""Tests for reproducibility and cache configuration."""

import os
import random

import numpy as np

from gridiron.config import configure_environment, seed_everything


def test_configure_environment_sets_nflreadpy_cache(monkeypatch):
    for variable in (
        "NFLREADPY_CACHE",
        "NFLREADPY_CACHE_DIR",
        "NFLREADPY_CACHE_DURATION",
    ):
        monkeypatch.delenv(variable, raising=False)

    cache_dir = configure_environment()

    assert os.environ["NFLREADPY_CACHE"] == "filesystem"
    assert os.environ["NFLREADPY_CACHE_DIR"] == str(cache_dir)
    assert os.environ["NFLREADPY_CACHE_DURATION"] == "86400"


def test_configure_environment_preserves_overrides(monkeypatch, tmp_path):
    custom_cache = tmp_path / "nfl-cache"
    monkeypatch.setenv("NFLREADPY_CACHE", "off")
    monkeypatch.setenv("NFLREADPY_CACHE_DIR", str(custom_cache))
    monkeypatch.setenv("NFLREADPY_CACHE_DURATION", "60")

    assert configure_environment() == custom_cache.resolve()
    assert os.environ["NFLREADPY_CACHE"] == "off"
    assert os.environ["NFLREADPY_CACHE_DURATION"] == "60"


def test_seed_everything_is_deterministic():
    seed_everything(123)
    first = (random.random(), np.random.random())

    seed_everything(123)
    second = (random.random(), np.random.random())

    assert first == second
    assert os.environ["PYTHONHASHSEED"] == "123"
