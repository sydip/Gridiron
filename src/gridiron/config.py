"""Runtime configuration shared by data and modeling code."""

import os
import random
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RANDOM_SEED = int(os.getenv("GRIDIRON_RANDOM_SEED", "42"))


def configure_environment() -> Path:
    """Configure nflreadpy caching without replacing explicit user settings.

    Returns the resolved filesystem cache directory. Environment variables are
    set before importing nflreadpy in application code so its settings loader
    observes them.
    """
    cache_dir = Path(
        os.getenv("NFLREADPY_CACHE_DIR", PROJECT_ROOT / ".cache" / "nflreadpy")
    ).expanduser()
    if not cache_dir.is_absolute():
        cache_dir = PROJECT_ROOT / cache_dir

    os.environ.setdefault("NFLREADPY_CACHE", "filesystem")
    os.environ.setdefault("NFLREADPY_CACHE_DIR", str(cache_dir.resolve()))
    os.environ.setdefault("NFLREADPY_CACHE_DURATION", "86400")
    return cache_dir.resolve()


def seed_everything(seed: int = DEFAULT_RANDOM_SEED) -> int:
    """Seed Python and NumPy and expose the seed for child Python processes."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    return seed
