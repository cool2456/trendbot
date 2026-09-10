from __future__ import annotations

import pytest

from trendbot.config import load_config
from trendbot.engine.validation import synthetic_prices


@pytest.fixture(scope="session")
def cfg():
    return load_config()


@pytest.fixture
def synth(cfg):
    return synthetic_prices(cfg.universe, seed=0, n_days=2000)
