"""Shared fixtures for the offline test suite.

Nothing here touches the network. Price panels come from
:func:`trendbot.engine.validation.synthetic_prices`, which is pure numpy.
"""

from __future__ import annotations

import pytest

from trendbot.config import load_config
from trendbot.engine.validation import synthetic_prices


@pytest.fixture(scope="session")
def cfg():
    """The frozen configuration parsed out of PREREGISTRATION.md."""
    return load_config()


@pytest.fixture
def synth(cfg):
    """A driftless synthetic price panel over the pre-registered universe.

    Function-scoped deliberately: it costs ~12 ms to build and tests are free to
    slice, copy and perturb it without leaking state into one another.
    """
    return synthetic_prices(cfg.universe, seed=0, n_days=2000)
