"""Shared test helpers. Don't edit; the tests import from here."""

import math

import pytest

import config


def skip_unless_track(*tracks):
    """Skip the current test unless config.TRACK is one of `tracks`."""
    if config.TRACK is None:
        pytest.skip("Set TRACK in config.py")
    if config.TRACK not in tracks:
        pytest.skip(f"not a Track {config.TRACK} test")


def same(got, expected, tol=1e-9) -> bool:
    """Equality that treats nan == nan and inf == inf, for floats and dicts of floats."""
    if isinstance(expected, dict):
        if not isinstance(got, dict) or set(got) != set(expected):
            return False
        return all(same(got[k], expected[k], tol) for k in expected)
    try:
        g, e = float(got), float(expected)
    except (TypeError, ValueError):
        return False
    if math.isnan(e):
        return math.isnan(g)
    if math.isinf(e):
        return math.isinf(g) and (g > 0) == (e > 0)
    return abs(g - e) <= tol
