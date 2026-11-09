"""Checks that predict() honours the contract in src/predict.py.

Before D2.6 the model part of predict() raises NotImplementedError and the
contract test skips, so this file doesn't turn CI red for three weeks. Once
your model is wired in, it runs for real: keys, types and an allowed
decision for your track. The demo (app.py) relies on exactly the same
contract, so green here means the demo will show your system.

The input-validation tests run from day one; that code is provided.
"""

import numpy as np
import pandas as pd
import pytest

import config
from conftest import skip_unless_track
from src import data, predict
from src.decision import DECISIONS

FIXTURES = config.ROOT / "tests" / "fixtures"


def sample_input(track):
    """One valid input for the track, built from the fixtures."""
    if track in (1, 2):
        pool, _ = data.load_track01(data_dir=FIXTURES)
        card = pool["cc_num"].iloc[0]
        return pool[pool["cc_num"] == card].sort_values("unix_time").head(5)
    if track == 3:
        pool, _ = data.load_track03(data_dir=FIXTURES)
        return pool.iloc[0].to_dict()
    if track == 4:
        pool, _ = data.load_track04(data_dir=FIXTURES)
        return pool.iloc[0].to_dict()
    if track == 6:
        return "https://www.example.com/login?user=1"
    if track == 7:
        pool, _ = data.load_track07(data_dir=FIXTURES)
        return pool.iloc[0].to_dict()
    if track == 9:
        pool, _, _ = data.load_track09(data_dir=FIXTURES)
        return int(pool["txId"].iloc[0])
    if track == 10:
        return 4
    raise ValueError(track)


def bad_inputs(track):
    """Inputs predict() must reject, with the exception it should raise."""
    if track in (1, 2):
        pool, _ = data.load_track01(data_dir=FIXTURES)
        two_cards = pool.groupby("cc_num").head(1).head(2)
        return [("a dict instead of a DataFrame", {"amt": 1.0}, TypeError),
                ("an empty DataFrame", pool.iloc[0:0], ValueError),
                ("rows from two cards", two_cards, ValueError)]
    if track in (3, 4, 7):
        return [("a string instead of a dict", "nope", TypeError),
                ("a dict with the fields missing", {"foo": 1}, ValueError)]
    if track == 6:
        return [("an empty string", "", TypeError), ("a number", 42, TypeError)]
    if track == 9:
        return [("a string", "12345", TypeError), ("a float", 1.5, TypeError)]
    if track == 10:
        return [("a string", "3", TypeError), ("a month that doesn't exist", 12, ValueError)]
    return []


def test_bad_inputs_are_rejected():
    skip_unless_track(*DECISIONS)
    for label, x, exc in bad_inputs(config.TRACK):
        with pytest.raises(exc):
            predict.validate_input(config.TRACK, x)


def test_contract():
    skip_unless_track(*DECISIONS)
    track = config.TRACK
    x = sample_input(track)
    try:
        out = predict.predict(x)
    except NotImplementedError:
        pytest.skip("predict() not wired yet (D2.6)")
    except FileNotFoundError as e:
        pytest.skip(f"model files not found in this checkout: {e}")
    predict.validate_output(track, out)
    assert isinstance(out["score"], (float, np.floating, int, np.integer))
    assert out["decision"] in DECISIONS[track]
    assert isinstance(out["reasons"], list)
    if track == 6:
        assert out["host"] == "example.com"


def test_validate_output_catches_mistakes():
    """The checker itself, so a wrong output is caught before the demo shows it."""
    skip_unless_track(*DECISIONS)
    track = config.TRACK
    good = {"score": 0.5, "decision": DECISIONS[track][0], "reasons": []}
    if track == 6:
        good["host"] = "example.com"
    predict.validate_output(track, good)
    with pytest.raises((TypeError, ValueError)):
        predict.validate_output(track, {**good, "decision": "maybe"})
    with pytest.raises((TypeError, ValueError)):
        predict.validate_output(track, {**good, "score": "high"})
    with pytest.raises((TypeError, ValueError)):
        predict.validate_output(track, {**good, "reasons": "because"})
    with pytest.raises((TypeError, ValueError)):
        predict.validate_output(track, {k: v for k, v in good.items() if k != "score"})
