"""One worked example of a leakage test, on the provided test split.

It checks two things about the split your loader returns, using the tiny
fixtures instead of the real data:

  1. no identifier (card, hostname, transaction) appears on both sides
  2. where there's a time column, everything in the train pool happens
     before everything in the test set

The last test runs the same two checks on *your* validation split, the one
in src/split.py. It skips until make_validation_split() is written and
fails if the split leaks. That's the done signal for D1.2.
"""

import pytest

import config
from conftest import skip_unless_track
from src import data, split

FIXTURES = config.ROOT / "tests" / "fixtures"


def _pool_and_test(track):
    out = data.load_track(track, data_dir=FIXTURES)
    if track == 10:
        pool, deployment = out
        # For the month loop, "test" is every deployment month together.
        import pandas as pd
        return pool, pd.concat(deployment.values(), ignore_index=True)
    return out[0], out[1]


def test_no_identifier_in_both_splits():
    skip_unless_track(*data.LOADERS)
    track = config.TRACK
    id_col = data.TRACK_COLUMNS[track]["id"]
    if id_col is None:
        pytest.skip(f"Track {track} has no row identifier to check")
    pool, test = _pool_and_test(track)
    shared = set(pool[id_col]) & set(test[id_col])
    assert not shared, f"{len(shared)} values of {id_col} are in both the train pool and the test set, e.g. {sorted(shared)[:3]}"


def test_train_pool_ends_before_test_begins():
    skip_unless_track(*data.LOADERS)
    track = config.TRACK
    time_col = data.TRACK_COLUMNS[track]["time"]
    if time_col is None:
        pytest.skip(f"Track {track} has no time column; the split is random and stratified")
    pool, test = _pool_and_test(track)
    assert pool[time_col].max() < test[time_col].min(), (
        f"Train pool runs to {time_col}={pool[time_col].max()} but the test set starts at "
        f"{test[time_col].min()}; the test set must lie entirely in the future"
    )


def test_fraud_rate_is_in_range():
    """Not a leakage check, a sanity check: a split with no fraud in one part
    can't be scored. The fixtures are tiny, so the bar is low."""
    skip_unless_track(*data.LOADERS)
    track = config.TRACK
    label = data.TRACK_COLUMNS[track]["label"]
    pool, test = _pool_and_test(track)
    assert (pool[label] == 1).sum() >= 1, "no positives in the train pool fixture"


def test_validation_split_does_not_leak():
    """Your split (D1.2): no identifier in both parts, train before valid."""
    skip_unless_track(*data.LOADERS)
    track = config.TRACK
    pool, _ = _pool_and_test(track)
    try:
        train, valid = split.make_validation_split(pool)
    except NotImplementedError:
        pytest.skip("make_validation_split() not written yet (D1.2)")
    id_col, time_col = data.TRACK_COLUMNS[track]["id"], data.TRACK_COLUMNS[track]["time"]
    split.check_split(train, valid, id_col=id_col, time_col=time_col)
    assert len(train) + len(valid) <= len(pool), "train and valid together are bigger than the pool: rows are duplicated"
