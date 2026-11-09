"""Your validation split (D1.2). One function, imported everywhere, so that
every teammate, every notebook and every model uses the same split.

The loader gives you the train pool and the fixed test set. This file
carves a validation set out of the pool. Two patterns:

  Time-based (Tracks 1, 2, 3, 4, 7, 9, 10): validation is the latest part of
  the training period, so the model is judged on data from after what it
  learned from. Pick a cut on the time column (a date, a step, a month, a
  time step) and say in the report why you cut there: enough fraud on the
  validation side to measure, enough training before it to learn from.

  Stratified random (Track 6): no time, so a random split with the phishing
  share kept equal in both parts (sklearn.model_selection.train_test_split
  with stratify=). Same seed every run: config.SEED.

Either way, check_split() below runs the two checks the deliverable asks
for, and tests/test_leakage.py runs them on your split once this is written.
"""

import pandas as pd

import config
from src import data


def make_validation_split(pool: pd.DataFrame):
    """Return (train, valid), both DataFrames, carved from the train pool.

    TODO D1.2: implement, then run `pytest tests/test_leakage.py`. The
    checks there (and in check_split) are the ones your report describes.
    """
    raise NotImplementedError("TODO D1.2: implement make_validation_split in src/split.py")


def check_split(train: pd.DataFrame, valid: pd.DataFrame, id_col=None, time_col=None) -> None:
    """The two leakage checks, as asserts with readable messages.

    id_col:   a column whose values must not appear in both parts
              (a transaction id, a hostname). None to skip.
    time_col: a column where everything in train must come before everything
              in valid. None to skip.
    Defaults for your track are in data.TRACK_COLUMNS.
    """
    assert len(train) > 0 and len(valid) > 0, "one of the parts is empty"
    if id_col is not None:
        shared = set(train[id_col]) & set(valid[id_col])
        assert not shared, f"{len(shared)} values of {id_col} are in both train and valid, e.g. {sorted(shared)[:3]}"
    if time_col is not None:
        assert train[time_col].max() < valid[time_col].min(), (
            f"train runs to {time_col}={train[time_col].max()} but valid starts at {valid[time_col].min()}; "
            "validation must lie entirely after training"
        )


def default_columns():
    """(id_col, time_col) for the track in config.py, for check_split()."""
    cols = data.TRACK_COLUMNS[config.require_track()]
    return cols["id"], cols["time"]
