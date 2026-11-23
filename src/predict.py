"""predict(): one function that the demo and the evaluation both call.

The contract is fixed so that app.py, tests/test_predict.py and your own
evaluation code all agree on what goes in and what comes out. The model
part is yours (D2.6).

Output, for every track:

    {"score": float, "decision": str, "reasons": list[str]}

plus "host" (the hostname that was scored) on Track 6, and optionally
"gate" (which rule fired) on Track 1. "decision" must be one of
decision.DECISIONS[track]. "reasons" is for the person reading the demo:
short strings, most important first, and an empty list is allowed.

Input, by track:

    1, 2   a DataFrame of one card's recent transactions, oldest first, with
           the Sparkov columns (at least cc_num, amt, unix_time)
    3      one transfer as a dict of PaySim columns
    4      one application as a dict of Base.csv columns
    6      a URL as a string; predict() reduces it to the hostname
    7      one transaction as a dict with V1..V28, Amount and Time
    9      an Elliptic txId (int); predict() looks it up in the data
    10     a month index (int); score is the drift measure for that month
"""

from __future__ import annotations

import numpy as np
import pandas as pd

import config
from src import data
from src.decision import DECISIONS

import json
import torch
from src.models import FraudLSTM

from src import features

REQUIRED_KEYS = ("score", "decision", "reasons")

# The minimum each dict-shaped input has to carry. The loaders have the full
# column lists; these are the ones a model can't do without.
_REQUIRED_FIELDS = {
    3: ("step", "type", "amount"),
    4: ("month", "customer_age"),
    7: tuple(f"V{i}" for i in range(1, 29)) + ("Amount", "Time"),
}


def validate_input(track: int, x) -> None:
    """Raise TypeError or ValueError if x isn't the right shape for the track."""
    if track in (1, 2):
        if not isinstance(x, pd.DataFrame):
            raise TypeError(f"Track {track} expects a DataFrame of one card's transactions, got {type(x).__name__}")
        if len(x) == 0:
            raise ValueError("Empty DataFrame: predict() needs at least one transaction")
        missing = {"cc_num", "amt", "unix_time"} - set(x.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")
        if x["cc_num"].nunique() != 1:
            raise ValueError("predict() scores one card at a time; the rows have more than one cc_num")
        if not x["unix_time"].is_monotonic_increasing:
            raise ValueError("Rows must be oldest first (sort by unix_time)")
    elif track in (3, 4, 7):
        if not isinstance(x, dict):
            raise TypeError(f"Track {track} expects a dict of one row's columns, got {type(x).__name__}")
        missing = [f for f in _REQUIRED_FIELDS[track] if f not in x]
        if missing:
            raise ValueError(f"Missing fields: {missing}")
    elif track == 6:
        if not isinstance(x, str) or not x.strip():
            raise TypeError("Track 6 expects a non-empty URL string")
    elif track in (9, 10):
        if isinstance(x, bool) or not isinstance(x, (int, np.integer)):
            raise TypeError(f"Track {track} expects an int, got {type(x).__name__}")
        if track == 10 and not 0 <= int(x) <= 7:
            raise ValueError("Month index must be between 0 and 7")
    else:
        raise ValueError(f"Unknown track {track}")


def validate_output(track: int, out) -> dict:
    """Raise if `out` doesn't meet the contract; return it unchanged if it does.

    Used by app.py and tests/test_predict.py. Call it on your own output
    while you're wiring things up: it tells you what's off.
    """
    if not isinstance(out, dict):
        raise TypeError(f"predict() must return a dict, got {type(out).__name__}")
    missing = [k for k in REQUIRED_KEYS if k not in out]
    if missing:
        raise ValueError(f"predict() output is missing keys: {missing}")
    score = out["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float, np.integer, np.floating)):
        raise TypeError(f"'score' must be a float, got {type(score).__name__}")
    if not np.isfinite(float(score)):
        raise ValueError(f"'score' must be a finite number, got {score}")
    if out["decision"] not in DECISIONS[track]:
        raise ValueError(f"'decision' must be one of {DECISIONS[track]}, got {out['decision']!r}")
    reasons = out["reasons"]
    if not isinstance(reasons, list) or not all(isinstance(r, str) for r in reasons):
        raise TypeError("'reasons' must be a list of strings")
    if track == 6 and not isinstance(out.get("host"), str):
        raise ValueError("Track 6 output must include 'host', the hostname that was scored")
    return out


def load_model():
  """Load the trained LSTM, its input settings and the hold threshold from models/."""
  folder = config.ROOT / "models"
  with open(folder / "lstm_n10.json") as f:
    settings = json.load(f)
  with open(folder / "thresholds.json") as f:
    thresholds = json.load(f)

  lstm = FraudLSTM(len(settings["features"]), settings["hidden_dim"])
  lstm.load_state_dict(torch.load(folder / "lstm_n10.pt", map_location="cpu"))
  lstm.eval()

  return {
    "lstm": lstm,
    "features": settings["features"],
    "n": settings["n"],
    "mean": np.array(settings["mean"], dtype="float32"),
    "std": np.array(settings["std"], dtype="float32"),
    "threshold": thresholds["hold"],
  }


_MODEL = None

def _reasons(x, f):
  """Short, plain-language flags about the newest transaction, most important first.
  They describe what the window looks like; they are not the LSTM's internals."""
  reasons = []
  t = pd.to_datetime(x["trans_date_trans_time"])
  last_amt = float(x["amt"].iloc[-1])
  prev = x["amt"].iloc[:-1].astype(float)

  n_24h = int((t >= t.iloc[-1] - pd.Timedelta(hours=24)).sum())
  if n_24h >= 4:
    reasons.append(f"{n_24h} charges in the last 24 hours")
  if len(prev) and last_amt > 3 * prev.mean():
    reasons.append(f"${last_amt:.0f} is over 3x this card's recent average")
  if f["log_seconds_since_prev"].iloc[-1] < np.log1p(10 * 60):
    reasons.append("within 10 minutes of the previous charge")
  if f["is_night"].iloc[-1] == 1:
    reasons.append("late-night charge (22:00-04:59)")
  cat = str(x["category"].iloc[-1])
  if cat.endswith("_net"):
    reasons.append(f"online purchase ({cat})")
  return reasons

def predict(x, model=None) -> dict:
    """Score one input and turn the score into a decision. See the module
    docstring for the contract.

    model: pass an object to use instead of load_model(). Handy for tests.
    """
    track = config.require_track()
    validate_input(track, x)

    global _MODEL
    if model is None:
        if _MODEL is None:
            _MODEL = load_model()
        model = _MODEL

    if track == 6:
        host = data.extract_hostname(x)  # the model sees this, not the URL; return it as "host"

    # TODO D2.6: build the features for x (src/features.py), score them with
    # model, turn the score into a decision with your thresholds
    # (src/decision.py), fill in reasons, and return the dict. Run it through
    # validate_output(track, out) before returning.
        # same per-transaction columns the LSTM was trained on
    f = x.join(features.build_features(x))
    f["log_amt"] = np.log1p(f["amt"].astype("float32"))

    # last n transactions -> scaled -> padded at the start, like make_sequences
    rows = f[model["features"]].to_numpy(dtype="float32")[-model["n"]:]
    rows = (rows - model["mean"]) / model["std"]
    window = np.zeros((model["n"], rows.shape[1]), dtype="float32")
    window[-len(rows):] = rows

    with torch.no_grad():
      logit = model["lstm"](torch.from_numpy(window[None]))   # batch of 1
    score = float(torch.sigmoid(logit).item())

    out = {
      "score": score,
      "decision": "hold" if score >= model["threshold"] else "allow",
      "reasons": _reasons(x, f),
    }
    return validate_output(track, out)
