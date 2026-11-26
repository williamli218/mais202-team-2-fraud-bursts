"""The cost of each kind of mistake, and the thresholds that follow from it.

The table format is provided. The numbers are yours (D1.6), and the rules
that turn scores into decisions are yours too (D3.1).
"""

import config
import numpy as np
import pandas as pd
import json

# The decisions each track's system can make. predict() must return one of
# these for your track; test_predict.py checks it.
DECISIONS = {
    1: ("allow", "block"),
    2: ("allow", "hold"),
    3: ("release", "hold"),
    4: ("approve", "verify", "decline"),
    6: ("open", "warn", "block"),
    7: ("queue", "skip"),
    9: ("escalate", "watch", "clear"),
    10: ("keep", "alert", "retrain"),
}

# What was actually true, per track. "fraud" covers phishing and illicit too.
TRUTHS = ("legit", "fraud")

# One entry per (truth, decision) pair, in dollars, for your track. A cost
# can be a number or the string "amount", meaning the transaction's own
# amount (a missed fraud usually costs what it took).
#
# Fill in every pair for your track. The two defaults the deliverable
# document gives: a missed fraud costs its amount, and a wrongly blocked
# customer costs $15, unless you can argue for other numbers. Tracks with a
# middle decision (verify, warn, watch, alert) have to price that step too:
# a manual check isn't free, and a warning that's ignored isn't a block.
#
# TODO D1.6: replace every None below with a number or "amount", and justify
# each one in the report.
COSTS = {
    ("legit", "allow"): 0,
    ("legit", "hold"): 15,
    ("fraud", "allow"): "amount",
    ("fraud", "hold"): 0,
}


def cost_table_template(track: int | None = None) -> dict:
    """Every (truth, decision) pair for a track, with None as the cost.

    Copy the printed result into COSTS above and fill in the numbers.
    """
    track = config.require_track() if track is None else track
    return {(truth, decision): None for truth in TRUTHS for decision in DECISIONS[track]}


def check_costs(track: int | None = None) -> None:
    """Raise if COSTS is missing a pair or still has a None in it."""
    track = config.require_track() if track is None else track
    missing = [pair for pair in cost_table_template(track) if pair not in COSTS]
    empty = [pair for pair, cost in COSTS.items() if cost is None]
    if missing or empty:
        raise ValueError(
            f"COSTS in src/decision.py is incomplete (D1.6). Missing: {missing}; still None: {empty}"
        )


def _cost_value(cost, amounts):
  """
  A COSTS entry as one number per row: the row's amount, or a fixed number.
  """
  return np.asarray(amounts, dtype=float) if cost == "amount" else np.full(len(amounts), float(cost))


def cost_curve(y_true, scores, amounts, card_ids):
  """
  Total cost on validation at every candidate threshold, for a card-hold system.

  A card is held from its first transaction with score >= t; that transaction
  and everything after it is blocked. Costs from COSTS: each honest card held
  costs ("legit", "hold"), each fraud charge that gets through before the hold
  costs ("fraud", "allow"). Rows must be in time order within each card.
  """
  y = np.asarray(y_true).astype(int)
  s = np.asarray(scores, dtype=float)
  amt = np.asarray(amounts, dtype=float)
  df = pd.DataFrame({"card": np.asarray(card_ids), "y": y, "s": s})

  run_max = df.groupby("card", sort=False)["s"].cummax().to_numpy()
  per_card = df.groupby("card", sort=False).agg(attacked=("y", "max"), max_score=("s", "max"))
  legit_max = np.sort(per_card.loc[per_card["attacked"] == 0, "max_score"].to_numpy())

  fraud = y == 1
  order = np.argsort(run_max[fraud], kind="stable")
  f_run = run_max[fraud][order]
  f_cost = _cost_value(COSTS[("fraud", "allow")], amt[fraud])[order]
  missed_cum = np.concatenate([[0.0], np.cumsum(f_cost)])

  cand = np.append(np.unique(s), np.inf)
  n_missed = np.searchsorted(f_run, cand, side="left")
  n_held = len(legit_max) - np.searchsorted(legit_max, cand, side="left")

  return pd.DataFrame({
    "threshold": cand,
    "legit_cards_held": n_held,
    "legit_hold_rate": n_held / max(len(legit_max), 1),
    "frauds_missed": n_missed,
    "missed_fraud_cost": missed_cum[n_missed],
    "total_cost": float(COSTS[("legit", "hold")]) * n_held + missed_cum[n_missed],
  })


def choose_thresholds(y_true, scores, amounts=None, card_ids=None,
                      max_legit_hold_rate=0.01, hold_hours=72, path=None):
  """
  D3.1: the cheapest hold threshold on validation that holds at most
  max_legit_hold_rate of legitimate cards (the track's target).
  Costs are computed with cost_curve (card held until the end of the data);
  the 72h hold period is checked separately with simulate_holds/evaluate.
  Returns {"hold": t, "hold_hours": hold_hours}; saves it to `path` if given.
  """
  curve = cost_curve(y_true, scores, amounts, card_ids)
  allowed = curve[curve["legit_hold_rate"] <= max_legit_hold_rate]
  best = allowed.loc[allowed["total_cost"].idxmin()]
  out = {"hold": float(best["threshold"]), "hold_hours": hold_hours}
  if path:
    with open(path, "w") as f:
      json.dump(out, f, indent=2)
  return out

def simulate_holds(scores, card_ids, times, threshold, hold_hours=72):
  """
  Replay the decision layer in time order: a transaction scoring >= threshold
  holds its card for hold_hours, blocking every charge in that window; after
  that the card is released and scored again (and re-held if needed).
  Rows must be in time order within each card. Returns "allow"/"hold" per row.
  """
  s = np.asarray(scores, dtype=float)
  cards = np.asarray(card_ids)
  t = pd.to_datetime(pd.Series(times)).to_numpy()
  hold = np.timedelta64(int(hold_hours * 3600), "s")
  decision = np.empty(len(s), dtype=object)
  held_until = {}
  for i in range(len(s)):
    c = cards[i]
    until = held_until.get(c)
    if until is not None and t[i] < until:
      decision[i] = "hold"
    elif s[i] >= threshold:
      decision[i] = "hold"
      held_until[c] = t[i] + hold
    else:
      decision[i] = "allow"
  return decision

def evaluate(scores, card_ids, labels, amounts, times, threshold, hold_hours=72):
  """
  Replay a fixed threshold + hold period and report the numbers for the
  D3 comparison table: median frauds before hold, honest cards held, fraud
  charges through and total cost under COSTS. Rows in time order within each card.
  """
  dec = simulate_holds(scores, card_ids, times, threshold, hold_hours)
  y = np.asarray(labels).astype(int)
  amt = np.asarray(amounts, dtype=float)
  slipped = (y == 1) & (dec == "allow")
  g = pd.DataFrame({"card": np.asarray(card_ids), "y": y,
                    "slip": slipped, "held": dec == "hold"}).groupby("card", sort=False)
  attacked = g["y"].max() == 1
  legit_held = int((~attacked & g["held"].max()).sum())
  missed_cost = _cost_value(COSTS[("fraud", "allow")], amt[slipped]).sum()
  return {
    "median frauds before hold": float(g["slip"].sum()[attacked].median()),
    "honest cards held": legit_held,
    "honest hold rate": round(legit_held / max(int((~attacked).sum()), 1), 4),
    "fraud charges through": int(slipped.sum()),
    "total cost $": round(float(COSTS[("legit", "hold")]) * legit_held + missed_cost),
  }
