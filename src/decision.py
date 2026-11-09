"""The cost of each kind of mistake, and the thresholds that follow from it.

The table format is provided. The numbers are yours (D1.6), and the rules
that turn scores into decisions are yours too (D3.1).
"""

import config

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
    # (truth, decision): cost
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


def choose_thresholds(y_true, scores, amounts=None):
    """Pick the thresholds (one, or two for tracks with a middle decision)
    that minimise total cost under COSTS on the validation set.

    Returns a dict like {"block": 0.91} or {"verify": 0.4, "decline": 0.8},
    which predict() reads to turn a score into a decision.

    TODO D3.1: implement. Sweep candidate thresholds, compute total cost at
    each with COSTS (and the amounts, where a cost is "amount"), keep the
    cheapest. Save the result to models/thresholds.json.
    """
    raise NotImplementedError("TODO D3.1: implement choose_thresholds in src/decision.py")
