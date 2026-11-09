"""Settings shared by every notebook, test and script in this repository.

One person on the team sets TRACK and commits this file. Nothing else here
needs changing.
"""

from pathlib import Path

# Your team's track number: 1, 2, 3, 4, 6, 7, 9 or 10.
# Until this is set, the tests skip and the data loaders refuse to run.
TRACK = 2

# Every random split, shuffle and model init uses this seed, so that two
# teammates running the same notebook get the same numbers.
SEED = 42

# Folders, resolved relative to this file so they work from a notebook,
# from pytest and from app.py alike.
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"          # ignored by Git; filled by the download code
MODELS_DIR = ROOT / "models"      # small saved weights and thresholds only
EXPERIMENT_LOG = ROOT / "experiments" / "log.csv"

# The template version, from the VERSION file at the repository root. Staff
# bump that file when a fix ships; the changelog is in README.md.
TEMPLATE_VERSION = (ROOT / "VERSION").read_text().strip()

TRACK_NAMES = {
    1: "Scoring card transactions",
    2: "Catching bursts of fraud as they happen",
    3: "Account takeover on money transfers",
    4: "Fraudulent account sign-ups, with a fairness check",
    6: "Phishing links",
    7: "Ranking fraud alerts without labels",
    9: "Money laundering on a payment network",
    10: "Noticing when a fraud model goes stale",
}


def require_track():
    """Return TRACK, or raise a message that says what to do about it."""
    if TRACK not in TRACK_NAMES:
        raise ValueError(
            "Set TRACK in config.py to your team's track number "
            f"({', '.join(str(t) for t in TRACK_NAMES)}). Current value: {TRACK!r}"
        )
    return TRACK
