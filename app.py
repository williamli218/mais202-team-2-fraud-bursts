"""The demo. Runs on day one with a placeholder predictor, and switches to
your system the moment src/predict.py returns the contract (D2.6).

    python app.py                      # local
    import app; app.demo.launch(share=True)   # from a Colab cell, gives a public link

Nothing in here needs editing to make the demo show your model. What you
may want to edit for D3.4: the title, the example inputs, and how the
reasons are shown.
"""

from __future__ import annotations

import io
import json
import sys

import gradio as gr
import pandas as pd

import config
from src import data
from src.decision import DECISIONS
from src.predict import predict, validate_input, validate_output

TRACK = config.require_track()
NAME = config.TRACK_NAMES[TRACK]
FIXTURES = config.ROOT / "tests" / "fixtures"
TITLE = "Track 2: Catching fraud bursts as they happen"

EXAMPLES = [
    ("Attack caught: this charge triggers a hold", "examples/attack_caught.csv"),
    ("Still on hold: 72 h after an earlier fraud", "examples/still_on_hold.csv"),
    ("Normal card: allowed", "examples/normal_card.csv"),
]

EXAMPLE_TEXT = {label: (config.ROOT / path).read_text() for label, path in EXAMPLES
                if (config.ROOT / path).exists()}

# ---------------------------------------------------------------------------
# The placeholder, used until predict() is implemented
# ---------------------------------------------------------------------------

def placeholder_predict(x) -> dict:
    """Same contract as predict(), constant score, no model behind it."""
    validate_input(TRACK, x)
    out = {"score": 0.5, "decision": DECISIONS[TRACK][0],
           "reasons": ["placeholder: src/predict.py isn't wired yet (D2.6)"]}
    if TRACK == 6:
        out["host"] = data.extract_hostname(x)
    return validate_output(TRACK, out)


def score(x) -> tuple[dict, bool]:
    """Try the team's predict(); fall back to the placeholder. Returns (output, is_placeholder)."""
    try:
        return validate_output(TRACK, predict(x)), False
    except NotImplementedError:
        return placeholder_predict(x), True


# ---------------------------------------------------------------------------
# Example inputs and parsers, one shape per track
# ---------------------------------------------------------------------------

def example_input() -> str:
    """A valid example for the input box, taken from the fixtures."""
    if TRACK in (1, 2):
        if EXAMPLE_TEXT:
            return next(iter(EXAMPLE_TEXT.values()))
        pool, _ = data.load_track01(data_dir=FIXTURES)
        card = pool["cc_num"].iloc[0]
        rows = pool[pool["cc_num"] == card].sort_values("unix_time").head(5)
        return rows.to_csv(index=False)
    if TRACK == 3:
        pool, _ = data.load_track03(data_dir=FIXTURES)
        return json.dumps(_jsonable(pool.iloc[0].to_dict()), indent=1)
    if TRACK == 4:
        pool, _ = data.load_track04(data_dir=FIXTURES)
        return json.dumps(_jsonable(pool.iloc[0].to_dict()), indent=1)
    if TRACK == 6:
        return "https://www.example.com/login?user=1"
    if TRACK == 7:
        pool, _ = data.load_track07(data_dir=FIXTURES)
        return json.dumps(_jsonable(pool.iloc[0].to_dict()), indent=1)
    if TRACK == 9:
        pool, _, _ = data.load_track09(data_dir=FIXTURES)
        return str(int(pool["txId"].iloc[0]))
    if TRACK == 10:
        return "4"
    raise ValueError(TRACK)


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if hasattr(v, "item"):
            v = v.item()
        out[k] = v if isinstance(v, (int, float, str, bool)) or v is None else str(v)
    return out


def parse_input(text: str):
    """Turn what's in the box into the object predict() expects."""
    text = text.strip()
    if TRACK in (1, 2):
        df = pd.read_csv(io.StringIO(text), dtype={"cc_num": str})
        return df.sort_values("unix_time").reset_index(drop=True)
    if TRACK in (3, 4, 7):
        return json.loads(text)
    if TRACK == 6:
        return text
    if TRACK in (9, 10):
        return int(text)
    raise ValueError(TRACK)

def to_table(text: str) -> pd.DataFrame:
    """The input as a readable table for the demo: when, where, what, how much."""
    try:
        df = parse_input(text)
    except Exception:
        return pd.DataFrame({"note": ["Input isn't valid CSV yet."]})
    cols = [c for c in ["trans_date_trans_time", "merchant", "category", "amt"] if c in df.columns]
    out = df[cols].rename(columns={"trans_date_trans_time": "time", "amt": "amount ($)"})
    if "merchant" in out:
        out["merchant"] = out["merchant"].str.replace("^fraud_", "", regex=True)   # Sparkov prefix, not a label
    return out


INPUT_HELP = {
    1: "One card's recent transactions as CSV with a header row, oldest first.",
    2: "One card's recent transactions as CSV with a header row, oldest first.",
    3: "One transfer as a JSON object with the PaySim columns.",
    4: "One application as a JSON object with the Base.csv columns.",
    6: "A URL. The system scores its hostname.",
    7: "One transaction as a JSON object: V1 to V28, Amount, Time.",
    9: "An Elliptic transaction ID (txId).",
    10: "A month index, 0 to 7.",
}


# ---------------------------------------------------------------------------
# The interface
# ---------------------------------------------------------------------------

def run(text: str):
    try:
        x = parse_input(text)
        out, is_placeholder = score(x)
    except Exception as e:  # show the problem in the page rather than in the console
        return "error", 0.0, f"{type(e).__name__}: {e}", "", "The input couldn't be scored."
    status = ("Placeholder predictor. Implement src/predict.py (D2.6) and this page will show your system."
              if is_placeholder else "Scored by src/predict.py.")
    reasons = "\n".join(f"- {r}" for r in out["reasons"]) or "(no reasons given)"
    extra = ", ".join(f"{k}: {v}" for k, v in out.items() if k not in ("score", "decision", "reasons"))
    return out["decision"], float(out["score"]), reasons, extra, status


with gr.Blocks(title=TITLE) as demo:
    gr.Markdown(f"# {TITLE}\n\n{INPUT_HELP[TRACK]} Decisions: "
                + ", ".join(f"`{d}`" for d in DECISIONS[TRACK])
                + ". A card is held for 72 hours once a charge scores at or above the threshold.")

    with gr.Tabs():
        with gr.Tab("Table"):
            table = gr.Dataframe(value=to_table(example_input()), interactive=False, wrap=True)
        with gr.Tab("Raw input (CSV)"):
            box = gr.Textbox(label="Input", value=example_input(), lines=12)

    with gr.Row():
        for label, text in EXAMPLE_TEXT.items():
            gr.Button(label, size="sm").click(lambda t=text: t, outputs=box)
        button = gr.Button("Score", variant="primary")

    with gr.Row():
        decision = gr.Textbox(label="Decision", interactive=False)
        score_out = gr.Number(label="Score", interactive=False)
        extra = gr.Textbox(label="Other fields", interactive=False)
    reasons = gr.Markdown(label="Reasons")
    status = gr.Markdown()

    button.click(run, inputs=box, outputs=[decision, score_out, reasons, extra, status])
    box.change(to_table, inputs=box, outputs=table)


if __name__ == "__main__":
    in_colab = "google.colab" in sys.modules
    demo.launch(share=in_colab)
