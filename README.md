# Catching fraud bursts as they happen (MAIS 202, Track 2, Team 02)

When a credit card is compromised, the fraudster usually makes a **burst** of charges.
This project watches each card as a **sequence** of transactions and **holds the card**
as early as possible in an attack, while wrongly holding at most **1% of honest cards**.

- **Data:** Sparkov simulated card transactions ([Kaggle: kartik2112/fraud-detection](https://www.kaggle.com/datasets/kartik2112/fraud-detection)),
  `fraudTrain.csv` for training/validation, `fraudTest.csv` for the final test.
- **Model:** an LSTM (PyTorch) that reads each card's last 10 transactions.
- **Decision:** hold the card for 72 hours when a charge scores at or above the threshold.
- **Headline metric:** median number of fraudulent charges that get through on an attacked
  card before it is held.

Team members are listed in [`TEAM.md`](TEAM.md).

---

## Results

**Test set** (Jun 21 – Dec 31, 2020), run once with thresholds fixed on validation:

| | Median frauds before hold | Honest cards held | Fraud charges through | Total cost |
|---|---|---|---|---|
| Doing nothing | 10 | 0 | 2,145 | $1,133,325 |
| LSTM, D2 threshold (hold forever) | 2 | 12 (1.7%) | 569 | $221,563 |
| **Decision layer** (cost threshold + 72 h hold) | **2** | 12 (1.7%) | 577 | $224,923 |

On validation (Jan 1 – Jun 21, 2020) the decision layer held 5 honest cards (0.72%) with the
same median of 2. The higher hold rate on test is likely seasonal: the test months include
the holiday season. Costs: $15 per honest card held, a missed fraud costs its amount.

**Decision rule in plain words:** when a charge on a card scores **0.99918 or higher**, hold
the card for **72 hours** and block all charges in that window. Contact the cardholder: if
they confirm the charges, release the card early; if not, cancel and reissue it. After
72 hours the card is released and scored again.

Full write-ups: [`reports/`](reports/) (Project Proposal, Preliminary Results, Final Results).
Every run is logged in [`experiments/log.csv`](experiments/log.csv).

---

## Run everything from a fresh Colab session

### 1. One-time setup (per person)

1. Get access to this repository (it is private).
2. Create a **GitHub token** (Settings → Developer settings → Personal access tokens →
   Tokens (classic), tick `repo`) and a **Kaggle API key** (kaggle.com → Settings → API).
3. In Colab, open the **key icon** in the left sidebar (Secrets) and add, with
   *Notebook access* switched on:
   - `GITHUB_TOKEN`
   - `KAGGLE_USERNAME`
   - `KAGGLE_KEY`

### 2. Run the pipeline

1. Open [`notebooks/workflow.ipynb`](notebooks/workflow.ipynb) in Colab
   (File → Open notebook → GitHub tab → this repository).
2. Switch to a GPU: **Runtime → Change runtime type → T4 GPU** (the LSTM trains in about 3 minutes).
3. **Run all cells, top to bottom, once.** The notebook:
   1. clones the repo, installs the requirements and downloads the data (`data.load_track02()`)
   2. makes the validation split (Jan 1, 2020) and checks it for leakage
   3. builds features and trains the baseline (logistic regression + score smoothing)
   4. trains the LSTM with early stopping and saves it to `models/`
   5. chooses the hold threshold on validation (cheapest cost within 1% of honest cards held)
      and the hold period, and saves them to `models/thresholds.json`
   6. runs the one-time test-set comparison

Run the setup cell only once per session (running it twice clones the repo inside itself).
Logging to `experiments/log.csv` is off by default (`LOG = False` at the top of the notebook).

Retraining overwrites the files in `models/`. GPU training is not bit-for-bit reproducible,
so retrained numbers can differ very slightly from the tables above; the threshold step
re-chooses the threshold for the new model.

### 3. Run the tests

```
!python -m pytest -q tests/
```

---

## Demo

The demo (`app.py`) scores one card's recent transactions with the same `predict()` used in
the evaluation, and shows the decision (`hold` / `allow`), the score and plain-language reasons.
It uses only the saved files in `models/` and the examples in `examples/`, so it needs **no
dataset download and no Kaggle key**.

**From Colab** (after the setup cell):

```python
import app
app.demo.launch(share=True)     # open the printed https://....gradio.live link
```

**On a laptop:**

```
git clone https://github.com/williamli218/mais202-team-2-fraud-bursts
cd mais202-team-2-fraud-bursts
pip install -r requirements.txt torch
python app.py                   # then open http://127.0.0.1:7860
```

**What to try:** three example buttons fill the input with real cards from the validation period
(labels removed):

- **Attack caught**: the third fraudulent charge in a burst scores above the threshold, so the card is held.
- **Still on hold**: the newest charge looks normal, but the card is within a 72-hour hold from an earlier fraud.
- **Normal card**: allowed.

The **Table** tab shows the transactions readably; the **Raw input (CSV)** tab is what `predict()`
reads, and you can paste your own card's transactions there (CSV with a header, oldest first,
one card). A recording is in [`reports/demo.mp4`](reports/demo.mp4).

Note: every merchant name in Sparkov starts with `fraud_`. That is a quirk of the data
generator, not a label; the demo's table hides it.

---

## Repository layout

| Path | What it holds |
|---|---|
| `config.py` | track number, seed, paths |
| `src/data.py` | course data loader and `make_sequences` (provided) |
| `src/split.py` | validation split (everything from Jan 1, 2020) |
| `src/features.py` | logistic-regression features (`build_features`) and LSTM inputs (`build_lstm_inputs`) |
| `src/metrics.py` | headline metric, `median_frauds_before_hold` |
| `src/models.py` | logistic regression, score smoothing, LSTM data prep, `FraudLSTM`, training and scoring |
| `src/decision.py` | cost table, cost curve, threshold choice, 72-hour hold replay, evaluation |
| `src/predict.py` | `predict()`: one card in, score + decision + reasons out |
| `app.py` | Gradio demo |
| `models/` | saved logistic regression, LSTM weights and settings, `thresholds.json` |
| `examples/` | the three demo inputs |
| `experiments/log.csv` | every logged run |
| `reports/` | reports, figures, demo recording, fair slide |
| `tests/` | course tests (metric, leakage, `predict()` contract) |
