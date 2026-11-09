# MAIS 202 Fall 2026 final project template

This is the repository every team starts from. It already has the data
loaders and the fixed test split for each track, the stubs you fill in for
the three deliverables, tests that check your metric against known answers,
a demo app that runs before you've trained anything, and the report
outlines. There is nothing to pull later: everything for D1, D2 and D3 is
here from the start.

The `VERSION` file says which version of the template you have. Every
notebook's setup cell prints it, and the changelog at the bottom of this
page says what changed when.

## Start here

One person on the team clicks **Use this template** on GitHub, names the
repository after the team, and invites the rest of the team, your TPM, Ben
and Pierre as collaborators (usernames are on Slack). That same person
opens `config.py`, sets `TRACK` to your track number, and pushes. One team,
one repository.

Then, each of you:

1. Create a GitHub token and a Kaggle API token, and save them in Colab
   Secrets as `GITHUB_TOKEN`, `KAGGLE_USERNAME` and `KAGGLE_KEY`. Secrets
   belong to your Google account, not to the repository, so everyone does
   this once. Track 6 (Phishing links) doesn't need the Kaggle ones. Never
   paste a token into a notebook cell.
2. Open `notebooks/00_getting_started.ipynb` in Colab (File → Open notebook
   → GitHub tab → paste the repository URL) and run it. It takes about an
   hour and ends with your first commit.
3. Open your track's starter, `notebooks/starter_track<NN>_*.ipynb`, and
   run it from top to bottom. It downloads the data, loads it, prints the
   basic numbers and hands you the fixed test split. The cells marked
   `TODO D1.x` are where your work begins.

## What's where

```
config.py            TRACK (set it), SEED, paths. Nothing else to change.
notebooks/           00_getting_started, one starter per track, and your own notebooks
src/data.py          download, load and the fixed test split, every track  (provided; don't edit)
src/split.py         your validation split (D1.2), one function every teammate imports
src/features.py      your features (D1.4); one worked example is there
src/metrics.py       your track's metric (D1.5); stubs with the full definition in each docstring
src/models.py        the trivial baseline (provided); your models (D2.1, D2.2)
src/decision.py      the cost table format (provided); your costs (D1.6) and thresholds (D3.1)
src/predict.py       the predict() contract (provided); your wiring (D2.6)
app.py               the Gradio demo; runs on a placeholder until predict() works
tests/               test_metrics.py (your metric), test_leakage.py (the split), test_predict.py (the contract)
tests/fixtures/      tiny synthetic files the tests run on; no real data
experiments/log.csv  one row per model you train, starting with the baseline
reports/             the three report outlines, with the headings already in
models/              small saved weights and thresholds only
data/                ignored by Git; filled by the loaders
TEAM.md              one line per teammate, added as each person's first commit
```

The rule that keeps a team of four out of trouble: code goes in `src/` as
functions, and everyone works in their own notebook. Git can merge two
people's edits to a `.py` file line by line. It can't do that for a
`.ipynb`, so two people saving the same notebook costs an afternoon. Save
notebooks with File → Save a copy in GitHub, and push changes to `src/`
with git from a cell; the getting-started notebook shows the commands.

## Where each deliverable's work goes

| Deliverable | Code | Report |
|---|---|---|
| D1 Project Proposal | `src/split.py`, with `tests/test_leakage.py` green; `src/features.py`; `src/metrics.py` with its tests green; costs in `src/decision.py` | `reports/Project Proposal.md`, exported to PDF |
| D2 Preliminary Results | `src/models.py`; `predict()` in `src/predict.py`; rows in `experiments/log.csv` | `reports/Preliminary Results.md`, exported to PDF |
| D3 Final Results and Demo | thresholds in `src/decision.py`; `app.py` showing your system; this README rewritten for your project | `reports/Final Results.md`, exported to PDF |

## What changes at each stage

Some files are touched once, some keep evolving, and a few are frozen after
a deadline because the number they produce gets compared across
deliverables.

| File | D1 | D2 | D3 | After that |
|---|---|---|---|---|
| `config.py` | set `TRACK` once | | | never |
| `src/data.py` | | | | never (provided) |
| `src/split.py` | write `make_validation_split` (D1.2) | | | frozen: moving the split voids every number you've reported |
| `src/features.py` | three features (D1.4) | whatever the models need | | |
| `src/metrics.py` | the metric, tests green (D1.5) | | | frozen after D1 |
| `src/decision.py` | the costs (D1.6) | | `choose_thresholds` (D3.1) | |
| `src/models.py` | | baseline and main model (D2.1, D2.2) | | |
| `src/predict.py` | | wire the model in (D2.6) | wire the thresholds in (D3.1) | |
| `models/` | | small saved weights | `thresholds.json` | |
| `experiments/log.csv` | the baseline row (optional) | one row per run | | |
| `app.py` | | | title and examples (D3.4) | |
| `reports/` | Project Proposal | Preliminary Results | Final Results | |
| `README.md` | | | rewritten for your project (D3.5) | |
| `notebooks/` | exploration plots, in your own notebook | training runs | the final comparison | |

If a frozen thing has to change anyway, say so in the next report and
re-report the earlier numbers under the new version.

## Writing the report

The `.md` files in `reports/` are outlines, not where you write: the
headings, and a line under each on what goes there. Write the report
wherever your team writes (Google Docs is fine), keep the headings, export
it as PDF, and put the PDF in `reports/` under the exact name the
deliverable asks for. The easiest way to get a PDF into the repository is
GitHub's web interface: open the `reports/` folder, *Add file → Upload
files*, commit. Plot images go in `reports/figures/`, so they live with the
report.

`experiments/log.csv` gets one row per model you train. The header is the
format, and the first row to add is the trivial baseline on your
validation set, something like

```
2026-11-10,amira,always_legitimate,-,valid,recall_at_fpr_0.01,0.0,baseline from src/models.py
```

## Running the tests

From the repository root, `pytest -q`. From a Colab cell after the setup
cell, `!python -m pytest -q`. GitHub Actions runs the same thing on every
push and shows a tick or a cross next to the commit.

Reading the result:

- Everything skipped means `TRACK` is still `None` in `config.py`.
- The metric tests fail until D1.5 is done. That's expected; the message
  names the function to write and the case it got wrong.
  `pytest -q tests/test_metrics.py` runs just those.
- `test_predict.py` skips with "not wired yet (D2.6)" until your model is
  in `predict()`.
- `test_validation_split_does_not_leak` skips until `src/split.py` is
  written. When it passes, D1.2's checks pass.
- The other leakage tests check the split the template provides. If one of
  those fails, tell your TPM.

A red cross on a commit doesn't block anything. It's information.

## Files not to edit

`src/data.py`, everything in `tests/` (other than tests you add yourself),
`.github/workflows/tests.yml`, `VERSION`, and `config.py` beyond the
`TRACK` line. The same copies sit in all eight team repositories. If one of
them has a bug, the staff fix it once and push the fix to every repository
as a commit called something like "Template fix 2026-11-09". `git pull`
(the setup cell does it) and you have it. An edit of your own to one of
these files would be overwritten by that push, which is the reason not to
make one. If you think a file is wrong, tell your TPM, with the input and
the output you got.

When a starter notebook is fixed, only the original in `notebooks/` is
replaced. Your own copies aren't touched, so the announcement will say
which cell to update in them.

## The demo

```python
import app
app.demo.launch(share=True)
```

in a Colab cell after the setup cell gives you a public link that lasts as
long as the session. Until D2.6 it shows a placeholder decision. Once
`predict()` returns the contract, it shows your system, with no change to
`app.py`. For D3.4, edit the title and the example inputs at the top of
`app.py`.

## When something goes wrong

**"Colab Secret X is missing or not shared with this notebook."** Open the
key icon in the left sidebar. Either the secret isn't there, or its
*Notebook access* switch is off for this notebook.

**Kaggle download fails with 401 or 403.** The username or key is wrong, or
the Kaggle account isn't verified yet (Kaggle → Settings → verify with a
phone number). Regenerate the key, update the secret, restart the runtime.

**`git push` is rejected.** Someone pushed before you. `git pull`, then
push again. If the pull reports a conflict in a notebook, two people have
been editing the same `.ipynb`: keep one copy and rename the other.

**The session ran out of memory.** Don't just rerun the cell. Load fewer
columns, pass `nrows=` to the loader while you explore, or work on
`df.sample(...)`. Section 3 of the getting-started notebook has the
pattern.

**A token got committed.** Revoke it first (GitHub token settings, or
Kaggle → Settings → Expire API token), make a new one, update the secret.
Only then fix the notebook; deleting the cell doesn't delete it from the
history. Don't count on GitHub revoking it for you, which only happens in
public repositories.

## Changelog

- 2026-11-06: first release.