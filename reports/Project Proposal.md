# Project Proposal (Deliverable 1)

Team:  
Track:  

Write under each heading, keep the whole thing to one page, then export to
`reports/Project Proposal.pdf` and push both files. The deliverable document
says what each part needs; the one-line reminders here are just that.

## 1. Problem Statement

Three or four sentences: what you predict, from what information, which
decision the score feeds and who acts on it.

## 2. Data and Splitting

Where your validation set comes from, how the test set stays untouched until
D3, and the two leakage checks you ran.

## 3. Exploratory Data Analysis

Two plots with a sentence or two under each: class balance, and one way
fraud differs from legitimate activity.

## 4. Features or Representation

Three features (`src/features.py`), each with why it should help and why it
can't see the future. Text tracks: how a raw input becomes model input.

## 5. Evaluation Metric

Your track's metric is in `src/metrics.py` and `pytest tests/test_metrics.py`
passes. The all-legitimate baseline's metric and its accuracy on your
validation set, side by side, and what the gap says.

## 6. Cost of Errors

The costs in `src/decision.py`, each with a sentence behind it.

## 7. Mentor Feedback

What was raised in the first mentor session and what changed, or why it
didn't.
