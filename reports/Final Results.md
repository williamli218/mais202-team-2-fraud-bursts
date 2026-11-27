# Final Results and Demo (Deliverable 3)

Team: 02
Track: Catching bursts of fraud as they happen

Export to `reports/Final Results.pdf` and push both files. This is the one
report that uses the test set, once, for the comparison in section 3.

## 1. Decision Layer

We chose the threshold on validation as the cheapest under the D1 costs ($15 per honest card held, plus the amount of each fraud charge before the hold) among those holding at most 1% of honest cards. It resulted in a threshold of 0.99918 (0.72% held, strictly less than 1%, $233,469). Held cards are blocked for 72 hours, then released and re-scored, since all training attacks lasted under 48h.

## 2. In Plain Words

When a charge scores 0.99918 or higher, hold the card for 72 hours and block all its charges. Contact the cardholder: if they confirm the charges, release the card; if not, cancel and reissue it. If no one responds, the card is released after 72 hours and scored again.

## 3. Comparison on the Test Set

| Test set (run once) | Median frauds before hold | Honest cards held | Total cost |
|---|---|---|---|
| Doing nothing | 10 | 0 | $1,133,325 |
| LSTM, D2 threshold (hold forever) | 2 | 12 (1.7%) | $221,563 |
| **Decision layer** | **2** | 12 (1.7%) | $224,923 |

## 4. Demo

app.py runs the same predict() and shows the decision, the score and the reasons. Run it with python app.py, or from Colab with app.demo.launch(share=True). Its three examples show an attack caught, a card still on hold, and a normal card allowed. Recording: reports/demo.mp4.

## 5. README

The repository README explains what's in the repo and how to run it; say so
here and point to it.

## 6. Showcase

Headline on our slide: a median of 2 fraud charges get through per attacked card, vs 10 with no system (test set). Slide: reports/fair_slide.pdf.

## 7. Mentor Notes

What the last mentor session raised and what changed because of it.
