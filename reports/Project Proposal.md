# Project Proposal (Deliverable 1)

Team: 02
Track: Catching bursts of fraud as they happen

Write under each heading, keep the whole thing to one page, then export to
`reports/Project Proposal.pdf` and push both files. The deliverable document
says what each part needs; the one-line reminders here are just that.

## 1. Problem Statement

Each time a card is charged, our model scores whether the card is under attack using its last 5–20 transactions. If the score crosses a threshold that holds at most 1% of legitimate cards, the card is held and its next charges are blocked. The cardholder is notified, and RBC’s fraud team reviews held cards to release false alarms.

## 2. Data and Splitting

We train on Sparkov’s fraudTrain.csv (January 2019 to December 2019 inclusive) and test on fraudTest.csv, untouched until D3. Validation is everything from Jan 1st, 2020, onward (235 attacked cards), leaving a full year for training. Both checks pass: no leakage from validation, and training ends before validation starts.

## 3. Exploratory Data Analysis

<img width="975" height="286" alt="image" src="https://github.com/user-attachments/assets/35c7f1ee-b112-4534-bf36-8b78ccf37bef" />
Class distribution: only 0.58% of transactions are fraud. Predicting “legitimate” everywhere is therefore 99.4% accurate yet catches nothing. 
Fraud patterns: fraud peaks late at night. Time of day might be a useful signal.

## 4. Features or Representation

Each uses only the current charge and that card’s earlier charges, so nothing comes from the future or from other splits:
-	Spend in past 24h (median $244 legit vs $2,261 fraud): attacks spike spending.
-	Hour of day (2 pm vs 10 pm): fraud clusters at night.
-	Time since previous charge (4.4 h vs 1.4 h): fraud charges come fast.

## 5. Evaluation Metric

Our metric consists of the median number of fraudulent transactions that go through on an attacked card before it is held. Our implementation also passed the metrics tests. The all-legitimate baseline gets 99.4% accuracy but lets 9 fraud charges through per attacked card (median). Accuracy rewards ignoring rare fraud, whereas our metric measures how fast attacks are stopped.

## 6. Cost of Errors

-	Missed fraud costs its amount (median about $391)
-	A wrongly held card costs $15 (support call and customer friction)
-	Correct decisions cost $0.

## 7. Mentor Feedback

-	Typo relating to data splitting dates
-	Make sure to keep track of different time zones
-	Make sure to use a percentage value of legitimate cards that we falsely hold to +-1% (it is set to this percentage by default, but we might need to fine tune this value)
-	Keep track of computation cost, license cost, storage cost, etc.
-	Implement some more complex features

