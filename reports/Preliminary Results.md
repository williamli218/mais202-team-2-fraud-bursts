# Preliminary Results (Deliverable 2)

Team: 02
Track: Catching bursts of fraud as they happen

Export to `reports/Preliminary Results.pdf` and push both files. Every
number here comes from your validation set; the test set is still untouched.

## 1. Baseline

Logistic regression on our features (balanced class weights) ranks transactions well but matches the all-legitimate baseline (median 9 frauds before hold): one large honest purchase pushes a legitimate card's max score so high that the 1% budget forces a threshold above every attack. Averaging each card's last 3 scores cuts this to 3.

## 2. Main Model

PyTorch LSTM (1 layer, hidden 32, linear output) over each card's last 10 transactions, with 17 standardised inputs (log amount, log time since the previous charge, night flag, 14 categories). Adam (lr 1e-3), batch 2048, pos_weight ≈ 176 for imbalance, early stopping on validation loss (epoch 11). Settings are standard defaults. 

## 3. Required Experiment

<img width="975" height="349" alt="image" src="https://github.com/user-attachments/assets/efd4c2df-bfca-44bb-b4ee-51fde7143ac5" />
(Figure 1) Sequence model vs hand-built features: the LSTM lets 2 frauds through, LR 9 and LR with smoothing 3. LR underfits (bad on train and validation); the LSTM scores 2 on both, and its loss curves show overfitting starting after epoch 11, which early stopping avoids.

## 4. Results Table

| Model | Settings | Median frauds before hold | Precision | Recall |
|---|---|---|---|---|
| All-legitimate | score 0 everywhere | 9 | n/a | 0.00 |
| Logistic regression | 25 features, balanced class weights | 9 | 0.50 | 0.02 |
| LR + smoothing | same + mean of last 3 scores | 3 | 0.97 | 0.92 |
| **LSTM** | n = 10, hidden 32, 1 layer, lr 1e-3, batch 2048, pos_weight 176, early stop (epoch 11) | **2** | **0.97** | **0.96** |
(Table 1) Card-level precision and recall at the threshold holding ≤1% of legitimate cards.

## 5. Error Analysis

Missed: mostly the first charge of an attack (232 of 565), often small gas-station test charges after a quiet gap.
False holds: mostly $4,000+ late-night shopping, above anything fraudsters spend here.

## 6. Code

Models in src/models.py, weights in models/, and predict() passes tests/test_predict.py. Runs are logged in experiments/log.csv.

## 7. Next Steps

1.	Give the LSTM amount bands, so it learns that huge amounts are safe.
2.	Window length n = 5/10/20.
3.	Not trying: location, job and age (constant per card, or no difference between fraud and legitimate).

## 8. Mentor Notes

1.	How precision and recall were calculated and if they were coming from validation data
2.	Why 10 transactions were chosen and what changes with 5 or 20?
