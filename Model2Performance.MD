# Model 2 Performance — Isolation Forest + Neural Network

**Script:** [`src/model2_performance.py`](src/model2_performance.py)
(shared helpers in [`src/metrics_utils.py`](src/metrics_utils.py))

These results are from the **real dataset** described in Part A: the full
"Credit Card Transactions Fraud Detection Dataset" (Shenoy, 2020), with
the Isolation Forest fit on the 1,289,169 legitimate training rows and the
Neural Network trained on the SMOTE-resampled training set (1,933,753 rows
after resampling), evaluated on all 555,719 real test rows.

## What it does

Loads both halves of the trained hybrid model
(`models/model2_isolation_forest.joblib` and
`models/model2_neural_network.keras`) and evaluates them together on the
held-out test set:

1. **Regenerates the 3 Isolation Forest anomaly features**
   (`anomaly_score`, `anomaly_flag`, `anomaly_bin`) for the test set,
   using the *already-fitted* Isolation Forest from training — it is not
   refit here, only applied, so no test-set information leaks into it.
2. **Scores the Neural Network** on the expanded feature set (original
   features + the 3 anomaly features) to get fraud probabilities.
3. Computes the same suite of metrics as Model 1 — point-estimate metrics
   at threshold 0.5, the F1-optimal threshold, and bootstrap confidence
   intervals for ROC-AUC and PR-AUC — so the two models are directly
   comparable using identical methodology.
4. ROC and Precision-Recall curve plots.

## How to run

```bash
python src/model2_performance.py \
    --iso-model models/model2_isolation_forest.joblib \
    --nn-model models/model2_neural_network.keras \
    --test data/processed/test_features.csv \
    --output-dir reports \
    --n-boot 200
```

**Inputs:** both trained halves of Model 2 from Part B, and the engineered
test set (`data/processed/test_features.csv`).

**Outputs** (written to `reports/`):
- `model2_results.json` — all metrics and CIs, machine-readable
- `model2_predictions.csv` — raw `(y_true, predicted_probability)` pairs,
  used by `compare_models.py`
- `model2_roc_curve.png`, `model2_pr_curve.png`

## Results

| Metric | Value | 95% Bootstrap CI |
|---|---|---|
| ROC-AUC | 0.9723 | [0.9682, 0.9760] |
| PR-AUC (average precision) | 0.6149 | [0.5934, 0.6338] |

**At the default 0.5 threshold:** precision 0.3538, recall 0.6918, F1
0.4681, accuracy 0.9939. Confusion matrix: 1,484 true positives, 661 false
negatives, 2,711 false positives, 550,863 true negatives.

**At the F1-optimal threshold (0.8860):** precision 0.7502, recall 0.5193,
F1 0.6138, accuracy 0.9975. Confusion matrix: 1,114 true positives, 1,031
false negatives, 371 false positives, 553,203 true negatives.

### Training behaviour

The Neural Network converged cleanly and quickly over the 5 documented
epochs — training accuracy rose from 85.2% to 98.1%, and training loss
fell from 1.461 to 0.060, with validation accuracy tracking closely
throughout (97.4% by the final epoch), showing no signs of overfitting at
this epoch count.

### Interpretation

Model 2's PR-AUC (0.615) is **substantially higher than Model 1's (0.335)**
— nearly double — despite Model 2's ROC-AUC being marginally lower (0.972
vs 0.983). This is the expected behaviour of a hybrid anomaly-aware model
on severely imbalanced data: ROC-AUC treats the (much larger) legitimate
class generously, so a small ROC-AUC gap can hide a large real difference
in how well a model handles the rare positive class — which is exactly
what PR-AUC surfaces here.

At its F1-optimal threshold, Model 2 reaches **75% precision at 52%
recall**, with only 371 false positives across the entire 555,719-row
test set — a dramatically more usable operating point for a real
fraud-review queue than Model 1's F1-optimal precision of 33%. This
matches the direction (if not the exact magnitude) of the finding in
Al-Hajabed (2026), whose hybrid Isolation Forest + Neural Network
architecture this replicates: combining an unsupervised anomaly signal
with a supervised classifier meaningfully improved precision over a
single-model baseline on this same dataset.

![ROC Curve](reports/model2_roc_curve.png)
![PR Curve](reports/model2_pr_curve.png)
