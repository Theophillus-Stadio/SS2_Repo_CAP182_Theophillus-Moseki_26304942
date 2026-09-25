# Model 1 Performance — Random Forest

**Script:** [`src/model1_performance.py`](src/model1_performance.py)
(shared helpers in [`src/metrics_utils.py`](src/metrics_utils.py))

These results are from the **real dataset** described in Part A: the full
"Credit Card Transactions Fraud Detection Dataset" (Shenoy, 2020), trained
on all 1,296,675 training rows and evaluated on all 555,719 test rows.

## What it does

Loads the trained Model 1 (`models/model1_random_forest.joblib`) and scores
it on the held-out test set, reporting:

1. **Point-estimate metrics** at the default 0.5 threshold: accuracy,
   precision, recall, F1, ROC-AUC, PR-AUC, and the confusion matrix.
2. **F1-optimal threshold**: since 0.5 is an arbitrary cutoff, the script
   also searches the precision-recall curve for the threshold that
   maximises F1 on the fraud class, and reports metrics there too — this
   is closer to how the threshold would actually be tuned for a
   production auto-flag/manual-review decision.
3. **Bootstrap confidence intervals** (200 resamples with replacement) for
   ROC-AUC and PR-AUC. A single point estimate on a ~0.58%-fraud dataset
   can look precise but actually rest on very few positive examples in the
   test set (only 2,145 fraud cases out of 555,719 rows); the bootstrap CI
   shows how much that estimate could plausibly vary.
4. ROC and Precision-Recall curve plots.

## How to run

```bash
python src/model1_performance.py \
    --model models/model1_random_forest.joblib \
    --test data/processed/test_features.csv \
    --output-dir reports \
    --n-boot 200
```

(`--n-boot 200` is used instead of the 1,000-resample default purely for
runtime — see the note on bootstrap resample count at the bottom of this
file. 200 resamples still gives a stable, usable confidence interval.)

**Inputs:** the trained model from Part B (`models/model1_random_forest.joblib`)
and the engineered test set (`data/processed/test_features.csv`).

**Outputs** (written to `reports/`):
- `model1_results.json` — all metrics and CIs, machine-readable
- `model1_predictions.csv` — raw `(y_true, predicted_probability)` pairs,
  used by `compare_models.py`
- `model1_roc_curve.png`, `model1_pr_curve.png`

## Results

| Metric | Value | 95% Bootstrap CI |
|---|---|---|
| ROC-AUC | 0.9829 | [0.9799, 0.9852] |
| PR-AUC (average precision) | 0.3352 | [0.3152, 0.3610] |

**At the default 0.5 threshold:** precision 0.2607, recall 0.6718, F1
0.3757, accuracy 0.9914. Confusion matrix: 1,441 true positives, 704 false
negatives, 4,086 false positives, 549,488 true negatives.

**At the F1-optimal threshold (0.5894):** precision 0.3290, recall 0.5366,
F1 0.4079, accuracy 0.9940. Confusion matrix: 1,151 true positives, 994
false negatives, 2,347 false positives, 551,227 true negatives.

### Interpretation

Random Forest achieves a very strong ROC-AUC (0.983), meaning it ranks
fraudulent transactions above legitimate ones highly reliably across all
thresholds. However, **PR-AUC is far more informative here** given how
rare fraud is (0.386% of the test set): PR-AUC of 0.335 shows that, when
forced to actually commit to positive predictions, the model's precision
degrades quickly as recall increases — at the default threshold, roughly
3 out of every 4 transactions it flags as fraud are actually legitimate
(4,086 false positives vs. 1,441 true positives). Raising the threshold to
the F1-optimal point trades some recall (0.672 → 0.537) for meaningfully
better precision (0.261 → 0.329), which is the kind of trade-off a
STADIOalot fraud-review team would need to tune based on how many manual
reviews they can realistically staff for.

A note on the bootstrap resample count: the assignment's original design
used 1,000 resamples, but on the full 555,719-row test set that took
roughly 5+ minutes to compute per metric in this environment, so 200
resamples were used instead to keep runtimes practical — the resulting
confidence intervals are narrow and stable at 200 resamples already (see
`bootstrap_mean` vs `point_estimate` in `model1_results.json`, which are
within 0.001 of each other for both metrics), so this does not materially
change the conclusions.

![ROC Curve](reports/model1_roc_curve.png)
![PR Curve](reports/model1_pr_curve.png)
