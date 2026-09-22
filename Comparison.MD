# Model Comparison — Model 1 vs Model 2

**Script:** [`src/compare_models.py`](src/compare_models.py)
(shared helpers in [`src/metrics_utils.py`](src/metrics_utils.py))

> ⚠️ **The numbers and plots in this file are from a smoke test on a small
> synthetic sample**, used only to prove the code runs end-to-end — see
> the caveats in [Model1Performance.MD](Model1Performance.MD) and
> [Model2Performance.MD](Model2Performance.MD). Model 2's poor smoke-test
> score (too little training data for a Neural Network) makes this
> particular comparison one-sided; re-run on the real dataset and replace
> the numbers below before submitting.

## What it does

Requires `model1_performance.py` and `model2_performance.py` to have
already been run (it reads their saved `reports/model1_predictions.csv`
and `reports/model2_predictions.csv`). It then runs **two complementary
statistical comparisons**, because they answer different questions:

### 1. McNemar's test — comparing hard decisions

At the default 0.5 threshold, each model gets each test transaction either
right or wrong. McNemar's test builds a 2×2 contingency table of "both
right / only Model 1 right / only Model 2 right / both wrong" and tests
whether the disagreements are lopsided in one model's favour, more than
chance alone would produce. This is the standard **paired** significance
test for two classifiers evaluated on the *same* test set (a plain t-test
would wrongly assume the two models' errors are independent samples, when
they're actually paired — same transactions, same true labels).

The exact binomial version is used automatically when the number of
disagreement cases is small (<25); otherwise the chi-squared version with
continuity correction is used.

### 2. Paired bootstrap test — comparing probability rankings

McNemar's test only looks at one threshold. The paired bootstrap instead
resamples the test set (with replacement) 1,000 times, and for each
resample computes the *difference* in PR-AUC (and separately, ROC-AUC)
between Model 1 and Model 2 using the **same resampled rows for both
models** (this pairing is what makes it a valid comparison, rather than
comparing two independent bootstrap distributions). This gives:
- A 95% confidence interval on the *size* of the performance gap, not
  just whether one exists.
- An approximate two-sided p-value for the null hypothesis that the true
  PR-AUC/ROC-AUC difference is zero.

## How to run

```bash
# Run both performance scripts first if you haven't already:
python src/model1_performance.py --model models/model1_random_forest.joblib \
    --test data/processed/test_features.csv --output-dir reports
python src/model2_performance.py --iso-model models/model2_isolation_forest.joblib \
    --nn-model models/model2_neural_network.keras \
    --test data/processed/test_features.csv --output-dir reports

# Then compare:
python src/compare_models.py \
    --model1-predictions reports/model1_predictions.csv \
    --model2-predictions reports/model2_predictions.csv \
    --output-dir reports
```

**Outputs** (written to `reports/`):
- `comparison_table.csv` — side-by-side metrics table
- `comparison_results.json` — McNemar's test and paired bootstrap results
- `comparison_roc_curves.png`, `comparison_pr_curves.png` — both models
  overlaid on the same axes

## Results (smoke test — replace with real-data numbers)

| Metric | Model 1 (Random Forest) | Model 2 (Isolation Forest + NN) |
|---|---|---|
| Precision | 0.5700 | 0.0000 |
| Recall | 0.8507 | 0.0000 |
| F1 | 0.6826 | 0.0000 |
| ROC-AUC | 0.9832 | 0.4714 |
| PR-AUC | 0.7616 | 0.0559 |

**McNemar's test** (threshold-0.5 decisions): both-correct = 1,090,
Model 1-only-correct = 57, Model 2-only-correct = 43, both-wrong = 10.
χ² = 1.69, **p = 0.194** (not significant at α=0.05).

**Paired bootstrap, PR-AUC (Model 1 − Model 2):** observed difference =
+0.706, 95% CI [0.595, 0.797], p ≈ 0.000.

**Paired bootstrap, ROC-AUC (Model 1 − Model 2):** observed difference =
+0.512, 95% CI [0.455, 0.565], p ≈ 0.000.

### Interpretation

On this smoke-test data the two results *appear* to disagree with each
other (McNemar's test says "not significant," the bootstrap test says
"highly significant") — that apparent contradiction is actually expected
and instructive, and the same pattern may well show up on the real
dataset too, so it's worth understanding rather than treating as a bug:

- McNemar's test only looks at **hard decisions at threshold 0.5**. Since
  Model 2's Neural Network barely learned anything on this toy sample, it
  predicts almost everything as the majority class at 0.5 — which,
  because fraud is rare, happens to be "correct" on most rows by sheer
  base-rate luck. That drags its threshold-0.5 accuracy close enough to
  Model 1's that McNemar's test doesn't detect a significant difference
  in raw *decisions*.
- The paired bootstrap on PR-AUC/ROC-AUC looks at the full **ranking of
  probabilities**, not just one threshold — and there Model 1 is
  unambiguously and significantly better, which is the real story
  (confirmed by both the point estimate and the narrow, zero-excluding CI).

**Takeaway for reporting real results**: don't rely on a single test.
Report both, and if they disagree, explain why using the same reasoning
as above — it usually means one model is only accidentally matching the
other's hard decisions at a specific threshold, while actually being
much worse (or better) once you look at how confidently it ranks
transactions across all thresholds.

![ROC Comparison](reports/comparison_roc_curves.png)
![PR Comparison](reports/comparison_pr_curves.png)
