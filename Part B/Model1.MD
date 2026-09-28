# Model 1 — Random Forest

**Script:** [`src/model1_random_forest.py`](src/model1_random_forest.py)

## Technique

A `RandomForestClassifier` (scikit-learn), trained on the engineered
feature set produced by `feature_engineering.py`.

## Why Random Forest

Random Forest is chosen as the interpretable baseline because it was the
top-performing single model in **two** of the four Part A sources:
- **Carneiro (2016)** compared Logistic Regression, Random Forest and SVM
  (RBF kernel) via 10-fold grid-search cross-validation on a real e-tail
  fraud dataset; Random Forest won on both AUC-PR and AUC-ROC and was the
  one deployed as a live fraud-scoring API.
- **Nagagopiraju et al. (2025)** likewise reported Random Forest as the
  most effective of the algorithms compared, citing its accuracy,
  precision, and resistance to overfitting.

It was also consistently among the strongest individual base learners in
**Moradi et al. (2025)**'s ensemble-stacking study, even though a full
stacked ensemble ultimately beat it there.

Beyond accuracy, Random Forest gives `feature_importances_`, which lets a
fraud-review team see *why* a transaction was flagged — important for a
production deployment where analysts need to justify blocking/reviewing an
order, not just receive a black-box score.

## Hyperparameters

```python
RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)
```

- `max_depth=12`, `min_samples_split=10`, `min_samples_leaf=4` — capped to
  reduce overfitting to the majority (legitimate) class, given the ~0.57%
  fraud rate.
- `class_weight="balanced"` — reweights the split criterion so the rare
  fraud class isn't drowned out by the majority class during training.
- `max_features="sqrt"` — standard Random Forest decorrelation setting,
  each tree only considers a random subset of features per split.
- An optional `--use-smote` flag additionally oversamples the training set
  to a 1:10 fraud:legitimate ratio via SMOTE before training, so both the
  "class-weighting only" and "class-weighting + resampling" approaches can
  be compared — Carneiro (2016) found resampling made no statistically
  significant difference to Random Forest specifically, which is worth
  re-testing on this dataset rather than assuming.

## Evaluation

Reported on the (naturally imbalanced) test set: classification report,
ROC-AUC, **PR-AUC (average precision)** — the primary metric, since
accuracy is close to meaningless at a 0.57% fraud rate — confusion matrix,
and the top 10 feature importances.

## How to run

```bash
python src/model1_random_forest.py \
    --train data/processed/train_features.csv \
    --test data/processed/test_features.csv \
    --output-model models/model1_random_forest.joblib \
    --use-smote          # optional
```

**Inputs:** `data/processed/train_features.csv`,
`data/processed/test_features.csv` (outputs of `feature_engineering.py` —
see [FeatureEngineering.MD](FeatureEngineering.MD)).

**Output:** `models/model1_random_forest.joblib` (trained model, loadable
via `joblib.load(...)` for inference on new transactions).
