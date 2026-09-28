# Model 2 — Hybrid Isolation Forest + Neural Network

**Script:** [`src/model2_hybrid_iforest_nn.py`](src/model2_hybrid_iforest_nn.py)

## Technique

A two-stage hybrid model:

1. **Isolation Forest** (unsupervised), fit on *legitimate transactions
   only* from the training set. It scores every transaction (train and
   test) and produces three new features:
   - `anomaly_score` — continuous outlier score (higher = more anomalous)
   - `anomaly_flag` — binary, 1 if the forest classifies the point as an
     outlier
   - `anomaly_bin` — the anomaly score bucketed into 5 equal-frequency bins

   These three columns are appended to the engineered feature set.

2. **Feedforward Neural Network** (Keras/TensorFlow), trained on the
   expanded feature set (original features + the 3 anomaly features) to
   produce the final fraud probability.

## Why this architecture

This is a direct, simplified replication of the hybrid architecture in
**Al-Hajabed (2026)** — the one Part A source built on this exact dataset.
Their key finding was that the hybrid beat a standalone Neural Network by
a wide margin:

| Model | PR-AUC | Precision | Recall |
|---|---|---|---|
| Neural Network alone | 0.8782 | 0.44 | 0.95 |
| Hybrid (Isolation Forest + NN) | **0.9004** | **0.78** | 0.88 |

Precision nearly doubled for a modest recall trade-off — a strong argument
for testing the same hybrid design here rather than jumping straight to a
single supervised model. Model 2 exists specifically to check whether that
same lift holds up on our own preprocessing/feature-engineering pipeline.

## Hyperparameters

**Isolation Forest:**
```python
IsolationForest(
    n_estimators=200,
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)
```
Fit only on rows where `is_fraud == 0`, so it learns what "normal" looks
like and scores deviations from that — rather than learning to separate
two classes directly.

**Neural Network:**
```
Dense(64, activation="relu")
Dropout(0.3)
Dense(32, activation="relu")
Dropout(0.3)
Dense(1, activation="sigmoid")

optimizer = Adam(learning_rate=0.001)
loss = "binary_crossentropy"
epochs = 5
batch_size = 2048
```
- **30% dropout** on each hidden layer is the regularisation technique
  used to prevent overfitting, matching Al-Hajabed (2026)'s setup exactly.
- **SMOTE** (`sampling_strategy=0.5`) is applied to the *training set
  only*, after the Isolation Forest features are added, bringing fraud up
  to ~33% of the training data. The test set is deliberately left in its
  natural, severely imbalanced state — evaluating on an artificially
  balanced test set would overstate real-world performance.

## Evaluation

Same metrics as Model 1 (classification report, ROC-AUC, PR-AUC, confusion
matrix) on the untouched, naturally imbalanced test set, so the two models
are directly comparable.

## How to run

```bash
python src/model2_hybrid_iforest_nn.py \
    --train data/processed/train_features.csv \
    --test data/processed/test_features.csv \
    --output-dir models
```

**Inputs:** `data/processed/train_features.csv`,
`data/processed/test_features.csv` (outputs of `feature_engineering.py` —
see [FeatureEngineering.MD](FeatureEngineering.MD)).

**Outputs:**
- `models/model2_isolation_forest.joblib` (fitted Isolation Forest)
- `models/model2_neural_network.keras` (trained Keras model)

Both are needed for inference: new transactions must first be scored by
the saved Isolation Forest to generate the 3 anomaly features, then passed
to the saved Neural Network.
