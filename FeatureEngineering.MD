# Feature Engineering

**Script:** [`src/feature_engineering.py`](src/feature_engineering.py)

## What it does

Takes the cleaned CSVs from the preprocessing stage and builds four families
of engineered features on top of them. The choice of families is directly
informed by Al-Hajabed (2026) — the Part A source that used this exact
dataset — whose feature-ablation experiment found that removing
**contextual** features caused the single largest drop in precision-recall
performance (−0.49), followed by **behavioural-velocity** features (−0.28).
Both families are implemented in full here.

### 1. Temporal features
- `trans_hour`, `trans_dayofweek`
- `is_weekend`, `is_night` (00:00–05:00) flags

### 2. Contextual features
- `age_years` and a 5-bucket `age_group`, from cardholder date of birth
- `merchant_distance_km` — great-circle (haversine) distance between the
  cardholder's registered location and the merchant's location
- `amt_log` — log-transformed transaction amount (raw `amt` is heavily
  right-skewed)

### 3. Behavioural velocity features (per card)
- `txn_count_24h`, `txn_amt_sum_24h` — rolling count/sum of that card's
  transactions in the trailing 24 hours
- `mins_since_last_txn` — minutes since that card's previous transaction

### 4. Cardholder history
- `cardholder_amt_zscore` — how unusual the current transaction amount is
  relative to that card's own running average spend

## Leakage prevention

All four velocity/history features are computed using a **backward-looking
window only**: for every transaction, the rolling/expanding statistics are
`shift(1)`-ed so they only ever include transactions that happened *before*
the current one on that card. No feature is allowed to "see the future" —
this is the same principle Al-Hajabed (2026) enforces via historical-only
aggregation windows.

## How to run

```bash
python src/feature_engineering.py \
    --train data/processed/train_clean.csv \
    --test data/processed/test_clean.csv \
    --output-dir data/processed
```

**Inputs:** `data/processed/train_clean.csv`, `data/processed/test_clean.csv`
(outputs of `preprocessing.py` — see [Preprocessing.MD](Preprocessing.MD)).

**Outputs:**
- `data/processed/train_features.csv`
- `data/processed/test_features.csv`

The script prints the final feature count as a sanity check.
