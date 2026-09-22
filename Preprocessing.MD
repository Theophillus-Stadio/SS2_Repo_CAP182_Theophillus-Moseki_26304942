# Preprocessing

**Script:** [`src/preprocessing.py`](src/preprocessing.py)

## What it does

Takes the two raw Kaggle CSVs (`fraudTrain.csv`, `fraudTest.csv`) and produces
two cleaned, encoded, and scaled CSVs ready for feature engineering.

1. **Drops non-predictive identifier/PII columns**: `Unnamed: 0`, `first`,
   `last`, `street`, `trans_num`, `unix_time` (a duplicate of the
   transaction timestamp).
2. **Parses dates**: `trans_date_trans_time` and `dob` are converted from
   string to `datetime`.
3. **Handles missing values defensively**: the source dataset is documented
   as complete, but the script checks for and median/mode-imputes any nulls
   found, rather than assuming a clean file.
4. **Encodes `gender`** as a binary 0/1.
5. **Risk-groups high-cardinality categoricals** (`merchant`, `job`, `city`,
   `state`, `zip`): instead of one-hot encoding hundreds of unique values,
   each category is bucketed into one of 4 risk tiers (Low → Very High)
   based on its **historical fraud rate computed on the training set only**.
   This mirrors Carneiro (2016)'s risk-level grouping of high-cardinality
   categoricals (they used Jenks natural-breaks clustering on a fitted risk
   score; we approximate the same idea with train-only quantile binning of
   the fraud rate, which is simpler to implement and equally leakage-safe).
6. **One-hot encodes** the one low-cardinality categorical, `category`
   (transaction type, e.g. `grocery_pos`, `shopping_net`) — small enough
   (~14 values) that one-hot encoding doesn't blow up dimensionality.
7. **Scales numeric columns** (`amt`, `city_pop`, `lat`, `long`, `merch_lat`,
   `merch_long`) with a `StandardScaler` **fit on the training set only**
   and then applied to the test set — never fit on test data, to avoid
   leaking test-set statistics into training.

## Why fit-on-train-only matters

Every statistic used anywhere in this pipeline (risk-tier boundaries, the
scaler's mean/std) is fitted on the training split and only *applied* to
the test split. This follows the leakage-prevention practice explicitly
highlighted as a design decision in Al-Hajabed (2026), who used this exact
dataset — fitting on the full dataset (train+test combined) would let
information about future/held-out transactions leak into the training
process and produce an optimistic, unrealistic evaluation.

## How to run

```bash
python src/preprocessing.py \
    --train data/raw/fraudTrain.csv \
    --test data/raw/fraudTest.csv \
    --output-dir data/processed \
    --artifacts-dir models
```

**Inputs:** `data/raw/fraudTrain.csv`, `data/raw/fraudTest.csv` (downloaded
from Kaggle — see the main [README](README.md)).

**Outputs:**
- `data/processed/train_clean.csv`
- `data/processed/test_clean.csv`
- `models/scaler.joblib` (fitted `StandardScaler`, reusable for inference)
- `models/risk_maps.joblib` (fitted category→risk-tier mappings, reusable
  for inference)

The script prints the resulting fraud rate in each split as a sanity check.
