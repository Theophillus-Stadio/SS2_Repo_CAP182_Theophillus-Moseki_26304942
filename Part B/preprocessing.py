"""
preprocessing.py

Cleans and encodes the raw "Credit Card Transactions Fraud Detection Dataset"
(Shenoy, 2020 / Sparkov Data Generation) ready for feature engineering.

Design decisions are informed by two of the sources in Part A:
  - Carneiro (2016): risk-level grouping of high-cardinality categoricals
    (we approximate this with train-only fraud-rate quantile grouping,
    since Jenks natural-breaks clustering needs a fitted continuous risk
    score first).
  - Al-Hajabed (2026): all statistics (means, fraud rates, scalers) are
    fitted on the TRAINING split only and then applied to the test split,
    to avoid data leakage from the future into the past.

Usage
-----
python src/preprocessing.py \
    --train data/raw/fraudTrain.csv \
    --test data/raw/fraudTest.csv \
    --output-dir data/processed \
    --artifacts-dir models

Outputs
-------
data/processed/train_clean.csv
data/processed/test_clean.csv
models/scaler.joblib
models/risk_maps.joblib
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Columns that are identifiers/PII with no predictive value on their own,
# or that would leak the label indirectly (e.g. trans_num is unique per row).
DROP_COLS = [
    "Unnamed: 0", "first", "last", "street", "trans_num", "unix_time",
]

# Columns we will risk-group by historical (train-only) fraud rate rather
# than one-hot encode, because they have too many unique values.
HIGH_CARDINALITY_COLS = ["merchant", "job", "city", "state", "zip"]

# Low-cardinality categoricals suitable for one-hot encoding.
LOW_CARDINALITY_COLS = ["category"]

NUMERIC_COLS = ["amt", "city_pop", "lat", "long", "merch_lat", "merch_long"]


def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Drop unneeded columns, parse dates, encode simple binaries."""
    df = df.copy()
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    df["trans_date_trans_time"] = pd.to_datetime(df["trans_date_trans_time"])
    df["dob"] = pd.to_datetime(df["dob"])

    # Report and impute missing values (the source dataset is documented as
    # complete, but we handle nulls defensively for robustness).
    n_missing = df.isna().sum().sum()
    if n_missing > 0:
        print(f"[preprocessing] Found {n_missing} missing values — imputing.")
        for col in df.select_dtypes(include=[np.number]).columns:
            df[col] = df[col].fillna(df[col].median())
        for col in df.select_dtypes(include=["object"]).columns:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"M": 0, "F": 1})

    return df


def fit_risk_maps(train_df: pd.DataFrame, target_col: str = "is_fraud") -> dict:
    """
    For each high-cardinality categorical, compute the TRAIN-ONLY historical
    fraud rate per category, then bucket those rates into 4 risk tiers
    (Low/Medium/High/Very High) via quantiles. Returns a dict of maps so the
    exact same buckets can be applied to the test set without leakage.
    """
    risk_maps = {}
    global_rate = train_df[target_col].mean()

    for col in HIGH_CARDINALITY_COLS:
        rate_per_cat = train_df.groupby(col)[target_col].mean()
        # Quantile-bin the fraud rates into 4 risk tiers.
        try:
            tiers = pd.qcut(rate_per_cat, q=4, labels=[0, 1, 2, 3], duplicates="drop")
        except ValueError:
            tiers = pd.Series(0, index=rate_per_cat.index)
        risk_maps[col] = {
            "category_to_tier": tiers.to_dict(),
            "global_fallback_tier": 1,  # median-ish default for unseen categories
            "global_rate": global_rate,
        }
    return risk_maps


def apply_risk_maps(df: pd.DataFrame, risk_maps: dict) -> pd.DataFrame:
    df = df.copy()
    for col, mapping in risk_maps.items():
        cat_to_tier = mapping["category_to_tier"]
        fallback = mapping["global_fallback_tier"]
        df[f"{col}_risk_tier"] = df[col].map(cat_to_tier).fillna(fallback).astype(int)
        df = df.drop(columns=[col])
    return df


def one_hot_encode(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    combined = pd.concat([train_df, test_df], keys=["train", "test"])
    combined = pd.get_dummies(combined, columns=LOW_CARDINALITY_COLS, prefix=LOW_CARDINALITY_COLS)
    train_out = combined.xs("train")
    test_out = combined.xs("test")
    # Align columns in case a category is missing from one split.
    test_out = test_out.reindex(columns=train_out.columns, fill_value=0)
    return train_out, test_out


def scale_numeric(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    scaler = StandardScaler()
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df[cols] = scaler.fit_transform(train_df[cols])
    test_df[cols] = scaler.transform(test_df[cols])
    return train_df, test_df, scaler


def main():
    parser = argparse.ArgumentParser(description="Preprocess the fraud detection dataset.")
    parser.add_argument("--train", required=True, help="Path to raw training CSV.")
    parser.add_argument("--test", required=True, help="Path to raw testing CSV.")
    parser.add_argument("--output-dir", default="data/processed", help="Where to write cleaned CSVs.")
    parser.add_argument("--artifacts-dir", default="models", help="Where to save the scaler/risk maps.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.artifacts_dir, exist_ok=True)

    print("[preprocessing] Loading data...")
    train_df, test_df = load_data(args.train, args.test)

    print("[preprocessing] Basic cleaning...")
    train_df = basic_clean(train_df)
    test_df = basic_clean(test_df)

    print("[preprocessing] Fitting risk maps on TRAIN ONLY...")
    risk_maps = fit_risk_maps(train_df, target_col="is_fraud")
    train_df = apply_risk_maps(train_df, risk_maps)
    test_df = apply_risk_maps(test_df, risk_maps)

    print("[preprocessing] One-hot encoding low-cardinality categoricals...")
    train_df, test_df = one_hot_encode(train_df, test_df)

    print("[preprocessing] Scaling numeric columns (fit on train only)...")
    train_df, test_df, scaler = scale_numeric(train_df, test_df, NUMERIC_COLS)

    train_path = os.path.join(args.output_dir, "train_clean.csv")
    test_path = os.path.join(args.output_dir, "test_clean.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    joblib.dump(scaler, os.path.join(args.artifacts_dir, "scaler.joblib"))
    joblib.dump(risk_maps, os.path.join(args.artifacts_dir, "risk_maps.joblib"))

    print(f"[preprocessing] Done. Wrote {train_path} and {test_path}.")
    print(f"[preprocessing] Train fraud rate: {train_df['is_fraud'].mean():.4%}")
    print(f"[preprocessing] Test fraud rate: {test_df['is_fraud'].mean():.4%}")


if __name__ == "__main__":
    main()
