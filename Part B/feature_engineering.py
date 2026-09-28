"""
feature_engineering.py

Builds the engineered feature set on top of the cleaned data produced by
preprocessing.py. Feature families are chosen because Al-Hajabed (2026) —
who used this exact dataset — found that shuffling (removing the signal
from) "contextual" and "behavioural velocity" features caused the largest
drops in precision-recall performance of any feature family tested.

Feature families implemented:
  1. Temporal features        — hour, day-of-week, weekend/night flags
  2. Contextual features      — cardholder age, age group, merchant distance
  3. Behavioural velocity     — rolling transaction count/spend in the last
                                 24h per card, and time since the previous
                                 transaction on that card
  4. Cardholder history       — expanding (running) mean/std of spend per
                                 card, used to z-score the current amount

Leakage prevention (per Al-Hajabed, 2026): every rolling/expanding
statistic below is computed using only transactions that occurred BEFORE
the current one for that card (via groupby + shift), so no feature ever
"sees the future".

Usage
-----
python src/feature_engineering.py \
    --train data/processed/train_clean.csv \
    --test data/processed/test_clean.csv \
    --output-dir data/processed
"""

import argparse
import os

import numpy as np
import pandas as pd


def haversine_km(lat1, lon1, lat2, lon2) -> pd.Series:
    """Great-circle distance between cardholder and merchant, in km."""
    r = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return r * c


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = df["trans_date_trans_time"]
    df["trans_hour"] = ts.dt.hour
    df["trans_dayofweek"] = ts.dt.dayofweek
    df["is_weekend"] = df["trans_dayofweek"].isin([5, 6]).astype(int)
    df["is_night"] = df["trans_hour"].between(0, 5).astype(int)
    return df


def add_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["age_years"] = (df["trans_date_trans_time"] - df["dob"]).dt.days / 365.25
    df["age_years"] = df["age_years"].clip(lower=0, upper=120)
    df["age_group"] = pd.cut(
        df["age_years"], bins=[-0.1, 25, 40, 55, 70, 120],
        labels=[0, 1, 2, 3, 4], include_lowest=True,
    ).astype(int)
    df["merchant_distance_km"] = haversine_km(
        df["lat"], df["long"], df["merch_lat"], df["merch_long"]
    )
    df["amt_log"] = np.log1p(df["amt"].clip(lower=0))
    return df


def add_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-card rolling behaviour, computed strictly backward-looking.
    Requires df to be sorted by cc_num then trans_date_trans_time.
    """
    df = df.sort_values(["cc_num", "trans_date_trans_time"]).copy()
    df = df.set_index("trans_date_trans_time")

    def _per_card(group: pd.DataFrame) -> pd.DataFrame:
        # Count/sum of transactions in the trailing 24h, EXCLUDING the
        # current transaction (shift after rolling achieves this).
        roll_count = group["amt"].rolling("24h").count()
        roll_sum = group["amt"].rolling("24h").sum()
        group["txn_count_24h"] = roll_count.shift(1).fillna(0)
        group["txn_amt_sum_24h"] = roll_sum.shift(1).fillna(0)

        # Time since the previous transaction on this card, in minutes.
        time_diff = group.index.to_series().diff().dt.total_seconds() / 60.0
        group["mins_since_last_txn"] = time_diff.fillna(time_diff.median())

        # Expanding (running) mean/std of spend on this card, EXCLUDING
        # the current row, used to z-score the current amount.
        expanding_mean = group["amt"].expanding().mean().shift(1)
        expanding_std = group["amt"].expanding().std().shift(1)
        group["cardholder_amt_zscore"] = (
            (group["amt"] - expanding_mean) / expanding_std.replace(0, np.nan)
        ).fillna(0)
        return group

    df = df.groupby("cc_num", group_keys=False).apply(_per_card)
    df = df.reset_index().rename(columns={"index": "trans_date_trans_time"})
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = add_temporal_features(df)
    df = add_contextual_features(df)
    df = add_velocity_features(df)
    # Drop raw columns now fully represented by engineered equivalents.
    drop_now = ["trans_date_trans_time", "dob", "lat", "long", "merch_lat", "merch_long", "cc_num"]
    df = df.drop(columns=[c for c in drop_now if c in df.columns])
    return df


def main():
    parser = argparse.ArgumentParser(description="Engineer features for the fraud model.")
    parser.add_argument("--train", required=True, help="Path to cleaned training CSV.")
    parser.add_argument("--test", required=True, help="Path to cleaned testing CSV.")
    parser.add_argument("--output-dir", default="data/processed", help="Where to write engineered CSVs.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[feature_engineering] Loading cleaned data...")
    train_df = pd.read_csv(args.train, parse_dates=["trans_date_trans_time", "dob"])
    test_df = pd.read_csv(args.test, parse_dates=["trans_date_trans_time", "dob"])

    print("[feature_engineering] Building features for training set...")
    train_out = build_features(train_df)
    print("[feature_engineering] Building features for testing set...")
    test_out = build_features(test_df)

    train_path = os.path.join(args.output_dir, "train_features.csv")
    test_path = os.path.join(args.output_dir, "test_features.csv")
    train_out.to_csv(train_path, index=False)
    test_out.to_csv(test_path, index=False)

    print(f"[feature_engineering] Done. Wrote {train_path} and {test_path}.")
    print(f"[feature_engineering] Final feature count: {train_out.shape[1] - 1}")


if __name__ == "__main__":
    main()
