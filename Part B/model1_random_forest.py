"""
model1_random_forest.py

Model 1: Random Forest classifier — an interpretable, strong tabular
baseline. Chosen because Random Forest was the top-performing single
model in two of the four Part A sources (Carneiro, 2016 and the
Nagagopiraju et al. 2025 comparison), and was consistently among the
strongest base learners in a third (Moradi et al., 2025). It also gives
feature importances, which is valuable for explaining flagged
transactions to a STADIOalot fraud-review team.

Usage
-----
Train and evaluate:
    python src/model1_random_forest.py \
        --train data/processed/train_features.csv \
        --test data/processed/test_features.csv \
        --output-model models/model1_random_forest.joblib

Options:
    --use-smote     Apply SMOTE oversampling to the training set only
                     (recommended given the ~0.57% fraud rate).
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

TARGET_COL = "is_fraud"

# Hyperparameters. max_depth and min_samples_leaf are capped to reduce
# overfitting on the majority class given the severe (~0.57%) imbalance;
# class_weight='balanced' further reweights the loss toward the fraud class.
RF_PARAMS = dict(
    n_estimators=300,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    max_features="sqrt",
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
)


def load_xy(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def maybe_smote(X: pd.DataFrame, y: pd.Series, use_smote: bool):
    if not use_smote:
        return X, y
    from imblearn.over_sampling import SMOTE
    print("[model1] Applying SMOTE to the training set...")
    smote = SMOTE(sampling_strategy=0.1, random_state=42)  # fraud -> ~10% of majority
    X_res, y_res = smote.fit_resample(X, y)
    print(f"[model1] Post-SMOTE class balance: {y_res.value_counts(normalize=True).to_dict()}")
    return X_res, y_res


def evaluate(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print("\n[model1] Classification report (threshold = 0.5):")
    print(classification_report(y_test, preds, digits=4))
    print(f"[model1] ROC-AUC:  {roc_auc_score(y_test, proba):.4f}")
    print(f"[model1] PR-AUC (average precision): {average_precision_score(y_test, proba):.4f}")
    print("[model1] Confusion matrix:")
    print(confusion_matrix(y_test, preds))

    top_features = pd.Series(model.feature_importances_, index=X_test.columns)
    print("\n[model1] Top 10 feature importances:")
    print(top_features.sort_values(ascending=False).head(10))


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate Model 1 (Random Forest).")
    parser.add_argument("--train", required=True, help="Path to engineered training CSV.")
    parser.add_argument("--test", required=True, help="Path to engineered testing CSV.")
    parser.add_argument("--output-model", default="models/model1_random_forest.joblib")
    parser.add_argument("--use-smote", action="store_true", help="Apply SMOTE to the training data.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_model), exist_ok=True)

    print("[model1] Loading data...")
    X_train, y_train = load_xy(args.train)
    X_test, y_test = load_xy(args.test)

    X_train, y_train = maybe_smote(X_train, y_train, args.use_smote)

    print(f"[model1] Training RandomForestClassifier with params: {RF_PARAMS}")
    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X_train, y_train)

    evaluate(model, X_test, y_test)

    joblib.dump(model, args.output_model)
    print(f"\n[model1] Saved trained model to {args.output_model}")


if __name__ == "__main__":
    main()
