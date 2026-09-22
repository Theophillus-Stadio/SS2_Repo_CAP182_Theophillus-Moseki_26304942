"""
model2_hybrid_iforest_nn.py

Model 2: Hybrid Isolation Forest -> Neural Network.

This is a direct, simplified replication of the architecture in
Al-Hajabed (2026) "Guarding the Digital Checkout" — the one Part A source
that used this exact dataset. Their hybrid design (unsupervised anomaly
detector feeding a supervised classifier) beat a standalone Neural
Network by a large margin (PR-AUC 0.8782 -> 0.9004; precision
0.44 -> 0.78), so it is reproduced here as Model 2 to test that same
finding on our own preprocessing/feature-engineering pipeline.

Pipeline
--------
1. Fit an Isolation Forest on LEGITIMATE transactions only (train split).
2. Score every transaction (train and test) with that forest, producing:
     - anomaly_score  (continuous; higher = more anomalous)
     - anomaly_flag   (binary; 1 if the forest predicts it as an outlier)
     - anomaly_bin    (5 equal-frequency bins of the anomaly score)
3. Append these 3 columns to the engineered feature set.
4. Apply SMOTE to the TRAINING set only (sampling_strategy=0.5, i.e.
   fraud becomes ~33% of the training data) — the test set is left in
   its natural, severely imbalanced state, matching Al-Hajabed's setup.
5. Train a small feedforward Neural Network:
     Dense(64, relu) -> Dropout(0.3) -> Dense(32, relu) -> Dropout(0.3)
     -> Dense(1, sigmoid)
   Adam optimiser (lr=0.001), binary cross-entropy loss,
   5 epochs, batch size 2048.

Usage
-----
python src/model2_hybrid_iforest_nn.py \
    --train data/processed/train_features.csv \
    --test data/processed/test_features.csv \
    --output-dir models
"""

import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

TARGET_COL = "is_fraud"

ISO_FOREST_PARAMS = dict(
    n_estimators=200,
    contamination="auto",
    random_state=42,
    n_jobs=-1,
)

NN_PARAMS = dict(
    hidden_units=(64, 32),
    dropout_rate=0.3,
    learning_rate=0.001,
    epochs=5,
    batch_size=2048,
)


def load_xy(path: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def add_isolation_forest_features(
    X_train, y_train, X_test
) -> tuple[pd.DataFrame, pd.DataFrame, IsolationForest]:
    """Fit on legitimate-only training rows; score train and test."""
    print("[model2] Fitting Isolation Forest on legitimate transactions only...")
    legit_mask = y_train == 0
    iso = IsolationForest(**ISO_FOREST_PARAMS)
    iso.fit(X_train[legit_mask])

    def score(X):
        X = X.copy()
        X["anomaly_score"] = -iso.score_samples(X.drop(columns=[], errors="ignore"))
        X["anomaly_flag"] = (iso.predict(X.drop(columns=["anomaly_score"])) == -1).astype(int)
        X["anomaly_bin"] = pd.qcut(X["anomaly_score"], q=5, labels=False, duplicates="drop")
        return X

    X_train_scored = score(X_train)
    X_test_scored = score(X_test)
    return X_train_scored, X_test_scored, iso


def maybe_smote(X: pd.DataFrame, y: pd.Series):
    from imblearn.over_sampling import SMOTE
    print("[model2] Applying SMOTE to the training set (target ratio 0.5)...")
    smote = SMOTE(sampling_strategy=0.5, random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print(f"[model2] Post-SMOTE class balance: {y_res.value_counts(normalize=True).to_dict()}")
    return X_res, y_res


def build_nn(input_dim: int):
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(NN_PARAMS["hidden_units"][0], activation="relu"),
        layers.Dropout(NN_PARAMS["dropout_rate"]),
        layers.Dense(NN_PARAMS["hidden_units"][1], activation="relu"),
        layers.Dropout(NN_PARAMS["dropout_rate"]),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=NN_PARAMS["learning_rate"]),
        loss="binary_crossentropy",
        metrics=[keras.metrics.AUC(name="pr_auc", curve="PR"), "accuracy"],
    )
    return model


def evaluate(model, X_test, y_test):
    proba = model.predict(X_test, batch_size=4096).ravel()
    preds = (proba >= 0.5).astype(int)

    print("\n[model2] Classification report (threshold = 0.5):")
    print(classification_report(y_test, preds, digits=4))
    print(f"[model2] ROC-AUC:  {roc_auc_score(y_test, proba):.4f}")
    print(f"[model2] PR-AUC (average precision): {average_precision_score(y_test, proba):.4f}")
    print("[model2] Confusion matrix:")
    print(confusion_matrix(y_test, preds))


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate Model 2 (Isolation Forest + Neural Network).")
    parser.add_argument("--train", required=True, help="Path to engineered training CSV.")
    parser.add_argument("--test", required=True, help="Path to engineered testing CSV.")
    parser.add_argument("--output-dir", default="models", help="Where to save the fitted Isolation Forest and NN.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[model2] Loading data...")
    X_train, y_train = load_xy(args.train)
    X_test, y_test = load_xy(args.test)

    X_train, X_test, iso = add_isolation_forest_features(X_train, y_train, X_test)
    X_train, y_train = maybe_smote(X_train, y_train)

    print(f"[model2] Building Neural Network with params: {NN_PARAMS}")
    nn = build_nn(input_dim=X_train.shape[1])
    nn.fit(
        X_train, y_train,
        epochs=NN_PARAMS["epochs"],
        batch_size=NN_PARAMS["batch_size"],
        validation_split=0.1,
        verbose=2,
    )

    evaluate(nn, X_test, y_test)

    joblib.dump(iso, os.path.join(args.output_dir, "model2_isolation_forest.joblib"))
    nn.save(os.path.join(args.output_dir, "model2_neural_network.keras"))
    print(f"\n[model2] Saved Isolation Forest and Neural Network to {args.output_dir}/")


if __name__ == "__main__":
    main()
