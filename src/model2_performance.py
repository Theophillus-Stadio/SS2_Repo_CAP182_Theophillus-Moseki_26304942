"""
model2_performance.py

Evaluates the trained Model 2 (Isolation Forest + Neural Network) on the
held-out test set. Since Model 2's Neural Network was trained on features
that INCLUDE the Isolation Forest's anomaly outputs, this script first
regenerates those three anomaly columns for the test set using the saved,
already-fitted Isolation Forest, then scores the saved Neural Network.

Usage
-----
python src/model2_performance.py \
    --iso-model models/model2_isolation_forest.joblib \
    --nn-model models/model2_neural_network.keras \
    --test data/processed/test_features.csv \
    --output-dir reports
"""

import argparse
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from metrics_utils import (
    average_precision_score,
    best_f1_threshold,
    bootstrap_ci,
    get_roc_pr_points,
    point_metrics,
    roc_auc_score,
    save_results,
)

TARGET_COL = "is_fraud"


def load_xy(path: str):
    df = pd.read_csv(path)
    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])
    return X, y


def add_isolation_forest_features(iso, X: pd.DataFrame) -> pd.DataFrame:
    """Re-derive the same 3 anomaly columns used at Model 2 training time."""
    X = X.copy()
    X["anomaly_score"] = -iso.score_samples(X)
    X["anomaly_flag"] = (iso.predict(X.drop(columns=["anomaly_score"])) == -1).astype(int)
    X["anomaly_bin"] = pd.qcut(X["anomaly_score"], q=5, labels=False, duplicates="drop")
    return X


def plot_curves(y_test, proba, output_dir: str, label: str = "Model 2 (Isolation Forest + NN)"):
    points = get_roc_pr_points(y_test, proba)

    plt.figure(figsize=(6, 5))
    plt.plot(points["fpr"], points["tpr"], label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Model 2 — ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model2_roc_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(points["recall"], points["precision"], label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Model 2 — Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model2_pr_curve.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Model 2 (Isolation Forest + NN) performance.")
    parser.add_argument("--iso-model", required=True, help="Path to the trained Isolation Forest .joblib file.")
    parser.add_argument("--nn-model", required=True, help="Path to the trained Neural Network .keras file.")
    parser.add_argument("--test", required=True, help="Path to the engineered test CSV.")
    parser.add_argument("--output-dir", default="reports", help="Where to write results/plots.")
    parser.add_argument("--n-boot", type=int, default=1000, help="Number of bootstrap resamples.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[model2_performance] Loading models and test data...")
    from tensorflow import keras
    iso = joblib.load(args.iso_model)
    nn = keras.models.load_model(args.nn_model)
    X_test, y_test = load_xy(args.test)

    print("[model2_performance] Regenerating Isolation Forest anomaly features...")
    X_test_scored = add_isolation_forest_features(iso, X_test)

    print("[model2_performance] Scoring with Neural Network...")
    proba = nn.predict(X_test_scored, batch_size=4096).ravel()

    print("[model2_performance] Computing point-estimate metrics at threshold 0.5...")
    metrics_default = point_metrics(y_test, proba, threshold=0.5)

    print("[model2_performance] Finding F1-optimal threshold...")
    best_thresh, best_f1 = best_f1_threshold(y_test, proba)
    metrics_best = point_metrics(y_test, proba, threshold=best_thresh)

    print(f"[model2_performance] Running {args.n_boot}-resample bootstrap CIs...")
    roc_ci = bootstrap_ci(y_test, proba, roc_auc_score, n_boot=args.n_boot)
    pr_ci = bootstrap_ci(y_test, proba, average_precision_score, n_boot=args.n_boot)

    plot_curves(y_test, proba, args.output_dir)

    results = {
        "model": "Model 2 - Isolation Forest + Neural Network",
        "metrics_at_threshold_0.5": metrics_default,
        "metrics_at_f1_optimal_threshold": metrics_best,
        "roc_auc_bootstrap_ci": roc_ci,
        "pr_auc_bootstrap_ci": pr_ci,
    }
    save_results(results, os.path.join(args.output_dir, "model2_results.json"))

    pd.DataFrame({"y_true": y_test, "proba": proba}).to_csv(
        os.path.join(args.output_dir, "model2_predictions.csv"), index=False
    )

    print("\n[model2_performance] Summary:")
    print(f"  ROC-AUC: {roc_ci['point_estimate']:.4f}  "
          f"(95% CI: {roc_ci['ci_lower']:.4f}-{roc_ci['ci_upper']:.4f})")
    print(f"  PR-AUC:  {pr_ci['point_estimate']:.4f}  "
          f"(95% CI: {pr_ci['ci_lower']:.4f}-{pr_ci['ci_upper']:.4f})")
    print(f"  F1-optimal threshold: {best_thresh:.4f} (F1={best_f1:.4f})")


if __name__ == "__main__":
    main()
