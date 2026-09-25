"""
model1_performance.py

Evaluates the trained Model 1 (Random Forest) on the held-out test set
and reports both point-estimate metrics and statistical measures
(bootstrap confidence intervals), since a single point estimate on an
imbalanced fraud dataset can be misleading on its own.

Usage
-----
python src/model1_performance.py \
    --model models/model1_random_forest.joblib \
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


def plot_curves(y_test, proba, output_dir: str, label: str = "Model 1 (Random Forest)"):
    points = get_roc_pr_points(y_test, proba)

    plt.figure(figsize=(6, 5))
    plt.plot(points["fpr"], points["tpr"], label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Model 1 — ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model1_roc_curve.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(points["recall"], points["precision"], label=label)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Model 1 — Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model1_pr_curve.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate Model 1 (Random Forest) performance.")
    parser.add_argument("--model", required=True, help="Path to the trained Model 1 .joblib file.")
    parser.add_argument("--test", required=True, help="Path to the engineered test CSV.")
    parser.add_argument("--output-dir", default="reports", help="Where to write results/plots.")
    parser.add_argument("--n-boot", type=int, default=1000, help="Number of bootstrap resamples.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[model1_performance] Loading model and test data...")
    model = joblib.load(args.model)
    X_test, y_test = load_xy(args.test)
    proba = model.predict_proba(X_test)[:, 1]

    print("[model1_performance] Computing point-estimate metrics at threshold 0.5...")
    metrics_default = point_metrics(y_test, proba, threshold=0.5)

    print("[model1_performance] Finding F1-optimal threshold...")
    best_thresh, best_f1 = best_f1_threshold(y_test, proba)
    metrics_best = point_metrics(y_test, proba, threshold=best_thresh)

    print(f"[model1_performance] Running {args.n_boot}-resample bootstrap CIs...")
    roc_ci = bootstrap_ci(y_test, proba, roc_auc_score, n_boot=args.n_boot)
    pr_ci = bootstrap_ci(y_test, proba, average_precision_score, n_boot=args.n_boot)

    plot_curves(y_test, proba, args.output_dir)

    results = {
        "model": "Model 1 - Random Forest",
        "metrics_at_threshold_0.5": metrics_default,
        "metrics_at_f1_optimal_threshold": metrics_best,
        "roc_auc_bootstrap_ci": roc_ci,
        "pr_auc_bootstrap_ci": pr_ci,
    }
    save_results(results, os.path.join(args.output_dir, "model1_results.json"))

    # Also save raw probabilities for compare_models.py
    pd.DataFrame({"y_true": y_test, "proba": proba}).to_csv(
        os.path.join(args.output_dir, "model1_predictions.csv"), index=False
    )

    print("\n[model1_performance] Summary:")
    print(f"  ROC-AUC: {roc_ci['point_estimate']:.4f}  "
          f"(95% CI: {roc_ci['ci_lower']:.4f}-{roc_ci['ci_upper']:.4f})")
    print(f"  PR-AUC:  {pr_ci['point_estimate']:.4f}  "
          f"(95% CI: {pr_ci['ci_lower']:.4f}-{pr_ci['ci_upper']:.4f})")
    print(f"  F1-optimal threshold: {best_thresh:.4f} (F1={best_f1:.4f})")


if __name__ == "__main__":
    main()
