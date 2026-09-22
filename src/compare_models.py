"""
compare_models.py

Statistically compares Model 1 (Random Forest) and Model 2 (Isolation
Forest + Neural Network) on the same held-out test set. Uses TWO
complementary statistical tests, since they answer slightly different
questions:

  1. McNemar's test — compares the two models' HARD classification
     decisions (at threshold 0.5) on the same test set. It asks: "of the
     cases where the two models disagree, is one model wrong more often
     than the other, more than chance would predict?" This is the
     standard paired test for comparing two classifiers on identical
     test data (unlike an unpaired test, it doesn't assume independent
     samples for each model).

  2. Paired bootstrap test on PR-AUC/ROC-AUC — compares the two models'
     PROBABILITY rankings rather than a single hard threshold, and gives
     a confidence interval on the size of the performance gap, not just
     a yes/no significance verdict.

Requires model1_performance.py and model2_performance.py to have been run
first (this script reads their saved *_predictions.csv outputs).

Usage
-----
python src/compare_models.py \
    --model1-predictions reports/model1_predictions.csv \
    --model2-predictions reports/model2_predictions.csv \
    --output-dir reports
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar

from metrics_utils import (
    average_precision_score,
    get_roc_pr_points,
    paired_bootstrap_diff,
    point_metrics,
    roc_auc_score,
    save_results,
)


def run_mcnemar(y_true, preds_a, preds_b) -> dict:
    """
    Builds the 2x2 contingency table of (model A correct/incorrect) x
    (model B correct/incorrect) and runs McNemar's exact test on it.
    """
    a_correct = (preds_a == y_true)
    b_correct = (preds_b == y_true)

    both_correct = int(np.sum(a_correct & b_correct))
    a_only_correct = int(np.sum(a_correct & ~b_correct))
    b_only_correct = int(np.sum(~a_correct & b_correct))
    both_wrong = int(np.sum(~a_correct & ~b_correct))

    table = [[both_correct, a_only_correct],
             [b_only_correct, both_wrong]]

    # Exact binomial version is preferred when the discordant-pair count
    # is small; otherwise the chi-squared version (with continuity
    # correction) is used automatically by statsmodels.
    discordant = a_only_correct + b_only_correct
    result = mcnemar(table, exact=(discordant < 25), correction=True)

    return {
        "contingency_table": {
            "both_correct": both_correct,
            "model1_correct_only": a_only_correct,
            "model2_correct_only": b_only_correct,
            "both_wrong": both_wrong,
        },
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "significant_at_0.05": bool(result.pvalue < 0.05),
    }


def plot_combined_curves(y_true, proba1, proba2, output_dir: str):
    p1 = get_roc_pr_points(y_true, proba1)
    p2 = get_roc_pr_points(y_true, proba2)

    plt.figure(figsize=(6, 5))
    plt.plot(p1["fpr"], p1["tpr"], label="Model 1 — Random Forest")
    plt.plot(p2["fpr"], p2["tpr"], label="Model 2 — Isolation Forest + NN")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Chance")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_roc_curves.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(6, 5))
    plt.plot(p1["recall"], p1["precision"], label="Model 1 — Random Forest")
    plt.plot(p2["recall"], p2["precision"], label="Model 2 — Isolation Forest + NN")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve Comparison")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comparison_pr_curves.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare Model 1 and Model 2 statistically.")
    parser.add_argument("--model1-predictions", required=True, help="reports/model1_predictions.csv")
    parser.add_argument("--model2-predictions", required=True, help="reports/model2_predictions.csv")
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--n-boot", type=int, default=1000)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("[compare_models] Loading saved predictions...")
    df1 = pd.read_csv(args.model1_predictions)
    df2 = pd.read_csv(args.model2_predictions)

    if not np.array_equal(df1["y_true"].values, df2["y_true"].values):
        raise ValueError(
            "model1 and model2 predictions were scored on different test "
            "sets/orderings — re-run both performance scripts on the same "
            "test CSV before comparing."
        )

    y_true = df1["y_true"].values
    proba1 = df1["proba"].values
    proba2 = df2["proba"].values

    print("[compare_models] Computing point metrics for both models...")
    m1 = point_metrics(y_true, proba1)
    m2 = point_metrics(y_true, proba2)

    print("[compare_models] Running McNemar's test on threshold-0.5 decisions...")
    preds1 = (proba1 >= 0.5).astype(int)
    preds2 = (proba2 >= 0.5).astype(int)
    mcnemar_result = run_mcnemar(y_true, preds1, preds2)

    print(f"[compare_models] Running paired bootstrap ({args.n_boot} resamples) "
          f"on PR-AUC and ROC-AUC differences...")
    pr_auc_diff = paired_bootstrap_diff(y_true, proba1, proba2, average_precision_score, n_boot=args.n_boot)
    roc_auc_diff = paired_bootstrap_diff(y_true, proba1, proba2, roc_auc_score, n_boot=args.n_boot)

    plot_combined_curves(y_true, proba1, proba2, args.output_dir)

    comparison_table = pd.DataFrame({
        "Model 1 (Random Forest)": {
            "Precision": m1["precision"], "Recall": m1["recall"], "F1": m1["f1"],
            "ROC-AUC": m1["roc_auc"], "PR-AUC": m1["pr_auc"],
        },
        "Model 2 (Isolation Forest + NN)": {
            "Precision": m2["precision"], "Recall": m2["recall"], "F1": m2["f1"],
            "ROC-AUC": m2["roc_auc"], "PR-AUC": m2["pr_auc"],
        },
    }).round(4)
    comparison_table.to_csv(os.path.join(args.output_dir, "comparison_table.csv"))

    results = {
        "model1_metrics": m1,
        "model2_metrics": m2,
        "mcnemar_test": mcnemar_result,
        "paired_bootstrap_pr_auc_diff_model1_minus_model2": pr_auc_diff,
        "paired_bootstrap_roc_auc_diff_model1_minus_model2": roc_auc_diff,
    }
    save_results(results, os.path.join(args.output_dir, "comparison_results.json"))

    print("\n[compare_models] Comparison table:")
    print(comparison_table)
    print("\n[compare_models] McNemar's test:", mcnemar_result)
    print("[compare_models] PR-AUC difference (Model 1 - Model 2):", pr_auc_diff)
    print("[compare_models] ROC-AUC difference (Model 1 - Model 2):", roc_auc_diff)


if __name__ == "__main__":
    main()
