"""
metrics_utils.py

Shared helper functions used by model1_performance.py, model2_performance.py
and compare_models.py, so the same metric/statistical-test logic is not
duplicated three times.
"""

import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def point_metrics(y_true, proba, threshold: float = 0.5) -> dict:
    """Standard point-estimate classification metrics at a given threshold."""
    preds = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
    return {
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, preds),
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "roc_auc": roc_auc_score(y_true, proba),
        "pr_auc": average_precision_score(y_true, proba),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def best_f1_threshold(y_true, proba) -> tuple[float, float]:
    """Find the probability threshold that maximises F1 on the fraud class."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, proba)
    f1s = np.where(
        (precisions + recalls) > 0,
        2 * precisions * recalls / (precisions + recalls + 1e-12),
        0,
    )
    best_idx = int(np.argmax(f1s[:-1]))  # last point has no matching threshold
    return float(thresholds[best_idx]), float(f1s[best_idx])


def bootstrap_ci(
    y_true, proba, metric_fn, n_boot: int = 1000, seed: int = 42, ci: float = 0.95
) -> dict:
    """
    Non-parametric bootstrap confidence interval for a scalar metric
    (e.g. roc_auc_score or average_precision_score), resampling the test
    set with replacement n_boot times.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    n = len(y_true)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_s, p_s = y_true[idx], proba[idx]
        if len(np.unique(y_s)) < 2:
            continue  # skip resamples with only one class present
        scores.append(metric_fn(y_s, p_s))
    scores = np.array(scores)
    alpha = (1 - ci) / 2
    return {
        "point_estimate": float(metric_fn(y_true, proba)),
        "bootstrap_mean": float(scores.mean()),
        "ci_lower": float(np.quantile(scores, alpha)),
        "ci_upper": float(np.quantile(scores, 1 - alpha)),
        "n_boot": int(len(scores)),
        "ci_level": ci,
    }


def paired_bootstrap_diff(
    y_true, proba_a, proba_b, metric_fn, n_boot: int = 1000, seed: int = 42, ci: float = 0.95
) -> dict:
    """
    Paired bootstrap test for the DIFFERENCE in a metric between two models
    scored on the same test set (same resampled indices used for both, so
    the comparison is paired rather than independent). Returns a CI on the
    difference (model_a - model_b) and a two-sided bootstrap p-value for
    the null hypothesis that the true difference is zero.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    proba_a = np.asarray(proba_a)
    proba_b = np.asarray(proba_b)
    n = len(y_true)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        y_s = y_true[idx]
        if len(np.unique(y_s)) < 2:
            continue
        diffs.append(metric_fn(y_s, proba_a[idx]) - metric_fn(y_s, proba_b[idx]))
    diffs = np.array(diffs)
    alpha = (1 - ci) / 2
    observed_diff = metric_fn(y_true, proba_a) - metric_fn(y_true, proba_b)
    # Two-sided bootstrap p-value: proportion of resamples on the opposite
    # side of zero from the observed difference, doubled.
    if observed_diff >= 0:
        p_value = 2 * min((diffs <= 0).mean(), 0.5)
    else:
        p_value = 2 * min((diffs >= 0).mean(), 0.5)
    return {
        "observed_diff": float(observed_diff),
        "ci_lower": float(np.quantile(diffs, alpha)),
        "ci_upper": float(np.quantile(diffs, 1 - alpha)),
        "p_value_approx": float(min(p_value, 1.0)),
        "n_boot": int(len(diffs)),
        "ci_level": ci,
    }


def save_results(results: dict, path: str):
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"[metrics_utils] Saved results to {path}")


def get_roc_pr_points(y_true, proba):
    fpr, tpr, _ = roc_curve(y_true, proba)
    precision, recall, _ = precision_recall_curve(y_true, proba)
    return {"fpr": fpr, "tpr": tpr, "precision": precision, "recall": recall}
