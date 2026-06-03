"""
Statistical significance testing for publication-ready comparisons.

Implemented:
  - McNemar's test: pairwise comparison of two classifiers on the same test set
  - Bootstrap confidence intervals: for any scalar metric
  - k-fold cross-validation evaluation
"""

from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import chi2
from sklearn.model_selection import StratifiedKFold

from .metrics import compute_all_metrics


# ── McNemar's test ────────────────────────────────────────────

def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    alpha: float = 0.05,
) -> Dict:
    """
    McNemar's test for two classifiers on the same test set.

    H0: the two classifiers have the same error rate.
    Significant p-value → reject H0 → classifiers differ significantly.

    Returns a dict with: statistic, p_value, significant, n01, n10
    """
    # n01: A wrong, B correct; n10: A correct, B wrong
    a_correct = (y_pred_a == y_true)
    b_correct = (y_pred_b == y_true)

    n01 = np.sum(~a_correct & b_correct)
    n10 = np.sum(a_correct & ~b_correct)

    # Continuity-corrected McNemar statistic
    if n01 + n10 == 0:
        return {"statistic": 0.0, "p_value": 1.0, "significant": False, "n01": 0, "n10": 0}

    statistic = (abs(n01 - n10) - 1.0) ** 2 / (n01 + n10)
    p_value = 1.0 - chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": bool(p_value < alpha),
        "n01": int(n01),
        "n10": int(n10),
    }


def mcnemar_table(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    alpha: float = 0.05,
) -> Dict[Tuple[str, str], Dict]:
    """
    Run all pairwise McNemar tests.

    Returns dict keyed by (model_a, model_b) → test result dict.
    """
    names = list(predictions.keys())
    results = {}
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            na, nb = names[i], names[j]
            results[(na, nb)] = mcnemar_test(
                y_true, predictions[na], predictions[nb], alpha
            )
    return results


# ── Bootstrap confidence intervals ────────────────────────────

def bootstrap_ci(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Non-parametric bootstrap confidence interval for a scalar metric.

    Returns (point_estimate, lower_bound, upper_bound).
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    boot_scores = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            score = metric_fn(y_true[idx], y_pred[idx])
        boot_scores.append(score)

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_scores, alpha * 100))
    upper = float(np.percentile(boot_scores, (1.0 - alpha) * 100))
    return float(point), lower, upper


def bootstrap_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    num_classes: int = 4,
    class_names: Optional[List[str]] = None,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Bootstrap CI for accuracy, macro_f1, weighted_f1, cohen_kappa, mcc.
    Returns {metric_name: (point, lower, upper)}.
    """
    from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef

    metric_fns = {
        "accuracy": accuracy_score,
        "macro_f1": lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        "weighted_f1": lambda yt, yp: f1_score(yt, yp, average="weighted", zero_division=0),
        "cohen_kappa": lambda yt, yp: cohen_kappa_score(yt, yp, weights="quadratic"),
        "mcc": matthews_corrcoef,
    }

    results = {}
    for name, fn in metric_fns.items():
        results[name] = bootstrap_ci(fn, y_true, y_pred, n_bootstrap, ci, seed)

    return results


# ── K-fold cross-validation ───────────────────────────────────

def kfold_evaluate(
    build_fn: Callable,
    samples: List[Tuple[str, int]],
    cfg,
    device,
    k: int = 5,
    seed: int = 42,
) -> Dict[str, List[float]]:
    """
    Stratified k-fold cross-validation.

    build_fn(train_samples, val_samples, cfg) → (model, val_metrics_dict)
    Returns a dict of {metric: [fold_1_val, ..., fold_k_val]}.
    """
    labels = np.array([s[1] for s in samples])
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)

    fold_metrics: Dict[str, List[float]] = {}

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(samples, labels)):
        print(f"\n{'─'*50}")
        print(f"  K-Fold: fold {fold_idx + 1}/{k}")
        print(f"{'─'*50}")

        train_s = [samples[i] for i in train_idx]
        val_s = [samples[i] for i in val_idx]

        _, metrics = build_fn(train_s, val_s, cfg, device)

        for k_name, v in metrics.items():
            if isinstance(v, (int, float)):
                fold_metrics.setdefault(k_name, []).append(float(v))

    return fold_metrics


def summarise_kfold(fold_metrics: Dict[str, List[float]]) -> Dict[str, Dict]:
    """Return mean ± std for each metric across folds."""
    summary = {}
    for name, values in fold_metrics.items():
        arr = np.array(values)
        summary[name] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1))}
    return summary
