"""
Fairness Measures for TUS Demographic Fairness
================================================
Adapted from FairEM (Shahbazi et al., VLDB 2023) measures.py
for entity matching, re-targeted to annotator-level fairness
in Table Union Search.

Instead of entity pairs with left/right sensitive attributes,
we evaluate fairness per annotator demographic group against
the ground truth (ActualAnswer).
"""

import numpy as np


# ===== Confusion matrix primitives =====

def AP(TP, FP, TN, FN):
    """Accuracy Parity: (TP+TN) / total"""
    total = TP + TN + FP + FN
    return (TP + TN) / total if total > 0 else 1.0


def SP(TP, FP, TN, FN):
    """Statistical Parity: positive prediction rate = TP / total"""
    total = TP + FP + TN + FN
    return TP / total if total > 0 else 1.0


def TPR(TP, FP, TN, FN):
    """True Positive Rate (Sensitivity / Recall)"""
    denom = TP + FN
    return TP / denom if denom > 0 else 1.0


def FPR(TP, FP, TN, FN):
    """False Positive Rate"""
    denom = FP + TN
    return FP / denom if denom > 0 else 0.0


def FNR(TP, FP, TN, FN):
    """False Negative Rate"""
    denom = FN + TP
    return FN / denom if denom > 0 else 0.0


def TNR(TP, FP, TN, FN):
    """True Negative Rate (Specificity)"""
    denom = TN + FP
    return TN / denom if denom > 0 else 1.0


def PPV(TP, FP, TN, FN):
    """Positive Predictive Value (Precision)"""
    denom = TP + FP
    return TP / denom if denom > 0 else 1.0


def NPV(TP, FP, TN, FN):
    """Negative Predictive Value"""
    denom = TN + FN
    return TN / denom if denom > 0 else 1.0


def FDR(TP, FP, TN, FN):
    """False Discovery Rate"""
    denom = TP + FP
    return FP / denom if denom > 0 else 0.0


def FOR(TP, FP, TN, FN):
    """False Omission Rate"""
    denom = TN + FN
    return FN / denom if denom > 0 else 0.0


# ===== Metric registry =====

HIGHER_IS_BETTER = {
    "accuracy_parity": AP,
    "statistical_parity": SP,
    "true_positive_rate_parity": TPR,
    "true_negative_rate_parity": TNR,
    "positive_predictive_value_parity": PPV,
    "negative_predictive_value_parity": NPV,
}

LOWER_IS_BETTER = {
    "false_positive_rate_parity": FPR,
    "false_negative_rate_parity": FNR,
    "false_discovery_rate_parity": FDR,
    "false_omission_rate_parity": FOR,
}

ALL_MEASURES = {**HIGHER_IS_BETTER, **LOWER_IS_BETTER}


def compute_measure(measure_name: str, TP, FP, TN, FN) -> float:
    """Compute a named measure from confusion matrix counts."""
    if measure_name not in ALL_MEASURES:
        raise ValueError(f"Unknown measure: {measure_name}")
    return ALL_MEASURES[measure_name](TP, FP, TN, FN)


# ===== Calibration Gap (specific to TUS) =====

def calibration_gap(confidences: np.ndarray, accuracies: np.ndarray) -> float:
    """
    Compute the average calibration gap:
    mean(confidence - accuracy) per group.
    A positive gap means overconfidence.
    """
    if len(confidences) == 0:
        return 0.0
    return float(np.mean(confidences - accuracies))


def wasserstein_score_bias(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """
    Wasserstein distance between TUS model score distributions
    for Group A vs Group B. Higher = more bias.
    """
    from scipy.stats import wasserstein_distance
    if len(scores_a) == 0 or len(scores_b) == 0:
        return 0.0
    return wasserstein_distance(scores_a, scores_b)
