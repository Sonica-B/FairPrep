"""
Phase 2: The Fairness Audit (The "What")
=========================================
Applies Accuracy Parity, TPRP, and other metrics from the Entity Matching
paper (FairEM, Shahbazi et al. VLDB 2023) to annotator demographic groups
in the TUS context.

Key questions:
  - Does majority-vote ground truth systematically disagree with Non-Native speakers?
  - Do Non-Native speakers have lower TPR on "text-heavy" tables?
  - Is there a calibration gap between demographic groups?
  - Is there score bias in TUS model outputs (Starmie/SANTOS/D3L)?
"""

import numpy as np
import pandas as pd
from collections import defaultdict

from . import measures


class TUSFairnessAuditor:
    """
    Fairness auditing engine for Table Union Search.
    Adapted from FairEM class in FairEMRepro.

    Instead of entity-matching predictions, we audit:
      - Human annotator decisions (SurveyAnswer) vs ground truth (ActualAnswer)
      - TUS model scores (Starmie, Santos, D3L) across demographic groups
    """

    def __init__(self, df: pd.DataFrame, threshold: float = 0.1, alpha: float = 0.05):
        """
        Args:
            df: DataFrame with demographic partitions already applied.
            threshold: fairness tolerance threshold (disparity limit).
            alpha: significance level for statistical tests.
        """
        self.df = df
        self.threshold = threshold
        self.alpha = alpha

    # ----- Confusion matrix per group -----

    def _confusion_matrix_for_group(
        self,
        group_df: pd.DataFrame,
        pred_col: str = "SurveyAnswer",
        label_col: str = "ActualAnswer",
    ) -> tuple:
        """
        Compute TP, FP, TN, FN for a group.
        Prediction and label are binary (1 = unionable, 0 = not unionable).
        """
        preds = group_df[pred_col].values
        labels = group_df[label_col].values
        TP = int(np.sum((preds == 1) & (labels == 1)))
        FP = int(np.sum((preds == 1) & (labels == 0)))
        TN = int(np.sum((preds == 0) & (labels == 0)))
        FN = int(np.sum((preds == 0) & (labels == 1)))
        return TP, FP, TN, FN

    def _confusion_matrix_overall(
        self, pred_col: str = "SurveyAnswer", label_col: str = "ActualAnswer"
    ) -> tuple:
        return self._confusion_matrix_for_group(self.df, pred_col, label_col)

    # ----- Core audit method -----

    def audit_demographic_fairness(
        self,
        group_col: str,
        measure_names: list = None,
        pred_col: str = "SurveyAnswer",
        label_col: str = "ActualAnswer",
    ) -> pd.DataFrame:
        """
        For each demographic subgroup defined by `group_col`, compute
        fairness metrics and the disparity from the overall population.

        Returns a DataFrame with columns:
            [Group, Measure, GroupValue, OverallValue, Disparity, IsFair]
        """
        if measure_names is None:
            measure_names = ["accuracy_parity", "true_positive_rate_parity",
                             "false_positive_rate_parity", "positive_predictive_value_parity"]

        overall_cm = self._confusion_matrix_overall(pred_col, label_col)
        groups = self.df.groupby(group_col)

        rows = []
        for group_name, group_df in groups:
            group_cm = self._confusion_matrix_for_group(group_df, pred_col, label_col)
            n_samples = len(group_df)

            for m_name in measure_names:
                group_val = measures.compute_measure(m_name, *group_cm)
                overall_val = measures.compute_measure(m_name, *overall_cm)
                disparity = group_val - overall_val

                # Fairness check (adapted from FairEM.is_fair_measure_specific)
                if m_name in measures.HIGHER_IS_BETTER:
                    is_fair = disparity >= -self.threshold
                else:
                    is_fair = disparity <= self.threshold

                rows.append({
                    "Group": group_name,
                    "Measure": m_name,
                    "GroupValue": round(group_val, 4),
                    "OverallValue": round(overall_val, 4),
                    "Disparity": round(disparity, 4),
                    "IsFair": is_fair,
                    "N_Samples": n_samples,
                })

        return pd.DataFrame(rows)

    # ----- TUS Model Score Audit -----

    def audit_tus_model_scores(
        self,
        group_col: str,
        score_cols: list = None,
    ) -> pd.DataFrame:
        """
        Compare TUS model score distributions across demographic groups
        using Wasserstein distance (score bias metric from PS2, Phase 2).

        Returns DataFrame with pairwise Wasserstein distances between groups
        for each TUS model.
        """
        if score_cols is None:
            score_cols = ["Starnie", "Santos", "D3L"]

        groups = self.df.groupby(group_col)
        group_names = sorted(groups.groups.keys())

        rows = []
        for i, g1 in enumerate(group_names):
            for g2 in group_names[i + 1:]:
                df1 = groups.get_group(g1)
                df2 = groups.get_group(g2)
                for score_col in score_cols:
                    s1 = df1[score_col].dropna().values
                    s2 = df2[score_col].dropna().values
                    w_dist = measures.wasserstein_score_bias(s1, s2)
                    rows.append({
                        "GroupA": g1,
                        "GroupB": g2,
                        "TUS_Model": score_col,
                        "Wasserstein_Distance": round(w_dist, 6),
                    })

        return pd.DataFrame(rows)

    # ----- Calibration Gap Audit -----

    def audit_calibration_gap(
        self,
        group_col: str,
        confidence_col: str = "ConfidenceLevel",
        label_col: str = "ActualAnswer",
        pred_col: str = "SurveyAnswer",
    ) -> pd.DataFrame:
        """
        Compute calibration gap (Confidence - Accuracy) per demographic group.
        Hypothesis: humans are more overconfident on "easy" tables and
        miscalibrated on "hard" ones, with variation by demographic group.
        """
        groups = self.df.groupby(group_col)
        rows = []

        for group_name, group_df in groups:
            confidences = group_df[confidence_col].values
            # Per-row accuracy: 1 if prediction matches label, else 0
            accuracies = (group_df[pred_col].values == group_df[label_col].values).astype(float)
            gap = measures.calibration_gap(confidences, accuracies)
            mean_conf = float(np.mean(confidences))
            mean_acc = float(np.mean(accuracies))
            rows.append({
                "Group": group_name,
                "MeanConfidence": round(mean_conf, 4),
                "MeanAccuracy": round(mean_acc, 4),
                "CalibrationGap": round(gap, 4),
                "N_Samples": len(group_df),
            })

        return pd.DataFrame(rows)

    # ----- Majority Vote Disagreement Analysis -----

    def audit_majority_disagreement(
        self,
        group_col: str,
        pred_col: str = "SurveyAnswer",
        majority_col: str = "Majority",
    ) -> pd.DataFrame:
        """
        Measure how often each demographic group disagrees with majority vote.
        Key Question: Does majority vote systematically disagree with
        Non-Native speakers?
        """
        groups = self.df.groupby(group_col)
        rows = []

        for group_name, group_df in groups:
            total = len(group_df)
            disagree = int(np.sum(group_df[pred_col] != group_df[majority_col]))
            agree = total - disagree
            rows.append({
                "Group": group_name,
                "TotalDecisions": total,
                "AgreesWithMajority": agree,
                "DisagreesWithMajority": disagree,
                "DisagreementRate": round(disagree / total, 4) if total > 0 else 0,
            })

        return pd.DataFrame(rows)
