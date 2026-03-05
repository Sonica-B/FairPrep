"""
Phase 5: Fair Aggregation
==========================
Create a "Fair Ground Truth" instead of simple majority voting
(which drowns out minorities). Use weighted aggregation where votes
from minority groups are upweighted if the LLM confirms their
interpretation is semantically valid.

References:
  - Li et al. (KDD 2020): "Towards Fair Truth Discovery from Biased Crowdsourced Answers"
  - CrowdTruth 2.0 (Aroyo & Welty, 2018): disagreement-preserving annotation
  - Capturing Perspectives (2023): multi-perspective classification
"""

import numpy as np
import pandas as pd
from collections import defaultdict


class FairAggregator:
    """
    Fair aggregation engine for annotator decisions.
    Replaces simple majority voting with demographically-aware
    weighted voting.
    """

    def __init__(self, minority_boost: float = 1.5, llm_validation_boost: float = 2.0):
        """
        Args:
            minority_boost: Weight multiplier for minority group votes.
            llm_validation_boost: Additional weight if LLM validates minority view.
        """
        self.minority_boost = minority_boost
        self.llm_validation_boost = llm_validation_boost

    def simple_majority(self, decisions: np.ndarray) -> int:
        """Standard majority voting (baseline)."""
        if len(decisions) == 0:
            return 0
        return int(np.round(np.mean(decisions)))

    def weighted_vote(
        self,
        decisions: np.ndarray,
        weights: np.ndarray,
    ) -> float:
        """Weighted voting. Returns soft score in [0, 1]."""
        if len(decisions) == 0 or np.sum(weights) == 0:
            return 0.5
        return float(np.average(decisions, weights=weights))

    def compute_fair_weights(
        self,
        df: pd.DataFrame,
        group_col: str,
        minority_groups: list,
        llm_validated: pd.Series = None,
    ) -> np.ndarray:
        """
        Compute per-annotator weights based on demographic group membership
        and optional LLM validation.

        Args:
            df: DataFrame with one row per annotation
            group_col: Column identifying demographic group
            minority_groups: List of group names considered "minority"
            llm_validated: Optional boolean Series indicating LLM validated minority view
        """
        weights = np.ones(len(df))

        is_minority = df[group_col].isin(minority_groups)
        weights[is_minority] *= self.minority_boost

        if llm_validated is not None:
            llm_boost_mask = is_minority & llm_validated
            weights[llm_boost_mask] *= self.llm_validation_boost

        return weights

    def aggregate_per_question(
        self,
        df: pd.DataFrame,
        group_col: str,
        minority_groups: list,
        question_col: str = "QuestionNum",
        decision_col: str = "SurveyAnswer",
        llm_validated: pd.Series = None,
    ) -> pd.DataFrame:
        """
        Produce fair ground truth per question using weighted aggregation.

        Returns DataFrame with columns:
          [QuestionNum, MajorityVote, FairVote, FairScore, N_Annotators,
           N_Minority, MinorityDecision, MajorityGroupDecision]
        """
        weights = self.compute_fair_weights(df, group_col, minority_groups, llm_validated)
        df = df.copy()
        df["_weight"] = weights

        rows = []
        for q, q_df in df.groupby(question_col):
            decisions = q_df[decision_col].values
            w = q_df["_weight"].values

            majority_vote = self.simple_majority(decisions)
            fair_score = self.weighted_vote(decisions, w)
            fair_vote = int(np.round(fair_score))

            is_minority = q_df[group_col].isin(minority_groups)
            n_minority = int(is_minority.sum())
            minority_dec = float(np.mean(decisions[is_minority])) if n_minority > 0 else np.nan
            majority_group_dec = float(np.mean(decisions[~is_minority])) if (~is_minority).sum() > 0 else np.nan

            rows.append({
                "QuestionNum": q,
                "MajorityVote": majority_vote,
                "FairVote": fair_vote,
                "FairScore": round(fair_score, 4),
                "N_Annotators": len(q_df),
                "N_Minority": n_minority,
                "MinorityMeanDecision": round(minority_dec, 4) if not np.isnan(minority_dec) else None,
                "MajorityGroupMeanDecision": round(majority_group_dec, 4) if not np.isnan(majority_group_dec) else None,
                "VoteFlipped": majority_vote != fair_vote,
            })

        return pd.DataFrame(rows)

    def evaluate_aggregation(
        self,
        results_df: pd.DataFrame,
        label_col: str = "ActualAnswer",
        original_df: pd.DataFrame = None,
    ) -> dict:
        """
        Evaluate fair aggregation against actual ground truth.
        Compares majority vote accuracy vs fair vote accuracy.
        """
        if original_df is not None:
            # Merge actual answers
            q_to_label = original_df.groupby("QuestionNum")["ActualAnswer"].first().to_dict()
            results_df = results_df.copy()
            results_df["ActualAnswer"] = results_df["QuestionNum"].map(q_to_label)

        if label_col not in results_df.columns:
            return {"error": "No ground truth labels available for evaluation"}

        valid = results_df.dropna(subset=[label_col])

        majority_correct = (valid["MajorityVote"] == valid[label_col]).mean()
        fair_correct = (valid["FairVote"] == valid[label_col]).mean()
        n_flipped = int(valid["VoteFlipped"].sum())

        return {
            "majority_vote_accuracy": round(float(majority_correct), 4),
            "fair_vote_accuracy": round(float(fair_correct), 4),
            "accuracy_delta": round(float(fair_correct - majority_correct), 4),
            "n_questions": len(valid),
            "n_vote_flipped": n_flipped,
            "flip_rate": round(n_flipped / len(valid), 4) if len(valid) > 0 else 0,
        }
