"""
Phase 3: Behavioral Signal Detection
======================================
Train a lightweight ML model to predict correctness using only
behavioral signals (metacognitive data):

  a. Click Patterns: Count of clicks (the "4-click drop" phenomenon)
  b. Decision Time: Initial hesitation vs total submission time
  c. Confidence Slider: Self-reported certainty

This model acts as a "Gatekeeper" — it flags when a human judgment
is likely unreliable due to structural or demographic complexity.

References:
  - Marimuthu, Klimenkova, Shraga (HILDA 2025): "Humans, ML, and LMs in Union"
  - Sap et al. (2022): "Annotators with Attitudes"
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score


# Behavioral feature columns from Feature_Engineered.csv
BEHAVIORAL_FEATURES = [
    "LastClick",
    "IsSingleClick",
    "DecisionTime",
    "ClickCount",
    "ConfidenceLevel",
    "DecisionTimeFract",
    "TimeDiff_FCnLC",
    "NoCY",         # Number of confidence-Yes clicks
    "NoCN",         # Number of confidence-No clicks
]


class BehavioralGatekeeper:
    """
    Predicts whether a human annotation is correct based on
    behavioral signals alone. Flags unreliable judgments.
    """

    def __init__(self, features: list = None, model_type: str = "rf"):
        self.features = features or BEHAVIORAL_FEATURES
        self.model_type = model_type
        self.scaler = StandardScaler()
        self.model = self._create_model(model_type)
        self.is_fitted = False

    def _create_model(self, model_type: str):
        if model_type == "rf":
            return RandomForestClassifier(
                n_estimators=100, max_depth=8, random_state=42, n_jobs=-1
            )
        elif model_type == "gb":
            return GradientBoostingClassifier(
                n_estimators=100, max_depth=5, random_state=42
            )
        elif model_type == "lr":
            return LogisticRegression(max_iter=1000, random_state=42)
        else:
            raise ValueError(f"Unknown model type: {model_type}")

    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract and clean behavioral features from dataframe."""
        available = [f for f in self.features if f in df.columns]
        X = df[available].copy()
        X = X.fillna(0).replace([np.inf, -np.inf], 0)
        return X.values, available

    def _prepare_target(self, df: pd.DataFrame) -> np.ndarray:
        """Target: 1 if annotator is correct, 0 if incorrect."""
        return (df["SurveyAnswer"] == df["ActualAnswer"]).astype(int).values

    def fit(self, df: pd.DataFrame):
        """Train the gatekeeper model on behavioral signals."""
        X, used_features = self._prepare_features(df)
        y = self._prepare_target(df)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.is_fitted = True
        self.used_features = used_features
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict whether each annotation is correct."""
        X, _ = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Predict probability of correctness."""
        X, _ = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def flag_unreliable(self, df: pd.DataFrame, confidence_threshold: float = 0.5) -> np.ndarray:
        """
        Flag annotations where the gatekeeper predicts low confidence
        in correctness. These are candidates for LLM arbitration.

        Returns boolean array: True = unreliable (should flag).
        """
        proba = self.predict_proba(df)
        return proba < confidence_threshold

    def evaluate(self, df: pd.DataFrame) -> dict:
        """
        Cross-validated evaluation of the gatekeeper model.
        Returns dict with accuracy, per-class metrics.
        """
        X, _ = self._prepare_features(df)
        y = self._prepare_target(df)
        X_scaled = self.scaler.fit_transform(X)

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring="accuracy")

        return {
            "cv_accuracy_mean": round(float(np.mean(scores)), 4),
            "cv_accuracy_std": round(float(np.std(scores)), 4),
            "cv_scores": [round(s, 4) for s in scores],
        }

    def feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return feature importances after fitting."""
        if not self.is_fitted:
            self.fit(df)

        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        elif hasattr(self.model, "coef_"):
            importances = np.abs(self.model.coef_[0])
        else:
            return pd.DataFrame()

        return pd.DataFrame({
            "Feature": self.used_features,
            "Importance": importances
        }).sort_values("Importance", ascending=False).reset_index(drop=True)


class DemographicDissonanceDetector:
    """
    Detects "Demographic Dissonance" — where behavioral patterns
    diverge significantly between demographic groups for the same
    table pair. Flags table pairs as "Demographically Ambiguous."

    Example: If Native speakers match a table in 10s with 1 click,
    but Non-Native speakers take 60s with 4 clicks, this gap
    signals a demographic fairness barrier.
    """

    def __init__(self, time_col: str = "DecisionTime",
                 click_col: str = "ClickCount",
                 confidence_col: str = "ConfidenceLevel"):
        self.time_col = time_col
        self.click_col = click_col
        self.confidence_col = confidence_col

    def detect_dissonance(
        self,
        df: pd.DataFrame,
        group_col: str,
        question_col: str = "QuestionNum",
        time_ratio_threshold: float = 3.0,
        click_diff_threshold: float = 2.0,
    ) -> pd.DataFrame:
        """
        For each question, compare behavioral signals across demographic groups.
        Flag questions where there is large behavioral divergence.

        Returns DataFrame of flagged questions with dissonance metrics.
        """
        groups = df[group_col].unique()
        if len(groups) < 2:
            return pd.DataFrame()

        # Aggregate per question per group
        agg = df.groupby([question_col, group_col]).agg({
            self.time_col: "mean",
            self.click_col: "mean",
            self.confidence_col: "mean",
        }).reset_index()

        flagged = []
        for q, q_df in agg.groupby(question_col):
            if len(q_df) < 2:
                continue

            time_vals = q_df[self.time_col].values
            click_vals = q_df[self.click_col].values
            conf_vals = q_df[self.confidence_col].values

            time_ratio = max(time_vals) / max(min(time_vals), 1e-10)
            click_diff = max(click_vals) - min(click_vals)
            conf_diff = max(conf_vals) - min(conf_vals)

            is_dissonant = (
                time_ratio >= time_ratio_threshold or
                click_diff >= click_diff_threshold
            )

            if is_dissonant:
                flagged.append({
                    "QuestionNum": q,
                    "TimeRatio": round(time_ratio, 2),
                    "ClickDiff": round(click_diff, 2),
                    "ConfidenceDiff": round(conf_diff, 4),
                    "IsDissonant": True,
                    "Groups": dict(zip(
                        q_df[group_col].values,
                        q_df[self.time_col].values
                    ))
                })

        return pd.DataFrame(flagged) if flagged else pd.DataFrame()
