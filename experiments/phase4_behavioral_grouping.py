"""
Phase 4: Behavioral Grouping Analysis (Tasks #4 & #5)
======================================================
Addresses April 7 meeting action items:
  "Create groups based on decision times and see whether there is
   fairness/unfairness between fast responders and others. Analyze
   based on their specific decisions -- fast decisions, medium
   decisions, slow decisions."

Part A -- Participant-Level Behavioral Grouping (Task #4):
  B1: Decision Time terciles per annotator
  B2: Confidence terciles per annotator
  B3: Click Engagement terciles per annotator
  B4: Composite Behavioral Score (engagement index)

Part B -- Decision-Level Behavioral Grouping (Task #5):
  B5: Per-decision time terciles
  B6: Per-decision confidence brackets (Low/Medium/High)
  B7: Per-decision click terciles

Part C -- Cross Behavioral x Demographic Analysis:
  B8:  Behavioral x Linguistic interaction
  B9:  Behavioral x Difficulty interaction
  B10: Summary comparison table (behavioral vs demographic disparities)

Expected findings (professor's hypothesis):
  - Fast decisions are less accurate (acceptable bias)
  - Higher confidence decisions are more accurate
  - If we DON'T see this, that itself is interesting

Uses full 10-metric FairEM disparity analysis (one-sided test, threshold=0.1).

Usage:
    cd FairPrep
    python experiments/phase4_behavioral_grouping.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact, chi2_contingency

from src.excel_data_loader import load_excel_data
from src.data_cleaning import clean_data
from src.measures import (
    AP, SP, TPR, FPR, FNR, TNR, PPV, NPV, FDR, FOR,
    calibration_gap,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXCEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Export_and_Compiled.xlsx"
)
RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "..", "results", "behavioral_grouping"
)
os.makedirs(RESULTS_DIR, exist_ok=True)

DEMOGRAPHIC_COLUMNS = ["LinguisticGroup", "ExpertiseGroup", "ExperienceGroup", "AgeGroup"]
FAIRNESS_THRESHOLD = 0.1
SEP = "=" * 72

ALL_METRICS = [
    ("AP",  AP,  "higher"),
    ("SP",  SP,  "higher"),
    ("TPR", TPR, "higher"),
    ("TNR", TNR, "higher"),
    ("PPV", PPV, "higher"),
    ("NPV", NPV, "higher"),
    ("FPR", FPR, "lower"),
    ("FNR", FNR, "lower"),
    ("FDR", FDR, "lower"),
    ("FOR", FOR, "lower"),
]

# Hard questions from phase3 performance-based difficulty.
# Q6 is hard across all versions; Q2 and Q5 are hard in most versions.
HARD_QUESTIONS = [2, 5, 6]

# Nina's per-version difficulty (from phase3_performance_difficulty.py).
# We load this dynamically if the CSV is available; otherwise fall back to
# static classification above.
NINA_DIFF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Fairness", "Fairness",
    "question_difficulty_summary_second_survey.csv",
)


def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def safe_qcut(series, q, labels):
    """
    pd.qcut that handles degenerate distributions (too many ties).
    When duplicates="drop" produces fewer bins than labels, falls back
    to rank-based assignment so we always get len(labels) groups.
    """
    try:
        return pd.qcut(series, q=q, labels=labels, duplicates="drop")
    except ValueError:
        pass
    # Fallback: rank-based assignment into q groups
    ranks = series.rank(method="first")
    n = len(series)
    group_size = n / q
    result = pd.Categorical(
        [labels[min(int((r - 1) // group_size), q - 1)] for r in ranks],
        categories=labels, ordered=True,
    )
    return result


# ---------------------------------------------------------------------------
# Confusion matrix + FairEM helpers (same as phase1_excel_experiments.py)
# ---------------------------------------------------------------------------

def confusion_counts(preds, labels):
    p, l = np.asarray(preds), np.asarray(labels)
    TP = int(np.sum((p == 1) & (l == 1)))
    FP = int(np.sum((p == 1) & (l == 0)))
    TN = int(np.sum((p == 0) & (l == 0)))
    FN = int(np.sum((p == 0) & (l == 1)))
    return TP, FP, TN, FN


def is_unfair(disparity: float, direction: str, threshold: float = FAIRNESS_THRESHOLD) -> bool:
    """One-sided FairEM fairness test."""
    if direction == "higher":
        return disparity < -threshold
    else:
        return disparity > threshold


def fairness_label(disparity: float, direction: str) -> str:
    return "UNFAIR" if is_unfair(disparity, direction) else "fair  "


def compute_fairem_disparity(df, group_col):
    """
    Run the full 10-metric FairEM disparity analysis for a given grouping column.
    Returns a DataFrame with one row per group, all metrics and disparity columns.
    """
    # Overall confusion matrix
    all_TP, all_FP, all_TN, all_FN = confusion_counts(
        df["SurveyAnswer"], df["ActualAnswer"]
    )
    overall_vals = {name: fn(all_TP, all_FP, all_TN, all_FN)
                    for name, fn, _ in ALL_METRICS}
    overall_acc = df["Accuracy"].mean()

    results = []
    for gname, gdf in df.groupby(group_col):
        TP, FP, TN, FN = confusion_counts(gdf["SurveyAnswer"], gdf["ActualAnswer"])
        row = {
            "Group": gname,
            "N": len(gdf),
            "Accuracy": round(gdf["Accuracy"].mean(), 4),
            "Acc_Disparity": round(gdf["Accuracy"].mean() - overall_acc, 4),
            "TP": TP, "FP": FP, "TN": TN, "FN": FN,
        }
        unfair_metrics = []
        for mname, mfn, direction in ALL_METRICS:
            val = mfn(TP, FP, TN, FN)
            disp = val - overall_vals[mname]
            row[mname] = round(val, 4)
            row[f"{mname}_d"] = round(disp, 4)
            row[f"{mname}_unfair"] = is_unfair(disp, direction)
            if is_unfair(disp, direction):
                unfair_metrics.append(f"{mname}(d={disp:+.3f})")
        row["Unfair_Metrics"] = "; ".join(unfair_metrics) if unfair_metrics else ""
        row["N_Unfair"] = len(unfair_metrics)

        # Behavioral stats
        for col in ["DecisionTime", "ConfidenceLevel", "ClickCount"]:
            if col in gdf.columns:
                row[f"Mean_{col}"] = round(float(gdf[col].mean()), 2)
                row[f"Median_{col}"] = round(float(gdf[col].median()), 2)

        results.append(row)

    return pd.DataFrame(results)


def print_fairem_table(rdf, group_col_name):
    """Print a compact FairEM disparity table to stdout."""
    print(f"\n  {'Group':22s} | {'N':>5s} | {'Acc':>6s} | {'AP_d':>7s} | {'TPR_d':>7s} | "
          f"{'FPR_d':>7s} | {'FNR_d':>7s} | Unfair Metrics")
    print("  " + "-" * 100)
    for _, r in rdf.iterrows():
        flag = r.get("Unfair_Metrics", "")
        ap_d = r.get("AP_d", 0)
        tpr_d = r.get("TPR_d", 0)
        fpr_d = r.get("FPR_d", 0)
        fnr_d = r.get("FNR_d", 0)
        print(f"  {str(r['Group']):22s} | {r['N']:5d} | {r['Accuracy']:.4f} | "
              f"{ap_d:+.4f} | {tpr_d:+.4f} | {fpr_d:+.4f} | {fnr_d:+.4f} | "
              f"{flag if flag else 'FAIR (all 10)'}")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_and_prepare():
    """Load data, apply cleaning, merge difficulty."""
    hdr("Loading & Cleaning Data")

    df = load_excel_data(EXCEL_PATH)
    print(f"  Raw: {len(df)} rows, {df['ResponseId'].nunique()} annotators")

    df, report = clean_data(df)
    print(f"  Cleaned: {len(df)} rows, {df['ResponseId'].nunique()} annotators")

    # Merge Nina's performance-based difficulty if available
    if os.path.exists(NINA_DIFF_PATH):
        nina = pd.read_csv(NINA_DIFF_PATH)
        nina = nina[["SurveyVersion", "QuestionNum", "difficulty_performance"]]
        nina = nina.rename(columns={"difficulty_performance": "DiffPerformance"})
        df = df.merge(nina, on=["SurveyVersion", "QuestionNum"], how="left")
        df["IsHard_Perf"] = (df["DiffPerformance"] == "Hard").astype(int)
        print(f"  Performance-based difficulty merged from Nina's CSV")
    else:
        # Fallback to static hard questions
        df["DiffPerformance"] = np.where(
            df["QuestionNum"].isin(HARD_QUESTIONS), "Hard", "Easy"
        )
        df["IsHard_Perf"] = df["QuestionNum"].isin(HARD_QUESTIONS).astype(int)
        print(f"  Using static difficulty classification (Q2,Q5,Q6 = Hard)")

    # Also keep static difficulty for comparison
    df["IsHard_Static"] = df["QuestionNum"].isin(HARD_QUESTIONS).astype(int)

    return df


# ===================================================================
# PART A: PARTICIPANT-LEVEL BEHAVIORAL GROUPING (Task #4)
# ===================================================================

# -------------------------------------------------------------------
# B1: Decision Time Grouping (Participant-Level)
# -------------------------------------------------------------------

def experiment_B1_time_participant(df):
    hdr("B1: Participant-Level Decision Time Grouping")
    print("  Method: Compute each annotator's median DecisionTime, classify into terciles")
    print("  Expected: Fast < Medium < Slow in accuracy (acceptable bias)")

    # Compute per-annotator median decision time
    ann_stats = df.groupby("ResponseId").agg(
        MedianTime=("DecisionTime", "median"),
        MeanTime=("DecisionTime", "mean"),
        MeanAcc=("Accuracy", "mean"),
    ).reset_index()

    # Tercile classification
    ann_stats["TimeGroup"] = pd.qcut(
        ann_stats["MedianTime"], q=3, labels=["Fast", "Medium", "Slow"]
    )

    # Print tercile boundaries
    boundaries = ann_stats.groupby("TimeGroup")["MedianTime"].agg(["min", "max"])
    print(f"\n  Tercile boundaries (median DecisionTime per annotator):")
    for g in ["Fast", "Medium", "Slow"]:
        if g in boundaries.index:
            print(f"    {g:8s}: {boundaries.loc[g, 'min']:.1f}s - {boundaries.loc[g, 'max']:.1f}s "
                  f"({len(ann_stats[ann_stats['TimeGroup'] == g])} annotators)")

    # Merge grouping back onto full data
    df = df.merge(ann_stats[["ResponseId", "TimeGroup"]], on="ResponseId", how="left")

    # Run FairEM disparity
    rdf = compute_fairem_disparity(df, "TimeGroup")
    rdf.insert(0, "Partition", "TimeGroup_Participant")
    print_fairem_table(rdf, "TimeGroup")

    # Check professor's expectation
    fast_acc = rdf[rdf["Group"] == "Fast"]["Accuracy"].values[0]
    slow_acc = rdf[rdf["Group"] == "Slow"]["Accuracy"].values[0]
    print(f"\n  Expected pattern (Fast < Medium < Slow accuracy):")
    print(f"    Fast={fast_acc:.4f}, Slow={slow_acc:.4f}, Gap={slow_acc - fast_acc:+.4f}")
    if fast_acc < slow_acc:
        print(f"    --> CONFIRMED: Slower annotators are more accurate (acceptable bias)")
    else:
        print(f"    --> UNEXPECTED: Fast annotators are NOT less accurate!")

    rdf.to_csv(os.path.join(RESULTS_DIR, "B1_time_participant.csv"), index=False)
    print(f"\n  -> B1_time_participant.csv")
    return df, rdf


# -------------------------------------------------------------------
# B2: Confidence Grouping (Participant-Level)
# -------------------------------------------------------------------

def experiment_B2_confidence_participant(df):
    hdr("B2: Participant-Level Confidence Grouping")
    print("  Method: Compute each annotator's median ConfidenceLevel, classify into terciles")
    print("  Expected: Higher confidence = higher accuracy")

    ann_stats = df.groupby("ResponseId").agg(
        MedianConf=("ConfidenceLevel", "median"),
        MeanAcc=("Accuracy", "mean"),
    ).reset_index()

    ann_stats["ConfGroup"] = safe_qcut(
        ann_stats["MedianConf"], q=3,
        labels=["Low-Confidence", "Med-Confidence", "High-Confidence"],
    )

    boundaries = ann_stats.groupby("ConfGroup")["MedianConf"].agg(["min", "max"])
    print(f"\n  Tercile boundaries (median ConfidenceLevel per annotator):")
    for g in ["Low-Confidence", "Med-Confidence", "High-Confidence"]:
        if g in boundaries.index:
            print(f"    {g:18s}: {boundaries.loc[g, 'min']:.0f} - {boundaries.loc[g, 'max']:.0f} "
                  f"({len(ann_stats[ann_stats['ConfGroup'] == g])} annotators)")

    df = df.merge(ann_stats[["ResponseId", "ConfGroup"]], on="ResponseId", how="left")

    rdf = compute_fairem_disparity(df, "ConfGroup")
    rdf.insert(0, "Partition", "ConfGroup_Participant")
    print_fairem_table(rdf, "ConfGroup")

    # Check expectation
    groups_sorted = rdf.sort_values("Mean_ConfidenceLevel") if "Mean_ConfidenceLevel" in rdf.columns else rdf
    lo = rdf[rdf["Group"] == "Low-Confidence"]["Accuracy"].values
    hi = rdf[rdf["Group"] == "High-Confidence"]["Accuracy"].values
    if len(lo) > 0 and len(hi) > 0:
        print(f"\n  Expected pattern (Low-Conf < High-Conf accuracy):")
        print(f"    Low-Conf={lo[0]:.4f}, High-Conf={hi[0]:.4f}, Gap={hi[0] - lo[0]:+.4f}")
        if lo[0] < hi[0]:
            print(f"    --> CONFIRMED: Higher confidence = higher accuracy")
        else:
            print(f"    --> UNEXPECTED: Low confidence NOT less accurate!")

    rdf.to_csv(os.path.join(RESULTS_DIR, "B2_confidence_participant.csv"), index=False)
    print(f"\n  -> B2_confidence_participant.csv")
    return df, rdf


# -------------------------------------------------------------------
# B3: Click Engagement Grouping (Participant-Level)
# -------------------------------------------------------------------

def experiment_B3_click_participant(df):
    hdr("B3: Participant-Level Click Engagement Grouping")
    print("  Method: Compute each annotator's median ClickCount, classify into terciles")

    ann_stats = df.groupby("ResponseId").agg(
        MedianClicks=("ClickCount", "median"),
        MeanAcc=("Accuracy", "mean"),
    ).reset_index()

    ann_stats["ClickGroup"] = safe_qcut(
        ann_stats["MedianClicks"], q=3,
        labels=["Low-Engagement", "Med-Engagement", "High-Engagement"],
    )

    boundaries = ann_stats.groupby("ClickGroup")["MedianClicks"].agg(["min", "max"])
    print(f"\n  Tercile boundaries (median ClickCount per annotator):")
    for g in ["Low-Engagement", "Med-Engagement", "High-Engagement"]:
        if g in boundaries.index:
            print(f"    {g:18s}: {boundaries.loc[g, 'min']:.0f} - {boundaries.loc[g, 'max']:.0f} "
                  f"({len(ann_stats[ann_stats['ClickGroup'] == g])} annotators)")

    df = df.merge(ann_stats[["ResponseId", "ClickGroup"]], on="ResponseId", how="left")

    rdf = compute_fairem_disparity(df, "ClickGroup")
    rdf.insert(0, "Partition", "ClickGroup_Participant")
    print_fairem_table(rdf, "ClickGroup")

    rdf.to_csv(os.path.join(RESULTS_DIR, "B3_click_participant.csv"), index=False)
    print(f"\n  -> B3_click_participant.csv")
    return df, rdf


# -------------------------------------------------------------------
# B4: Composite Behavioral Score (Participant-Level)
# -------------------------------------------------------------------

def experiment_B4_composite(df):
    hdr("B4: Composite Behavioral Engagement Score (Participant-Level)")
    print("  Method: Combine normalized DecisionTime + ClickCount + ConfidenceLevel")
    print("  Higher composite = more engaged (more time + more clicks + higher confidence)")

    ann_stats = df.groupby("ResponseId").agg(
        MedianTime=("DecisionTime", "median"),
        MedianClicks=("ClickCount", "median"),
        MedianConf=("ConfidenceLevel", "median"),
        MeanAcc=("Accuracy", "mean"),
    ).reset_index()

    # Normalize each component to [0, 1] using min-max scaling
    for col in ["MedianTime", "MedianClicks", "MedianConf"]:
        mn, mx = ann_stats[col].min(), ann_stats[col].max()
        if mx > mn:
            ann_stats[f"{col}_norm"] = (ann_stats[col] - mn) / (mx - mn)
        else:
            ann_stats[f"{col}_norm"] = 0.5

    # Composite = equal-weight average of the three normalized components
    ann_stats["CompositeScore"] = (
        ann_stats["MedianTime_norm"]
        + ann_stats["MedianClicks_norm"]
        + ann_stats["MedianConf_norm"]
    ) / 3.0

    ann_stats["EngagementGroup"] = safe_qcut(
        ann_stats["CompositeScore"], q=3,
        labels=["Disengaged", "Moderate", "Highly-Engaged"],
    )

    boundaries = ann_stats.groupby("EngagementGroup")["CompositeScore"].agg(["min", "max"])
    print(f"\n  Composite score tercile boundaries:")
    for g in ["Disengaged", "Moderate", "Highly-Engaged"]:
        if g in boundaries.index:
            n = len(ann_stats[ann_stats["EngagementGroup"] == g])
            mean_acc = ann_stats[ann_stats["EngagementGroup"] == g]["MeanAcc"].mean()
            print(f"    {g:18s}: score={boundaries.loc[g, 'min']:.3f}-{boundaries.loc[g, 'max']:.3f} "
                  f"({n} annotators, mean_acc={mean_acc:.3f})")

    df = df.merge(ann_stats[["ResponseId", "EngagementGroup", "CompositeScore"]], on="ResponseId", how="left")

    rdf = compute_fairem_disparity(df, "EngagementGroup")
    rdf.insert(0, "Partition", "EngagementGroup_Participant")
    print_fairem_table(rdf, "EngagementGroup")

    rdf.to_csv(os.path.join(RESULTS_DIR, "B4_composite_participant.csv"), index=False)
    print(f"\n  -> B4_composite_participant.csv")
    return df, rdf


# ===================================================================
# PART B: DECISION-LEVEL BEHAVIORAL GROUPING (Task #5)
# ===================================================================

# -------------------------------------------------------------------
# B5: Per-Decision Time Grouping
# -------------------------------------------------------------------

def experiment_B5_time_decision(df):
    hdr("B5: Decision-Level Time Grouping")
    print("  Method: Classify EACH decision (row) as Fast/Medium/Slow by DecisionTime terciles")
    print("  Same annotator can have both fast and slow decisions")

    df["DecisionSpeedGroup"] = pd.qcut(
        df["DecisionTime"], q=3, labels=["Fast", "Medium", "Slow"],
        duplicates="drop",
    )

    boundaries = df.groupby("DecisionSpeedGroup")["DecisionTime"].agg(["min", "max"])
    print(f"\n  Tercile boundaries (per-decision DecisionTime):")
    for g in ["Fast", "Medium", "Slow"]:
        if g in boundaries.index:
            n = len(df[df["DecisionSpeedGroup"] == g])
            acc = df[df["DecisionSpeedGroup"] == g]["Accuracy"].mean()
            print(f"    {g:8s}: {boundaries.loc[g, 'min']:.1f}s - {boundaries.loc[g, 'max']:.1f}s "
                  f"({n} decisions, acc={acc:.3f})")

    rdf = compute_fairem_disparity(df, "DecisionSpeedGroup")
    rdf.insert(0, "Partition", "DecisionSpeedGroup_Decision")
    print_fairem_table(rdf, "DecisionSpeedGroup")

    # Compare with participant-level (B1)
    print(f"\n  Decision-level vs Participant-level comparison:")
    print(f"    Decision-level captures within-annotator variation")
    print(f"    N annotators with decisions in multiple speed groups: ", end="")
    ann_groups = df.groupby("ResponseId")["DecisionSpeedGroup"].nunique()
    multi = (ann_groups > 1).sum()
    print(f"{multi}/{len(ann_groups)} ({multi/len(ann_groups)*100:.0f}%)")

    rdf.to_csv(os.path.join(RESULTS_DIR, "B5_time_decision.csv"), index=False)
    print(f"\n  -> B5_time_decision.csv")
    return df, rdf


# -------------------------------------------------------------------
# B6: Per-Decision Confidence Grouping
# -------------------------------------------------------------------

def experiment_B6_confidence_decision(df):
    hdr("B6: Decision-Level Confidence Grouping")
    print("  Method: Classify each decision by ConfidenceLevel: Low (<40), Med (40-70), High (>70)")

    df["DecisionConfGroup"] = pd.cut(
        df["ConfidenceLevel"],
        bins=[0, 40, 70, 100],
        labels=["Low-Conf", "Med-Conf", "High-Conf"],
        include_lowest=True,
    )

    for g in ["Low-Conf", "Med-Conf", "High-Conf"]:
        sub = df[df["DecisionConfGroup"] == g]
        if len(sub) > 0:
            print(f"    {g:12s}: {len(sub)} decisions, acc={sub['Accuracy'].mean():.3f}, "
                  f"conf={sub['ConfidenceLevel'].mean():.1f}")

    rdf = compute_fairem_disparity(df, "DecisionConfGroup")
    rdf.insert(0, "Partition", "DecisionConfGroup_Decision")
    print_fairem_table(rdf, "DecisionConfGroup")

    rdf.to_csv(os.path.join(RESULTS_DIR, "B6_confidence_decision.csv"), index=False)
    print(f"\n  -> B6_confidence_decision.csv")
    return df, rdf


# -------------------------------------------------------------------
# B7: Per-Decision Click Grouping
# -------------------------------------------------------------------

def experiment_B7_click_decision(df):
    hdr("B7: Decision-Level Click Grouping")
    print("  Method: Classify each decision by ClickCount terciles")

    df["DecisionClickGroup"] = safe_qcut(
        df["ClickCount"], q=3,
        labels=["Low-Clicks", "Med-Clicks", "High-Clicks"],
    )

    for g in ["Low-Clicks", "Med-Clicks", "High-Clicks"]:
        sub = df[df["DecisionClickGroup"] == g]
        if len(sub) > 0:
            print(f"    {g:14s}: {len(sub)} decisions, acc={sub['Accuracy'].mean():.3f}, "
                  f"clicks={sub['ClickCount'].mean():.1f}")

    rdf = compute_fairem_disparity(df, "DecisionClickGroup")
    rdf.insert(0, "Partition", "DecisionClickGroup_Decision")
    print_fairem_table(rdf, "DecisionClickGroup")

    rdf.to_csv(os.path.join(RESULTS_DIR, "B7_click_decision.csv"), index=False)
    print(f"\n  -> B7_click_decision.csv")
    return df, rdf


# ===================================================================
# PART C: CROSS BEHAVIORAL x DEMOGRAPHIC ANALYSIS
# ===================================================================

# -------------------------------------------------------------------
# B8: Behavioral x Linguistic Interaction
# -------------------------------------------------------------------

def experiment_B8_behavioral_x_linguistic(df):
    hdr("B8: Decision-Level Speed x Linguistic Group Interaction")
    print("  Key question: Do fast Non-Native decisions show the same bias")
    print("  as slow Non-Native decisions?")

    if "DecisionSpeedGroup" not in df.columns:
        df["DecisionSpeedGroup"] = pd.qcut(
            df["DecisionTime"], q=3, labels=["Fast", "Medium", "Slow"],
            duplicates="drop",
        )

    results = []
    overall_acc = df["Accuracy"].mean()

    print(f"\n  Overall accuracy: {overall_acc:.4f}")
    print(f"\n  {'Speed':8s} x {'Linguistic':12s} | {'N':>5s} | {'Acc':>6s} | {'Acc_d':>7s} | Fisher p")
    print("  " + "-" * 70)

    for speed in ["Fast", "Medium", "Slow"]:
        for ling in ["Native", "Non-Native"]:
            sub = df[(df["DecisionSpeedGroup"] == speed) & (df["LinguisticGroup"] == ling)]
            if len(sub) == 0:
                continue

            acc = sub["Accuracy"].mean()
            acc_d = acc - overall_acc

            # Fisher exact: this cell vs rest
            n_correct = int(sub["Accuracy"].sum())
            n_wrong = len(sub) - n_correct
            rest = df[~((df["DecisionSpeedGroup"] == speed) & (df["LinguisticGroup"] == ling))]
            rest_correct = int(rest["Accuracy"].sum())
            rest_wrong = len(rest) - rest_correct
            table = [[n_correct, n_wrong], [rest_correct, rest_wrong]]
            try:
                odds, p = fisher_exact(table)
            except ValueError:
                odds, p = 1.0, 1.0

            sig = " *" if p < 0.05 else "  **" if p < 0.01 else ""
            print(f"  {speed:8s} x {ling:12s} | {len(sub):5d} | {acc:.4f} | {acc_d:+.4f} | "
                  f"p={p:.4f}{sig}")

            results.append({
                "SpeedGroup": speed, "LinguisticGroup": ling,
                "N": len(sub), "Accuracy": round(acc, 4),
                "Acc_Disparity": round(acc_d, 4),
                "Fisher_p": round(p, 4), "OddsRatio": round(odds, 4),
            })

    rdf = pd.DataFrame(results)

    # Compute Native-NonNative gap within each speed group
    print(f"\n  Native vs Non-Native gap within each speed group:")
    for speed in ["Fast", "Medium", "Slow"]:
        nat_row = rdf[(rdf["SpeedGroup"] == speed) & (rdf["LinguisticGroup"] == "Native")]
        nn_row = rdf[(rdf["SpeedGroup"] == speed) & (rdf["LinguisticGroup"] == "Non-Native")]
        if len(nat_row) > 0 and len(nn_row) > 0:
            gap = nn_row.iloc[0]["Accuracy"] - nat_row.iloc[0]["Accuracy"]
            # Fisher exact for the gap
            nat_sub = df[(df["DecisionSpeedGroup"] == speed) & (df["LinguisticGroup"] == "Native")]
            nn_sub = df[(df["DecisionSpeedGroup"] == speed) & (df["LinguisticGroup"] == "Non-Native")]
            table2 = [
                [int(nat_sub["Accuracy"].sum()), len(nat_sub) - int(nat_sub["Accuracy"].sum())],
                [int(nn_sub["Accuracy"].sum()), len(nn_sub) - int(nn_sub["Accuracy"].sum())],
            ]
            try:
                odds2, p2 = fisher_exact(table2)
            except ValueError:
                odds2, p2 = 1.0, 1.0
            sig = "***" if p2 < 0.01 else "**" if p2 < 0.05 else "*" if p2 < 0.1 else ""
            print(f"    {speed:8s}: NN-Nat gap = {gap:+.4f}, Fisher p = {p2:.4f} {sig}")

    rdf.to_csv(os.path.join(RESULTS_DIR, "B8_speed_x_linguistic.csv"), index=False)
    print(f"\n  -> B8_speed_x_linguistic.csv")
    return rdf


# -------------------------------------------------------------------
# B9: Behavioral x Difficulty Interaction
# -------------------------------------------------------------------

def experiment_B9_behavioral_x_difficulty(df):
    hdr("B9: Decision-Level Speed x Question Difficulty Interaction")
    print("  Key question: Are fast decisions on Hard questions worse than slow on Hard?")

    if "DecisionSpeedGroup" not in df.columns:
        df["DecisionSpeedGroup"] = pd.qcut(
            df["DecisionTime"], q=3, labels=["Fast", "Medium", "Slow"],
            duplicates="drop",
        )

    # Use performance-based difficulty if available, else static
    diff_col = "DiffPerformance" if "DiffPerformance" in df.columns else None
    if diff_col is None:
        df["DiffPerformance"] = np.where(
            df["QuestionNum"].isin(HARD_QUESTIONS), "Hard", "Easy"
        )
        diff_col = "DiffPerformance"

    difficulty_levels = sorted(df[diff_col].dropna().unique())

    results = []
    overall_acc = df["Accuracy"].mean()

    print(f"\n  {'Speed':8s} x {'Difficulty':10s} | {'N':>5s} | {'Acc':>6s} | {'Acc_d':>7s}")
    print("  " + "-" * 55)

    for speed in ["Fast", "Medium", "Slow"]:
        for diff in difficulty_levels:
            sub = df[(df["DecisionSpeedGroup"] == speed) & (df[diff_col] == diff)]
            if len(sub) == 0:
                continue
            acc = sub["Accuracy"].mean()
            acc_d = acc - overall_acc
            print(f"  {speed:8s} x {diff:10s} | {len(sub):5d} | {acc:.4f} | {acc_d:+.4f}")
            results.append({
                "SpeedGroup": speed, "Difficulty": diff,
                "N": len(sub), "Accuracy": round(acc, 4),
                "Acc_Disparity": round(acc_d, 4),
            })

    rdf = pd.DataFrame(results)

    # Key comparison: Fast-Hard vs Slow-Hard
    fast_hard = rdf[(rdf["SpeedGroup"] == "Fast") & (rdf["Difficulty"] == "Hard")]
    slow_hard = rdf[(rdf["SpeedGroup"] == "Slow") & (rdf["Difficulty"] == "Hard")]
    if len(fast_hard) > 0 and len(slow_hard) > 0:
        gap = slow_hard.iloc[0]["Accuracy"] - fast_hard.iloc[0]["Accuracy"]
        print(f"\n  Key comparison:")
        print(f"    Fast-Hard acc = {fast_hard.iloc[0]['Accuracy']:.4f}")
        print(f"    Slow-Hard acc = {slow_hard.iloc[0]['Accuracy']:.4f}")
        print(f"    Gap (Slow-Fast) = {gap:+.4f}")
        if gap > 0:
            print(f"    --> Slower decisions on hard questions ARE more accurate")
        else:
            print(f"    --> UNEXPECTED: Slow decisions on hard questions NOT more accurate")

    # Also cross with Linguistic for a 3-way interaction
    print(f"\n  3-way interaction: Speed x Difficulty x Linguistic")
    print(f"  {'Speed':8s} x {'Diff':6s} x {'Ling':12s} | {'N':>4s} | {'Acc':>6s}")
    print("  " + "-" * 55)
    three_way = []
    for speed in ["Fast", "Medium", "Slow"]:
        for diff in ["Hard", "Easy"] if "Hard" in difficulty_levels else difficulty_levels:
            for ling in ["Native", "Non-Native"]:
                sub = df[(df["DecisionSpeedGroup"] == speed) &
                         (df[diff_col] == diff) &
                         (df["LinguisticGroup"] == ling)]
                if len(sub) < 2:
                    continue
                acc = sub["Accuracy"].mean()
                print(f"  {speed:8s} x {diff:6s} x {ling:12s} | {len(sub):4d} | {acc:.4f}")
                three_way.append({
                    "SpeedGroup": speed, "Difficulty": diff, "LinguisticGroup": ling,
                    "N": len(sub), "Accuracy": round(acc, 4),
                })

    rdf_3way = pd.DataFrame(three_way)
    rdf.to_csv(os.path.join(RESULTS_DIR, "B9_speed_x_difficulty.csv"), index=False)
    rdf_3way.to_csv(os.path.join(RESULTS_DIR, "B9_3way_interaction.csv"), index=False)
    print(f"\n  -> B9_speed_x_difficulty.csv + B9_3way_interaction.csv")
    return rdf, rdf_3way


# -------------------------------------------------------------------
# B10: Summary Comparison Table
# -------------------------------------------------------------------

def experiment_B10_summary(df, b1_df, b2_df, b3_df, b4_df, b5_df, b6_df, b7_df):
    hdr("B10: Summary — Behavioral vs Demographic Grouping Comparison")

    # Collect all behavioral grouping results
    all_behavioral = pd.concat([b1_df, b2_df, b3_df, b4_df, b5_df, b6_df, b7_df], ignore_index=True)

    # Run demographic groupings for comparison
    demographic_results = []
    for gc in DEMOGRAPHIC_COLUMNS:
        rdf = compute_fairem_disparity(df, gc)
        rdf.insert(0, "Partition", gc)
        demographic_results.append(rdf)
    all_demographic = pd.concat(demographic_results, ignore_index=True)

    # Summary: max absolute AP disparity per partition
    print(f"\n  Maximum |AP Disparity| per Grouping:")
    print(f"  {'Partition':35s} | {'Max |AP_d|':>10s} | {'Worst Group':22s} | {'N_Unfair':>8s}")
    print("  " + "-" * 85)

    summary_rows = []
    for label, results_df in [("BEHAVIORAL", all_behavioral), ("DEMOGRAPHIC", all_demographic)]:
        for partition in results_df["Partition"].unique():
            sub = results_df[results_df["Partition"] == partition]
            max_ap_d_row = sub.loc[sub["AP_d"].abs().idxmax()]
            max_tpr_d_row = sub.loc[sub["TPR_d"].abs().idxmax()]
            n_unfair = int(sub["N_Unfair"].sum())
            print(f"  {partition:35s} | {abs(max_ap_d_row['AP_d']):10.4f} | "
                  f"{str(max_ap_d_row['Group']):22s} | {n_unfair:8d}")
            summary_rows.append({
                "Type": label,
                "Partition": partition,
                "Max_AP_d": round(abs(max_ap_d_row["AP_d"]), 4),
                "Worst_Group_AP": max_ap_d_row["Group"],
                "Max_TPR_d": round(abs(max_tpr_d_row["TPR_d"]), 4),
                "Worst_Group_TPR": max_tpr_d_row["Group"],
                "N_Unfair_Metrics": n_unfair,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "B10_summary.csv"), index=False)

    # Master comparison table
    master = pd.concat([all_behavioral, all_demographic], ignore_index=True)
    master.to_csv(os.path.join(RESULTS_DIR, "behavioral_grouping_summary.csv"), index=False)

    # Key question: which produces larger disparities?
    beh_max = all_behavioral["AP_d"].abs().max()
    dem_max = all_demographic["AP_d"].abs().max()
    print(f"\n  Overall max behavioral AP disparity : {beh_max:.4f}")
    print(f"  Overall max demographic AP disparity: {dem_max:.4f}")
    if beh_max > dem_max:
        print(f"  --> Behavioral grouping produces LARGER disparities than demographics")
        print(f"      This suggests behavioral patterns (speed, engagement) are a stronger")
        print(f"      source of performance variation than demographic identity")
    else:
        print(f"  --> Demographic grouping produces LARGER disparities than behavioral")
        print(f"      Demographic bias is NOT fully explained by behavioral differences")

    # Is demographic bias explained away by behavioral differences?
    print(f"\n  Correlation between behavioral engagement and demographic groups:")
    for gc in DEMOGRAPHIC_COLUMNS:
        if "CompositeScore" in df.columns:
            group_scores = df.groupby(gc)["CompositeScore"].mean()
            groups = sorted(group_scores.index)
            if len(groups) == 2:
                g1_scores = df[df[gc] == groups[0]]["CompositeScore"]
                g2_scores = df[df[gc] == groups[1]]["CompositeScore"]
                from scipy.stats import mannwhitneyu
                try:
                    u_stat, u_p = mannwhitneyu(g1_scores, g2_scores, alternative="two-sided")
                except ValueError:
                    u_p = 1.0
                print(f"    {gc:25s}: {groups[0]}={g1_scores.mean():.3f} vs "
                      f"{groups[1]}={g2_scores.mean():.3f} (MWU p={u_p:.4f})")

    print(f"\n  -> B10_summary.csv + behavioral_grouping_summary.csv")
    return summary_df


# ===================================================================
# Visualizations
# ===================================================================

def plot_behavioral_vs_demographic(df, b1_df, b2_df, b5_df, b6_df):
    """Create comparison visualization: behavioral vs demographic disparities."""
    hdr("Generating Visualizations")

    fig, axes = plt.subplots(2, 3, figsize=(20, 12))

    # ---- Panel 1: Participant-level time groups (B1) ----
    ax = axes[0, 0]
    if len(b1_df) > 0:
        colors_time = {"Fast": "#EF5350", "Medium": "#FFA726", "Slow": "#66BB6A"}
        bars = ax.bar(b1_df["Group"], b1_df["Accuracy"],
                      color=[colors_time.get(g, "#90A4AE") for g in b1_df["Group"]],
                      alpha=0.85)
        for i, (_, r) in enumerate(b1_df.iterrows()):
            ax.text(i, r["Accuracy"] + 0.01, f'{r["Accuracy"]:.3f}\n(n={r["N"]})',
                    ha="center", fontsize=9)
        ax.set_title("B1: Participant Time Groups", fontsize=10, fontweight="bold")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.0)
        ax.axhline(df["Accuracy"].mean(), color="gray", linestyle="--", linewidth=0.8,
                   label=f"Overall ({df['Accuracy'].mean():.3f})")
        ax.legend(fontsize=8)

    # ---- Panel 2: Participant-level confidence groups (B2) ----
    ax = axes[0, 1]
    if len(b2_df) > 0:
        colors_conf = {"Low-Confidence": "#EF5350", "Med-Confidence": "#FFA726",
                       "High-Confidence": "#66BB6A"}
        bars = ax.bar(b2_df["Group"], b2_df["Accuracy"],
                      color=[colors_conf.get(g, "#90A4AE") for g in b2_df["Group"]],
                      alpha=0.85)
        for i, (_, r) in enumerate(b2_df.iterrows()):
            ax.text(i, r["Accuracy"] + 0.01, f'{r["Accuracy"]:.3f}\n(n={r["N"]})',
                    ha="center", fontsize=9)
        ax.set_title("B2: Participant Confidence Groups", fontsize=10, fontweight="bold")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.0)
        ax.axhline(df["Accuracy"].mean(), color="gray", linestyle="--", linewidth=0.8)

    # ---- Panel 3: Decision-level time vs participant-level time ----
    ax = axes[0, 2]
    if len(b1_df) > 0 and len(b5_df) > 0:
        x = np.arange(3)
        w = 0.35
        part_acc = [b1_df[b1_df["Group"] == g]["Accuracy"].values[0]
                    if len(b1_df[b1_df["Group"] == g]) > 0 else 0
                    for g in ["Fast", "Medium", "Slow"]]
        dec_acc = [b5_df[b5_df["Group"] == g]["Accuracy"].values[0]
                   if len(b5_df[b5_df["Group"] == g]) > 0 else 0
                   for g in ["Fast", "Medium", "Slow"]]
        ax.bar(x - w/2, part_acc, w, label="Participant-Level", color="#1E88E5", alpha=0.8)
        ax.bar(x + w/2, dec_acc, w, label="Decision-Level", color="#FF8F00", alpha=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Fast", "Medium", "Slow"])
        ax.set_title("B1 vs B5: Participant vs Decision Time Groups", fontsize=10, fontweight="bold")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.0)
        ax.legend(fontsize=8)

    # ---- Panel 4: AP Disparity comparison — behavioral vs demographic ----
    ax = axes[1, 0]
    # Collect behavioral AP disparities
    beh_partitions = {
        "Time(Part)": b1_df,
        "Conf(Part)": b2_df,
        "Time(Dec)": b5_df,
        "Conf(Dec)": b6_df,
    }
    beh_max_d = []
    for name, rdf in beh_partitions.items():
        if len(rdf) > 0:
            max_d = rdf["AP_d"].abs().max()
            beh_max_d.append({"Partition": name, "Max_AP_d": max_d, "Type": "Behavioral"})

    # Demographic AP disparities
    for gc in DEMOGRAPHIC_COLUMNS:
        rdf = compute_fairem_disparity(df, gc)
        max_d = rdf["AP_d"].abs().max()
        short = gc.replace("Group", "")
        beh_max_d.append({"Partition": short, "Max_AP_d": max_d, "Type": "Demographic"})

    comp_df = pd.DataFrame(beh_max_d)
    colors_type = {"Behavioral": "#1E88E5", "Demographic": "#EF5350"}
    for i, (_, r) in enumerate(comp_df.iterrows()):
        ax.barh(i, r["Max_AP_d"], color=colors_type[r["Type"]], alpha=0.8)
    ax.set_yticks(range(len(comp_df)))
    ax.set_yticklabels([f'{r["Partition"]} ({r["Type"][0]})' for _, r in comp_df.iterrows()],
                       fontsize=8)
    ax.axvline(FAIRNESS_THRESHOLD, color="red", linestyle=":", linewidth=1,
               label=f"Threshold={FAIRNESS_THRESHOLD}")
    ax.set_xlabel("Max |AP Disparity|")
    ax.set_title("Max AP Disparity: Behavioral vs Demographic", fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # ---- Panel 5: Decision-level confidence bar ----
    ax = axes[1, 1]
    if len(b6_df) > 0:
        colors_c = {"Low-Conf": "#EF5350", "Med-Conf": "#FFA726", "High-Conf": "#66BB6A"}
        bars = ax.bar(b6_df["Group"], b6_df["Accuracy"],
                      color=[colors_c.get(g, "#90A4AE") for g in b6_df["Group"]],
                      alpha=0.85)
        for i, (_, r) in enumerate(b6_df.iterrows()):
            ax.text(i, r["Accuracy"] + 0.01, f'{r["Accuracy"]:.3f}\n(n={r["N"]})',
                    ha="center", fontsize=9)
        ax.set_title("B6: Decision-Level Confidence Groups", fontsize=10, fontweight="bold")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.0)
        ax.axhline(df["Accuracy"].mean(), color="gray", linestyle="--", linewidth=0.8)

    # ---- Panel 6: Stacked hypothesis check ----
    ax = axes[1, 2]
    hypotheses = []
    # H1: Fast < Slow in accuracy?
    fast_acc_p = b1_df[b1_df["Group"] == "Fast"]["Accuracy"].values[0] if len(b1_df[b1_df["Group"] == "Fast"]) > 0 else None
    slow_acc_p = b1_df[b1_df["Group"] == "Slow"]["Accuracy"].values[0] if len(b1_df[b1_df["Group"] == "Slow"]) > 0 else None
    if fast_acc_p is not None and slow_acc_p is not None:
        hypotheses.append(("Fast < Slow (Part)", fast_acc_p < slow_acc_p))
    # H2: Low-Conf < High-Conf?
    lo_acc = b2_df[b2_df["Group"] == "Low-Confidence"]["Accuracy"].values
    hi_acc = b2_df[b2_df["Group"] == "High-Confidence"]["Accuracy"].values
    if len(lo_acc) > 0 and len(hi_acc) > 0:
        hypotheses.append(("LowConf < HighConf (Part)", lo_acc[0] < hi_acc[0]))
    # H3: Fast < Slow (Decision)?
    fast_d = b5_df[b5_df["Group"] == "Fast"]["Accuracy"].values
    slow_d = b5_df[b5_df["Group"] == "Slow"]["Accuracy"].values
    if len(fast_d) > 0 and len(slow_d) > 0:
        hypotheses.append(("Fast < Slow (Dec)", fast_d[0] < slow_d[0]))
    # H4: Low-Conf < High-Conf (Decision)?
    lo_d = b6_df[b6_df["Group"] == "Low-Conf"]["Accuracy"].values
    hi_d = b6_df[b6_df["Group"] == "High-Conf"]["Accuracy"].values
    if len(lo_d) > 0 and len(hi_d) > 0:
        hypotheses.append(("LowConf < HighConf (Dec)", lo_d[0] < hi_d[0]))

    if hypotheses:
        names, confirmed = zip(*hypotheses)
        colors_h = ["#66BB6A" if c else "#EF5350" for c in confirmed]
        ax.barh(range(len(hypotheses)), [1] * len(hypotheses), color=colors_h, alpha=0.8)
        ax.set_yticks(range(len(hypotheses)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xticks([])
        for i, c in enumerate(confirmed):
            label = "CONFIRMED" if c else "NOT CONFIRMED"
            ax.text(0.5, i, label, ha="center", va="center", fontsize=10,
                    fontweight="bold", color="white")
        ax.set_title("Hypothesis Check", fontsize=10, fontweight="bold")

    plt.suptitle("Phase 4: Behavioral Grouping Analysis — Overview\n"
                 "Blue = Behavioral, Red = Demographic; Green = confirmed hypothesis",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "behavioral_vs_demographic.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> behavioral_vs_demographic.png")


def plot_decision_level_heatmap(df):
    """Create heatmap of accuracy by (Speed x Linguistic x Difficulty)."""
    if "DecisionSpeedGroup" not in df.columns:
        return

    diff_col = "DiffPerformance" if "DiffPerformance" in df.columns else None
    if diff_col is None:
        return

    # Build cross-tab
    rows = []
    for speed in ["Fast", "Medium", "Slow"]:
        for ling in ["Native", "Non-Native"]:
            for diff in sorted(df[diff_col].dropna().unique()):
                sub = df[(df["DecisionSpeedGroup"] == speed) &
                         (df["LinguisticGroup"] == ling) &
                         (df[diff_col] == diff)]
                if len(sub) >= 2:
                    rows.append({
                        "Row": f"{speed}-{ling}",
                        "Difficulty": diff,
                        "Accuracy": round(sub["Accuracy"].mean(), 3),
                        "N": len(sub),
                    })

    if not rows:
        return

    hm_df = pd.DataFrame(rows)
    pivot_acc = hm_df.pivot(index="Row", columns="Difficulty", values="Accuracy")
    pivot_n = hm_df.pivot(index="Row", columns="Difficulty", values="N")

    # Build annotation text
    annot = pivot_acc.copy().astype(str)
    for r in annot.index:
        for c in annot.columns:
            acc_val = pivot_acc.loc[r, c] if pd.notna(pivot_acc.loc[r, c]) else 0
            n_val = pivot_n.loc[r, c] if pd.notna(pivot_n.loc[r, c]) else 0
            annot.loc[r, c] = f"{acc_val:.2f}\n(n={int(n_val)})"

    # Sort rows for consistent ordering
    row_order = [f"{s}-{l}" for s in ["Fast", "Medium", "Slow"] for l in ["Native", "Non-Native"]]
    existing_rows = [r for r in row_order if r in pivot_acc.index]
    pivot_acc = pivot_acc.loc[existing_rows]
    annot = annot.loc[existing_rows]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pivot_acc, ax=ax, annot=annot, fmt="",
                cmap="RdYlGn", vmin=0.2, vmax=1.0, linewidths=0.5,
                cbar_kws={"shrink": 0.7, "label": "Accuracy"})
    ax.set_title("Decision-Level Accuracy Heatmap\n(Speed x Linguistic x Difficulty)",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Question Difficulty")
    ax.set_ylabel("Speed Group - Linguistic Group")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "decision_level_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> decision_level_heatmap.png")


# ===================================================================
# Summary printout
# ===================================================================

def print_final_summary(df, b1, b2, b5, b6, summary_df):
    hdr("PHASE 4 FINAL SUMMARY: Behavioral Grouping Analysis")

    print("\n  PROFESSOR'S HYPOTHESES:")
    print("  -----------------------")

    # Hypothesis 1: Fast decisions less accurate
    fast_part = b1[b1["Group"] == "Fast"]["Accuracy"].values
    slow_part = b1[b1["Group"] == "Slow"]["Accuracy"].values
    fast_dec = b5[b5["Group"] == "Fast"]["Accuracy"].values
    slow_dec = b5[b5["Group"] == "Slow"]["Accuracy"].values

    print(f"\n  H1: 'Fast decisions are less accurate'")
    if len(fast_part) > 0 and len(slow_part) > 0:
        gap_p = slow_part[0] - fast_part[0]
        status_p = "CONFIRMED" if fast_part[0] < slow_part[0] else "NOT CONFIRMED"
        print(f"    Participant-level: Fast={fast_part[0]:.4f}, Slow={slow_part[0]:.4f}, "
              f"Gap={gap_p:+.4f} -- {status_p}")
    if len(fast_dec) > 0 and len(slow_dec) > 0:
        gap_d = slow_dec[0] - fast_dec[0]
        status_d = "CONFIRMED" if fast_dec[0] < slow_dec[0] else "NOT CONFIRMED"
        print(f"    Decision-level:   Fast={fast_dec[0]:.4f}, Slow={slow_dec[0]:.4f}, "
              f"Gap={gap_d:+.4f} -- {status_d}")

    # Hypothesis 2: Higher confidence = higher accuracy
    lo_part = b2[b2["Group"] == "Low-Confidence"]["Accuracy"].values
    hi_part = b2[b2["Group"] == "High-Confidence"]["Accuracy"].values
    lo_dec = b6[b6["Group"] == "Low-Conf"]["Accuracy"].values
    hi_dec = b6[b6["Group"] == "High-Conf"]["Accuracy"].values

    print(f"\n  H2: 'Higher confidence = higher accuracy'")
    if len(lo_part) > 0 and len(hi_part) > 0:
        gap_p = hi_part[0] - lo_part[0]
        status_p = "CONFIRMED" if lo_part[0] < hi_part[0] else "NOT CONFIRMED"
        print(f"    Participant-level: Low={lo_part[0]:.4f}, High={hi_part[0]:.4f}, "
              f"Gap={gap_p:+.4f} -- {status_p}")
    if len(lo_dec) > 0 and len(hi_dec) > 0:
        gap_d = hi_dec[0] - lo_dec[0]
        status_d = "CONFIRMED" if lo_dec[0] < hi_dec[0] else "NOT CONFIRMED"
        print(f"    Decision-level:   Low={lo_dec[0]:.4f}, High={hi_dec[0]:.4f}, "
              f"Gap={gap_d:+.4f} -- {status_d}")

    # FairEM unfairness flags
    print(f"\n  FAIREM UNFAIRNESS FLAGS (|disparity| > {FAIRNESS_THRESHOLD}):")
    total_beh_unfair = 0
    total_dem_unfair = 0
    if summary_df is not None:
        for _, r in summary_df.iterrows():
            if r["N_Unfair_Metrics"] > 0:
                print(f"    {r['Partition']:35s}: {r['N_Unfair_Metrics']} unfair metric(s), "
                      f"worst AP group = {r['Worst_Group_AP']} (|d|={r['Max_AP_d']:.4f})")
                if r["Type"] == "BEHAVIORAL":
                    total_beh_unfair += r["N_Unfair_Metrics"]
                else:
                    total_dem_unfair += r["N_Unfair_Metrics"]

    print(f"\n  Total behavioral UNFAIR flags : {total_beh_unfair}")
    print(f"  Total demographic UNFAIR flags: {total_dem_unfair}")

    # Acceptable bias assessment
    print(f"\n  ACCEPTABLE BIAS ASSESSMENT:")
    if len(fast_dec) > 0 and len(slow_dec) > 0:
        if fast_dec[0] < slow_dec[0]:
            print(f"    Fast-decision bias IS present (gap={slow_dec[0] - fast_dec[0]:+.4f})")
            print(f"    This is ACCEPTABLE bias -- faster decisions are naturally less considered")
        else:
            print(f"    Fast-decision bias is NOT present -- this is noteworthy")
            print(f"    Annotators who decide quickly are not systematically less accurate")

    print(f"\n  Does behavioral grouping explain demographic bias?")
    if summary_df is not None:
        beh_max = summary_df[summary_df["Type"] == "BEHAVIORAL"]["Max_AP_d"].max()
        dem_max = summary_df[summary_df["Type"] == "DEMOGRAPHIC"]["Max_AP_d"].max()
        if beh_max > dem_max:
            print(f"    Behavioral disparities ({beh_max:.4f}) > Demographic ({dem_max:.4f})")
            print(f"    --> Behavioral patterns may PARTIALLY EXPLAIN demographic differences")
        else:
            print(f"    Behavioral disparities ({beh_max:.4f}) < Demographic ({dem_max:.4f})")
            print(f"    --> Demographic bias is NOT explained by behavioral differences alone")


# ===================================================================
# Main
# ===================================================================

def main():
    hdr("Phase 4: Behavioral Grouping Analysis (Tasks #4 & #5)")
    print(f"  Excel path : {os.path.abspath(EXCEL_PATH)}")
    print(f"  Results dir: {os.path.abspath(RESULTS_DIR)}")

    df = load_and_prepare()

    # Part A: Participant-level grouping
    print(f"\n{'='*72}")
    print(f"  PART A: PARTICIPANT-LEVEL BEHAVIORAL GROUPING (Task #4)")
    print(f"{'='*72}")
    df, b1_df = experiment_B1_time_participant(df)
    df, b2_df = experiment_B2_confidence_participant(df)
    df, b3_df = experiment_B3_click_participant(df)
    df, b4_df = experiment_B4_composite(df)

    # Part B: Decision-level grouping
    print(f"\n{'='*72}")
    print(f"  PART B: DECISION-LEVEL BEHAVIORAL GROUPING (Task #5)")
    print(f"{'='*72}")
    df, b5_df = experiment_B5_time_decision(df)
    df, b6_df = experiment_B6_confidence_decision(df)
    df, b7_df = experiment_B7_click_decision(df)

    # Part C: Cross behavioral x demographic
    print(f"\n{'='*72}")
    print(f"  PART C: CROSS BEHAVIORAL x DEMOGRAPHIC ANALYSIS")
    print(f"{'='*72}")
    b8_df = experiment_B8_behavioral_x_linguistic(df)
    b9_df, b9_3way = experiment_B9_behavioral_x_difficulty(df)
    summary_df = experiment_B10_summary(df, b1_df, b2_df, b3_df, b4_df, b5_df, b6_df, b7_df)

    # Visualizations
    plot_behavioral_vs_demographic(df, b1_df, b2_df, b5_df, b6_df)
    plot_decision_level_heatmap(df)

    # Final summary
    print_final_summary(df, b1_df, b2_df, b5_df, b6_df, summary_df)

    print(f"\n{SEP}")
    print(f"  All Phase 4 results saved to: {os.path.abspath(RESULTS_DIR)}")
    print(f"  Files: B1-B10 CSVs + behavioral_vs_demographic.png + decision_level_heatmap.png")
    print(SEP)


if __name__ == "__main__":
    main()
