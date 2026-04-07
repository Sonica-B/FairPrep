"""
Phase 3: Re-Analysis with Performance-Based Difficulty Classification
=====================================================================
Addresses meeting action items (March 24, 2026):
  1. Redefine hard vs easy using PERFORMANCE-BASED method (Nina's classification)
  2. Deeper Non-Native intersectional analysis
  3. Test if findings translate across both survey versions (cross-version)

Key shift: Difficulty is now PER (SurveyVersion, QuestionNum), not per QuestionNum alone.
Example: Q6 is Hard in V1 (50% acc) but Easy in V4 (78.6% acc).

Uses Nina's question_difficulty_summary_first_survey.csv for classification.

Usage:
    cd FairPrep
    python experiments/phase3_performance_difficulty.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.excel_data_loader import load_excel_data, get_annotator_demographics
from src.measures import AP, TPR, FPR, FNR, TNR, PPV, NPV, FDR, FOR, SP

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Export_and_Compiled.xlsx")
NINA_DIFF_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Fairness", "Fairness",
    "question_difficulty_summary_second_survey.csv"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "phase3_perf_difficulty")
os.makedirs(RESULTS_DIR, exist_ok=True)

GROUP_COLUMNS = ["LinguisticGroup", "ExpertiseGroup", "ExperienceGroup", "AgeGroup"]
FAIRNESS_THRESHOLD = 0.1
SEP = "=" * 72

ALL_METRICS = [
    ("AP",  AP,  "higher"), ("SP",  SP,  "higher"), ("TPR", TPR, "higher"),
    ("TNR", TNR, "higher"), ("PPV", PPV, "higher"), ("NPV", NPV, "higher"),
    ("FPR", FPR, "lower"),  ("FNR", FNR, "lower"),  ("FDR", FDR, "lower"),
    ("FOR", FOR, "lower"),
]


def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


def confusion_counts(preds, labels):
    p, l = np.asarray(preds), np.asarray(labels)
    TP = int(np.sum((p == 1) & (l == 1)))
    FP = int(np.sum((p == 1) & (l == 0)))
    TN = int(np.sum((p == 0) & (l == 0)))
    FN = int(np.sum((p == 0) & (l == 1)))
    return TP, FP, TN, FN


def compute_all_metrics(preds, labels):
    TP, FP, TN, FN = confusion_counts(preds, labels)
    return {name: fn(TP, FP, TN, FN) for name, fn, _ in ALL_METRICS}


def is_unfair(disparity, direction, threshold=FAIRNESS_THRESHOLD):
    if direction == "higher":
        return disparity < -threshold
    else:
        return disparity > threshold


# ---------------------------------------------------------------------------
# Data loading + cleaning + Nina's difficulty merge
# ---------------------------------------------------------------------------

def load_and_prepare():
    """Load cleaned data and merge Nina's performance-based difficulty."""
    hdr("Loading Data + Nina's Performance-Based Difficulty")

    df = load_excel_data(EXCEL_PATH)
    n_before = len(df)
    n_ann_before = df["ResponseId"].nunique()

    # Apply same cleaning as before
    # Step 1: IQR time outliers
    q1 = df["DecisionTime"].quantile(0.25)
    q3 = df["DecisionTime"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 3 * iqr
    df = df[df["DecisionTime"] <= upper]

    # Step 2: Speed clickers
    df = df[~((df["DecisionTime"] < 3) & (df["ClickCount"] <= 1))]

    # Step 3: Low-accuracy annotators (<= 25%)
    ann_acc = df.groupby("ResponseId")["Accuracy"].mean()
    bad_ann = ann_acc[ann_acc <= 0.25].index
    df = df[~df["ResponseId"].isin(bad_ann)]

    # Step 4: Straightliners
    for rid in df["ResponseId"].unique():
        answers = df[df["ResponseId"] == rid]["SurveyAnswer"].values
        if len(set(answers)) == 1 and len(answers) >= 8:
            df = df[df["ResponseId"] != rid]

    # Step 5: Confidence capping
    df["ConfidenceLevel"] = df["ConfidenceLevel"].clip(5, 95)
    df["ConfidenceLevelNorm"] = df["ConfidenceLevel"] / 100.0

    print(f"  Cleaned: {n_before} -> {len(df)} rows, {n_ann_before} -> {df['ResponseId'].nunique()} annotators")

    # Merge Nina's performance-based difficulty
    nina = pd.read_csv(NINA_DIFF_PATH)
    nina = nina[["SurveyVersion", "QuestionNum", "accuracy", "difficulty_disagreement", "difficulty_performance"]]
    nina = nina.rename(columns={
        "accuracy": "VersionQuestionAccuracy",
        "difficulty_disagreement": "DiffDisagreement",
        "difficulty_performance": "DiffPerformance",
    })

    df = df.merge(nina, on=["SurveyVersion", "QuestionNum"], how="left")

    # Binary difficulty flags
    df["IsHard_Performance"] = (df["DiffPerformance"] == "Hard").astype(int)
    df["IsHard_Disagreement"] = (df["DiffDisagreement"] == "Hard").astype(int)
    # Also keep our old static definition for comparison
    df["IsHard_Old"] = df["QuestionNum"].isin([2, 5, 6]).astype(int)

    # Summary
    perf_counts = df["DiffPerformance"].value_counts()
    print(f"\n  Performance-based difficulty distribution:")
    for k, v in perf_counts.items():
        print(f"    {k:8s}: {v:4d} rows ({v/len(df)*100:.1f}%)")

    old_hard = df["IsHard_Old"].sum()
    new_hard = df["IsHard_Performance"].sum()
    print(f"\n  Old hard rows: {old_hard} ({old_hard/len(df)*100:.1f}%)")
    print(f"  New hard rows: {new_hard} ({new_hard/len(df)*100:.1f}%)")

    # Show per-version-question difficulty
    print("\n  Performance difficulty by (Version, Question):")
    for v in sorted(df["SurveyVersion"].unique()):
        labels = []
        for q in range(1, 9):
            sub = df[(df["SurveyVersion"] == v) & (df["QuestionNum"] == q)]
            if len(sub) > 0:
                d = sub["DiffPerformance"].iloc[0]
                labels.append(f"Q{q}={d[0]}")
            else:
                labels.append(f"Q{q}=?")
        print(f"    V{v}: {', '.join(labels)}")

    return df


# ===========================================================================
# Exp P1: Old vs New Difficulty — Comparison
# ===========================================================================

def exp_p1_difficulty_comparison(df):
    hdr("P1: Old vs New Difficulty Classification Comparison")

    results = []
    for v in sorted(df["SurveyVersion"].unique()):
        for q in range(1, 9):
            sub = df[(df["SurveyVersion"] == v) & (df["QuestionNum"] == q)]
            if len(sub) == 0:
                continue
            row = sub.iloc[0]
            overall_acc = sub["Accuracy"].mean()
            results.append({
                "Version": v, "Question": q,
                "N": len(sub),
                "Accuracy": round(overall_acc, 3),
                "VersionAccuracy": round(row["VersionQuestionAccuracy"], 3),
                "Old_Difficulty": "Hard" if q in [2, 5, 6] else "Easy",
                "Perf_Difficulty": row["DiffPerformance"],
                "Disagree_Difficulty": row["DiffDisagreement"],
                "Match_Old_Perf": ("Hard" if q in [2, 5, 6] else "Easy") == row["DiffPerformance"],
            })

    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(RESULTS_DIR, "P1_difficulty_comparison.csv"), index=False)

    n_match = rdf["Match_Old_Perf"].sum()
    n_total = len(rdf)
    print(f"\n  Old vs Performance match: {n_match}/{n_total} ({n_match/n_total*100:.1f}%)")
    print(f"  Mismatches:")
    mismatches = rdf[~rdf["Match_Old_Perf"]]
    for _, r in mismatches.iterrows():
        print(f"    V{r['Version']}Q{r['Question']}: Old={r['Old_Difficulty']}, "
              f"Perf={r['Perf_Difficulty']} (acc={r['VersionAccuracy']:.3f})")

    # Visualization: heatmap of accuracy by (Version, Question) with difficulty labels
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for ax_idx, (diff_col, title) in enumerate([
        ("Old_Difficulty", "Old (Static Q2,Q5,Q6)"),
        ("Perf_Difficulty", "Performance-Based (Nina)"),
        ("Disagree_Difficulty", "Disagreement-Based"),
    ]):
        pivot_acc = rdf.pivot(index="Question", columns="Version", values="Accuracy")
        pivot_diff = rdf.pivot(index="Question", columns="Version", values=diff_col)

        annot_text = pd.DataFrame("", index=pivot_acc.index, columns=pivot_acc.columns)
        for q in annot_text.index:
            for v in annot_text.columns:
                acc = pivot_acc.loc[q, v]
                d = pivot_diff.loc[q, v]
                marker = "H" if d == "Hard" else "M" if d == "Medium" else "E"
                annot_text.loc[q, v] = f"{acc:.2f}\n[{marker}]"

        sns.heatmap(pivot_acc, ax=axes[ax_idx], annot=annot_text, fmt="",
                    cmap="RdYlGn", vmin=0.1, vmax=1.0, linewidths=0.5,
                    cbar_kws={"shrink": 0.7})
        axes[ax_idx].set_title(f"{title}", fontsize=10, fontweight="bold")
        axes[ax_idx].set_xlabel("Survey Version")
        axes[ax_idx].set_ylabel("Question")

    plt.suptitle("P1 -- Question Difficulty: Old Static vs Performance-Based vs Disagreement-Based\n"
                 "Cell = accuracy, [H]=Hard, [M]=Medium, [E]=Easy",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "P1_difficulty_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  -> P1_difficulty_comparison.csv + .png")
    return rdf


# ===========================================================================
# Exp P2: Conditional Fairness with Performance-Based Difficulty
# ===========================================================================

def exp_p2_conditional_fairness(df):
    hdr("P2: Conditional Fairness — Performance-Based Hard/Medium/Easy")

    tiers = {
        "Hard": df[df["DiffPerformance"] == "Hard"],
        "Medium": df[df["DiffPerformance"] == "Medium"],
        "Easy": df[df["DiffPerformance"] == "Easy"],
    }

    results = []
    for tier_name, tier_df in tiers.items():
        if len(tier_df) == 0:
            continue
        overall_acc = tier_df["Accuracy"].mean()
        overall_tpr_val = compute_all_metrics(tier_df["SurveyAnswer"], tier_df["ActualAnswer"])["TPR"]

        print(f"\n  === {tier_name} (n={len(tier_df)}, overall_acc={overall_acc:.3f}) ===")

        for gc in GROUP_COLUMNS:
            for gname, gdf in tier_df.groupby(gc):
                if len(gdf) == 0:
                    continue
                acc = gdf["Accuracy"].mean()
                m = compute_all_metrics(gdf["SurveyAnswer"], gdf["ActualAnswer"])
                acc_d = acc - overall_acc
                tpr_d = m["TPR"] - overall_tpr_val
                flag = "*** UNFAIR ***" if abs(acc_d) > 0.1 or abs(tpr_d) > 0.1 else ""
                print(f"    {gc:25s} {gname:15s}: Acc={acc:.3f}(d={acc_d:+.3f}) "
                      f"TPR={m['TPR']:.3f}(d={tpr_d:+.3f}) n={len(gdf)} {flag}")
                results.append({
                    "Tier": tier_name, "Partition": gc, "Group": gname,
                    "N": len(gdf), "Accuracy": round(acc, 4),
                    "TPR": round(m["TPR"], 4),
                    "Acc_Disparity": round(acc_d, 4),
                    "TPR_Disparity": round(tpr_d, 4),
                    "UNFAIR_ACC": abs(acc_d) > 0.1,
                    "UNFAIR_TPR": abs(tpr_d) > 0.1,
                })

    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(RESULTS_DIR, "P2_conditional_fairness.csv"), index=False)

    # Visualization
    n_gc = len(GROUP_COLUMNS)
    fig, axes = plt.subplots(1, n_gc, figsize=(5 * n_gc, 6))
    tier_colors = {"Hard": "#EF5350", "Medium": "#FFA726", "Easy": "#66BB6A"}

    for i, gc in enumerate(GROUP_COLUMNS):
        sub = rdf[rdf["Partition"] == gc]
        groups = sorted(sub["Group"].unique())
        x = np.arange(len(groups))
        w = 0.25
        for j, tier in enumerate(["Hard", "Medium", "Easy"]):
            tier_sub = sub[sub["Tier"] == tier]
            vals = [tier_sub[tier_sub["Group"] == g]["Accuracy"].values[0]
                    if len(tier_sub[tier_sub["Group"] == g]) > 0 else 0
                    for g in groups]
            bars = axes[i].bar(x + (j - 1) * w, vals, w,
                              label=tier, color=tier_colors[tier], alpha=0.85)
            for k, v in enumerate(vals):
                axes[i].text(x[k] + (j - 1) * w, v + 0.01, f"{v:.2f}",
                            ha="center", fontsize=7)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(groups, rotation=15, ha="right", fontsize=9)
        axes[i].set_title(gc, fontsize=10, fontweight="bold")
        axes[i].set_ylabel("Accuracy")
        axes[i].set_ylim(0, 1.0)
        axes[i].axhline(0.5, color="gray", linestyle=":", linewidth=0.5)
        axes[i].legend(fontsize=8)

    plt.suptitle("P2 -- Conditional Fairness: Performance-Based Hard/Medium/Easy\n"
                 "(3-tier split per SurveyVersion x QuestionNum)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "P2_conditional_fairness.png"), dpi=150, bbox_inches="tight")
    plt.close()

    unfair = rdf[rdf["UNFAIR_ACC"] | rdf["UNFAIR_TPR"]]
    if len(unfair):
        print(f"\n  UNFAIR flags ({len(unfair)}):")
        for _, r in unfair.iterrows():
            print(f"    [{r['Tier']}] {r['Partition']} / {r['Group']}: "
                  f"Acc_d={r['Acc_Disparity']:+.4f}, TPR_d={r['TPR_Disparity']:+.4f}")

    print(f"\n  -> P2_conditional_fairness.csv + .png")
    return rdf


# ===========================================================================
# Exp P3: Deeper Non-Native Intersectional Analysis (Action Item 2)
# ===========================================================================

def exp_p3_nonnative_deep(df):
    hdr("P3: Deep Non-Native Intersectional Analysis")
    print("  Action Item: Explore Non-Native + STEM/Non-STEM, Education, Age")

    nn_df = df[df["LinguisticGroup"] == "Non-Native"]
    nat_df = df[df["LinguisticGroup"] == "Native"]

    results = []
    cross_partitions = ["ExpertiseGroup", "ExperienceGroup", "AgeGroup"]

    for gc in cross_partitions:
        print(f"\n  --- Non-Native x {gc} ---")
        for gname in sorted(df[gc].unique()):
            # Non-Native subgroup
            nn_sub = nn_df[nn_df[gc] == gname]
            nat_sub = nat_df[nat_df[gc] == gname]
            if len(nn_sub) == 0:
                continue

            nn_acc = nn_sub["Accuracy"].mean()
            nat_acc = nat_sub["Accuracy"].mean()
            gap = nn_acc - nat_acc

            # By difficulty tier
            for tier in ["Hard", "Medium", "Easy"]:
                nn_tier = nn_sub[nn_sub["DiffPerformance"] == tier]
                nat_tier = nat_sub[nat_sub["DiffPerformance"] == tier]
                if len(nn_tier) == 0:
                    continue

                nn_tier_acc = nn_tier["Accuracy"].mean()
                nat_tier_acc = nat_tier["Accuracy"].mean() if len(nat_tier) > 0 else np.nan
                tier_gap = nn_tier_acc - nat_tier_acc if not np.isnan(nat_tier_acc) else np.nan

                results.append({
                    "CrossPartition": gc, "CrossGroup": gname,
                    "Tier": tier,
                    "NN_N": len(nn_tier), "NN_Acc": round(nn_tier_acc, 4),
                    "Nat_N": len(nat_tier), "Nat_Acc": round(nat_tier_acc, 4) if not np.isnan(nat_tier_acc) else None,
                    "Gap_NN_minus_Nat": round(tier_gap, 4) if not np.isnan(tier_gap) else None,
                    "NN_Overall_Acc": round(nn_acc, 4),
                    "Nat_Overall_Acc": round(nat_acc, 4),
                    "Overall_Gap": round(gap, 4),
                })

            # Fisher exact for Hard tier specifically
            nn_hard = nn_sub[nn_sub["DiffPerformance"] == "Hard"]
            nat_hard = nat_sub[nat_sub["DiffPerformance"] == "Hard"]
            if len(nn_hard) > 0 and len(nat_hard) > 0:
                table = [
                    [int(nat_hard["Accuracy"].sum()), len(nat_hard) - int(nat_hard["Accuracy"].sum())],
                    [int(nn_hard["Accuracy"].sum()), len(nn_hard) - int(nn_hard["Accuracy"].sum())],
                ]
                try:
                    odds, p = fisher_exact(table)
                except ValueError:
                    odds, p = 1.0, 1.0
                print(f"    {gname:15s}: NN_hard={nn_hard['Accuracy'].mean():.3f}(n={len(nn_hard)}) "
                      f"vs Nat_hard={nat_hard['Accuracy'].mean():.3f}(n={len(nat_hard)}) "
                      f"gap={nn_hard['Accuracy'].mean() - nat_hard['Accuracy'].mean():+.3f} "
                      f"Fisher_p={p:.4f}")

    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(RESULTS_DIR, "P3_nonnative_deep.csv"), index=False)

    # Visualization: grouped bar of NN vs Native accuracy by tier, for each cross-partition
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, gc in enumerate(cross_partitions):
        ax = axes[i]
        sub = rdf[rdf["CrossPartition"] == gc]
        groups = sorted(sub["CrossGroup"].unique())
        tiers = ["Hard", "Medium", "Easy"]
        x_pos = 0
        xticks, xlabels = [], []

        for g in groups:
            for t_idx, tier in enumerate(tiers):
                row = sub[(sub["CrossGroup"] == g) & (sub["Tier"] == tier)]
                if len(row) == 0:
                    x_pos += 1
                    continue
                row = row.iloc[0]
                nn_val = row["NN_Acc"] if row["NN_Acc"] is not None else 0
                nat_val = row["Nat_Acc"] if row["Nat_Acc"] is not None else 0
                w = 0.35
                ax.bar(x_pos - w/2, nat_val, w, color="#1E88E5", alpha=0.8,
                       label="Native" if x_pos == 0 else "")
                ax.bar(x_pos + w/2, nn_val, w, color="#FF8F00", alpha=0.8,
                       label="Non-Native" if x_pos == 0 else "")
                # Gap annotation
                if row["Gap_NN_minus_Nat"] is not None:
                    gap_val = row["Gap_NN_minus_Nat"]
                    color = "red" if gap_val < -0.1 else "black"
                    ax.text(x_pos, max(nn_val, nat_val) + 0.02,
                           f"{gap_val:+.2f}", ha="center", fontsize=7,
                           color=color, fontweight="bold" if abs(gap_val) > 0.1 else "normal")
                xticks.append(x_pos)
                xlabels.append(f"{g[:8]}\n{tier}")
                x_pos += 1
            x_pos += 0.5  # gap between groups

        ax.set_xticks(xticks)
        ax.set_xticklabels(xlabels, fontsize=7, rotation=30, ha="right")
        ax.set_title(f"Non-Native x {gc}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Accuracy")
        ax.set_ylim(0, 1.05)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5)
        if i == 0:
            ax.legend(fontsize=8)

    plt.suptitle("P3 -- Non-Native vs Native Accuracy by Intersecting Partition & Difficulty Tier",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "P3_nonnative_deep.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  -> P3_nonnative_deep.csv + .png")
    return rdf


# ===========================================================================
# Exp P4: Cross-Version Consistency (Action Item 3)
# ===========================================================================

def exp_p4_cross_version(df):
    hdr("P4: Cross-Version Consistency — Do findings replicate?")

    results = []
    overall_metrics = compute_all_metrics(df["SurveyAnswer"], df["ActualAnswer"])

    for v in sorted(df["SurveyVersion"].unique()):
        vdf = df[df["SurveyVersion"] == v]
        v_overall = compute_all_metrics(vdf["SurveyAnswer"], vdf["ActualAnswer"])

        print(f"\n  === Version {v} (n={len(vdf)}, {vdf['ResponseId'].nunique()} annotators, "
              f"acc={vdf['Accuracy'].mean():.3f}) ===")

        for gc in GROUP_COLUMNS:
            for gname, gdf in vdf.groupby(gc):
                if len(gdf) < 3:
                    continue
                m = compute_all_metrics(gdf["SurveyAnswer"], gdf["ActualAnswer"])
                acc_d = m["AP"] - v_overall["AP"]
                tpr_d = m["TPR"] - v_overall["TPR"]

                # Also hard-only within this version
                hard_gdf = gdf[gdf["DiffPerformance"] == "Hard"]
                hard_vdf = vdf[vdf["DiffPerformance"] == "Hard"]
                hard_acc_d = None
                if len(hard_gdf) >= 2 and len(hard_vdf) >= 5:
                    hard_g_acc = hard_gdf["Accuracy"].mean()
                    hard_v_acc = hard_vdf["Accuracy"].mean()
                    hard_acc_d = hard_g_acc - hard_v_acc

                results.append({
                    "Version": v, "Partition": gc, "Group": gname,
                    "N": len(gdf), "AP_d": round(acc_d, 4), "TPR_d": round(tpr_d, 4),
                    "UNFAIR_AP": abs(acc_d) > 0.1,
                    "UNFAIR_TPR": abs(tpr_d) > 0.1,
                    "Hard_Acc_d": round(hard_acc_d, 4) if hard_acc_d is not None else None,
                })

                flag = ""
                if abs(acc_d) > 0.1 or abs(tpr_d) > 0.1:
                    flag = " ** UNFAIR"
                hard_str = f"{hard_acc_d:+.3f}" if hard_acc_d is not None else "N/A"
                print(f"    {gc:20s} {gname:15s}: AP_d={acc_d:+.3f} TPR_d={tpr_d:+.3f} "
                      f"Hard_d={hard_str:>6s}{flag}")

    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(RESULTS_DIR, "P4_cross_version.csv"), index=False)

    # Visualization: Non-Native AP_d and TPR_d across versions
    nn = rdf[(rdf["Partition"] == "LinguisticGroup") & (rdf["Group"] == "Non-Native")]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: AP_d by version for all partitions
    ax = axes[0]
    for gc in GROUP_COLUMNS:
        for gname in sorted(df[gc].unique()):
            sub = rdf[(rdf["Partition"] == gc) & (rdf["Group"] == gname)]
            if len(sub) == 0:
                continue
            marker = "o" if gname in ["Non-Native", "Young-18-34"] else "s"
            color = "#FF8F00" if gname == "Non-Native" else "#90A4AE"
            lw = 2.5 if gname == "Non-Native" else 0.8
            ax.plot(sub["Version"], sub["AP_d"], marker=marker, label=f"{gname}",
                   color=color, linewidth=lw, markersize=6, alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axhline(-0.1, color="red", linewidth=0.8, linestyle=":")
    ax.axhline(0.1, color="red", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Survey Version")
    ax.set_ylabel("AP Disparity")
    ax.set_title("AP Disparity by Version\n(Non-Native highlighted)", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6, ncol=2, loc="lower left")

    # Right: Non-Native hard accuracy across versions
    ax2 = axes[1]
    nn_hard_d = nn["Hard_Acc_d"].dropna()
    if len(nn_hard_d) > 0:
        versions = nn.dropna(subset=["Hard_Acc_d"])["Version"]
        ax2.bar(versions, nn["Hard_Acc_d"].dropna(), color="#EF5350", alpha=0.8)
        ax2.axhline(0, color="black", linewidth=0.5)
        ax2.axhline(-0.1, color="orange", linewidth=0.8, linestyle=":")
        for v_val, d_val in zip(versions, nn["Hard_Acc_d"].dropna()):
            ax2.text(v_val, d_val + 0.01 * np.sign(d_val), f"{d_val:+.3f}",
                    ha="center", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Survey Version")
    ax2.set_ylabel("Non-Native Hard Accuracy Disparity")
    ax2.set_title("Non-Native Performance on HARD Questions by Version\n(Performance-based difficulty)",
                  fontsize=10, fontweight="bold")

    plt.suptitle("P4 -- Cross-Version Consistency", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "P4_cross_version.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Summary: does Non-Native unfairness replicate?
    print(f"\n  Cross-version Non-Native summary:")
    for _, r in nn.iterrows():
        hd = f"Hard_d={r['Hard_Acc_d']:+.3f}" if r['Hard_Acc_d'] is not None else "Hard_d=N/A"
        print(f"    V{r['Version']}: AP_d={r['AP_d']:+.3f}, TPR_d={r['TPR_d']:+.3f}, {hd}")

    print(f"\n  -> P4_cross_version.csv + .png")
    return rdf


# ===========================================================================
# Exp P5: Regression with Performance-Based Difficulty
# ===========================================================================

def exp_p5_regression(df):
    hdr("P5: Logistic Regression — Performance-Based Difficulty")

    features = {
        "IsNonNative": (df["LinguisticGroup"] == "Non-Native").astype(int),
        "IsSTEM": (df["ExpertiseGroup"] == "STEM").astype(int),
        "IsHighEdu": (df["ExperienceGroup"] == "High-Edu").astype(int),
        "IsOlder35": (df["AgeGroup"] == "Older-35plus").astype(int),
        "ConfidenceNorm": df["ConfidenceLevelNorm"],
        "ClickCount": df["ClickCount"],
        "IsHard_Perf": df["IsHard_Performance"],
        "IsMedium": (df["DiffPerformance"] == "Medium").astype(int),
    }
    X = pd.DataFrame(features)
    y = df["Accuracy"].values

    # Interactions
    X["NonNative_x_Hard"] = X["IsNonNative"] * X["IsHard_Perf"]
    X["NonNative_x_Medium"] = X["IsNonNative"] * X["IsMedium"]
    X["HighEdu_x_Hard"] = X["IsHighEdu"] * X["IsHard_Perf"]

    # Standardize continuous
    scaler = StandardScaler()
    X_scaled = X.copy()
    for col in ["ConfidenceNorm", "ClickCount"]:
        X_scaled[col] = scaler.fit_transform(X_scaled[[col]])

    model = LogisticRegression(max_iter=1000, random_state=42, penalty="l2", C=1.0)
    model.fit(X_scaled, y)

    coefs = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0],
        "Odds_Ratio": np.exp(model.coef_[0]),
        "Abs_Coef": np.abs(model.coef_[0]),
    }).sort_values("Abs_Coef", ascending=False)

    print(f"\n  Model accuracy: {model.score(X_scaled, y):.4f}")
    print(f"\n  {'Feature':25s} | {'Coef':>8s} | {'OR':>8s} | Strength")
    for _, r in coefs.iterrows():
        strength = "STRONG" if r["Abs_Coef"] > 0.2 else "moderate" if r["Abs_Coef"] > 0.1 else "weak"
        direction = "decreases acc" if r["Coefficient"] < 0 else "increases acc"
        print(f"  {r['Feature']:25s} | {r['Coefficient']:+.4f} | {r['Odds_Ratio']:.4f} | "
              f"{direction} [{strength}]")

    coefs.to_csv(os.path.join(RESULTS_DIR, "P5_regression.csv"), index=False)

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#EF5350" if c < 0 else "#43A047" for c in coefs["Coefficient"]]
    ax.barh(range(len(coefs)), coefs["Coefficient"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(coefs)))
    ax.set_yticklabels(coefs["Feature"])
    ax.set_xlabel("Logistic Regression Coefficient")
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(-0.1, color="orange", linewidth=0.8, linestyle=":", label="Active threshold")
    ax.axvline(0.1, color="orange", linewidth=0.8, linestyle=":")

    for j, (_, r) in enumerate(coefs.iterrows()):
        active = "ACTIVE" if r["Abs_Coef"] > 0.1 else "passive"
        ax.text(r["Coefficient"] + 0.02 * np.sign(r["Coefficient"]),
                j, f"OR={r['Odds_Ratio']:.2f} [{active}]", va="center", fontsize=8)

    ax.set_title("P5 -- Logistic Regression with Performance-Based Difficulty\n"
                 "Red = decreases accuracy, Green = increases accuracy",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "P5_regression.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  -> P5_regression.csv + .png")
    return coefs


# ===========================================================================
# Exp P6: Summary — Old vs New Comparison
# ===========================================================================

def exp_p6_summary(df, p2_df, p4_df, p5_df):
    hdr("P6: Summary — What Changed with Performance-Based Difficulty?")

    print("\n  1. DIFFICULTY DEFINITION CHANGE:")
    print(f"     Old: Static Q2,Q5,Q6 = Hard for all versions")
    print(f"     New: Per (Version, Question) from Nina's accuracy-based classification")
    n_hard_old = df["IsHard_Old"].sum()
    n_hard_new = df["IsHard_Performance"].sum()
    print(f"     Old Hard rows: {n_hard_old} ({n_hard_old/len(df)*100:.1f}%)")
    print(f"     New Hard rows: {n_hard_new} ({n_hard_new/len(df)*100:.1f}%)")

    print("\n  2. CONDITIONAL FAIRNESS (P2):")
    unfair_p2 = p2_df[p2_df["UNFAIR_ACC"] | p2_df["UNFAIR_TPR"]]
    if len(unfair_p2):
        for _, r in unfair_p2.iterrows():
            print(f"     UNFAIR: [{r['Tier']}] {r['Group']} ({r['Partition']}): "
                  f"Acc_d={r['Acc_Disparity']:+.4f}")
    else:
        print("     No UNFAIR flags in any tier.")

    print("\n  3. REGRESSION TOP FEATURES (P5):")
    for _, r in p5_df.head(5).iterrows():
        print(f"     {r['Feature']:25s}: coef={r['Coefficient']:+.4f} OR={r['Odds_Ratio']:.3f}")

    print("\n  4. CROSS-VERSION (P4):")
    nn_p4 = p4_df[(p4_df["Partition"] == "LinguisticGroup") & (p4_df["Group"] == "Non-Native")]
    n_unfair_versions = len(nn_p4[nn_p4["UNFAIR_AP"] | nn_p4["UNFAIR_TPR"]])
    print(f"     Non-Native UNFAIR in {n_unfair_versions}/{len(nn_p4)} versions")

    # Save summary
    lines = [
        "Phase 3: Performance-Based Difficulty Re-Analysis Summary",
        "=" * 55,
        f"Data: {len(df)} rows, {df['ResponseId'].nunique()} annotators (cleaned)",
        f"Difficulty: Nina's performance-based (per Version x Question)",
        f"Hard rows: {n_hard_new} ({n_hard_new/len(df)*100:.1f}%) vs old {n_hard_old} ({n_hard_old/len(df)*100:.1f}%)",
        "",
        "Key findings:",
    ]
    if len(unfair_p2):
        for _, r in unfair_p2.iterrows():
            lines.append(f"  UNFAIR: [{r['Tier']}] {r['Group']} Acc_d={r['Acc_Disparity']:+.4f}")
    lines.append(f"\nNon-Native UNFAIR in {n_unfair_versions}/{len(nn_p4)} versions")
    lines.append(f"\nTop regression feature: {p5_df.iloc[0]['Feature']} (coef={p5_df.iloc[0]['Coefficient']:+.4f})")

    with open(os.path.join(RESULTS_DIR, "P6_summary.txt"), "w") as f:
        f.write("\n".join(lines))

    print(f"\n  -> P6_summary.txt")


# ===========================================================================
# Main
# ===========================================================================

def main():
    hdr("Phase 3: Performance-Based Difficulty Re-Analysis")

    df = load_and_prepare()

    p1 = exp_p1_difficulty_comparison(df)
    p2 = exp_p2_conditional_fairness(df)
    p3 = exp_p3_nonnative_deep(df)
    p4 = exp_p4_cross_version(df)
    p5 = exp_p5_regression(df)
    exp_p6_summary(df, p2, p4, p5)

    print(f"\n{SEP}")
    print(f"  All Phase 3 results saved to: {os.path.abspath(RESULTS_DIR)}")
    print(SEP)


if __name__ == "__main__":
    main()
