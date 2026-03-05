"""
Phase 1: Demographic Partitioning — Initial Experiments
=========================================================
Problem Statement 2: Demographic Fairness Direction for TUS

This script runs the initial experiments for Phase 1, which partitions
the 58 expert annotators in the TUNE dataset into demographic subgroups
and produces exploratory results:

  1. Group distribution analysis (sizes, proportions)
  2. Per-group decision accuracy against ActualAnswer & Majority vote
  3. Per-group behavioral signal profiles (time, clicks, confidence)
  4. Cross-group disagreement rates
  5. Initial fairness indicators (Accuracy Parity, TPR Parity per group)
  6. TUS model score distributions per group

References:
  - FairEM (Shahbazi et al., VLDB 2023)
  - FairEM360 (Shahbazi et al., VLDB 2024)
  - TUNE Benchmark (Marimuthu, Klimenkova, Shraga, HILDA 2025)

Usage:
    python experiments/phase1_demographic_partitioning.py
"""

import sys
import os

# Add parent dir so we can import src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

from src.demographic_partitioning import (
    load_tune_data,
    apply_all_partitions,
    get_annotator_demographics,
    group_distribution_summary,
)
from src.measures import (
    AP, TPR, FPR, PPV, FNR,
    calibration_gap,
    wasserstein_score_bias,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "TUNE_Benchmark", "data", "Feature_Engineered.csv"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "phase1")
os.makedirs(RESULTS_DIR, exist_ok=True)

GROUP_COLUMNS = ["LinguisticGroup", "ExpertiseGroup", "ExperienceGroup"]
TUS_SCORE_COLS = ["Starnie", "Santos", "D3L"]

SEPARATOR = "=" * 70


def print_header(title):
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ===========================================================================
# Experiment 1: Group Distribution Analysis
# ===========================================================================
def experiment_1_group_distributions(df):
    """Analyze annotator demographics and group sizes."""
    print_header("Experiment 1: Demographic Group Distributions")

    annotators = get_annotator_demographics(df)
    print(f"\nTotal unique annotators: {len(annotators)}")
    print(f"Total annotation rows: {len(df)}")
    print(f"Unique questions: {df['QuestionNum'].nunique()}")

    summary = group_distribution_summary(df)
    results = []

    for group_col in GROUP_COLUMNS:
        print(f"\n--- {group_col} ---")
        counts = summary[group_col]
        total_annotators = sum(counts.values())
        for name, count in sorted(counts.items()):
            pct = count / total_annotators * 100
            n_rows = len(df[df[group_col] == name])
            print(f"  {name:15s}: {count:3d} annotators ({pct:5.1f}%), {n_rows:5d} annotations")
            results.append({
                "Partition": group_col,
                "Group": name,
                "N_Annotators": count,
                "Pct_Annotators": round(pct, 1),
                "N_Annotations": n_rows,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "exp1_group_distributions.csv"), index=False)
    print(f"\n  -> Saved to exp1_group_distributions.csv")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, group_col in enumerate(GROUP_COLUMNS):
        sub = results_df[results_df["Partition"] == group_col]
        axes[i].bar(sub["Group"], sub["N_Annotators"], color=["#2196F3", "#FF9800"])
        axes[i].set_title(group_col.replace("Group", " Groups"))
        axes[i].set_ylabel("Number of Annotators")
        for j, (_, row) in enumerate(sub.iterrows()):
            axes[i].text(j, row["N_Annotators"] + 0.5,
                         f'{row["Pct_Annotators"]:.0f}%', ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp1_group_distributions.png"), dpi=150)
    plt.close()
    print("  -> Saved exp1_group_distributions.png")

    return results_df


# ===========================================================================
# Experiment 2: Per-Group Decision Accuracy
# ===========================================================================
def experiment_2_group_accuracy(df):
    """Compute accuracy, TPR, FPR, PPV per demographic group."""
    print_header("Experiment 2: Per-Group Decision Accuracy (Fairness Indicators)")

    results = []
    for group_col in GROUP_COLUMNS:
        print(f"\n--- {group_col} ---")

        # Overall confusion matrix
        all_preds = df["SurveyAnswer"].values
        all_labels = df["ActualAnswer"].values
        overall_TP = int(np.sum((all_preds == 1) & (all_labels == 1)))
        overall_FP = int(np.sum((all_preds == 1) & (all_labels == 0)))
        overall_TN = int(np.sum((all_preds == 0) & (all_labels == 0)))
        overall_FN = int(np.sum((all_preds == 0) & (all_labels == 1)))

        overall_acc = AP(overall_TP, overall_FP, overall_TN, overall_FN)
        overall_tpr = TPR(overall_TP, overall_FP, overall_TN, overall_FN)
        overall_fpr = FPR(overall_TP, overall_FP, overall_TN, overall_FN)
        overall_ppv = PPV(overall_TP, overall_FP, overall_TN, overall_FN)

        print(f"  {'Overall':15s}: Acc={overall_acc:.4f}  TPR={overall_tpr:.4f}  "
              f"FPR={overall_fpr:.4f}  PPV={overall_ppv:.4f}")

        for group_name, group_df in df.groupby(group_col):
            preds = group_df["SurveyAnswer"].values
            labels = group_df["ActualAnswer"].values
            TP = int(np.sum((preds == 1) & (labels == 1)))
            FP = int(np.sum((preds == 1) & (labels == 0)))
            TN = int(np.sum((preds == 0) & (labels == 0)))
            FN = int(np.sum((preds == 0) & (labels == 1)))

            acc = AP(TP, FP, TN, FN)
            tpr = TPR(TP, FP, TN, FN)
            fpr = FPR(TP, FP, TN, FN)
            ppv = PPV(TP, FP, TN, FN)

            # Disparity from overall (FairEM-style)
            acc_disp = acc - overall_acc
            tpr_disp = tpr - overall_tpr
            fpr_disp = fpr - overall_fpr

            print(f"  {group_name:15s}: Acc={acc:.4f} (d={acc_disp:+.4f})  "
                  f"TPR={tpr:.4f} (d={tpr_disp:+.4f})  "
                  f"FPR={fpr:.4f} (d={fpr_disp:+.4f})  "
                  f"PPV={ppv:.4f}   [n={len(group_df)}]")

            results.append({
                "Partition": group_col,
                "Group": group_name,
                "N": len(group_df),
                "TP": TP, "FP": FP, "TN": TN, "FN": FN,
                "Accuracy": round(acc, 4),
                "TPR": round(tpr, 4),
                "FPR": round(fpr, 4),
                "PPV": round(ppv, 4),
                "Accuracy_Disparity": round(acc_disp, 4),
                "TPR_Disparity": round(tpr_disp, 4),
                "FPR_Disparity": round(fpr_disp, 4),
                "Overall_Accuracy": round(overall_acc, 4),
                "Overall_TPR": round(overall_tpr, 4),
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "exp2_group_accuracy.csv"), index=False)
    print(f"\n  -> Saved to exp2_group_accuracy.csv")

    # Visualization: grouped bar chart of accuracy and TPR
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["Accuracy", "TPR", "FPR"]
    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, group_col in enumerate(GROUP_COLUMNS):
            sub = results_df[results_df["Partition"] == group_col]
            x = np.arange(len(sub))
            width = 0.25
            ax.bar(x + j * width, sub[metric], width,
                   label=group_col.replace("Group", ""),
                   alpha=0.85)
            for k, (_, row) in enumerate(sub.iterrows()):
                ax.text(k + j * width, row[metric] + 0.01,
                        f'{row[metric]:.3f}', ha="center", fontsize=7, rotation=45)
        ax.set_title(f"{metric} by Demographic Group")
        ax.set_ylabel(metric)
        ax.set_xticks(x + width)
        # Use the groups from the last partition for labels (all have 2 groups)
        ax.set_xticklabels(["Group A", "Group B"])
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_group_accuracy.png"), dpi=150)
    plt.close()
    print("  -> Saved exp2_group_accuracy.png")

    return results_df


# ===========================================================================
# Experiment 3: Behavioral Signal Profiles per Group
# ===========================================================================
def experiment_3_behavioral_profiles(df):
    """Compare behavioral signals (time, clicks, confidence) across groups."""
    print_header("Experiment 3: Behavioral Signal Profiles per Group")

    behavioral_cols = ["DecisionTime", "ClickCount", "ConfidenceLevel",
                       "LastClick", "DecisionTimeFract"]

    results = []
    for group_col in GROUP_COLUMNS:
        print(f"\n--- {group_col} ---")
        for group_name, group_df in df.groupby(group_col):
            row = {"Partition": group_col, "Group": group_name, "N": len(group_df)}
            for col in behavioral_cols:
                if col in group_df.columns:
                    vals = group_df[col].dropna()
                    row[f"{col}_mean"] = round(float(vals.mean()), 6)
                    row[f"{col}_median"] = round(float(vals.median()), 6)
                    row[f"{col}_std"] = round(float(vals.std()), 6)
            results.append(row)
            print(f"  {group_name:15s}: DecTime_mean={row.get('DecisionTime_mean', 'N/A'):.6f}  "
                  f"Clicks_mean={row.get('ClickCount_mean', 'N/A'):.4f}  "
                  f"Conf_mean={row.get('ConfidenceLevel_mean', 'N/A'):.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "exp3_behavioral_profiles.csv"), index=False)
    print(f"\n  -> Saved to exp3_behavioral_profiles.csv")

    # Visualization: box plots per group
    fig, axes = plt.subplots(len(GROUP_COLUMNS), 3, figsize=(15, 4 * len(GROUP_COLUMNS)))
    plot_cols = ["DecisionTime", "ClickCount", "ConfidenceLevel"]

    for i, group_col in enumerate(GROUP_COLUMNS):
        for j, col in enumerate(plot_cols):
            ax = axes[i][j] if len(GROUP_COLUMNS) > 1 else axes[j]
            data_to_plot = []
            labels = []
            for group_name in sorted(df[group_col].unique()):
                vals = df[df[group_col] == group_name][col].dropna().values
                data_to_plot.append(vals)
                labels.append(group_name)
            ax.boxplot(data_to_plot, labels=labels)
            ax.set_title(f"{col}\n({group_col.replace('Group', '')})")
            ax.set_ylabel(col)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_behavioral_profiles.png"), dpi=150)
    plt.close()
    print("  -> Saved exp3_behavioral_profiles.png")

    return results_df


# ===========================================================================
# Experiment 4: Majority Vote Disagreement Analysis
# ===========================================================================
def experiment_4_majority_disagreement(df):
    """Measure how often each demographic group disagrees with majority vote."""
    print_header("Experiment 4: Majority Vote Disagreement per Group")
    print("  Key Question: Does the majority vote systematically disagree")
    print("  with Non-Native speakers or Non-STEM annotators?")

    results = []
    for group_col in GROUP_COLUMNS:
        print(f"\n--- {group_col} ---")
        for group_name, group_df in df.groupby(group_col):
            total = len(group_df)
            disagree_majority = int(np.sum(group_df["SurveyAnswer"] != group_df["Majority"]))
            disagree_actual = int(np.sum(group_df["SurveyAnswer"] != group_df["ActualAnswer"]))
            majority_wrong = int(np.sum(group_df["Majority"] != group_df["ActualAnswer"]))

            results.append({
                "Partition": group_col,
                "Group": group_name,
                "N": total,
                "DisagreesWithMajority": disagree_majority,
                "MajDisagreeRate": round(disagree_majority / total, 4) if total > 0 else 0,
                "AnnotatorErrorRate": round(disagree_actual / total, 4) if total > 0 else 0,
                "MajorityErrorRate": round(majority_wrong / total, 4) if total > 0 else 0,
            })

            print(f"  {group_name:15s}: MajDisagree={disagree_majority}/{total} "
                  f"({disagree_majority/total*100:.1f}%)  "
                  f"AnnotatorErr={disagree_actual/total*100:.1f}%  "
                  f"MajorityErr={majority_wrong/total*100:.1f}%")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "exp4_majority_disagreement.csv"), index=False)
    print(f"\n  -> Saved to exp4_majority_disagreement.csv")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for i, group_col in enumerate(GROUP_COLUMNS):
        sub = results_df[results_df["Partition"] == group_col]
        x = np.arange(len(sub))
        width = 0.35
        axes[i].bar(x - width/2, sub["MajDisagreeRate"], width,
                     label="Disagrees w/ Majority", color="#E53935")
        axes[i].bar(x + width/2, sub["AnnotatorErrorRate"], width,
                     label="Annotator Error Rate", color="#1E88E5")
        axes[i].set_title(group_col.replace("Group", " Groups"))
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(sub["Group"])
        axes[i].set_ylabel("Rate")
        axes[i].legend(fontsize=8)
        for j, (_, row) in enumerate(sub.iterrows()):
            axes[i].text(j - width/2, row["MajDisagreeRate"] + 0.01,
                         f'{row["MajDisagreeRate"]:.3f}', ha="center", fontsize=8)
            axes[i].text(j + width/2, row["AnnotatorErrorRate"] + 0.01,
                         f'{row["AnnotatorErrorRate"]:.3f}', ha="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp4_majority_disagreement.png"), dpi=150)
    plt.close()
    print("  -> Saved exp4_majority_disagreement.png")

    return results_df


# ===========================================================================
# Experiment 5: Calibration Gap per Group
# ===========================================================================
def experiment_5_calibration_gap(df):
    """Compute calibration gap (Confidence - Accuracy) per group."""
    print_header("Experiment 5: Calibration Gap per Group")
    print("  Hypothesis: Humans are more overconfident on 'easy' tables")
    print("  and miscalibrated on 'hard' ones, varying by demographic.")

    results = []
    for group_col in GROUP_COLUMNS:
        print(f"\n--- {group_col} ---")
        for group_name, group_df in df.groupby(group_col):
            confidences = group_df["ConfidenceLevel"].values
            is_correct = (group_df["SurveyAnswer"] == group_df["ActualAnswer"]).astype(float).values
            gap = calibration_gap(confidences, is_correct)
            mean_conf = float(np.mean(confidences))
            mean_acc = float(np.mean(is_correct))

            results.append({
                "Partition": group_col,
                "Group": group_name,
                "N": len(group_df),
                "MeanConfidence": round(mean_conf, 4),
                "MeanAccuracy": round(mean_acc, 4),
                "CalibrationGap": round(gap, 4),
                "IsOverconfident": gap > 0,
            })

            direction = "OVERCONFIDENT" if gap > 0 else "UNDERCONFIDENT"
            print(f"  {group_name:15s}: Conf={mean_conf:.4f}  Acc={mean_acc:.4f}  "
                  f"Gap={gap:+.4f} ({direction})")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "exp5_calibration_gap.csv"), index=False)
    print(f"\n  -> Saved to exp5_calibration_gap.csv")

    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, group_col in enumerate(GROUP_COLUMNS):
        sub = results_df[results_df["Partition"] == group_col]
        x = np.arange(len(sub))
        width = 0.3
        axes[i].bar(x - width, sub["MeanConfidence"], width, label="Confidence", color="#42A5F5")
        axes[i].bar(x, sub["MeanAccuracy"], width, label="Accuracy", color="#66BB6A")
        axes[i].bar(x + width, sub["CalibrationGap"], width, label="Gap (Conf-Acc)", color="#EF5350")
        axes[i].set_title(group_col.replace("Group", " Groups"))
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(sub["Group"])
        axes[i].set_ylabel("Value")
        axes[i].legend(fontsize=8)
        axes[i].axhline(y=0, color="black", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp5_calibration_gap.png"), dpi=150)
    plt.close()
    print("  -> Saved exp5_calibration_gap.png")

    return results_df


# ===========================================================================
# Experiment 6: TUS Model Score Bias (Wasserstein)
# ===========================================================================
def experiment_6_tus_score_bias(df):
    """Compare TUS model score distributions across demographic groups."""
    print_header("Experiment 6: TUS Model Score Bias (Wasserstein Distance)")

    results = []
    for group_col in GROUP_COLUMNS:
        print(f"\n--- {group_col} ---")
        groups = sorted(df[group_col].unique())
        if len(groups) < 2:
            continue
        g1, g2 = groups[0], groups[1]
        df1 = df[df[group_col] == g1]
        df2 = df[df[group_col] == g2]

        for score_col in TUS_SCORE_COLS:
            s1 = df1[score_col].dropna().values
            s2 = df2[score_col].dropna().values
            if len(s1) == 0 or len(s2) == 0:
                continue
            w_dist = wasserstein_score_bias(s1, s2)
            results.append({
                "Partition": group_col,
                "GroupA": g1,
                "GroupB": g2,
                "TUS_Model": score_col,
                "Wasserstein_Distance": round(w_dist, 6),
                "MeanA": round(float(np.mean(s1)), 4),
                "MeanB": round(float(np.mean(s2)), 4),
                "MeanDiff": round(float(np.mean(s1) - np.mean(s2)), 4),
            })
            print(f"  {score_col:10s}: {g1} vs {g2}  W_dist={w_dist:.6f}  "
                  f"mean_A={np.mean(s1):.4f}  mean_B={np.mean(s2):.4f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "exp6_tus_score_bias.csv"), index=False)
    print(f"\n  -> Saved to exp6_tus_score_bias.csv")
    return results_df


# ===========================================================================
# Summary Report
# ===========================================================================
def generate_summary(exp1, exp2, exp3, exp4, exp5, exp6):
    """Generate a summary of all Phase 1 findings."""
    print_header("PHASE 1 SUMMARY: Demographic Partitioning Results")

    print("\n1. GROUP DISTRIBUTIONS:")
    for _, row in exp1.iterrows():
        print(f"   {row['Partition']:25s} | {row['Group']:15s} | "
              f"{row['N_Annotators']:3.0f} annotators ({row['Pct_Annotators']:.0f}%)")

    print("\n2. FAIRNESS INDICATORS (Accuracy Parity, TPR Parity):")
    for _, row in exp2.iterrows():
        fair_flag = "FAIR" if abs(row["Accuracy_Disparity"]) < 0.1 else "UNFAIR"
        tpr_flag = "FAIR" if abs(row["TPR_Disparity"]) < 0.1 else "UNFAIR"
        print(f"   {row['Partition']:25s} | {row['Group']:15s} | "
              f"Acc_d={row['Accuracy_Disparity']:+.4f} [{fair_flag:6s}] | "
              f"TPR_d={row['TPR_Disparity']:+.4f} [{tpr_flag:6s}]")

    print("\n3. KEY FINDINGS:")

    # Find largest disparity
    max_acc_disp = exp2.loc[exp2["Accuracy_Disparity"].abs().idxmax()]
    max_tpr_disp = exp2.loc[exp2["TPR_Disparity"].abs().idxmax()]
    print(f"   - Largest Accuracy Disparity: {max_acc_disp['Group']} "
          f"({max_acc_disp['Partition']}) = {max_acc_disp['Accuracy_Disparity']:+.4f}")
    print(f"   - Largest TPR Disparity: {max_tpr_disp['Group']} "
          f"({max_tpr_disp['Partition']}) = {max_tpr_disp['TPR_Disparity']:+.4f}")

    # Check majority vote disagreement
    max_disagree = exp4.loc[exp4["MajDisagreeRate"].idxmax()]
    print(f"   - Highest Majority Disagreement: {max_disagree['Group']} "
          f"({max_disagree['Partition']}) = {max_disagree['MajDisagreeRate']:.4f}")

    # Calibration gap
    max_gap = exp5.loc[exp5["CalibrationGap"].abs().idxmax()]
    print(f"   - Largest Calibration Gap: {max_gap['Group']} "
          f"({max_gap['Partition']}) = {max_gap['CalibrationGap']:+.4f}")

    # Write summary to file
    summary_lines = [
        "Phase 1: Demographic Partitioning - Summary Report",
        "=" * 50,
        f"Total annotations analyzed: {exp2['N'].sum() // len(exp2)}",
        "",
        "Fairness Assessment (threshold = 0.1):",
    ]
    for _, row in exp2.iterrows():
        fair_flag = "FAIR" if abs(row["Accuracy_Disparity"]) < 0.1 else "UNFAIR"
        summary_lines.append(
            f"  {row['Partition']} / {row['Group']}: "
            f"Acc={row['Accuracy']:.4f} (d={row['Accuracy_Disparity']:+.4f}) [{fair_flag}]"
        )

    with open(os.path.join(RESULTS_DIR, "phase1_summary.txt"), "w") as f:
        f.write("\n".join(summary_lines))
    print(f"\n  -> Saved phase1_summary.txt")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print_header("FairPrep Phase 1: Demographic Partitioning Experiments")
    print(f"Data path: {DATA_PATH}")

    # Load and partition data
    print("\nLoading TUNE dataset...")
    df = load_tune_data(DATA_PATH)
    df = apply_all_partitions(df)
    print(f"Loaded {len(df)} rows, {df['ByWho'].nunique()} unique annotators")

    # Run all experiments
    exp1 = experiment_1_group_distributions(df)
    exp2 = experiment_2_group_accuracy(df)
    exp3 = experiment_3_behavioral_profiles(df)
    exp4 = experiment_4_majority_disagreement(df)
    exp5 = experiment_5_calibration_gap(df)
    exp6 = experiment_6_tus_score_bias(df)

    # Summary
    generate_summary(exp1, exp2, exp3, exp4, exp5, exp6)

    print(f"\n{SEPARATOR}")
    print(f"  All Phase 1 results saved to: {os.path.abspath(RESULTS_DIR)}")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
