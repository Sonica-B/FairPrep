"""
Generate PNG visualizations for all 9 cleaned Phase 1 experiments.
Reads CSVs from results/cleaned_phase1/ and saves PNGs to the same directory.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "results", "cleaned_phase1")
OUT_DIR = DATA_DIR  # save PNGs alongside the CSVs

PARTITION_ORDER = ["LinguisticGroup", "ExpertiseGroup", "ExperienceGroup", "AgeGroup"]
PARTITION_LABELS = {
    "LinguisticGroup": "Linguistic",
    "ExpertiseGroup": "Expertise",
    "ExperienceGroup": "Experience",
    "AgeGroup": "Age",
}

# A consistent color palette per group within each partition
GROUP_COLORS = sns.color_palette("Set2", 8)


def _group_label(row):
    """Return a short label: 'Partition\nGroup'."""
    p = PARTITION_LABELS.get(row["Partition"], row["Partition"])
    return f"{p}\n{row['Group']}"


# ===== Experiment 1 =====
def plot_exp1():
    df = pd.read_csv(os.path.join(DATA_DIR, "exp1_group_distributions.csv"))
    fig, ax = plt.subplots(figsize=(10, 5))

    partitions = df["Partition"].unique()
    x = np.arange(len(partitions))
    width = 0.35

    for i, part in enumerate(partitions):
        sub = df[df["Partition"] == part].sort_values("Group")
        bottom = 0
        for j, (_, row) in enumerate(sub.iterrows()):
            color = GROUP_COLORS[j]
            bar = ax.bar(x[i], row["N_Annotators"], width, bottom=bottom,
                         color=color, edgecolor="white")
            ax.text(x[i], bottom + row["N_Annotators"] / 2,
                    f"{row['Group']}\n(n={int(row['N_Annotators'])})",
                    ha="center", va="center", fontsize=8)
            bottom += row["N_Annotators"]

    ax.set_xticks(x)
    ax.set_xticklabels([PARTITION_LABELS.get(p, p) for p in partitions])
    ax.set_ylabel("Number of Annotators")
    ax.set_title("Exp 1: Annotator Group Distributions (Stacked)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp1_group_distributions.png"), dpi=150)
    plt.close(fig)
    print("  [OK] exp1_group_distributions.png")


# ===== Experiment 2 =====
def plot_exp2():
    df = pd.read_csv(os.path.join(DATA_DIR, "exp2_group_accuracy.csv"))
    df["Label"] = df.apply(_group_label, axis=1)

    metrics = ["AP_d", "TPR_d"]
    metric_labels = ["AP Disparity", "TPR Disparity"]
    unfair_cols = ["AP_unfair", "TPR_unfair"]

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    w = 0.35

    for k, (met, ufc, lbl) in enumerate(zip(metrics, unfair_cols, metric_labels)):
        offset = (k - 0.5) * w
        vals = df[met].values
        colors = []
        for _, row in df.iterrows():
            colors.append("red" if row[ufc] else GROUP_COLORS[k])
        bars = ax.bar(x + offset, vals, w, label=lbl, color=colors, edgecolor="grey", linewidth=0.5)

    ax.axhline(0.1, color="black", ls="--", lw=1, label="+-0.1 threshold")
    ax.axhline(-0.1, color="black", ls="--", lw=1)
    ax.axhline(0, color="grey", ls="-", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Label"], fontsize=8)
    ax.set_ylabel("Disparity")
    ax.set_title("Exp 2: AP and TPR Disparity per Group (red = unfair)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp2_group_accuracy.png"), dpi=150)
    plt.close(fig)
    print("  [OK] exp2_group_accuracy.png")


# ===== Experiment 3 =====
def plot_exp3():
    df = pd.read_csv(os.path.join(DATA_DIR, "exp3_behavioral_profiles.csv"))
    df["Label"] = df.apply(_group_label, axis=1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Panel 1: Decision Time (mean +/- std)
    ax = axes[0]
    x = np.arange(len(df))
    ax.bar(x, df["DecisionTime_mean"], yerr=df["DecisionTime_std"],
           color=sns.color_palette("Blues_d", len(df)), capsize=3, edgecolor="grey", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Label"], fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Seconds")
    ax.set_title("Decision Time (mean +/- std)")

    # Panel 2: Confidence Level
    ax = axes[1]
    ax.bar(x, df["ConfidenceLevel_mean"], yerr=df["ConfidenceLevel_std"],
           color=sns.color_palette("Greens_d", len(df)), capsize=3, edgecolor="grey", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Label"], fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Confidence (0-100)")
    ax.set_title("Confidence Level (mean +/- std)")

    # Panel 3: Click Count
    ax = axes[2]
    ax.bar(x, df["ClickCount_mean"], yerr=df["ClickCount_std"],
           color=sns.color_palette("Oranges_d", len(df)), capsize=3, edgecolor="grey", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Label"], fontsize=7, rotation=45, ha="right")
    ax.set_ylabel("Clicks")
    ax.set_title("Click Count (mean +/- std)")

    fig.suptitle("Exp 3: Behavioral Profiles per Group", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp3_behavioral_profiles.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  [OK] exp3_behavioral_profiles.png")


# ===== Experiment 4 =====
def plot_exp4():
    df = pd.read_csv(os.path.join(DATA_DIR, "exp4_majority_disagreement.csv"))
    df["Label"] = df.apply(_group_label, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(df))
    colors = sns.color_palette("coolwarm", len(df))
    ax.barh(y, df["DisagreementRate"], color=colors, edgecolor="grey", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(df["Label"], fontsize=9)
    ax.set_xlabel("Disagreement Rate")
    ax.set_title("Exp 4: Majority-Vote Disagreement Rate per Group")
    for i, v in enumerate(df["DisagreementRate"]):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp4_majority_disagreement.png"), dpi=150)
    plt.close(fig)
    print("  [OK] exp4_majority_disagreement.png")


# ===== Experiment 5 =====
def plot_exp5():
    df = pd.read_csv(os.path.join(DATA_DIR, "exp5_calibration_gap.csv"))
    df["Label"] = df.apply(_group_label, axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(df))
    w = 0.35

    ax.bar(x - w / 2, df["MeanConfidence"], w, label="Mean Confidence",
           color=sns.color_palette("Pastel1")[0], edgecolor="grey", linewidth=0.5)
    ax.bar(x + w / 2, df["MeanAccuracy"], w, label="Mean Accuracy",
           color=sns.color_palette("Pastel1")[1], edgecolor="grey", linewidth=0.5)

    # Annotate calibration gap
    for i, row in df.iterrows():
        gap = row["CalibrationGap"]
        flag = row["CalibFairFlag"]
        mid_y = (row["MeanConfidence"] + row["MeanAccuracy"]) / 2
        color = "red" if "UNFAIR" in str(flag) else "black"
        ax.annotate(f"gap={gap:.3f}", (x[i], mid_y), fontsize=7,
                    ha="center", color=color, fontweight="bold" if "UNFAIR" in str(flag) else "normal")

    ax.set_xticks(x)
    ax.set_xticklabels(df["Label"], fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Exp 5: Calibration Gap (Confidence vs Accuracy) -- red = unfair")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp5_calibration_gap.png"), dpi=150)
    plt.close(fig)
    print("  [OK] exp5_calibration_gap.png")


# ===== Experiment 6 =====
def plot_exp6():
    df = pd.read_csv(os.path.join(DATA_DIR, "exp6_explanation_rate.csv"))
    df["Label"] = df.apply(_group_label, axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(df))
    colors = sns.color_palette("Set2", len(df))
    ax.bar(x, df["ExplanationRate"], color=colors, edgecolor="grey", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(df["Label"], fontsize=8)
    ax.set_ylabel("Explanation Rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Exp 6: Explanation Rate per Group")
    for i, v in enumerate(df["ExplanationRate"]):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp6_explanation_rate.png"), dpi=150)
    plt.close(fig)
    print("  [OK] exp6_explanation_rate.png")


# ===== Experiment 7 =====
def plot_exp7():
    df = pd.read_csv(os.path.join(DATA_DIR, "exp7_per_question_disparity.csv"))

    partitions = df["Partition"].unique()
    n_part = len(partitions)
    ncols = 2
    nrows = (n_part + 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5 * nrows))
    axes = axes.flatten()

    for idx, part in enumerate(partitions):
        ax = axes[idx]
        sub = df[df["Partition"] == part]
        pivot = sub.pivot(index="QuestionNum", columns="Group", values="Disparity")
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                    ax=ax, linewidths=0.5, cbar_kws={"label": "Disparity"})
        ax.set_title(PARTITION_LABELS.get(part, part))
        ax.set_ylabel("Question")
        ax.set_xlabel("Group")

    # Hide extra axes
    for j in range(idx + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Exp 7: Per-Question Accuracy Disparity Heatmap", fontsize=13, y=1.01)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp7_per_question_disparity.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  [OK] exp7_per_question_disparity.png")


# ===== Experiment 8 =====
def plot_exp8():
    df = pd.read_csv(os.path.join(DATA_DIR, "exp8_conditional_fairness.csv"))

    partitions = df["Partition"].unique()
    fig, axes = plt.subplots(1, len(partitions), figsize=(5 * len(partitions), 5),
                             sharey=True)
    if len(partitions) == 1:
        axes = [axes]

    w = 0.35
    for idx, part in enumerate(partitions):
        ax = axes[idx]
        sub = df[df["Partition"] == part]
        groups = sub["Group"].unique()
        x = np.arange(len(groups))

        hard = sub[sub["Tier"] == "Hard"].set_index("Group").reindex(groups)
        easy = sub[sub["Tier"] == "Easy"].set_index("Group").reindex(groups)

        ax.bar(x - w / 2, hard["Accuracy"], w, label="Hard", color="#d9534f",
               edgecolor="grey", linewidth=0.5)
        ax.bar(x + w / 2, easy["Accuracy"], w, label="Easy", color="#5cb85c",
               edgecolor="grey", linewidth=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(groups, fontsize=8)
        ax.set_title(PARTITION_LABELS.get(part, part))
        if idx == 0:
            ax.set_ylabel("Accuracy")
        ax.legend(fontsize=8)

    fig.suptitle("Exp 8: Conditional Fairness (Hard vs Easy) per Group", fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp8_conditional_fairness.png"), dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print("  [OK] exp8_conditional_fairness.png")


# ===== Experiment 9 =====
def plot_exp9():
    df = pd.read_csv(os.path.join(DATA_DIR, "exp9_bootstrap_ci.csv"))
    df["Label"] = df.apply(_group_label, axis=1)

    fig, ax = plt.subplots(figsize=(8, 5))
    y = np.arange(len(df))

    # Compute error bars from CI
    centers = df["Obs_AccDisparity"].values
    lo = df["Acc_CI_lo"].values
    hi = df["Acc_CI_hi"].values
    err_lo = centers - lo
    err_hi = hi - centers

    colors = ["red" if sig else "steelblue" for sig in df["Acc_Significant"]]
    ax.errorbar(centers, y, xerr=[err_lo, err_hi], fmt="o", color="black",
                ecolor="grey", elinewidth=2, capsize=4, markersize=6, zorder=5)
    # Overlay colored dots
    for i in range(len(df)):
        ax.plot(centers[i], y[i], "o", color=colors[i], markersize=8, zorder=6)

    ax.axvline(0, color="grey", ls="--", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(df["Label"], fontsize=9)
    ax.set_xlabel("Accuracy Disparity (with 95% Bootstrap CI)")
    ax.set_title("Exp 9: Bootstrap Confidence Intervals (red = significant)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "exp9_bootstrap_ci.png"), dpi=150)
    plt.close(fig)
    print("  [OK] exp9_bootstrap_ci.png")


# ===== Main =====
def main():
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUT_DIR}")
    print()

    plot_exp1()
    plot_exp2()
    plot_exp3()
    plot_exp4()
    plot_exp5()
    plot_exp6()
    plot_exp7()
    plot_exp8()
    plot_exp9()

    print("\nAll 9 visualizations generated successfully.")


if __name__ == "__main__":
    main()
