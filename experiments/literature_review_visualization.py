"""
Literature Review Visualization — Bias Detection Techniques for Annotation Data
=================================================================================
Creates visual comparison charts of techniques applicable to Export_and_Compiled.xlsx
type of data (annotator demographics + behavioral signals + binary decisions).
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "literature_review")
os.makedirs(RESULTS_DIR, exist_ok=True)


def create_technique_comparison():
    """Create comprehensive comparison of bias detection techniques."""

    techniques = pd.DataFrame([
        # Name, Category, Can detect Active bias?, Small sample friendly?, Applicable to our data?, Complexity, Key Paper
        {"Technique": "FairEM (10-metric disparity)", "Category": "Statistical Fairness",
         "Detects_Active": 0.3, "Small_Sample": 0.6, "Data_Fit": 1.0, "Complexity": 0.3,
         "Phase": "Detection", "Status": "Implemented"},
        {"Technique": "Intersectional Fairness", "Category": "Statistical Fairness",
         "Detects_Active": 0.5, "Small_Sample": 0.3, "Data_Fit": 1.0, "Complexity": 0.4,
         "Phase": "Detection", "Status": "Implemented"},
        {"Technique": "Calibration Gap Analysis", "Category": "Statistical Fairness",
         "Detects_Active": 0.4, "Small_Sample": 0.5, "Data_Fit": 1.0, "Complexity": 0.2,
         "Phase": "Detection", "Status": "Implemented"},
        {"Technique": "Logistic Regression Decomposition", "Category": "Regression",
         "Detects_Active": 0.7, "Small_Sample": 0.5, "Data_Fit": 1.0, "Complexity": 0.4,
         "Phase": "Root Cause", "Status": "Implemented"},
        {"Technique": "Leave-One-Out Influence", "Category": "Individual Analysis",
         "Detects_Active": 0.8, "Small_Sample": 0.8, "Data_Fit": 1.0, "Complexity": 0.5,
         "Phase": "Root Cause", "Status": "Implemented"},
        {"Technique": "Ablation Study", "Category": "Experimental",
         "Detects_Active": 0.6, "Small_Sample": 0.4, "Data_Fit": 1.0, "Complexity": 0.4,
         "Phase": "Root Cause", "Status": "Implemented"},
        {"Technique": "Chi-Square / Cramer's V", "Category": "Statistical Tests",
         "Detects_Active": 0.3, "Small_Sample": 0.4, "Data_Fit": 1.0, "Complexity": 0.2,
         "Phase": "Detection", "Status": "Implemented"},
        {"Technique": "Bootstrap Confidence Intervals", "Category": "Statistical Tests",
         "Detects_Active": 0.3, "Small_Sample": 0.7, "Data_Fit": 1.0, "Complexity": 0.3,
         "Phase": "Validation", "Status": "Implemented"},
        {"Technique": "Dawid-Skene Model", "Category": "Annotator Modeling",
         "Detects_Active": 0.6, "Small_Sample": 0.5, "Data_Fit": 0.9, "Complexity": 0.6,
         "Phase": "Root Cause", "Status": "Recommended"},
        {"Technique": "MACE (Multi-Annotator CE)", "Category": "Annotator Modeling",
         "Detects_Active": 0.7, "Small_Sample": 0.5, "Data_Fit": 0.9, "Complexity": 0.6,
         "Phase": "Root Cause", "Status": "Recommended"},
        {"Technique": "Item Response Theory (IRT)", "Category": "Psychometrics",
         "Detects_Active": 0.8, "Small_Sample": 0.4, "Data_Fit": 0.95, "Complexity": 0.7,
         "Phase": "Root Cause", "Status": "Recommended"},
        {"Technique": "Blinder-Oaxaca Decomposition", "Category": "Regression",
         "Detects_Active": 0.9, "Small_Sample": 0.4, "Data_Fit": 0.85, "Complexity": 0.6,
         "Phase": "Root Cause", "Status": "Recommended"},
        {"Technique": "Causal Fairness (DAG-based)", "Category": "Causal Inference",
         "Detects_Active": 1.0, "Small_Sample": 0.3, "Data_Fit": 0.7, "Complexity": 0.9,
         "Phase": "Root Cause", "Status": "Future"},
        {"Technique": "CrowdTruth Framework", "Category": "Annotator Modeling",
         "Detects_Active": 0.5, "Small_Sample": 0.6, "Data_Fit": 0.8, "Complexity": 0.5,
         "Phase": "Detection", "Status": "Recommended"},
        {"Technique": "Fair Truth Discovery (Li et al.)", "Category": "Fair Aggregation",
         "Detects_Active": 0.6, "Small_Sample": 0.5, "Data_Fit": 0.9, "Complexity": 0.6,
         "Phase": "Mitigation", "Status": "Future"},
        {"Technique": "DEM-MoE (Xu et al.)", "Category": "Fair Aggregation",
         "Detects_Active": 0.7, "Small_Sample": 0.3, "Data_Fit": 0.6, "Complexity": 0.9,
         "Phase": "Mitigation", "Status": "Future"},
        {"Technique": "Behavioral Gatekeeper (RF/GB)", "Category": "Behavioral Analysis",
         "Detects_Active": 0.7, "Small_Sample": 0.3, "Data_Fit": 1.0, "Complexity": 0.5,
         "Phase": "Detection", "Status": "Scaffolded"},
        {"Technique": "Explanation NLP Analysis", "Category": "Qualitative",
         "Detects_Active": 0.6, "Small_Sample": 0.7, "Data_Fit": 0.95, "Complexity": 0.7,
         "Phase": "Root Cause", "Status": "Future"},
        {"Technique": "Permutation Fairness Test", "Category": "Statistical Tests",
         "Detects_Active": 0.4, "Small_Sample": 0.8, "Data_Fit": 1.0, "Complexity": 0.3,
         "Phase": "Validation", "Status": "Recommended"},
        {"Technique": "AIF360 / Fairlearn Toolkit", "Category": "Statistical Fairness",
         "Detects_Active": 0.4, "Small_Sample": 0.5, "Data_Fit": 0.8, "Complexity": 0.3,
         "Phase": "Detection", "Status": "Recommended"},
    ])

    # =========================================================================
    # Figure 1: Technique Comparison Bubble Chart
    # =========================================================================
    fig, ax = plt.subplots(figsize=(16, 10))

    status_colors = {
        "Implemented": "#43A047",
        "Scaffolded": "#FFA726",
        "Recommended": "#1E88E5",
        "Future": "#AB47BC",
    }
    phase_markers = {
        "Detection": "o",
        "Root Cause": "s",
        "Validation": "D",
        "Mitigation": "^",
    }

    for _, t in techniques.iterrows():
        ax.scatter(t["Detects_Active"], t["Small_Sample"],
                   s=t["Data_Fit"] * 400,
                   c=status_colors[t["Status"]],
                   marker=phase_markers[t["Phase"]],
                   alpha=0.7, edgecolors="black", linewidth=0.5)
        # Label
        offset_x = 0.015
        offset_y = 0.015
        ax.annotate(t["Technique"], (t["Detects_Active"] + offset_x, t["Small_Sample"] + offset_y),
                    fontsize=7, ha="left")

    ax.set_xlabel("Ability to Detect ACTIVE (Causal) Bias Sources", fontsize=11)
    ax.set_ylabel("Small Sample Friendliness (n=11 minority)", fontsize=11)
    ax.set_xlim(-0.05, 1.15)
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5)
    ax.axvline(0.5, color="gray", linestyle=":", linewidth=0.5)

    # Quadrant labels
    ax.text(0.25, 0.95, "PASSIVE DETECTION\n(good for small samples)", ha="center",
            fontsize=9, color="gray", style="italic")
    ax.text(0.75, 0.95, "IDEAL\n(active + small-sample)", ha="center",
            fontsize=9, color="green", style="italic", fontweight="bold")
    ax.text(0.25, 0.05, "LIMITED\n(neither causal nor small-sample)", ha="center",
            fontsize=9, color="red", style="italic")
    ax.text(0.75, 0.05, "CAUSAL BUT NEEDS DATA\n(active detection, needs large N)", ha="center",
            fontsize=9, color="orange", style="italic")

    # Legends
    status_patches = [mpatches.Patch(color=c, label=l) for l, c in status_colors.items()]
    phase_handles = [plt.Line2D([0], [0], marker=m, color="gray", markersize=8, linestyle="None",
                                label=l) for l, m in phase_markers.items()]
    legend1 = ax.legend(handles=status_patches, title="Implementation Status",
                        loc="lower left", fontsize=8)
    ax.add_artist(legend1)
    ax.legend(handles=phase_handles, title="Analysis Phase",
              loc="upper right", fontsize=8)

    ax.set_title("Literature Review: Bias Detection Techniques for Annotator Fairness\n"
                 "(Bubble size = Data Fit to Export_and_Compiled.xlsx)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "technique_comparison_bubble.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Figure 2: Pipeline Roadmap (which technique at which phase)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(18, 8))

    phases = ["Detection", "Root Cause", "Validation", "Mitigation"]
    phase_x = {p: i for i, p in enumerate(phases)}
    categories = techniques["Category"].unique()
    cat_y = {c: i for i, c in enumerate(sorted(categories))}

    for _, t in techniques.iterrows():
        x = phase_x[t["Phase"]]
        y = cat_y[t["Category"]]
        color = status_colors[t["Status"]]
        ax.scatter(x, y, s=300, c=color, alpha=0.8, edgecolors="black", linewidth=0.5, zorder=3)
        ax.annotate(t["Technique"].replace(" ", "\n", 1)[:30],
                    (x + 0.08, y), fontsize=7, va="center")

    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phases, fontsize=11, fontweight="bold")
    ax.set_yticks(range(len(cat_y)))
    ax.set_yticklabels(sorted(categories), fontsize=9)
    ax.set_xlabel("Analysis Pipeline Phase", fontsize=12)
    ax.set_ylabel("Technique Category", fontsize=12)

    # Phase arrows
    for i in range(len(phases) - 1):
        ax.annotate("", xy=(i + 0.8, -0.5), xytext=(i + 0.2, -0.5),
                     arrowprops=dict(arrowstyle="->", color="gray", lw=2))

    status_patches = [mpatches.Patch(color=c, label=l) for l, c in status_colors.items()]
    ax.legend(handles=status_patches, title="Status", loc="upper right", fontsize=9)
    ax.set_title("Bias Detection Pipeline: Techniques by Phase and Category\n"
                 "Green = Implemented | Blue = Recommended Next | Purple = Future Work",
                 fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "technique_pipeline_roadmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Figure 3: Technique Scoring Heatmap
    # =========================================================================
    score_cols = ["Detects_Active", "Small_Sample", "Data_Fit", "Complexity"]
    score_labels = ["Active Bias\nDetection", "Small Sample\nFriendly", "Data Fit\n(Our Data)", "Complexity\n(lower=simpler)"]
    pivot = techniques.set_index("Technique")[score_cols]
    pivot.columns = score_labels

    # Sort by total score (higher is better except complexity)
    pivot["Total"] = pivot.iloc[:, 0] + pivot.iloc[:, 1] + pivot.iloc[:, 2] - pivot.iloc[:, 3]
    pivot = pivot.sort_values("Total", ascending=False)
    pivot = pivot.drop("Total", axis=1)

    fig, ax = plt.subplots(figsize=(12, 12))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn",
                vmin=0, vmax=1, linewidths=0.5, ax=ax,
                cbar_kws={"label": "Score (0-1)"})

    # Add status color bars
    for i, (idx, row) in enumerate(techniques.set_index("Technique").loc[pivot.index].iterrows()):
        color = status_colors[row["Status"]]
        ax.add_patch(plt.Rectangle((-0.15, i), 0.1, 1, transform=ax.get_yaxis_transform(),
                                   color=color, clip_on=False))

    ax.set_title("Technique Scoring Matrix\n"
                 "(Higher = Better for all except Complexity where Lower = Better)\n"
                 "Left bar: Green=Implemented, Blue=Recommended, Orange=Scaffolded, Purple=Future",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "technique_scoring_heatmap.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    # =========================================================================
    # Figure 4: Our Findings Summary — Active vs Passive Parameters
    # =========================================================================
    fig = plt.figure(figsize=(18, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.3)

    # Panel A: Classification summary
    ax1 = fig.add_subplot(gs[0, 0])
    partitions = ["LinguisticGroup", "ExpertiseGroup", "ExperienceGroup", "AgeGroup"]
    classifications = ["STRONGLY\nACTIVE", "MODERATELY\nACTIVE", "ACTIVE", "ACTIVE"]
    scores = [4, 2, 3, 3]
    colors_cls = ["#D32F2F", "#FF9800", "#F57C00", "#F57C00"]
    bars = ax1.barh(range(len(partitions)), scores, color=colors_cls, alpha=0.8, edgecolor="black")
    ax1.set_yticks(range(len(partitions)))
    ax1.set_yticklabels(partitions, fontsize=10)
    ax1.set_xlabel("Active Evidence Score (out of 6)")
    ax1.set_title("(A) Active vs Passive Classification", fontsize=11, fontweight="bold")
    for bar, cls in zip(bars, classifications):
        ax1.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                 cls, va="center", fontsize=9, fontweight="bold")
    ax1.set_xlim(0, 6.5)
    ax1.axvline(3, color="red", linestyle=":", linewidth=0.8, label="Active threshold")
    ax1.legend(fontsize=8)

    # Panel B: Key finding — NonNative x HardQ interaction
    ax2 = fig.add_subplot(gs[0, 1])
    groups = ["Native\n(Easy)", "Non-Native\n(Easy)", "Native\n(Hard)", "Non-Native\n(Hard)"]
    accs = [0.743, 0.764, 0.578, 0.424]
    bar_colors = ["#43A047", "#43A047", "#1E88E5", "#E53935"]
    bars2 = ax2.bar(groups, accs, color=bar_colors, alpha=0.8, edgecolor="black")
    ax2.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, label="Chance level")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("(B) Critical Finding: NonNative x Hard Interaction\n"
                  "(Non-Native FAIR on easy, UNFAIR on hard questions)", fontsize=10, fontweight="bold")
    ax2.set_ylim(0, 1.0)
    for bar, acc in zip(bars2, accs):
        ax2.text(bar.get_x() + bar.get_width()/2, acc + 0.02, f"{acc:.1%}",
                 ha="center", fontsize=10, fontweight="bold")
    ax2.legend(fontsize=8)

    # Panel C: Q6 anomaly
    ax3 = fig.add_subplot(gs[1, 0])
    q_accs_native = [0.73, 0.65, 0.73, 0.78, 0.57, 0.57, 0.69, 0.86]
    q_accs_nn = [0.73, 0.45, 0.64, 0.64, 0.73, 0.09, 0.55, 1.00]
    questions = [f"Q{i}" for i in range(1, 9)]
    x = np.arange(8)
    w = 0.35
    ax3.bar(x - w/2, q_accs_native, w, label="Native", color="#1E88E5", alpha=0.8)
    ax3.bar(x + w/2, q_accs_nn, w, label="Non-Native", color="#FF8F00", alpha=0.8)
    ax3.set_xticks(x)
    ax3.set_xticklabels(questions)
    ax3.set_ylabel("Accuracy")
    ax3.set_title("(C) Per-Question Accuracy: Q6 is the Bias Hotspot\n"
                  "(Non-Native 9.1% vs Native 57%, p=0.006)", fontsize=10, fontweight="bold")
    ax3.axhline(0.5, color="gray", linestyle="--", linewidth=0.5)
    ax3.legend(fontsize=9)
    # Highlight Q6
    ax3.axvspan(4.6, 5.4, alpha=0.15, color="red")
    ax3.annotate("p=0.006\nOR=13.3", xy=(5, 0.09), fontsize=8, color="red",
                 fontweight="bold", ha="center")

    # Panel D: Regression coefficients
    ax4 = fig.add_subplot(gs[1, 1])
    features = ["IsHardQ", "NonNative\nx HardQ", "Confidence", "ClickCount",
                "HighEdu\nx HardQ", "IsHighEdu", "IsNonNative", "IsOlder35", "IsSTEM"]
    coefs = [-0.723, -0.587, 0.168, 0.152, -0.092, -0.077, 0.075, -0.052, 0.006]
    colors_reg = ["#E53935" if c < -0.1 else "#FF9800" if c < 0 else "#43A047" for c in coefs]
    ax4.barh(range(len(features)), coefs, color=colors_reg, alpha=0.8, edgecolor="black")
    ax4.set_yticks(range(len(features)))
    ax4.set_yticklabels(features, fontsize=9)
    ax4.set_xlabel("Logistic Regression Coefficient")
    ax4.axvline(0, color="black", linewidth=1)
    ax4.axvline(-0.1, color="red", linestyle=":", linewidth=0.8)
    ax4.axvline(0.1, color="red", linestyle=":", linewidth=0.8)
    ax4.set_title("(D) Feature Importance: What Drives Accuracy?\n"
                  "(Red = ACTIVE bias sources, beyond +/-0.1)", fontsize=10, fontweight="bold")
    for i, c in enumerate(coefs):
        label = "ACTIVE" if abs(c) > 0.1 else "passive"
        ax4.text(c + 0.02 * np.sign(c), i, label, va="center", fontsize=8,
                 fontweight="bold" if abs(c) > 0.1 else "normal",
                 color="red" if abs(c) > 0.1 else "gray")

    plt.suptitle("FairPrep Phase 2: Active vs Passive Bias Sources — Key Findings",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.savefig(os.path.join(RESULTS_DIR, "findings_summary_4panel.png"),
                dpi=150, bbox_inches="tight")
    plt.close()

    techniques.to_csv(os.path.join(RESULTS_DIR, "technique_comparison.csv"), index=False)
    print(f"  Saved 4 visualizations + CSV to {os.path.abspath(RESULTS_DIR)}")
    return techniques


if __name__ == "__main__":
    import matplotlib.gridspec as gridspec
    print("Creating literature review visualizations...")
    create_technique_comparison()
    print("Done!")
