"""
Phase 2: Active vs Passive Bias Investigation
===============================================
Deep-dive experiment to identify which demographic parameters and individual
annotators are "actively" causing bias (direct performance impact) vs
"passively" contributing (correlated but not causal).

Terminology:
  ACTIVE bias source  — a parameter/annotator whose removal or correction
                        materially changes the fairness outcome (>= 0.05 shift
                        in disparity). These are direct drivers of unfairness.
  PASSIVE bias source — a parameter/annotator that correlates with bias metrics
                        but whose removal does not materially change outcomes.
                        These are bystanders or confounded variables.

Experiments:
  A1 — Intersectional Fairness Analysis (crossing 2+ demographics)
  A2 — Leave-One-Out Annotator Influence (which individuals shift group metrics)
  A3 — Ablation Study: Parameter Isolation (remove each partition, measure change)
  A4 — Mutual Information & Chi-Square: Statistical association between demographics
  A5 — Regression-based Decomposition (logistic regression on accuracy)
  A6 — Per-Question xPer-Demographic Interaction Heatmap (granular)
  A7 — Active vs Passive Classification Summary

Usage:
    cd FairPrep
    python experiments/phase2_active_passive_bias.py
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import combinations
from scipy.stats import chi2_contingency, fisher_exact, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.excel_data_loader import load_excel_data, get_annotator_demographics
from src.measures import AP, SP, TPR, FPR, FNR, TNR, PPV, NPV, FDR, FOR

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXCEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Export_and_Compiled.xlsx"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "phase2_bias")
os.makedirs(RESULTS_DIR, exist_ok=True)

GROUP_COLUMNS = ["LinguisticGroup", "ExpertiseGroup", "ExperienceGroup", "AgeGroup"]
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

HARD_QUESTIONS = [2, 5, 6]
EASY_QUESTIONS = [1, 3, 4, 7, 8]


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


# ===========================================================================
# A1: Intersectional Fairness Analysis
# ===========================================================================

def experiment_A1_intersectional(df):
    hdr("A1: Intersectional Fairness Analysis")
    print("  Crosses pairs of demographic partitions to find compound disadvantage.")

    overall_metrics = compute_all_metrics(df["SurveyAnswer"], df["ActualAnswer"])
    results = []

    # All pairwise intersections
    for gc1, gc2 in combinations(GROUP_COLUMNS, 2):
        df["_intersect"] = df[gc1] + " x" + df[gc2]
        for combo, gdf in df.groupby("_intersect"):
            metrics = compute_all_metrics(gdf["SurveyAnswer"], gdf["ActualAnswer"])
            row = {
                "Intersection": f"{gc1} x{gc2}",
                "Group": combo,
                "N": len(gdf),
                "N_Annotators": gdf["ResponseId"].nunique(),
            }
            unfair_metrics = []
            for mname, _, direction in ALL_METRICS:
                disp = metrics[mname] - overall_metrics[mname]
                row[mname] = round(metrics[mname], 4)
                row[f"{mname}_d"] = round(disp, 4)
                if is_unfair(disp, direction):
                    unfair_metrics.append(f"{mname}({disp:+.3f})")
            row["Unfair_Metrics"] = "; ".join(unfair_metrics) if unfair_metrics else ""
            row["N_Unfair"] = len(unfair_metrics)
            results.append(row)

    df.drop("_intersect", axis=1, inplace=True)
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "A1_intersectional.csv"), index=False)

    # Top disadvantaged intersections
    results_df["Max_Neg_Disp"] = results_df[[f"{m}_d" for m, _, d in ALL_METRICS if d == "higher"]].min(axis=1)
    top_disadv = results_df.nsmallest(10, "Max_Neg_Disp")

    print("\n  Top 10 most disadvantaged intersectional groups:")
    print(f"  {'Group':50s} | {'N':>4s} | {'AP_d':>7s} | {'TPR_d':>7s} | {'#Unfair':>7s} | Unfair Metrics")
    for _, r in top_disadv.iterrows():
        print(f"  {r['Group']:50s} | {r['N']:4d} | {r['AP_d']:+.4f} | {r['TPR_d']:+.4f} | {r['N_Unfair']:7d} | {r['Unfair_Metrics'][:60]}")

    # Heatmap: intersectional AP disparity
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    axes = axes.flatten()
    pair_idx = 0
    for gc1, gc2 in combinations(GROUP_COLUMNS, 2):
        sub = results_df[results_df["Intersection"] == f"{gc1} x{gc2}"]
        groups1 = sorted(df[gc1].unique())
        groups2 = sorted(df[gc2].unique())
        pivot_ap = pd.DataFrame(index=groups1, columns=groups2, dtype=float)
        pivot_tpr = pd.DataFrame(index=groups1, columns=groups2, dtype=float)
        for _, r in sub.iterrows():
            parts = r["Group"].split(" x")
            if len(parts) == 2:
                pivot_ap.loc[parts[0], parts[1]] = r["AP_d"]
                pivot_tpr.loc[parts[0], parts[1]] = r["TPR_d"]

        ax = axes[pair_idx]
        pivot_ap = pivot_ap.astype(float)
        sns.heatmap(pivot_ap, ax=ax, annot=True, fmt=".3f", cmap="RdYlGn",
                    center=0, vmin=-0.15, vmax=0.15, linewidths=0.5)
        ax.set_title(f"AP Disparity\n{gc1} x{gc2}", fontsize=9, fontweight="bold")
        pair_idx += 1

    plt.suptitle("A1 — Intersectional Accuracy Parity Disparities (group − overall)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "A1_intersectional.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  -> A1_intersectional.csv + .png")
    return results_df


# ===========================================================================
# A2: Leave-One-Out Annotator Influence
# ===========================================================================

def experiment_A2_leave_one_out(df):
    hdr("A2: Leave-One-Out Annotator Influence Analysis")
    print("  For each annotator, compute how much group disparity changes when they're removed.")

    overall_metrics = compute_all_metrics(df["SurveyAnswer"], df["ActualAnswer"])
    annotators = df["ResponseId"].unique()

    results = []
    for rid in annotators:
        df_without = df[df["ResponseId"] != rid]
        ann_data = df[df["ResponseId"] == rid].iloc[0]
        ann_acc = float(df[df["ResponseId"] == rid]["Accuracy"].mean())

        row = {"ResponseId": rid, "Annotator_Accuracy": round(ann_acc, 4)}

        for gc in GROUP_COLUMNS:
            row[gc] = ann_data[gc]
            # Original group disparity
            grp_name = ann_data[gc]
            grp_orig = df[df[gc] == grp_name]
            grp_without = df_without[df_without[gc] == grp_name]
            overall_without = compute_all_metrics(df_without["SurveyAnswer"], df_without["ActualAnswer"])

            if len(grp_without) == 0:
                continue

            orig_metrics = compute_all_metrics(grp_orig["SurveyAnswer"], grp_orig["ActualAnswer"])
            new_metrics = compute_all_metrics(grp_without["SurveyAnswer"], grp_without["ActualAnswer"])

            orig_ap_d = orig_metrics["AP"] - overall_metrics["AP"]
            new_ap_d = new_metrics["AP"] - overall_without["AP"]
            orig_tpr_d = orig_metrics["TPR"] - overall_metrics["TPR"]
            new_tpr_d = new_metrics["TPR"] - overall_without["TPR"]

            row[f"{gc}_AP_shift"] = round(new_ap_d - orig_ap_d, 5)
            row[f"{gc}_TPR_shift"] = round(new_tpr_d - orig_tpr_d, 5)

        results.append(row)

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "A2_leave_one_out.csv"), index=False)

    # Find most influential annotators per partition
    print("\n  Most influential annotators (largest |AP_shift| when removed):")
    for gc in GROUP_COLUMNS:
        col = f"{gc}_AP_shift"
        if col not in results_df.columns:
            continue
        top = results_df.reindex(results_df[col].abs().nlargest(3).index)
        print(f"\n  --- {gc} ---")
        for _, r in top.iterrows():
            print(f"    {r['ResponseId'][:20]:20s} [{r[gc]:15s}] "
                  f"Acc={r['Annotator_Accuracy']:.3f}  AP_shift={r[col]:+.5f}  "
                  f"TPR_shift={r.get(f'{gc}_TPR_shift', 0):+.5f}")

    # Visualization: scatter of annotator accuracy vs influence
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    colors = {"Native": "#1E88E5", "Non-Native": "#FF8F00",
              "STEM": "#43A047", "Non-STEM": "#E53935",
              "High-Edu": "#8E24AA", "Lower-Edu": "#FB8C00",
              "Young-18-34": "#00ACC1", "Older-35plus": "#6D4C41"}

    for i, gc in enumerate(GROUP_COLUMNS):
        ax = axes[i]
        col = f"{gc}_AP_shift"
        if col not in results_df.columns:
            continue
        for gname in sorted(df[gc].unique()):
            sub = results_df[results_df[gc] == gname]
            ax.scatter(sub["Annotator_Accuracy"], sub[col].abs(),
                       label=gname, alpha=0.7, s=40,
                       color=colors.get(gname, "#90A4AE"))
        ax.set_xlabel("Annotator Accuracy")
        ax.set_ylabel("|AP Disparity Shift| when removed")
        ax.set_title(f"Annotator Influence on {gc}", fontsize=10, fontweight="bold")
        ax.legend(fontsize=8)
        ax.axhline(0.005, color="orange", linestyle=":", linewidth=0.8, label="Influence threshold")

    plt.suptitle("A2 — Leave-One-Out: Individual Annotator Influence on Group Disparity",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "A2_leave_one_out.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  -> A2_leave_one_out.csv + .png")
    return results_df


# ===========================================================================
# A3: Ablation Study — Remove each partition, measure fairness change
# ===========================================================================

def experiment_A3_ablation(df):
    hdr("A3: Ablation Study — Parameter Isolation")
    print("  Remove one demographic dimension at a time to measure its independent effect.")
    print("  'Active' = removal changes max disparity by >= 0.02")

    overall_metrics = compute_all_metrics(df["SurveyAnswer"], df["ActualAnswer"])
    baseline_disparities = {}
    for gc in GROUP_COLUMNS:
        max_disp = 0
        for gname, gdf in df.groupby(gc):
            m = compute_all_metrics(gdf["SurveyAnswer"], gdf["ActualAnswer"])
            for mname, _, direction in ALL_METRICS:
                d = m[mname] - overall_metrics[mname]
                if direction == "higher":
                    max_disp = max(max_disp, abs(min(d, 0)))
                else:
                    max_disp = max(max_disp, abs(max(d, 0)))
        baseline_disparities[gc] = max_disp

    # For each partition, equalize by resampling minority to match majority
    results = []
    for target_gc in GROUP_COLUMNS:
        groups = df.groupby(target_gc)
        group_sizes = groups.size()
        min_size = group_sizes.min()

        # Downsample to equalize
        balanced_dfs = []
        rng = np.random.default_rng(42)
        for gname, gdf in groups:
            if len(gdf) > min_size:
                idx = rng.choice(len(gdf), min_size, replace=False)
                balanced_dfs.append(gdf.iloc[idx])
            else:
                balanced_dfs.append(gdf)
        df_balanced = pd.concat(balanced_dfs, ignore_index=True)

        # Measure remaining disparity on OTHER partitions after balancing target
        for gc in GROUP_COLUMNS:
            if gc == target_gc:
                continue
            overall_bal = compute_all_metrics(df_balanced["SurveyAnswer"], df_balanced["ActualAnswer"])
            max_disp_after = 0
            for gname, gdf in df_balanced.groupby(gc):
                m = compute_all_metrics(gdf["SurveyAnswer"], gdf["ActualAnswer"])
                for mname, _, direction in ALL_METRICS:
                    d = m[mname] - overall_bal[mname]
                    if direction == "higher":
                        max_disp_after = max(max_disp_after, abs(min(d, 0)))
                    else:
                        max_disp_after = max(max_disp_after, abs(max(d, 0)))

            change = max_disp_after - baseline_disparities[gc]
            results.append({
                "Balanced_Partition": target_gc,
                "Measured_Partition": gc,
                "Baseline_MaxDisp": round(baseline_disparities[gc], 4),
                "After_MaxDisp": round(max_disp_after, 4),
                "Change": round(change, 4),
                "Interpretation": "ACTIVE (correlated)" if abs(change) >= 0.02 else "PASSIVE (independent)"
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "A3_ablation.csv"), index=False)

    print(f"\n  {'Balanced':25s} | {'Measured':25s} | {'Before':>7s} | {'After':>7s} | {'Change':>7s} | Status")
    for _, r in results_df.iterrows():
        print(f"  {r['Balanced_Partition']:25s} | {r['Measured_Partition']:25s} | "
              f"{r['Baseline_MaxDisp']:.4f} | {r['After_MaxDisp']:.4f} | "
              f"{r['Change']:+.4f} | {r['Interpretation']}")

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = results_df.pivot(index="Balanced_Partition", columns="Measured_Partition", values="Change")
    sns.heatmap(pivot, annot=True, fmt="+.3f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax)
    ax.set_title("A3 — Ablation: Change in Max Disparity After Balancing One Partition\n"
                 "(Positive = disparity increased, Negative = decreased)", fontsize=11, fontweight="bold")
    ax.set_xlabel("Measured Partition")
    ax.set_ylabel("Balanced Partition")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "A3_ablation.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  -> A3_ablation.csv + .png")
    return results_df


# ===========================================================================
# A4: Mutual Information & Chi-Square — Statistical Association
# ===========================================================================

def experiment_A4_statistical_association(df):
    hdr("A4: Statistical Association (Chi-Square & Cramér's V)")
    print("  Tests which demographics are statistically associated with accuracy.")

    results = []

    # 1. Chi-square: demographic group vs correct/incorrect
    for gc in GROUP_COLUMNS:
        ct = pd.crosstab(df[gc], df["Accuracy"])
        chi2, p, dof, expected = chi2_contingency(ct)
        n = len(df)
        k = min(ct.shape)
        cramers_v = np.sqrt(chi2 / (n * (k - 1))) if (n * (k - 1)) > 0 else 0

        results.append({
            "Test": "Chi2_vs_Accuracy",
            "Variable": gc,
            "Chi2": round(chi2, 4),
            "p_value": round(p, 4),
            "Cramers_V": round(cramers_v, 4),
            "Significant": p < 0.05,
            "Effect_Size": "Small" if cramers_v < 0.1 else "Medium" if cramers_v < 0.3 else "Large",
        })
        print(f"  {gc:25s} vs Accuracy: Chi2={chi2:.3f}, p={p:.4f}, V={cramers_v:.4f} "
              f"[{'SIG' if p < 0.05 else 'ns'}] [{results[-1]['Effect_Size']}]")

    # 2. Chi-square: pairwise between demographic partitions (confounding check)
    print("\n  Pairwise demographic associations (confounding check):")
    for gc1, gc2 in combinations(GROUP_COLUMNS, 2):
        ct = pd.crosstab(df[gc1], df[gc2])
        chi2, p, dof, expected = chi2_contingency(ct)
        n = len(df)
        k = min(ct.shape)
        cramers_v = np.sqrt(chi2 / (n * (k - 1))) if (n * (k - 1)) > 0 else 0

        results.append({
            "Test": "Chi2_Pairwise",
            "Variable": f"{gc1} x{gc2}",
            "Chi2": round(chi2, 4),
            "p_value": round(p, 4),
            "Cramers_V": round(cramers_v, 4),
            "Significant": p < 0.05,
            "Effect_Size": "Small" if cramers_v < 0.1 else "Medium" if cramers_v < 0.3 else "Large",
        })
        print(f"  {gc1:20s} x{gc2:20s}: Chi2={chi2:.3f}, p={p:.4f}, V={cramers_v:.4f} "
              f"[{'SIG -> CONFOUNDED' if p < 0.05 else 'independent'}]")

    # 3. Spearman correlations: numeric demographics vs accuracy
    print("\n  Spearman correlations (numeric demographics -> accuracy):")
    for col in ["Age", "Education", "EngProf", "Major"]:
        rho, p = spearmanr(df[col], df["Accuracy"])
        results.append({
            "Test": "Spearman_vs_Accuracy",
            "Variable": col,
            "Chi2": round(rho, 4),  # using Chi2 field for rho
            "p_value": round(p, 4),
            "Cramers_V": abs(round(rho, 4)),
            "Significant": p < 0.05,
            "Effect_Size": "Small" if abs(rho) < 0.1 else "Medium" if abs(rho) < 0.3 else "Large",
        })
        print(f"  {col:15s}: rho={rho:+.4f}, p={p:.4f} [{'SIG' if p < 0.05 else 'ns'}]")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "A4_statistical_association.csv"), index=False)

    # Visualization: Cramér's V heatmap for pairwise + accuracy
    labels = GROUP_COLUMNS + ["Accuracy"]
    n_labels = len(labels)
    v_matrix = np.zeros((n_labels, n_labels))
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if i == j:
                v_matrix[i, j] = 1.0
                continue
            col1 = df[l1] if l1 != "Accuracy" else df["Accuracy"].astype(str)
            col2 = df[l2] if l2 != "Accuracy" else df["Accuracy"].astype(str)
            ct = pd.crosstab(col1, col2)
            chi2, p, dof, exp = chi2_contingency(ct)
            k = min(ct.shape)
            v = np.sqrt(chi2 / (len(df) * (k - 1))) if (len(df) * (k - 1)) > 0 else 0
            v_matrix[i, j] = v

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(v_matrix, index=labels, columns=labels),
                annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=0.5,
                linewidths=0.5, ax=ax)
    ax.set_title("A4 — Cramér's V: Association Between Demographics & Accuracy\n"
                 "(Higher V = stronger association; V > 0.1 suggests confounding)",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "A4_statistical_association.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  -> A4_statistical_association.csv + .png")
    return results_df


# ===========================================================================
# A5: Logistic Regression Decomposition
# ===========================================================================

def experiment_A5_regression(df):
    hdr("A5: Logistic Regression Decomposition")
    print("  Which demographic features independently predict accuracy?")
    print("  Controls for confounding between demographics.")

    # Prepare features
    features = {
        "IsNonNative": (df["LinguisticGroup"] == "Non-Native").astype(int),
        "IsSTEM": (df["ExpertiseGroup"] == "STEM").astype(int),
        "IsHighEdu": (df["ExperienceGroup"] == "High-Edu").astype(int),
        "IsOlder35": (df["AgeGroup"] == "Older-35plus").astype(int),
        "ConfidenceNorm": df["ConfidenceLevelNorm"],
        "ClickCount": df["ClickCount"],
        "IsHardQ": df["QuestionNum"].isin(HARD_QUESTIONS).astype(int),
    }
    X = pd.DataFrame(features)
    y = df["Accuracy"].values

    # Add interaction: NonNative xHardQ
    X["NonNative_x_HardQ"] = X["IsNonNative"] * X["IsHardQ"]
    X["HighEdu_x_HardQ"] = X["IsHighEdu"] * X["IsHardQ"]

    # Standardize continuous features
    scaler = StandardScaler()
    X_scaled = X.copy()
    for col in ["ConfidenceNorm", "ClickCount"]:
        X_scaled[col] = scaler.fit_transform(X_scaled[[col]])

    # Fit logistic regression
    model = LogisticRegression(max_iter=1000, random_state=42, penalty="l2", C=1.0)
    model.fit(X_scaled, y)

    coefs = pd.DataFrame({
        "Feature": X.columns,
        "Coefficient": model.coef_[0],
        "Odds_Ratio": np.exp(model.coef_[0]),
        "Abs_Coef": np.abs(model.coef_[0]),
    }).sort_values("Abs_Coef", ascending=False)

    print(f"\n  Model accuracy: {model.score(X_scaled, y):.4f}")
    print(f"\n  {'Feature':25s} | {'Coef':>8s} | {'OR':>8s} | Interpretation")
    for _, r in coefs.iterrows():
        direction = "decreases accuracy" if r["Coefficient"] < 0 else "increases accuracy"
        strength = "STRONG" if r["Abs_Coef"] > 0.2 else "moderate" if r["Abs_Coef"] > 0.1 else "weak"
        active = "ACTIVE" if r["Abs_Coef"] > 0.1 else "passive"
        print(f"  {r['Feature']:25s} | {r['Coefficient']:+.4f} | {r['Odds_Ratio']:.4f} | "
              f"{direction} [{strength}] -> {active}")

    coefs.to_csv(os.path.join(RESULTS_DIR, "A5_regression.csv"), index=False)

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#E53935" if c < 0 else "#43A047" for c in coefs["Coefficient"]]
    bars = ax.barh(range(len(coefs)), coefs["Coefficient"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(coefs)))
    ax.set_yticklabels(coefs["Feature"])
    ax.set_xlabel("Logistic Regression Coefficient")
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(-0.1, color="orange", linewidth=0.8, linestyle=":", label="Active threshold (±0.1)")
    ax.axvline(0.1, color="orange", linewidth=0.8, linestyle=":")

    for bar, (_, r) in zip(bars, coefs.iterrows()):
        active = "ACTIVE" if r["Abs_Coef"] > 0.1 else "passive"
        ax.text(r["Coefficient"] + 0.01 * np.sign(r["Coefficient"]),
                bar.get_y() + bar.get_height() / 2,
                f"OR={r['Odds_Ratio']:.2f} [{active}]", va="center", fontsize=8)

    ax.set_title("A5 — Logistic Regression: Feature Contributions to Accuracy\n"
                 "Red = decreases accuracy, Green = increases accuracy",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "A5_regression.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  -> A5_regression.csv + .png")
    return coefs


# ===========================================================================
# A6: Per-Question xPer-Demographic Interaction
# ===========================================================================

def experiment_A6_question_demographic_interaction(df):
    hdr("A6: Per-Question xPer-Demographic Deep Interaction")
    print("  Identifies which specific (question, demographic) pairs drive bias.")

    results = []
    for gc in GROUP_COLUMNS:
        for q in range(1, 9):
            q_df = df[df["QuestionNum"] == q]
            overall_acc = q_df["Accuracy"].mean()

            for gname, gdf in q_df.groupby(gc):
                g_acc = gdf["Accuracy"].mean()
                n = len(gdf)
                disp = g_acc - overall_acc

                # Fisher exact test for this cell
                other = q_df[q_df[gc] != gname]
                table = [
                    [int(gdf["Accuracy"].sum()), n - int(gdf["Accuracy"].sum())],
                    [int(other["Accuracy"].sum()), len(other) - int(other["Accuracy"].sum())],
                ]
                try:
                    odds, p = fisher_exact(table)
                except ValueError:
                    odds, p = 1.0, 1.0

                classification = "ACTIVE-UNFAIR" if abs(disp) > 0.15 and p < 0.1 else \
                                 "BORDERLINE" if abs(disp) > 0.1 else \
                                 "PASSIVE" if abs(disp) > 0.05 else "FAIR"

                results.append({
                    "Partition": gc, "Group": gname, "QuestionNum": q,
                    "GroupAcc": round(g_acc, 4), "OverallQAcc": round(overall_acc, 4),
                    "Disparity": round(disp, 4), "N": n,
                    "Fisher_OR": round(odds, 4), "Fisher_p": round(p, 4),
                    "Classification": classification,
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "A6_question_interaction.csv"), index=False)

    # Count active/borderline/passive per partition
    print("\n  Classification summary per partition:")
    for gc in GROUP_COLUMNS:
        sub = results_df[results_df["Partition"] == gc]
        counts = sub["Classification"].value_counts()
        print(f"  {gc:25s}: " + ", ".join(f"{k}={v}" for k, v in counts.items()))

    # Print all ACTIVE-UNFAIR cells
    active = results_df[results_df["Classification"] == "ACTIVE-UNFAIR"]
    if len(active):
        print(f"\n  ACTIVE-UNFAIR cells ({len(active)}):")
        for _, r in active.iterrows():
            print(f"    Q{r['QuestionNum']} x{r['Group']:20s} ({r['Partition']:20s}): "
                  f"Acc={r['GroupAcc']:.3f} (d={r['Disparity']:+.3f}, OR={r['Fisher_OR']:.2f}, p={r['Fisher_p']:.4f})")

    # Visualization: combined heatmap
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    for i, gc in enumerate(GROUP_COLUMNS):
        sub = results_df[results_df["Partition"] == gc]
        pivot = sub.pivot(index="QuestionNum", columns="Group", values="Disparity")

        # Annotate with classification
        annot_df = sub.pivot(index="QuestionNum", columns="Group", values="Classification")
        annot_text = pd.DataFrame("", index=pivot.index, columns=pivot.columns)
        for q in annot_text.index:
            for g in annot_text.columns:
                d = pivot.loc[q, g]
                cls = annot_df.loc[q, g] if pd.notna(annot_df.loc[q, g]) else ""
                marker = "***" if cls == "ACTIVE-UNFAIR" else "**" if cls == "BORDERLINE" else ""
                annot_text.loc[q, g] = f"{d:+.2f}{marker}"

        sns.heatmap(pivot, ax=axes[i], annot=annot_text, fmt="", cmap="RdYlGn",
                    center=0, vmin=-0.4, vmax=0.4, linewidths=0.5)
        axes[i].set_title(f"{gc}\n(*** = ACTIVE-UNFAIR, ** = BORDERLINE)", fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Group")
        axes[i].set_ylabel("Question")

    plt.suptitle("A6 — Question xDemographic Disparity Heatmap",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "A6_question_interaction.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  -> A6_question_interaction.csv + .png")
    return results_df


# ===========================================================================
# A7: Active vs Passive Classification Summary
# ===========================================================================

def experiment_A7_summary(df, a1_df, a2_df, a3_df, a4_df, a5_df, a6_df):
    hdr("A7: Active vs Passive Bias Source Classification")

    classifications = []

    # For each demographic partition, aggregate evidence
    for gc in GROUP_COLUMNS:
        evidence = {
            "Partition": gc,
            "Active_Evidence": [],
            "Passive_Evidence": [],
        }

        # A1: Intersectional unfairness count
        gc_intersections = a1_df[a1_df["Intersection"].str.contains(gc)]
        n_unfair_intersect = gc_intersections["N_Unfair"].sum()
        if n_unfair_intersect > 2:
            evidence["Active_Evidence"].append(f"A1: {n_unfair_intersect} unfair intersections")
        else:
            evidence["Passive_Evidence"].append(f"A1: only {n_unfair_intersect} unfair intersections")

        # A2: High-influence annotators
        col = f"{gc}_AP_shift"
        if col in a2_df.columns:
            high_influence = (a2_df[col].abs() > 0.005).sum()
            if high_influence >= 3:
                evidence["Active_Evidence"].append(f"A2: {high_influence} high-influence annotators")
            else:
                evidence["Passive_Evidence"].append(f"A2: only {high_influence} high-influence annotators")

        # A3: Ablation effect
        ablation_involving = a3_df[
            (a3_df["Balanced_Partition"] == gc) | (a3_df["Measured_Partition"] == gc)
        ]
        n_active_ablation = (ablation_involving["Interpretation"].str.contains("ACTIVE")).sum()
        if n_active_ablation > 0:
            evidence["Active_Evidence"].append(f"A3: {n_active_ablation} active ablation effects")
        else:
            evidence["Passive_Evidence"].append("A3: no ablation effects")

        # A4: Chi-square significance
        chi2_row = a4_df[(a4_df["Test"] == "Chi2_vs_Accuracy") & (a4_df["Variable"] == gc)]
        if len(chi2_row) and chi2_row.iloc[0]["Significant"]:
            evidence["Active_Evidence"].append(f"A4: Chi2 significant (p={chi2_row.iloc[0]['p_value']:.4f})")
        else:
            p_val = chi2_row.iloc[0]["p_value"] if len(chi2_row) else "N/A"
            evidence["Passive_Evidence"].append(f"A4: Chi2 not significant (p={p_val})")

        # A5: Regression coefficient
        demo_map = {
            "LinguisticGroup": "IsNonNative",
            "ExpertiseGroup": "IsSTEM",
            "ExperienceGroup": "IsHighEdu",
            "AgeGroup": "IsOlder35",
        }
        feat = demo_map.get(gc)
        if feat:
            reg_row = a5_df[a5_df["Feature"] == feat]
            if len(reg_row) and reg_row.iloc[0]["Abs_Coef"] > 0.1:
                evidence["Active_Evidence"].append(
                    f"A5: regression coef={reg_row.iloc[0]['Coefficient']:+.3f} (STRONG)")
            else:
                coef = reg_row.iloc[0]["Coefficient"] if len(reg_row) else 0
                evidence["Passive_Evidence"].append(f"A5: regression coef={coef:+.3f} (weak)")

        # A6: Active-unfair question cells
        gc_cells = a6_df[a6_df["Partition"] == gc]
        n_active_cells = (gc_cells["Classification"] == "ACTIVE-UNFAIR").sum()
        n_borderline = (gc_cells["Classification"] == "BORDERLINE").sum()
        if n_active_cells >= 2:
            evidence["Active_Evidence"].append(f"A6: {n_active_cells} ACTIVE-UNFAIR + {n_borderline} BORDERLINE cells")
        elif n_active_cells >= 1 or n_borderline >= 2:
            evidence["Active_Evidence"].append(f"A6: {n_active_cells} ACTIVE + {n_borderline} BORDERLINE (moderate)")
        else:
            evidence["Passive_Evidence"].append(f"A6: {n_active_cells} active, {n_borderline} borderline cells")

        n_active = len(evidence["Active_Evidence"])
        n_passive = len(evidence["Passive_Evidence"])
        total = n_active + n_passive

        if n_active >= 4:
            classification = "STRONGLY ACTIVE"
        elif n_active >= 3:
            classification = "ACTIVE"
        elif n_active >= 2:
            classification = "MODERATELY ACTIVE"
        elif n_active >= 1:
            classification = "WEAKLY ACTIVE / BORDERLINE"
        else:
            classification = "PASSIVE"

        evidence["Classification"] = classification
        evidence["Active_Score"] = f"{n_active}/{total}"
        classifications.append(evidence)

    # Print summary
    print("\n  " + "=" * 80)
    print(f"  {'Partition':25s} | {'Classification':25s} | {'Score':>6s} | Evidence")
    print("  " + "-" * 80)
    for c in classifications:
        print(f"  {c['Partition']:25s} | {c['Classification']:25s} | {c['Active_Score']:>6s} |")
        for e in c["Active_Evidence"]:
            print(f"  {'':25s} | {'':25s} | {'':>6s} | [+] {e}")
        for e in c["Passive_Evidence"]:
            print(f"  {'':25s} | {'':25s} | {'':>6s} | [-] {e}")

    # Save summary
    summary_rows = []
    for c in classifications:
        summary_rows.append({
            "Partition": c["Partition"],
            "Classification": c["Classification"],
            "Active_Score": c["Active_Score"],
            "Active_Evidence": " | ".join(c["Active_Evidence"]),
            "Passive_Evidence": " | ".join(c["Passive_Evidence"]),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(RESULTS_DIR, "A7_classification_summary.csv"), index=False)

    # Visualization: radar/bar chart of evidence
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))

    # Left: evidence score bars
    ax = axes[0]
    partitions = [c["Partition"] for c in classifications]
    active_counts = [len(c["Active_Evidence"]) for c in classifications]
    passive_counts = [len(c["Passive_Evidence"]) for c in classifications]
    x = np.arange(len(partitions))
    w = 0.35
    ax.barh(x - w/2, active_counts, w, label="Active Evidence", color="#E53935", alpha=0.8)
    ax.barh(x + w/2, passive_counts, w, label="Passive Evidence", color="#90A4AE", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(partitions)
    ax.set_xlabel("Number of Evidence Lines")
    ax.set_title("Evidence Count: Active vs Passive", fontsize=11, fontweight="bold")
    ax.legend()
    for i, c in enumerate(classifications):
        ax.text(max(active_counts[i], passive_counts[i]) + 0.3, i,
                c["Classification"], va="center", fontsize=9, fontweight="bold",
                color="#E53935" if "ACTIVE" in c["Classification"] else "#666")

    # Right: combined disparity landscape
    ax2 = axes[1]
    overall = compute_all_metrics(df["SurveyAnswer"], df["ActualAnswer"])
    bar_data = []
    for gc in GROUP_COLUMNS:
        for gname, gdf in df.groupby(gc):
            m = compute_all_metrics(gdf["SurveyAnswer"], gdf["ActualAnswer"])
            bar_data.append({
                "Group": f"{gc[:12]}:{gname}",
                "AP_d": m["AP"] - overall["AP"],
                "TPR_d": m["TPR"] - overall["TPR"],
            })
    bar_df = pd.DataFrame(bar_data)
    x2 = np.arange(len(bar_df))
    w2 = 0.35
    ax2.bar(x2 - w2/2, bar_df["AP_d"], w2, label="AP Disparity", color="#1E88E5", alpha=0.8)
    ax2.bar(x2 + w2/2, bar_df["TPR_d"], w2, label="TPR Disparity", color="#FF8F00", alpha=0.8)
    ax2.set_xticks(x2)
    ax2.set_xticklabels(bar_df["Group"], rotation=45, ha="right", fontsize=8)
    ax2.axhline(-0.1, color="red", linestyle=":", linewidth=0.8, label="Unfairness threshold")
    ax2.axhline(0.1, color="red", linestyle=":", linewidth=0.8)
    ax2.axhline(0, color="black", linewidth=0.5)
    ax2.set_ylabel("Disparity")
    ax2.set_title("All Group Disparities (AP & TPR)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=8)

    plt.suptitle("A7 — Active vs Passive Bias Source Classification",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "A7_classification_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  -> A7_classification_summary.csv + .png")
    return summary_df


# ===========================================================================
# Main
# ===========================================================================

def main():
    hdr("FairPrep Phase 2: Active vs Passive Bias Investigation")
    print(f"  Excel path : {os.path.abspath(EXCEL_PATH)}")
    print(f"  Results dir: {os.path.abspath(RESULTS_DIR)}")

    print("\nLoading data...")
    df = load_excel_data(EXCEL_PATH)
    print(f"  {len(df)} rows | {df['ResponseId'].nunique()} annotators | "
          f"{df['QuestionNum'].nunique()} questions")

    a1 = experiment_A1_intersectional(df)
    a2 = experiment_A2_leave_one_out(df)
    a3 = experiment_A3_ablation(df)
    a4 = experiment_A4_statistical_association(df)
    a5 = experiment_A5_regression(df)
    a6 = experiment_A6_question_demographic_interaction(df)
    a7 = experiment_A7_summary(df, a1, a2, a3, a4, a5, a6)

    print(f"\n{SEP}")
    print(f"  All Phase 2 results saved to: {os.path.abspath(RESULTS_DIR)}")
    print(SEP)


if __name__ == "__main__":
    main()
