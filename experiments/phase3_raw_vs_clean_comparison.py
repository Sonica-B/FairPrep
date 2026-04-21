"""
Phase 3 Raw vs Clean Comparison
================================
Runs the Phase 3 performance-based difficulty analysis on BOTH raw
(480 rows, 60 annotators) and cleaned (427 rows, 56 annotators) data
side-by-side.  Produces a comprehensive comparison to validate whether
the Non-Native Hard-question disparity is robust to cleaning choices.

Motivated by Professor's request at the April 7 meeting.

Analyses performed on each dataset:
  R1: Overall accuracy by difficulty level for each demographic group
  R2: Non-Native vs Native accuracy on Hard questions
  R3: FairEM 10-metric disparity for Non-Native on Hard questions
  R4: Intersection: Non-Native x (Expertise, Experience, Age) on Hard Qs
  R5: Cross-version replication of Non-Native Hard-Q disparity
  R6: Logistic regression with NonNative x HardQ interaction

Usage:
    cd FairPrep
    python experiments/phase3_raw_vs_clean_comparison.py
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
from scipy.stats import fisher_exact
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.excel_data_loader import load_excel_data
from src.data_cleaning import clean_data
from src.measures import AP, TPR, FPR, FNR, TNR, PPV, NPV, FDR, FOR, SP

# ---------------------------------------------------------------------------
# Paths — resolve data files by walking up from the script's directory
# until we find the FairPrep project root (containing Export_and_Compiled.xlsx).
# This makes the script work both from the main repo and from worktrees.
# ---------------------------------------------------------------------------

def _find_project_root():
    """Walk upward from the script dir to find the folder containing the xlsx."""
    d = os.path.abspath(os.path.dirname(__file__))
    for _ in range(10):
        d = os.path.dirname(d)
        if os.path.isfile(os.path.join(d, "Export_and_Compiled.xlsx")):
            return d
    # Fallback: assume standard layout (experiments/ is one level below FairPrep/)
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


_PROJECT_ROOT = _find_project_root()
EXCEL_PATH = os.path.join(_PROJECT_ROOT, "Export_and_Compiled.xlsx")
NINA_DIFF_PATH = os.path.join(
    _PROJECT_ROOT, "Fairness", "Fairness",
    "question_difficulty_summary_second_survey.csv",
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "raw_vs_clean_comparison")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
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


def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def confusion_counts(preds, labels):
    """Return (TP, FP, TN, FN) from binary prediction/label arrays."""
    p, l = np.asarray(preds), np.asarray(labels)
    TP = int(np.sum((p == 1) & (l == 1)))
    FP = int(np.sum((p == 1) & (l == 0)))
    TN = int(np.sum((p == 0) & (l == 0)))
    FN = int(np.sum((p == 0) & (l == 1)))
    return TP, FP, TN, FN


def compute_all_metrics(preds, labels):
    """Compute all 10 FairEM metrics from prediction/label arrays."""
    TP, FP, TN, FN = confusion_counts(preds, labels)
    return {name: fn(TP, FP, TN, FN) for name, fn, _ in ALL_METRICS}


def is_unfair(disparity, direction, threshold=FAIRNESS_THRESHOLD):
    """Return True when disparity crosses the fairness threshold."""
    if direction == "higher":
        return disparity < -threshold
    else:
        return disparity > threshold


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _merge_difficulty(df):
    """Merge Nina's performance-based difficulty onto a DataFrame."""
    if os.path.exists(NINA_DIFF_PATH):
        nina = pd.read_csv(NINA_DIFF_PATH)
        nina = nina[["SurveyVersion", "QuestionNum", "accuracy",
                      "difficulty_disagreement", "difficulty_performance"]]
        nina = nina.rename(columns={
            "accuracy": "VersionQuestionAccuracy",
            "difficulty_disagreement": "DiffDisagreement",
            "difficulty_performance": "DiffPerformance",
        })
        df = df.merge(nina, on=["SurveyVersion", "QuestionNum"], how="left")
    else:
        # Hardcoded fallback from phase3_performance_difficulty.py comments:
        # Difficulty thresholds: Hard <53.3%, Medium <70.2%, Easy >=70.2%
        vq_acc = df.groupby(["SurveyVersion", "QuestionNum"])["Accuracy"].mean()
        vq_acc = vq_acc.reset_index().rename(columns={"Accuracy": "VersionQuestionAccuracy"})

        def _classify(acc):
            if acc < 0.533:
                return "Hard"
            elif acc < 0.702:
                return "Medium"
            else:
                return "Easy"

        vq_acc["DiffPerformance"] = vq_acc["VersionQuestionAccuracy"].apply(_classify)
        vq_acc["DiffDisagreement"] = vq_acc["DiffPerformance"]  # approximate
        df = df.merge(vq_acc, on=["SurveyVersion", "QuestionNum"], how="left")

    df["IsHard_Performance"] = (df["DiffPerformance"] == "Hard").astype(int)
    df["IsHard_Old"] = df["QuestionNum"].isin([2, 5, 6]).astype(int)
    return df


def load_raw():
    """Load raw data (no cleaning) with difficulty labels."""
    df = load_excel_data(EXCEL_PATH)
    # Ensure ConfidenceLevelNorm exists (loader already sets it, belt-and-suspenders)
    if "ConfidenceLevelNorm" not in df.columns:
        df["ConfidenceLevelNorm"] = df["ConfidenceLevel"] / 100.0
    df = _merge_difficulty(df)
    return df


def load_cleaned():
    """Load cleaned data with difficulty labels."""
    df = load_excel_data(EXCEL_PATH)
    df_clean, _report = clean_data(df, verbose=False)
    # Ensure ConfidenceLevelNorm exists after cleaning
    if "ConfidenceLevelNorm" not in df_clean.columns:
        df_clean["ConfidenceLevelNorm"] = df_clean["ConfidenceLevel"] / 100.0
    df_clean = _merge_difficulty(df_clean)
    return df_clean


# ===================================================================
# R1: Overall accuracy by difficulty level for each demographic group
# ===================================================================

def run_r1(df, label):
    """Return a DataFrame of accuracy by (Tier, Partition, Group)."""
    rows = []
    for tier in ["Hard", "Medium", "Easy"]:
        tier_df = df[df["DiffPerformance"] == tier]
        if len(tier_df) == 0:
            continue
        overall_acc = tier_df["Accuracy"].mean()

        for gc in GROUP_COLUMNS:
            for gname, gdf in tier_df.groupby(gc):
                if len(gdf) == 0:
                    continue
                acc = gdf["Accuracy"].mean()
                acc_d = acc - overall_acc
                rows.append({
                    "Dataset": label,
                    "Tier": tier,
                    "Partition": gc,
                    "Group": gname,
                    "N": len(gdf),
                    "Accuracy": round(acc, 4),
                    "Overall_Accuracy": round(overall_acc, 4),
                    "Acc_Disparity": round(acc_d, 4),
                    "UNFAIR": abs(acc_d) > FAIRNESS_THRESHOLD,
                })
    return pd.DataFrame(rows)


# ===================================================================
# R2: Non-Native vs Native accuracy on Hard questions
# ===================================================================

def run_r2(df, label):
    """Return key Non-Native vs Native numbers on Hard questions."""
    hard = df[df["DiffPerformance"] == "Hard"]
    nn = hard[hard["LinguisticGroup"] == "Non-Native"]
    nat = hard[hard["LinguisticGroup"] == "Native"]

    nn_acc = nn["Accuracy"].mean() if len(nn) > 0 else np.nan
    nat_acc = nat["Accuracy"].mean() if len(nat) > 0 else np.nan
    overall_hard_acc = hard["Accuracy"].mean() if len(hard) > 0 else np.nan
    gap = nn_acc - nat_acc if not (np.isnan(nn_acc) or np.isnan(nat_acc)) else np.nan

    # Fisher exact test
    fisher_p = np.nan
    odds_ratio = np.nan
    if len(nn) > 0 and len(nat) > 0:
        table = [
            [int(nat["Accuracy"].sum()), len(nat) - int(nat["Accuracy"].sum())],
            [int(nn["Accuracy"].sum()), len(nn) - int(nn["Accuracy"].sum())],
        ]
        try:
            odds_ratio, fisher_p = fisher_exact(table)
        except ValueError:
            pass

    return pd.DataFrame([{
        "Dataset": label,
        "N_Hard": len(hard),
        "N_NonNative_Hard": len(nn),
        "N_Native_Hard": len(nat),
        "N_Annotators_NN": nn["ResponseId"].nunique() if len(nn) > 0 else 0,
        "N_Annotators_Nat": nat["ResponseId"].nunique() if len(nat) > 0 else 0,
        "Overall_Hard_Acc": round(overall_hard_acc, 4) if not np.isnan(overall_hard_acc) else None,
        "NonNative_Hard_Acc": round(nn_acc, 4) if not np.isnan(nn_acc) else None,
        "Native_Hard_Acc": round(nat_acc, 4) if not np.isnan(nat_acc) else None,
        "Gap_NN_minus_Nat": round(gap, 4) if not np.isnan(gap) else None,
        "Disparity_NN_vs_Overall": round(nn_acc - overall_hard_acc, 4) if not (np.isnan(nn_acc) or np.isnan(overall_hard_acc)) else None,
        "Fisher_p": round(fisher_p, 4) if not np.isnan(fisher_p) else None,
        "Odds_Ratio": round(odds_ratio, 4) if not np.isnan(odds_ratio) else None,
        "UNFAIR_flag": abs(nn_acc - overall_hard_acc) > FAIRNESS_THRESHOLD if not (np.isnan(nn_acc) or np.isnan(overall_hard_acc)) else False,
    }])


# ===================================================================
# R3: FairEM 10-metric disparity for Non-Native on Hard questions
# ===================================================================

def run_r3(df, label):
    """Return all 10 FairEM metric disparities for Non-Native on Hard Qs."""
    hard = df[df["DiffPerformance"] == "Hard"]
    nn_hard = hard[hard["LinguisticGroup"] == "Non-Native"]
    if len(nn_hard) == 0 or len(hard) == 0:
        return pd.DataFrame()

    overall_metrics = compute_all_metrics(hard["SurveyAnswer"], hard["ActualAnswer"])
    nn_metrics = compute_all_metrics(nn_hard["SurveyAnswer"], nn_hard["ActualAnswer"])

    rows = []
    for name, _fn, direction in ALL_METRICS:
        disp = nn_metrics[name] - overall_metrics[name]
        rows.append({
            "Dataset": label,
            "Metric": name,
            "Direction": direction,
            "Overall_Hard": round(overall_metrics[name], 4),
            "NonNative_Hard": round(nn_metrics[name], 4),
            "Disparity": round(disp, 4),
            "UNFAIR": is_unfair(disp, direction),
        })
    return pd.DataFrame(rows)


# ===================================================================
# R4: Intersection analysis -- Non-Native x other partitions on Hard
# ===================================================================

def run_r4(df, label):
    """Non-Native intersected with Expertise, Experience, Age on Hard Qs."""
    hard = df[df["DiffPerformance"] == "Hard"]
    nn_hard = hard[hard["LinguisticGroup"] == "Non-Native"]
    nat_hard = hard[hard["LinguisticGroup"] == "Native"]

    cross_partitions = ["ExpertiseGroup", "ExperienceGroup", "AgeGroup"]
    rows = []

    for gc in cross_partitions:
        for gname in sorted(df[gc].unique()):
            nn_sub = nn_hard[nn_hard[gc] == gname]
            nat_sub = nat_hard[nat_hard[gc] == gname]
            if len(nn_sub) == 0:
                continue

            nn_acc = nn_sub["Accuracy"].mean()
            nat_acc = nat_sub["Accuracy"].mean() if len(nat_sub) > 0 else np.nan
            gap = nn_acc - nat_acc if not np.isnan(nat_acc) else np.nan

            # Fisher exact for this subgroup
            fisher_p = np.nan
            if len(nn_sub) > 0 and len(nat_sub) > 0:
                table = [
                    [int(nat_sub["Accuracy"].sum()), len(nat_sub) - int(nat_sub["Accuracy"].sum())],
                    [int(nn_sub["Accuracy"].sum()), len(nn_sub) - int(nn_sub["Accuracy"].sum())],
                ]
                try:
                    _, fisher_p = fisher_exact(table)
                except ValueError:
                    pass

            rows.append({
                "Dataset": label,
                "CrossPartition": gc,
                "CrossGroup": gname,
                "NN_N": len(nn_sub),
                "NN_Acc": round(nn_acc, 4),
                "Nat_N": len(nat_sub),
                "Nat_Acc": round(nat_acc, 4) if not np.isnan(nat_acc) else None,
                "Gap_NN_minus_Nat": round(gap, 4) if not np.isnan(gap) else None,
                "Fisher_p": round(fisher_p, 4) if not np.isnan(fisher_p) else None,
                "UNFAIR_gap": abs(gap) > FAIRNESS_THRESHOLD if not np.isnan(gap) else False,
            })
    return pd.DataFrame(rows)


# ===================================================================
# R5: Cross-version replication of Non-Native Hard-Q disparity
# ===================================================================

def run_r5(df, label):
    """Per-version Non-Native disparity on Hard questions."""
    rows = []
    for v in sorted(df["SurveyVersion"].unique()):
        vdf = df[df["SurveyVersion"] == v]
        v_hard = vdf[vdf["DiffPerformance"] == "Hard"]
        if len(v_hard) < 5:
            continue
        nn_hard = v_hard[v_hard["LinguisticGroup"] == "Non-Native"]
        nat_hard = v_hard[v_hard["LinguisticGroup"] == "Native"]
        if len(nn_hard) == 0:
            continue

        v_acc = v_hard["Accuracy"].mean()
        nn_acc = nn_hard["Accuracy"].mean()
        nat_acc = nat_hard["Accuracy"].mean() if len(nat_hard) > 0 else np.nan
        disp = nn_acc - v_acc

        # FairEM metrics for this version
        v_overall = compute_all_metrics(v_hard["SurveyAnswer"], v_hard["ActualAnswer"])
        nn_mets = compute_all_metrics(nn_hard["SurveyAnswer"], nn_hard["ActualAnswer"])
        ap_d = nn_mets["AP"] - v_overall["AP"]
        tpr_d = nn_mets["TPR"] - v_overall["TPR"]

        rows.append({
            "Dataset": label,
            "Version": v,
            "N_Hard": len(v_hard),
            "N_NN_Hard": len(nn_hard),
            "N_Nat_Hard": len(nat_hard),
            "Overall_Hard_Acc": round(v_acc, 4),
            "NN_Hard_Acc": round(nn_acc, 4),
            "Nat_Hard_Acc": round(nat_acc, 4) if not np.isnan(nat_acc) else None,
            "Acc_Disparity_NN": round(disp, 4),
            "AP_Disparity": round(ap_d, 4),
            "TPR_Disparity": round(tpr_d, 4),
            "UNFAIR_Acc": abs(disp) > FAIRNESS_THRESHOLD,
            "UNFAIR_TPR": abs(tpr_d) > FAIRNESS_THRESHOLD,
        })
    return pd.DataFrame(rows)


# ===================================================================
# R6: Logistic regression with NonNative x HardQ interaction
# ===================================================================

def run_r6(df, label):
    """Logistic regression with interaction terms; return coefficients."""
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

    # Interaction terms
    X["NonNative_x_Hard"] = X["IsNonNative"] * X["IsHard_Perf"]
    X["NonNative_x_Medium"] = X["IsNonNative"] * X["IsMedium"]
    X["HighEdu_x_Hard"] = X["IsHighEdu"] * X["IsHard_Perf"]

    scaler = StandardScaler()
    X_scaled = X.copy()
    for col in ["ConfidenceNorm", "ClickCount"]:
        X_scaled[col] = scaler.fit_transform(X_scaled[[col]])

    model = LogisticRegression(max_iter=1000, random_state=42, penalty="l2", C=1.0)
    model.fit(X_scaled, y)

    coefs = pd.DataFrame({
        "Dataset": label,
        "Feature": X.columns,
        "Coefficient": model.coef_[0],
        "Odds_Ratio": np.exp(model.coef_[0]),
        "Abs_Coef": np.abs(model.coef_[0]),
    }).sort_values("Abs_Coef", ascending=False)

    coefs["Model_Accuracy"] = model.score(X_scaled, y)
    return coefs


# ===================================================================
# Visualisation
# ===================================================================

def make_visualisation(r1_combined, r2_combined, r3_combined, r6_combined):
    """Create a 2x2 comparison figure saved as raw_vs_clean_comparison.png."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # --- Panel A: R1 accuracy by difficulty tier for LinguisticGroup ---
    ax = axes[0, 0]
    sub = r1_combined[r1_combined["Partition"] == "LinguisticGroup"]
    tiers = ["Hard", "Medium", "Easy"]
    datasets = ["Raw", "Cleaned"]
    groups = sorted(sub["Group"].unique())
    x = np.arange(len(tiers))
    total_bars = len(datasets) * len(groups)
    bar_w = 0.8 / total_bars
    colors = {"Raw": {"Native": "#1E88E5", "Non-Native": "#FF8F00"},
              "Cleaned": {"Native": "#90CAF9", "Non-Native": "#FFE082"}}
    idx = 0
    for ds in datasets:
        for g in groups:
            vals = []
            for t in tiers:
                row = sub[(sub["Dataset"] == ds) & (sub["Group"] == g) & (sub["Tier"] == t)]
                vals.append(row["Accuracy"].values[0] if len(row) > 0 else 0)
            offset = (idx - total_bars / 2 + 0.5) * bar_w
            bars = ax.bar(x + offset, vals, bar_w, label=f"{ds} {g}",
                          color=colors[ds][g], edgecolor="gray", linewidth=0.5)
            for k, v in enumerate(vals):
                ax.text(x[k] + offset, v + 0.01, f"{v:.2f}",
                        ha="center", fontsize=7, rotation=45)
            idx += 1
    ax.set_xticks(x)
    ax.set_xticklabels(tiers)
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("A) LinguisticGroup Accuracy by Difficulty\n(Raw vs Cleaned)",
                  fontsize=10, fontweight="bold")
    ax.legend(fontsize=7, loc="upper left")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5)

    # --- Panel B: R2 Non-Native Hard gap comparison ---
    ax = axes[0, 1]
    for i, ds in enumerate(datasets):
        row = r2_combined[r2_combined["Dataset"] == ds].iloc[0]
        nn_val = row["NonNative_Hard_Acc"] if row["NonNative_Hard_Acc"] is not None else 0
        nat_val = row["Native_Hard_Acc"] if row["Native_Hard_Acc"] is not None else 0
        gap_val = row["Gap_NN_minus_Nat"] if row["Gap_NN_minus_Nat"] is not None else 0
        x_pos = np.array([i * 3, i * 3 + 1])
        bar_colors = ["#1E88E5", "#FF8F00"]
        bars = ax.bar(x_pos, [nat_val, nn_val], 0.8, color=bar_colors, edgecolor="gray")
        ax.text(x_pos[0], nat_val + 0.01, f"{nat_val:.3f}", ha="center", fontsize=9)
        ax.text(x_pos[1], nn_val + 0.01, f"{nn_val:.3f}", ha="center", fontsize=9)
        # Gap annotation
        mid_y = max(nat_val, nn_val) + 0.06
        color = "red" if abs(gap_val) > 0.1 else "black"
        ax.annotate(f"gap={gap_val:+.3f}", xy=(i * 3 + 0.5, mid_y),
                    fontsize=10, fontweight="bold", color=color, ha="center")
    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(["Raw\n(Nat / NN)", "Cleaned\n(Nat / NN)"])
    ax.set_ylabel("Accuracy on Hard Questions")
    ax.set_ylim(0, 1.1)
    ax.set_title("B) Non-Native vs Native Hard-Q Accuracy\n(Raw vs Cleaned)",
                  fontsize=10, fontweight="bold")
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5)

    # --- Panel C: R3 FairEM disparities side-by-side ---
    ax = axes[1, 0]
    metrics_order = [name for name, _, _ in ALL_METRICS]
    y_pos = np.arange(len(metrics_order))
    w = 0.35
    for i, ds in enumerate(datasets):
        ds_df = r3_combined[r3_combined["Dataset"] == ds]
        disp_vals = []
        unfair_flags = []
        for m in metrics_order:
            row = ds_df[ds_df["Metric"] == m]
            if len(row) > 0:
                disp_vals.append(row["Disparity"].values[0])
                unfair_flags.append(row["UNFAIR"].values[0])
            else:
                disp_vals.append(0)
                unfair_flags.append(False)
        offset = -w / 2 if i == 0 else w / 2
        bar_colors = ["#EF5350" if u else "#90A4AE" for u in unfair_flags]
        ax.barh(y_pos + offset, disp_vals, w,
                color=bar_colors if i == 0 else ["#EF9A9A" if u else "#CFD8DC" for u in unfair_flags],
                edgecolor="gray", linewidth=0.5, label=ds)
        for j, (dv, uf) in enumerate(zip(disp_vals, unfair_flags)):
            marker = " *" if uf else ""
            ax.text(dv + 0.005 * np.sign(dv) if dv != 0 else 0.005,
                    y_pos[j] + offset,
                    f"{dv:+.3f}{marker}", va="center", fontsize=7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(metrics_order)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.axvline(-FAIRNESS_THRESHOLD, color="red", linewidth=0.8, linestyle=":")
    ax.axvline(FAIRNESS_THRESHOLD, color="red", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Non-Native Disparity (group - overall)")
    ax.set_title("C) FairEM 10-Metric Disparity: Non-Native Hard Qs\n"
                  "(Red = UNFAIR, * = exceeds threshold)",
                  fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    # --- Panel D: R6 Regression coefficient comparison ---
    ax = axes[1, 1]
    # Align features across both datasets
    raw_r6 = r6_combined[r6_combined["Dataset"] == "Raw"].set_index("Feature")
    clean_r6 = r6_combined[r6_combined["Dataset"] == "Cleaned"].set_index("Feature")
    all_features = raw_r6.index.tolist()  # already sorted by abs
    y_pos = np.arange(len(all_features))
    w = 0.35
    raw_vals = [raw_r6.loc[f, "Coefficient"] for f in all_features]
    clean_vals = [clean_r6.loc[f, "Coefficient"] if f in clean_r6.index else 0
                  for f in all_features]
    ax.barh(y_pos - w / 2, raw_vals, w, color="#42A5F5", alpha=0.8, label="Raw")
    ax.barh(y_pos + w / 2, clean_vals, w, color="#FFA726", alpha=0.8, label="Cleaned")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(all_features, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logistic Regression Coefficient")
    ax.set_title("D) Regression Coefficients\n(Raw vs Cleaned)",
                  fontsize=10, fontweight="bold")
    ax.legend(fontsize=8)

    plt.suptitle("Phase 3 Raw vs Clean Comparison\n"
                 "Validating Non-Native Hard-Q Disparity Robustness",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig_path = os.path.join(RESULTS_DIR, "raw_vs_clean_comparison.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {fig_path}")


# ===================================================================
# Master summary builder
# ===================================================================

def build_summary(r1_combined, r2_combined, r3_combined, r4_combined,
                  r5_combined, r6_combined):
    """One-row-per-finding master comparison table."""
    rows = []

    # --- From R1: Non-Native Hard accuracy disparity ---
    for ds in ["Raw", "Cleaned"]:
        sub = r1_combined[(r1_combined["Dataset"] == ds) &
                          (r1_combined["Partition"] == "LinguisticGroup") &
                          (r1_combined["Group"] == "Non-Native") &
                          (r1_combined["Tier"] == "Hard")]
        if len(sub) > 0:
            rows.append({
                "Finding": "R1_NonNative_Hard_Accuracy",
                "Dataset": ds,
                "Value": sub.iloc[0]["Accuracy"],
                "Disparity": sub.iloc[0]["Acc_Disparity"],
                "UNFAIR": sub.iloc[0]["UNFAIR"],
            })

    # --- From R2: Gap and Fisher p ---
    for ds in ["Raw", "Cleaned"]:
        sub = r2_combined[r2_combined["Dataset"] == ds]
        if len(sub) > 0:
            row = sub.iloc[0]
            rows.append({
                "Finding": "R2_NN_Hard_Gap",
                "Dataset": ds,
                "Value": row["Gap_NN_minus_Nat"],
                "Disparity": row["Disparity_NN_vs_Overall"],
                "UNFAIR": row["UNFAIR_flag"],
            })
            rows.append({
                "Finding": "R2_Fisher_p",
                "Dataset": ds,
                "Value": row["Fisher_p"],
                "Disparity": None,
                "UNFAIR": (row["Fisher_p"] is not None and row["Fisher_p"] < 0.05),
            })

    # --- From R3: key metrics (AP, TPR, FPR) ---
    for metric_name in ["AP", "TPR", "FPR"]:
        for ds in ["Raw", "Cleaned"]:
            sub = r3_combined[(r3_combined["Dataset"] == ds) &
                              (r3_combined["Metric"] == metric_name)]
            if len(sub) > 0:
                rows.append({
                    "Finding": f"R3_{metric_name}_Disparity",
                    "Dataset": ds,
                    "Value": sub.iloc[0]["NonNative_Hard"],
                    "Disparity": sub.iloc[0]["Disparity"],
                    "UNFAIR": sub.iloc[0]["UNFAIR"],
                })

    # --- From R5: count of UNFAIR versions ---
    for ds in ["Raw", "Cleaned"]:
        sub = r5_combined[r5_combined["Dataset"] == ds]
        if len(sub) > 0:
            n_unfair = sub["UNFAIR_Acc"].sum()
            n_total = len(sub)
            rows.append({
                "Finding": "R5_Versions_UNFAIR",
                "Dataset": ds,
                "Value": f"{n_unfair}/{n_total}",
                "Disparity": None,
                "UNFAIR": n_unfair > 0,
            })

    # --- From R6: interaction term ---
    for ds in ["Raw", "Cleaned"]:
        sub = r6_combined[(r6_combined["Dataset"] == ds) &
                          (r6_combined["Feature"] == "NonNative_x_Hard")]
        if len(sub) > 0:
            rows.append({
                "Finding": "R6_NonNative_x_Hard_Coef",
                "Dataset": ds,
                "Value": round(sub.iloc[0]["Coefficient"], 4),
                "Disparity": None,
                "UNFAIR": abs(sub.iloc[0]["Coefficient"]) > 0.1,
            })

    return pd.DataFrame(rows)


# ===================================================================
# Main
# ===================================================================

def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    hdr("Phase 3: Raw vs Clean Comparison")
    print(f"  Output directory: {os.path.abspath(RESULTS_DIR)}")

    # ------ Load datasets ------
    hdr("Loading datasets")
    print("  Loading raw data...")
    df_raw = load_raw()
    print(f"  Raw: {len(df_raw)} rows, {df_raw['ResponseId'].nunique()} annotators")
    print(f"       Hard rows: {(df_raw['DiffPerformance'] == 'Hard').sum()}")

    print("  Loading cleaned data...")
    df_clean = load_cleaned()
    print(f"  Cleaned: {len(df_clean)} rows, {df_clean['ResponseId'].nunique()} annotators")
    print(f"           Hard rows: {(df_clean['DiffPerformance'] == 'Hard').sum()}")

    # ------ R1 ------
    hdr("R1: Accuracy by Difficulty Level & Demographic Group")
    r1_raw = run_r1(df_raw, "Raw")
    r1_clean = run_r1(df_clean, "Cleaned")
    r1_combined = pd.concat([r1_raw, r1_clean], ignore_index=True)
    r1_combined.to_csv(os.path.join(RESULTS_DIR, "R1_accuracy_by_difficulty.csv"), index=False)

    # Print key R1 numbers
    for ds in ["Raw", "Cleaned"]:
        nn_hard = r1_combined[(r1_combined["Dataset"] == ds) &
                              (r1_combined["Partition"] == "LinguisticGroup") &
                              (r1_combined["Group"] == "Non-Native") &
                              (r1_combined["Tier"] == "Hard")]
        if len(nn_hard) > 0:
            r = nn_hard.iloc[0]
            flag = " *** UNFAIR ***" if r["UNFAIR"] else ""
            print(f"  [{ds:>7s}] Non-Native Hard: Acc={r['Accuracy']:.4f}, "
                  f"d={r['Acc_Disparity']:+.4f}{flag}")
    print(f"  -> R1_accuracy_by_difficulty.csv ({len(r1_combined)} rows)")

    # ------ R2 ------
    hdr("R2: Non-Native vs Native on Hard Questions")
    r2_raw = run_r2(df_raw, "Raw")
    r2_clean = run_r2(df_clean, "Cleaned")
    r2_combined = pd.concat([r2_raw, r2_clean], ignore_index=True)
    r2_combined.to_csv(os.path.join(RESULTS_DIR, "R2_nonnative_hard_comparison.csv"), index=False)

    for _, row in r2_combined.iterrows():
        ds = row["Dataset"]
        flag = " *** UNFAIR ***" if row["UNFAIR_flag"] else ""
        fisher = f"p={row['Fisher_p']:.4f}" if row["Fisher_p"] is not None else "p=N/A"
        print(f"  [{ds:>7s}] NN={row['NonNative_Hard_Acc']:.4f} vs Nat={row['Native_Hard_Acc']:.4f}, "
              f"gap={row['Gap_NN_minus_Nat']:+.4f}, {fisher}{flag}")
    print(f"  -> R2_nonnative_hard_comparison.csv")

    # ------ R3 ------
    hdr("R3: FairEM 10-Metric Disparity on Hard Qs")
    r3_raw = run_r3(df_raw, "Raw")
    r3_clean = run_r3(df_clean, "Cleaned")
    r3_combined = pd.concat([r3_raw, r3_clean], ignore_index=True)
    r3_combined.to_csv(os.path.join(RESULTS_DIR, "R3_fairem_disparity_comparison.csv"), index=False)

    for ds in ["Raw", "Cleaned"]:
        ds_df = r3_combined[r3_combined["Dataset"] == ds]
        unfair_count = ds_df["UNFAIR"].sum()
        unfair_names = ds_df[ds_df["UNFAIR"]]["Metric"].tolist()
        print(f"  [{ds:>7s}] UNFAIR on {unfair_count}/10 metrics: {unfair_names or 'none'}")
    print(f"  -> R3_fairem_disparity_comparison.csv")

    # ------ R4 ------
    hdr("R4: Non-Native Intersection Analysis on Hard Qs")
    r4_raw = run_r4(df_raw, "Raw")
    r4_clean = run_r4(df_clean, "Cleaned")
    r4_combined = pd.concat([r4_raw, r4_clean], ignore_index=True)
    r4_combined.to_csv(os.path.join(RESULTS_DIR, "R4_intersection_comparison.csv"), index=False)

    for ds in ["Raw", "Cleaned"]:
        ds_df = r4_combined[r4_combined["Dataset"] == ds]
        unfair_rows = ds_df[ds_df["UNFAIR_gap"]]
        print(f"  [{ds:>7s}] Intersection subgroups with |gap| > 0.1:")
        if len(unfair_rows) == 0:
            print(f"           None")
        for _, r in unfair_rows.iterrows():
            fp = f"Fisher p={r['Fisher_p']:.4f}" if r["Fisher_p"] is not None else ""
            print(f"           {r['CrossPartition']} / {r['CrossGroup']}: "
                  f"NN={r['NN_Acc']:.3f} vs Nat={r['Nat_Acc']}, "
                  f"gap={r['Gap_NN_minus_Nat']:+.3f} {fp}")
    print(f"  -> R4_intersection_comparison.csv")

    # ------ R5 ------
    hdr("R5: Cross-Version Replication")
    r5_raw = run_r5(df_raw, "Raw")
    r5_clean = run_r5(df_clean, "Cleaned")
    r5_combined = pd.concat([r5_raw, r5_clean], ignore_index=True)
    r5_combined.to_csv(os.path.join(RESULTS_DIR, "R5_cross_version_comparison.csv"), index=False)

    for ds in ["Raw", "Cleaned"]:
        ds_df = r5_combined[r5_combined["Dataset"] == ds]
        for _, r in ds_df.iterrows():
            flag = " UNFAIR" if r["UNFAIR_Acc"] else ""
            print(f"  [{ds:>7s}] V{r['Version']}: NN_Hard_Acc={r['NN_Hard_Acc']:.3f}, "
                  f"d={r['Acc_Disparity_NN']:+.3f}{flag}")
    print(f"  -> R5_cross_version_comparison.csv")

    # ------ R6 ------
    hdr("R6: Logistic Regression Comparison")
    r6_raw = run_r6(df_raw, "Raw")
    r6_clean = run_r6(df_clean, "Cleaned")
    r6_combined = pd.concat([r6_raw, r6_clean], ignore_index=True)
    r6_combined.to_csv(os.path.join(RESULTS_DIR, "R6_regression_comparison.csv"), index=False)

    # Print key features
    for ds in ["Raw", "Cleaned"]:
        ds_df = r6_combined[r6_combined["Dataset"] == ds].head(5)
        model_acc = ds_df["Model_Accuracy"].iloc[0]
        print(f"  [{ds:>7s}] Model accuracy: {model_acc:.4f}")
        for _, r in ds_df.iterrows():
            strength = "STRONG" if r["Abs_Coef"] > 0.2 else "moderate" if r["Abs_Coef"] > 0.1 else "weak"
            print(f"           {r['Feature']:25s}: coef={r['Coefficient']:+.4f}, "
                  f"OR={r['Odds_Ratio']:.3f} [{strength}]")
    print(f"  -> R6_regression_comparison.csv")

    # ------ Summary table ------
    hdr("Master Summary")
    summary = build_summary(r1_combined, r2_combined, r3_combined,
                            r4_combined, r5_combined, r6_combined)
    summary.to_csv(os.path.join(RESULTS_DIR, "summary_raw_vs_clean.csv"), index=False)
    print(f"  -> summary_raw_vs_clean.csv ({len(summary)} rows)")
    print(summary.to_string(index=False))

    # ------ Visualization ------
    hdr("Generating Comparison Visualization")
    make_visualisation(r1_combined, r2_combined, r3_combined, r6_combined)

    # ------ Final text verdict ------
    hdr("ROBUSTNESS VERDICT")

    # Assess robustness across multiple dimensions
    robust_checks = []

    # Check 1: R2 UNFAIR flag consistent?
    raw_unfair = r2_combined[r2_combined["Dataset"] == "Raw"]["UNFAIR_flag"].values[0]
    clean_unfair = r2_combined[r2_combined["Dataset"] == "Cleaned"]["UNFAIR_flag"].values[0]
    same_flag = raw_unfair == clean_unfair
    robust_checks.append(("R2 UNFAIR flag agrees", same_flag))
    print(f"  [{'PASS' if same_flag else 'DIFF'}] R2 UNFAIR flag: Raw={raw_unfair}, Cleaned={clean_unfair}")

    # Check 2: R2 gap direction consistent?
    raw_gap = r2_combined[r2_combined["Dataset"] == "Raw"]["Gap_NN_minus_Nat"].values[0]
    clean_gap = r2_combined[r2_combined["Dataset"] == "Cleaned"]["Gap_NN_minus_Nat"].values[0]
    same_dir = (raw_gap is not None and clean_gap is not None and
                np.sign(raw_gap) == np.sign(clean_gap))
    robust_checks.append(("R2 gap direction agrees", same_dir))
    print(f"  [{'PASS' if same_dir else 'DIFF'}] R2 gap direction: "
          f"Raw={raw_gap:+.4f}, Cleaned={clean_gap:+.4f}")

    # Check 3: R3 number of UNFAIR metrics similar?
    raw_unfair_n = r3_combined[r3_combined["Dataset"] == "Raw"]["UNFAIR"].sum()
    clean_unfair_n = r3_combined[r3_combined["Dataset"] == "Cleaned"]["UNFAIR"].sum()
    similar_metrics = abs(raw_unfair_n - clean_unfair_n) <= 2
    robust_checks.append(("R3 UNFAIR metric count similar", similar_metrics))
    print(f"  [{'PASS' if similar_metrics else 'DIFF'}] R3 UNFAIR metrics: "
          f"Raw={raw_unfair_n}, Cleaned={clean_unfair_n}")

    # Check 4: R6 interaction term sign consistent?
    raw_coef = r6_combined[(r6_combined["Dataset"] == "Raw") &
                           (r6_combined["Feature"] == "NonNative_x_Hard")]
    clean_coef = r6_combined[(r6_combined["Dataset"] == "Cleaned") &
                             (r6_combined["Feature"] == "NonNative_x_Hard")]
    if len(raw_coef) > 0 and len(clean_coef) > 0:
        rc = raw_coef.iloc[0]["Coefficient"]
        cc = clean_coef.iloc[0]["Coefficient"]
        same_sign = np.sign(rc) == np.sign(cc)
        robust_checks.append(("R6 interaction sign agrees", same_sign))
        print(f"  [{'PASS' if same_sign else 'DIFF'}] R6 NonNative_x_Hard: "
              f"Raw={rc:+.4f}, Cleaned={cc:+.4f}")

    # Check 5: R5 replication pattern similar?
    raw_v_unfair = r5_combined[r5_combined["Dataset"] == "Raw"]["UNFAIR_Acc"].sum()
    clean_v_unfair = r5_combined[r5_combined["Dataset"] == "Cleaned"]["UNFAIR_Acc"].sum()
    similar_versions = abs(raw_v_unfair - clean_v_unfair) <= 1
    robust_checks.append(("R5 cross-version pattern similar", similar_versions))
    print(f"  [{'PASS' if similar_versions else 'DIFF'}] R5 versions UNFAIR: "
          f"Raw={raw_v_unfair}, Cleaned={clean_v_unfair}")

    n_pass = sum(1 for _, ok in robust_checks if ok)
    n_total = len(robust_checks)
    print(f"\n  Overall: {n_pass}/{n_total} robustness checks PASS")

    if n_pass >= n_total - 1:
        print("\n  CONCLUSION: The Non-Native Hard-question disparity is ROBUST to cleaning.")
        print("  Cleaning does not materially alter the core fairness finding.")
    elif n_pass >= n_total // 2:
        print("\n  CONCLUSION: PARTIALLY robust. Some findings shift with cleaning.")
        print("  The direction holds but magnitude or significance may differ.")
    else:
        print("\n  CONCLUSION: NOT robust. Cleaning substantially changes the findings.")
        print("  The disparity may be an artifact of noisy/outlier data.")

    print(f"\n{SEP}")
    print(f"  All results saved to: {os.path.abspath(RESULTS_DIR)}")
    print(f"  Files: R1-R6 CSVs, summary_raw_vs_clean.csv, raw_vs_clean_comparison.png")
    print(SEP)


if __name__ == "__main__":
    main()
