"""
Phase 3 Statistical Significance Tests
========================================
Adds comprehensive p-values to all key findings from Phase 3 (performance-
based difficulty) and Phase 3b (Non-Native deep dive).

Requested at the April 7 2026 meeting: newer analyses lack the statistical
tests that older analyses had.

Seven experiments:
  S1  Permutation test for FairEM disparity significance (Non-Native, 10 metrics)
  S2  Fisher's exact test for every (group x difficulty) cell + BH FDR correction
  S3  Bootstrap confidence intervals for key disparities
  S4  Chi-square / Fisher tests for intersection cells on Hard questions
  S5  Logistic regression with proper inference (statsmodels)
  S6  Cross-version consistency test for Non-Native Hard disparity
  S7  Permutation test for calibration gap significance

Outputs saved to results/statistical_tests/.

Usage:
    cd FairPrep
    python experiments/phase3_statistical_tests.py
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import fisher_exact, chi2_contingency
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from src.excel_data_loader import load_excel_data
from src.measures import AP, SP, TPR, FPR, FNR, TNR, PPV, NPV, FDR, FOR

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
EXCEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Export_and_Compiled.xlsx")
NINA_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Fairness", "Fairness",
    "question_difficulty_summary_second_survey.csv",
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "statistical_tests")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
GROUP_COLUMNS = ["LinguisticGroup", "ExpertiseGroup", "ExperienceGroup", "AgeGroup"]
FAIRNESS_THRESHOLD = 0.1
N_PERM = 10_000
N_BOOT = 10_000
SEED = 42
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
# Confusion-matrix helpers
# ---------------------------------------------------------------------------

def confusion_counts(preds, labels):
    p, la = np.asarray(preds), np.asarray(labels)
    TP = int(np.sum((p == 1) & (la == 1)))
    FP_cnt = int(np.sum((p == 1) & (la == 0)))
    TN = int(np.sum((p == 0) & (la == 0)))
    FN = int(np.sum((p == 0) & (la == 1)))
    return TP, FP_cnt, TN, FN


def compute_metric(preds, labels, metric_fn):
    tp, fp, tn, fn = confusion_counts(preds, labels)
    return metric_fn(tp, fp, tn, fn)


# ---------------------------------------------------------------------------
# Data loading & cleaning (same pipeline as phase3 / phase3b)
# ---------------------------------------------------------------------------

def load_and_prepare():
    hdr("Loading & Cleaning Data")
    df = load_excel_data(EXCEL_PATH)
    n_before = len(df)

    # IQR time outliers
    q1 = df["DecisionTime"].quantile(0.25)
    q3 = df["DecisionTime"].quantile(0.75)
    df = df[df["DecisionTime"] <= q3 + 3 * (q3 - q1)]

    # Speed clickers
    df = df[~((df["DecisionTime"] < 3) & (df["ClickCount"] <= 1))]

    # Low-accuracy annotators
    ann_acc = df.groupby("ResponseId")["Accuracy"].mean()
    df = df[~df["ResponseId"].isin(ann_acc[ann_acc <= 0.25].index)]

    # Straightliners
    for rid in list(df["ResponseId"].unique()):
        answers = df[df["ResponseId"] == rid]["SurveyAnswer"].values
        if len(set(answers)) == 1 and len(answers) >= 8:
            df = df[df["ResponseId"] != rid]

    # Confidence capping
    df["ConfidenceLevel"] = df["ConfidenceLevel"].clip(5, 95)
    df["ConfidenceLevelNorm"] = df["ConfidenceLevel"] / 100.0

    # Merge Nina's performance-based difficulty
    nina = pd.read_csv(NINA_PATH)[["SurveyVersion", "QuestionNum",
                                    "accuracy", "difficulty_performance"]]
    nina = nina.rename(columns={"accuracy": "VersionQuestionAccuracy",
                                "difficulty_performance": "DiffPerf"})
    df = df.merge(nina, on=["SurveyVersion", "QuestionNum"], how="left")

    df["IsHard"] = (df["DiffPerf"] == "Hard").astype(int)
    df["IsMedium"] = (df["DiffPerf"] == "Medium").astype(int)
    df["IsNonNative"] = (df["LinguisticGroup"] == "Non-Native").astype(int)

    print(f"  Cleaned: {n_before} -> {len(df)} rows, "
          f"{df['ResponseId'].nunique()} annotators")
    print(f"  Non-Native: {df['IsNonNative'].sum()} rows "
          f"({df[df['IsNonNative'] == 1]['ResponseId'].nunique()} annotators)")
    diff_counts = df["DiffPerf"].value_counts()
    for k, v in diff_counts.items():
        print(f"    {k}: {v} rows ({v / len(df) * 100:.1f}%)")
    return df


# ===================================================================
# S1: Permutation Test for FairEM Disparity (Non-Native, all metrics)
# ===================================================================

def s1_permutation_fairem(df):
    hdr("S1: Permutation Test for FairEM Disparity Significance (Non-Native)")
    rng = np.random.RandomState(SEED)
    results = []

    preds_all = df["SurveyAnswer"].values
    labels_all = df["ActualAnswer"].values
    group_labels = df["LinguisticGroup"].values.copy()
    nn_mask = group_labels == "Non-Native"

    for mname, mfn, direction in ALL_METRICS:
        # Observed disparity
        overall_val = compute_metric(preds_all, labels_all, mfn)
        nn_val = compute_metric(preds_all[nn_mask], labels_all[nn_mask], mfn)
        obs_disp = nn_val - overall_val

        # Permutation null distribution
        null_disps = np.empty(N_PERM)
        for i in range(N_PERM):
            shuffled = rng.permutation(group_labels)
            shuf_mask = shuffled == "Non-Native"
            shuf_nn_val = compute_metric(preds_all[shuf_mask], labels_all[shuf_mask], mfn)
            null_disps[i] = shuf_nn_val - overall_val

        # Two-sided p-value
        p_val = np.mean(np.abs(null_disps) >= np.abs(obs_disp))
        ci_lo = np.percentile(null_disps, 2.5)
        ci_hi = np.percentile(null_disps, 97.5)
        null_mean = np.mean(null_disps)
        null_std = np.std(null_disps)

        is_unfair = (direction == "higher" and obs_disp < -FAIRNESS_THRESHOLD) or \
                    (direction == "lower" and obs_disp > FAIRNESS_THRESHOLD)
        sig = p_val < 0.05

        results.append({
            "Metric": mname, "Direction": direction,
            "Observed_Disparity": round(obs_disp, 5),
            "Null_Mean": round(null_mean, 5), "Null_Std": round(null_std, 5),
            "Null_CI_2.5": round(ci_lo, 5), "Null_CI_97.5": round(ci_hi, 5),
            "P_Value": round(p_val, 5),
            "Significant_0.05": sig,
            "Exceeds_Threshold": is_unfair,
        })
        flag = "***" if sig else ""
        print(f"  {mname:4s}: obs_d={obs_disp:+.4f}, p={p_val:.4f}, "
              f"null_CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] {flag}")

    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(RESULTS_DIR, "S1_permutation_fairem.csv"), index=False)
    print(f"\n  -> S1_permutation_fairem.csv")
    return rdf


# ===================================================================
# S2: Fisher's Exact Test for Each Group x Difficulty Cell
# ===================================================================

def s2_fisher_per_cell(df):
    hdr("S2: Fisher's Exact Test for (Group x Difficulty) Cells + BH FDR")
    results = []

    for gc in GROUP_COLUMNS:
        for gname in sorted(df[gc].unique()):
            for tier in ["Hard", "Medium", "Easy"]:
                tier_df = df[df["DiffPerf"] == tier]
                in_group = tier_df[tier_df[gc] == gname]
                out_group = tier_df[tier_df[gc] != gname]
                if len(in_group) == 0 or len(out_group) == 0:
                    continue

                # 2x2: [group, not-group] x [correct, incorrect]
                a = int(in_group["Accuracy"].sum())
                b = len(in_group) - a
                c = int(out_group["Accuracy"].sum())
                d = len(out_group) - c
                table = [[a, b], [c, d]]

                try:
                    odds_ratio, p_val = fisher_exact(table, alternative="two-sided")
                except ValueError:
                    odds_ratio, p_val = 1.0, 1.0

                # Confidence interval for odds ratio (Woolf's method)
                if a > 0 and b > 0 and c > 0 and d > 0:
                    log_or = np.log(odds_ratio)
                    se_log_or = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
                    or_ci_lo = np.exp(log_or - 1.96 * se_log_or)
                    or_ci_hi = np.exp(log_or + 1.96 * se_log_or)
                else:
                    or_ci_lo, or_ci_hi = np.nan, np.nan

                group_acc = in_group["Accuracy"].mean()
                other_acc = out_group["Accuracy"].mean()

                results.append({
                    "Partition": gc, "Group": gname, "Tier": tier,
                    "N_Group": len(in_group), "N_Other": len(out_group),
                    "Group_Acc": round(group_acc, 4),
                    "Other_Acc": round(other_acc, 4),
                    "Acc_Diff": round(group_acc - other_acc, 4),
                    "Odds_Ratio": round(odds_ratio, 4),
                    "OR_CI_Lo": round(or_ci_lo, 4) if not np.isnan(or_ci_lo) else None,
                    "OR_CI_Hi": round(or_ci_hi, 4) if not np.isnan(or_ci_hi) else None,
                    "P_Value_Raw": round(p_val, 6),
                })

    rdf = pd.DataFrame(results)

    # Benjamini-Hochberg FDR correction
    if len(rdf) > 0:
        reject, pvals_corrected, _, _ = multipletests(
            rdf["P_Value_Raw"], alpha=0.05, method="fdr_bh"
        )
        rdf["P_Value_BH"] = np.round(pvals_corrected, 6)
        rdf["Significant_BH_0.05"] = reject
    else:
        rdf["P_Value_BH"] = []
        rdf["Significant_BH_0.05"] = []

    rdf.to_csv(os.path.join(RESULTS_DIR, "S2_fisher_per_cell.csv"), index=False)

    sig = rdf[rdf["Significant_BH_0.05"] == True]
    print(f"\n  Total cells tested: {len(rdf)}")
    print(f"  Significant after BH correction (alpha=0.05): {len(sig)}")
    if len(sig) > 0:
        print(f"\n  Significant cells:")
        for _, r in sig.iterrows():
            print(f"    {r['Partition']:20s} {r['Group']:15s} [{r['Tier']:6s}]: "
                  f"Acc={r['Group_Acc']:.3f} vs {r['Other_Acc']:.3f}, "
                  f"OR={r['Odds_Ratio']:.3f}, p_raw={r['P_Value_Raw']:.4f}, "
                  f"p_BH={r['P_Value_BH']:.4f}")

    print(f"\n  -> S2_fisher_per_cell.csv")
    return rdf


# ===================================================================
# S3: Bootstrap CIs for Key Disparities
# ===================================================================

def s3_bootstrap_ci(df):
    hdr("S3: Bootstrap Confidence Intervals for Key Disparities")
    rng = np.random.RandomState(SEED)
    results = []

    # Define the key disparities to test
    disparities = []

    # 1. Non-Native TPR disparity (overall)
    def nn_tpr_disp(data):
        nn = data[data["LinguisticGroup"] == "Non-Native"]
        overall_tpr = compute_metric(data["SurveyAnswer"], data["ActualAnswer"], TPR)
        nn_tpr = compute_metric(nn["SurveyAnswer"], nn["ActualAnswer"], TPR)
        return nn_tpr - overall_tpr
    disparities.append(("NonNative_TPR_disparity", nn_tpr_disp, df))

    # 2. Non-Native accuracy on Hard questions
    hard_df = df[df["DiffPerf"] == "Hard"]
    def nn_hard_acc_diff(data):
        nn = data[data["LinguisticGroup"] == "Non-Native"]
        nat = data[data["LinguisticGroup"] == "Native"]
        if len(nn) == 0 or len(nat) == 0:
            return 0.0
        return nn["Accuracy"].mean() - nat["Accuracy"].mean()
    disparities.append(("NonNative_Hard_Acc_diff", nn_hard_acc_diff, hard_df))

    # 3. Non-Native accuracy on Hard vs overall
    def nn_hard_vs_overall(data):
        nn = data[data["LinguisticGroup"] == "Non-Native"]
        if len(nn) == 0:
            return 0.0
        return nn["Accuracy"].mean() - data["Accuracy"].mean()
    disparities.append(("NonNative_Hard_Acc_vs_overall", nn_hard_vs_overall, hard_df))

    # 4. Fluent speakers on Hard vs Native on Hard
    def fluent_hard_diff(data):
        fluent = data[(data["LinguisticGroup"] == "Non-Native") & (data["EngProf"] == 4)]
        native = data[data["LinguisticGroup"] == "Native"]
        if len(fluent) == 0 or len(native) == 0:
            return 0.0
        return fluent["Accuracy"].mean() - native["Accuracy"].mean()
    disparities.append(("Fluent_Hard_Acc_vs_Native", fluent_hard_diff, hard_df))

    # 5. Non-Native AP disparity (overall)
    def nn_ap_disp(data):
        nn = data[data["LinguisticGroup"] == "Non-Native"]
        overall_ap = compute_metric(data["SurveyAnswer"], data["ActualAnswer"], AP)
        nn_ap = compute_metric(nn["SurveyAnswer"], nn["ActualAnswer"], AP)
        return nn_ap - overall_ap
    disparities.append(("NonNative_AP_disparity", nn_ap_disp, df))

    # 6. Non-Native FNR disparity (overall)
    def nn_fnr_disp(data):
        nn = data[data["LinguisticGroup"] == "Non-Native"]
        overall_fnr = compute_metric(data["SurveyAnswer"], data["ActualAnswer"], FNR)
        nn_fnr = compute_metric(nn["SurveyAnswer"], nn["ActualAnswer"], FNR)
        return nn_fnr - overall_fnr
    disparities.append(("NonNative_FNR_disparity", nn_fnr_disp, df))

    # 7. STEM vs Non-STEM Hard accuracy diff
    def stem_hard_diff(data):
        stem = data[data["ExpertiseGroup"] == "STEM"]
        non_stem = data[data["ExpertiseGroup"] == "Non-STEM"]
        if len(stem) == 0 or len(non_stem) == 0:
            return 0.0
        return stem["Accuracy"].mean() - non_stem["Accuracy"].mean()
    disparities.append(("STEM_Hard_Acc_diff", stem_hard_diff, hard_df))

    # 8. High-Edu vs Lower-Edu Hard accuracy diff
    def edu_hard_diff(data):
        hi = data[data["ExperienceGroup"] == "High-Edu"]
        lo = data[data["ExperienceGroup"] == "Lower-Edu"]
        if len(hi) == 0 or len(lo) == 0:
            return 0.0
        return hi["Accuracy"].mean() - lo["Accuracy"].mean()
    disparities.append(("HighEdu_Hard_Acc_diff", edu_hard_diff, hard_df))

    for name, fn, data in disparities:
        obs = fn(data)
        boot_vals = np.empty(N_BOOT)
        n = len(data)
        for i in range(N_BOOT):
            sample = data.iloc[rng.choice(n, size=n, replace=True)]
            boot_vals[i] = fn(sample)
        ci_lo = np.percentile(boot_vals, 2.5)
        ci_hi = np.percentile(boot_vals, 97.5)
        excludes_zero = (ci_lo > 0) or (ci_hi < 0)
        boot_se = np.std(boot_vals)

        results.append({
            "Disparity": name,
            "Observed": round(obs, 5),
            "Boot_SE": round(boot_se, 5),
            "CI_2.5": round(ci_lo, 5),
            "CI_97.5": round(ci_hi, 5),
            "CI_Excludes_Zero": excludes_zero,
        })
        flag = "***" if excludes_zero else ""
        print(f"  {name:35s}: obs={obs:+.4f}, 95%CI=[{ci_lo:+.4f}, {ci_hi:+.4f}] {flag}")

    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(RESULTS_DIR, "S3_bootstrap_ci.csv"), index=False)
    print(f"\n  -> S3_bootstrap_ci.csv")
    return rdf


# ===================================================================
# S4: Intersection Tests on Hard Questions
# ===================================================================

def s4_intersection_tests(df):
    hdr("S4: Intersection Tests (NonNative x Demographics) on Hard Questions")
    hard = df[df["DiffPerf"] == "Hard"]
    nn_hard = hard[hard["LinguisticGroup"] == "Non-Native"]
    nat_hard = hard[hard["LinguisticGroup"] == "Native"]
    results = []

    cross_partitions = ["ExpertiseGroup", "ExperienceGroup", "AgeGroup"]

    # 2-way intersections: Non-Native x each demographic group, on Hard
    for gc in cross_partitions:
        for gname in sorted(hard[gc].unique()):
            nn_sub = nn_hard[nn_hard[gc] == gname]
            nat_sub = nat_hard[nat_hard[gc] == gname]
            if len(nn_sub) == 0:
                continue

            nn_correct = int(nn_sub["Accuracy"].sum())
            nn_wrong = len(nn_sub) - nn_correct
            nat_correct = int(nat_sub["Accuracy"].sum()) if len(nat_sub) > 0 else 0
            nat_wrong = len(nat_sub) - nat_correct if len(nat_sub) > 0 else 0

            table = [[nat_correct, nat_wrong], [nn_correct, nn_wrong]]

            try:
                odds_ratio, p_val = fisher_exact(table, alternative="two-sided")
            except ValueError:
                odds_ratio, p_val = np.nan, 1.0

            # Effect size: Cramer's V from chi-square
            cramers_v = np.nan
            if min(len(nn_sub), len(nat_sub)) > 0:
                full_table = np.array(table)
                if full_table.sum() > 0 and full_table.min() >= 0:
                    try:
                        chi2, _, _, _ = chi2_contingency(full_table,
                                                         correction=False)
                        n_total = full_table.sum()
                        cramers_v = np.sqrt(chi2 / n_total) if n_total > 0 else 0.0
                    except ValueError:
                        cramers_v = np.nan

            nn_acc = nn_sub["Accuracy"].mean()
            nat_acc = nat_sub["Accuracy"].mean() if len(nat_sub) > 0 else np.nan

            results.append({
                "Type": "2-way",
                "Intersection": f"NonNative+{gname}",
                "Cross1": gc, "Value1": gname,
                "Cross2": "", "Value2": "",
                "NN_N": len(nn_sub), "NN_Acc": round(nn_acc, 4),
                "Nat_N": len(nat_sub),
                "Nat_Acc": round(nat_acc, 4) if not np.isnan(nat_acc) else None,
                "Odds_Ratio": round(odds_ratio, 4) if not np.isnan(odds_ratio) else None,
                "Cramers_V": round(cramers_v, 4) if not np.isnan(cramers_v) else None,
                "P_Value_Raw": round(p_val, 6),
            })

    # 3-way intersections: Non-Native x dim1 x dim2, on Hard
    for d1, d2 in combinations(cross_partitions, 2):
        for v1 in sorted(hard[d1].unique()):
            for v2 in sorted(hard[d2].unique()):
                nn_sub = nn_hard[(nn_hard[d1] == v1) & (nn_hard[d2] == v2)]
                nat_sub = nat_hard[(nat_hard[d1] == v1) & (nat_hard[d2] == v2)]
                if len(nn_sub) == 0:
                    continue

                nn_correct = int(nn_sub["Accuracy"].sum())
                nn_wrong = len(nn_sub) - nn_correct
                nat_correct = int(nat_sub["Accuracy"].sum()) if len(nat_sub) > 0 else 0
                nat_wrong = len(nat_sub) - nat_correct if len(nat_sub) > 0 else 0

                table = [[nat_correct, nat_wrong], [nn_correct, nn_wrong]]

                try:
                    odds_ratio, p_val = fisher_exact(table, alternative="two-sided")
                except ValueError:
                    odds_ratio, p_val = np.nan, 1.0

                nn_acc = nn_sub["Accuracy"].mean()
                nat_acc = nat_sub["Accuracy"].mean() if len(nat_sub) > 0 else np.nan

                results.append({
                    "Type": "3-way",
                    "Intersection": f"NonNative+{v1}+{v2}",
                    "Cross1": d1, "Value1": v1,
                    "Cross2": d2, "Value2": v2,
                    "NN_N": len(nn_sub), "NN_Acc": round(nn_acc, 4),
                    "Nat_N": len(nat_sub),
                    "Nat_Acc": round(nat_acc, 4) if not np.isnan(nat_acc) else None,
                    "Odds_Ratio": round(odds_ratio, 4) if not np.isnan(odds_ratio) else None,
                    "Cramers_V": None,
                    "P_Value_Raw": round(p_val, 6),
                })

    rdf = pd.DataFrame(results)

    # BH correction
    if len(rdf) > 0:
        reject, pvals_corrected, _, _ = multipletests(
            rdf["P_Value_Raw"], alpha=0.05, method="fdr_bh"
        )
        rdf["P_Value_BH"] = np.round(pvals_corrected, 6)
        rdf["Significant_BH_0.05"] = reject
    else:
        rdf["P_Value_BH"] = []
        rdf["Significant_BH_0.05"] = []

    rdf.to_csv(os.path.join(RESULTS_DIR, "S4_intersection_tests.csv"), index=False)

    sig = rdf[rdf["Significant_BH_0.05"] == True]
    print(f"\n  Total intersection cells: {len(rdf)}")
    print(f"  Significant after BH (alpha=0.05): {len(sig)}")
    if len(sig) > 0:
        for _, r in sig.iterrows():
            print(f"    {r['Intersection']:35s}: NN_Acc={r['NN_Acc']:.3f}(n={r['NN_N']}) "
                  f"Nat_Acc={r['Nat_Acc']} (n={r['Nat_N']}), "
                  f"OR={r['Odds_Ratio']}, p_BH={r['P_Value_BH']:.4f}")
    else:
        print("  (No cells significant after correction -- expected with small n)")

    # Print borderline (raw p < 0.10)
    borderline = rdf[(rdf["P_Value_Raw"] < 0.10) & (~rdf["Significant_BH_0.05"])]
    if len(borderline) > 0:
        print(f"\n  Borderline (raw p < 0.10, not significant after BH):")
        for _, r in borderline.iterrows():
            print(f"    {r['Intersection']:35s}: p_raw={r['P_Value_Raw']:.4f}, "
                  f"p_BH={r['P_Value_BH']:.4f}, NN_Acc={r['NN_Acc']:.3f}")

    print(f"\n  -> S4_intersection_tests.csv")
    return rdf


# ===================================================================
# S5: Logistic Regression with Proper Inference (statsmodels)
# ===================================================================

def s5_regression_significance(df):
    hdr("S5: Logistic Regression with Proper Inference (statsmodels)")

    features = pd.DataFrame({
        "IsNonNative": (df["LinguisticGroup"] == "Non-Native").astype(int),
        "IsSTEM": (df["ExpertiseGroup"] == "STEM").astype(int),
        "IsHighEdu": (df["ExperienceGroup"] == "High-Edu").astype(int),
        "IsOlder35": (df["AgeGroup"] == "Older-35plus").astype(int),
        "ConfidenceNorm": df["ConfidenceLevelNorm"],
        "ClickCount": df["ClickCount"],
        "IsHard": df["IsHard"].astype(int),
        "IsMedium": df["IsMedium"].astype(int),
    })

    # Interaction terms
    features["NonNative_x_Hard"] = features["IsNonNative"] * features["IsHard"]
    features["NonNative_x_Medium"] = features["IsNonNative"] * features["IsMedium"]
    features["HighEdu_x_Hard"] = features["IsHighEdu"] * features["IsHard"]

    # Standardize continuous features for interpretability
    for col in ["ConfidenceNorm", "ClickCount"]:
        mean_val = features[col].mean()
        std_val = features[col].std()
        if std_val > 0:
            features[col] = (features[col] - mean_val) / std_val

    y = df["Accuracy"].values

    # Add constant for intercept
    X = sm.add_constant(features)

    # Fit logistic regression
    model = sm.Logit(y, X)
    result = model.fit(disp=0, maxiter=1000)

    print(f"\n  Model summary:")
    print(f"    N = {result.nobs:.0f}")
    print(f"    Pseudo R-sq = {result.prsquared:.4f}")
    print(f"    Log-likelihood = {result.llf:.2f}")
    print(f"    AIC = {result.aic:.2f}")

    rows = []
    print(f"\n  {'Feature':25s} | {'Coef':>8s} | {'SE':>8s} | {'z':>8s} | {'p-value':>8s} | {'OR':>8s} | {'95% CI':>16s}")
    print("  " + "-" * 105)
    for feat in X.columns:
        coef = result.params[feat]
        se = result.bse[feat]
        z = result.tvalues[feat]
        p = result.pvalues[feat]
        ci = result.conf_int().loc[feat]
        odds_ratio = np.exp(coef)
        or_ci = (np.exp(ci[0]), np.exp(ci[1]))

        sig_str = "***" if p < 0.001 else "** " if p < 0.01 else "*  " if p < 0.05 else ".  " if p < 0.10 else "   "
        print(f"  {feat:25s} | {coef:+.4f} | {se:.4f} | {z:+.3f} | {p:.5f} | "
              f"{odds_ratio:.4f} | [{or_ci[0]:.3f}, {or_ci[1]:.3f}] {sig_str}")

        rows.append({
            "Feature": feat,
            "Coefficient": round(coef, 5),
            "Std_Error": round(se, 5),
            "Z_Statistic": round(z, 4),
            "P_Value": round(p, 6),
            "Odds_Ratio": round(odds_ratio, 5),
            "OR_CI_Lo": round(or_ci[0], 5),
            "OR_CI_Hi": round(or_ci[1], 5),
            "Significant_0.05": p < 0.05,
            "Significant_0.01": p < 0.01,
        })

    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(RESULTS_DIR, "S5_regression_significance.csv"), index=False)

    # Highlight the interaction terms
    print(f"\n  KEY INTERACTION TERMS:")
    for feat in ["NonNative_x_Hard", "NonNative_x_Medium", "HighEdu_x_Hard"]:
        r = rdf[rdf["Feature"] == feat].iloc[0]
        sig = "SIGNIFICANT" if r["Significant_0.05"] else "not significant"
        print(f"    {feat}: coef={r['Coefficient']:+.4f}, p={r['P_Value']:.5f}, "
              f"OR={r['Odds_Ratio']:.4f} -> {sig}")

    print(f"\n  -> S5_regression_significance.csv")
    return rdf


# ===================================================================
# S6: Cross-Version Consistency Test
# ===================================================================

def s6_cross_version_consistency(df):
    hdr("S6: Cross-Version Consistency (Non-Native Hard Disparity)")

    hard = df[df["DiffPerf"] == "Hard"]
    versions = sorted(hard["SurveyVersion"].unique())
    results = []

    version_disparities = {}
    for v in versions:
        v_hard = hard[hard["SurveyVersion"] == v]
        nn = v_hard[v_hard["LinguisticGroup"] == "Non-Native"]
        nat = v_hard[v_hard["LinguisticGroup"] == "Native"]
        if len(nn) == 0:
            continue

        nn_acc = nn["Accuracy"].mean()
        nat_acc = nat["Accuracy"].mean() if len(nat) > 0 else np.nan
        disparity = nn_acc - nat_acc if len(nat) > 0 else np.nan
        version_disparities[v] = disparity

        # Fisher exact per version
        p_val = 1.0
        odds_ratio = 1.0
        if len(nat) > 0:
            table = [
                [int(nat["Accuracy"].sum()), len(nat) - int(nat["Accuracy"].sum())],
                [int(nn["Accuracy"].sum()), len(nn) - int(nn["Accuracy"].sum())],
            ]
            try:
                odds_ratio, p_val = fisher_exact(table, alternative="two-sided")
            except ValueError:
                pass

        results.append({
            "Version": v,
            "NN_N": len(nn), "NN_Acc": round(nn_acc, 4),
            "Nat_N": len(nat), "Nat_Acc": round(nat_acc, 4) if not np.isnan(nat_acc) else None,
            "Disparity": round(disparity, 4) if not np.isnan(disparity) else None,
            "Fisher_OR": round(odds_ratio, 4),
            "Fisher_P": round(p_val, 6),
            "Direction_Consistent": disparity < 0 if not np.isnan(disparity) else None,
        })
        sign = "NN < Nat" if disparity < 0 else "NN >= Nat"
        print(f"  V{v}: NN_acc={nn_acc:.3f}(n={len(nn)}) vs Nat_acc={nat_acc:.3f}(n={len(nat)}) "
              f"d={disparity:+.3f} Fisher_p={p_val:.4f} [{sign}]")

    rdf = pd.DataFrame(results)

    # Cochran-Mantel-Haenszel-like test: combine across versions
    # We use a meta-analytic approach: test if all version-specific ORs
    # point in the same direction and compute a combined p-value
    valid_versions = rdf.dropna(subset=["Disparity"])
    n_negative = (valid_versions["Disparity"] < 0).sum()
    n_total = len(valid_versions)
    # Sign test (binomial): under H0, P(negative) = 0.5
    from scipy.stats import binomtest
    try:
        sign_result = binomtest(n_negative, n_total, 0.5, alternative="greater")
        sign_p = sign_result.pvalue
    except Exception:
        sign_p = 1.0

    print(f"\n  Direction consistency: {n_negative}/{n_total} versions show NN < Native")
    print(f"  Sign test (one-sided, H1: consistently negative): p = {sign_p:.4f}")

    # Fisher's method to combine p-values across versions
    fisher_ps = valid_versions["Fisher_P"].values
    if len(fisher_ps) > 0:
        chi2_combined = -2 * np.sum(np.log(np.clip(fisher_ps, 1e-300, 1.0)))
        from scipy.stats import chi2 as chi2_dist
        combined_p = 1 - chi2_dist.cdf(chi2_combined, df=2 * len(fisher_ps))
        print(f"  Fisher's combined p-value (meta-analytic): {combined_p:.5f}")
    else:
        combined_p = 1.0

    # Add summary row
    results.append({
        "Version": "COMBINED",
        "NN_N": int(hard[hard["LinguisticGroup"] == "Non-Native"].shape[0]),
        "NN_Acc": round(hard[hard["LinguisticGroup"] == "Non-Native"]["Accuracy"].mean(), 4),
        "Nat_N": int(hard[hard["LinguisticGroup"] == "Native"].shape[0]),
        "Nat_Acc": round(hard[hard["LinguisticGroup"] == "Native"]["Accuracy"].mean(), 4),
        "Disparity": round(
            hard[hard["LinguisticGroup"] == "Non-Native"]["Accuracy"].mean()
            - hard[hard["LinguisticGroup"] == "Native"]["Accuracy"].mean(), 4),
        "Fisher_OR": None,
        "Fisher_P": round(combined_p, 6),
        "Direction_Consistent": n_negative == n_total,
    })

    rdf = pd.DataFrame(results)
    rdf["Sign_Test_P"] = round(sign_p, 6)
    rdf.to_csv(os.path.join(RESULTS_DIR, "S6_cross_version_consistency.csv"), index=False)
    print(f"\n  -> S6_cross_version_consistency.csv")
    return rdf


# ===================================================================
# S7: Calibration Gap Significance (Permutation)
# ===================================================================

def s7_calibration_significance(df):
    hdr("S7: Calibration Gap Significance (Permutation Test)")
    rng = np.random.RandomState(SEED)
    results = []

    # Overall calibration gap
    overall_gap = (df["ConfidenceLevelNorm"] - df["Accuracy"]).mean()
    print(f"  Overall calibration gap: {overall_gap:+.4f}")

    for gc in GROUP_COLUMNS:
        for gname in sorted(df[gc].unique()):
            group_mask = df[gc] == gname
            group_conf = df.loc[group_mask, "ConfidenceLevelNorm"].values
            group_acc = df.loc[group_mask, "Accuracy"].values
            group_gap = np.mean(group_conf - group_acc)
            obs_diff = group_gap - overall_gap

            # Permutation test: shuffle group labels
            all_conf = df["ConfidenceLevelNorm"].values
            all_acc = df["Accuracy"].values
            group_size = group_mask.sum()
            null_diffs = np.empty(N_PERM)
            indices = np.arange(len(df))

            for i in range(N_PERM):
                perm_idx = rng.choice(indices, size=group_size, replace=False)
                perm_gap = np.mean(all_conf[perm_idx] - all_acc[perm_idx])
                null_diffs[i] = perm_gap - overall_gap

            p_val = np.mean(np.abs(null_diffs) >= np.abs(obs_diff))
            ci_lo = np.percentile(null_diffs, 2.5)
            ci_hi = np.percentile(null_diffs, 97.5)

            is_calib_unfair = abs(group_gap) > FAIRNESS_THRESHOLD
            sig = p_val < 0.05

            results.append({
                "Partition": gc, "Group": gname,
                "N": int(group_mask.sum()),
                "Group_CalibGap": round(group_gap, 5),
                "Overall_CalibGap": round(overall_gap, 5),
                "Diff_from_Overall": round(obs_diff, 5),
                "Null_CI_2.5": round(ci_lo, 5),
                "Null_CI_97.5": round(ci_hi, 5),
                "P_Value": round(p_val, 5),
                "Significant_0.05": sig,
                "CALIB_UNFAIR": is_calib_unfair,
            })
            flag = "***" if sig else ""
            unfair_str = "CALIB-UNFAIR" if is_calib_unfair else ""
            print(f"  {gc:20s} {gname:15s}: gap={group_gap:+.4f} "
                  f"(d={obs_diff:+.4f}), p={p_val:.4f} {flag} {unfair_str}")

    rdf = pd.DataFrame(results)
    rdf.to_csv(os.path.join(RESULTS_DIR, "S7_calibration_significance.csv"), index=False)
    print(f"\n  -> S7_calibration_significance.csv")
    return rdf


# ===================================================================
# Master Summary Table
# ===================================================================

def build_summary_table(s1, s2, s3, s4, s5, s6, s7):
    hdr("MASTER SUMMARY: All Key Findings with P-Values")

    rows = []

    # -- From S1: FairEM permutation for Non-Native
    for _, r in s1.iterrows():
        rows.append({
            "Finding": f"NonNative {r['Metric']} disparity (FairEM permutation)",
            "Source": "S1",
            "Observed_Value": r["Observed_Disparity"],
            "P_Value": r["P_Value"],
            "CI_Lo": r["Null_CI_2.5"],
            "CI_Hi": r["Null_CI_97.5"],
            "Significant_0.05": r["Significant_0.05"],
            "Test_Type": "Permutation (10k)",
        })

    # -- From S2: significant Fisher cells
    for _, r in s2[s2["P_Value_Raw"] < 0.10].iterrows():
        rows.append({
            "Finding": f"{r['Group']} ({r['Partition']}) on {r['Tier']} Qs",
            "Source": "S2",
            "Observed_Value": r["Acc_Diff"],
            "P_Value": r["P_Value_BH"],
            "CI_Lo": None,
            "CI_Hi": None,
            "Significant_0.05": r["Significant_BH_0.05"],
            "Test_Type": f"Fisher exact + BH (OR={r['Odds_Ratio']:.2f})",
        })

    # -- From S3: bootstrap CIs
    for _, r in s3.iterrows():
        rows.append({
            "Finding": r["Disparity"],
            "Source": "S3",
            "Observed_Value": r["Observed"],
            "P_Value": None,
            "CI_Lo": r["CI_2.5"],
            "CI_Hi": r["CI_97.5"],
            "Significant_0.05": r["CI_Excludes_Zero"],
            "Test_Type": "Bootstrap 95% CI (10k)",
        })

    # -- From S4: significant intersections
    for _, r in s4[s4["P_Value_Raw"] < 0.10].iterrows():
        rows.append({
            "Finding": f"{r['Intersection']} on Hard Qs",
            "Source": "S4",
            "Observed_Value": r["NN_Acc"],
            "P_Value": r["P_Value_BH"],
            "CI_Lo": None,
            "CI_Hi": None,
            "Significant_0.05": r["Significant_BH_0.05"],
            "Test_Type": f"Fisher exact + BH ({r['Type']})",
        })

    # -- From S5: significant regression terms
    for _, r in s5[s5["Feature"] != "const"].iterrows():
        rows.append({
            "Finding": f"Regression: {r['Feature']}",
            "Source": "S5",
            "Observed_Value": r["Coefficient"],
            "P_Value": r["P_Value"],
            "CI_Lo": np.log(r["OR_CI_Lo"]) if r["OR_CI_Lo"] > 0 else None,
            "CI_Hi": np.log(r["OR_CI_Hi"]) if r["OR_CI_Hi"] > 0 else None,
            "Significant_0.05": r["Significant_0.05"],
            "Test_Type": f"Logit z-test (z={r['Z_Statistic']:.2f})",
        })

    # -- From S6: cross-version combined
    combined = s6[s6["Version"] == "COMBINED"]
    if len(combined) > 0:
        r = combined.iloc[0]
        rows.append({
            "Finding": "NonNative Hard disparity cross-version combined",
            "Source": "S6",
            "Observed_Value": r["Disparity"],
            "P_Value": r["Fisher_P"],
            "CI_Lo": None,
            "CI_Hi": None,
            "Significant_0.05": r["Fisher_P"] < 0.05 if r["Fisher_P"] is not None else False,
            "Test_Type": "Fisher combined meta-analytic",
        })

    # -- From S7: calibration gaps
    for _, r in s7.iterrows():
        rows.append({
            "Finding": f"{r['Group']} ({r['Partition']}) calibration gap",
            "Source": "S7",
            "Observed_Value": r["Diff_from_Overall"],
            "P_Value": r["P_Value"],
            "CI_Lo": r["Null_CI_2.5"],
            "CI_Hi": r["Null_CI_97.5"],
            "Significant_0.05": r["Significant_0.05"],
            "Test_Type": "Permutation (10k)",
        })

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(RESULTS_DIR, "statistical_tests_summary.csv"), index=False)

    # Print highlights
    n_sig = summary["Significant_0.05"].sum()
    n_total = len(summary)
    print(f"\n  Total findings tested: {n_total}")
    print(f"  Statistically significant (alpha=0.05): {n_sig} ({n_sig / n_total * 100:.1f}%)")

    print(f"\n  SIGNIFICANT FINDINGS:")
    for _, r in summary[summary["Significant_0.05"] == True].iterrows():
        p_str = f"p={r['P_Value']:.4f}" if r["P_Value"] is not None else "CI excl. zero"
        print(f"    [{r['Source']}] {r['Finding']}: val={r['Observed_Value']:+.4f}, "
              f"{p_str} ({r['Test_Type']})")

    print(f"\n  NON-SIGNIFICANT FINDINGS (key ones):")
    for _, r in summary[(summary["Significant_0.05"] == False) &
                        (summary["Source"].isin(["S1", "S3", "S5"]))].iterrows():
        p_str = f"p={r['P_Value']:.4f}" if r["P_Value"] is not None else "CI incl. zero"
        print(f"    [{r['Source']}] {r['Finding']}: val={r['Observed_Value']:+.4f}, "
              f"{p_str}")

    print(f"\n  -> statistical_tests_summary.csv")
    return summary


# ===================================================================
# Visualization: Significance Summary
# ===================================================================

def plot_significance_summary(summary):
    hdr("Generating Significance Summary Visualization")

    # Select a representative subset of findings for the plot
    # Group by source and pick important ones
    plot_data = []

    # S1: all FairEM metrics
    s1_rows = summary[summary["Source"] == "S1"]
    for _, r in s1_rows.iterrows():
        label = r["Finding"].replace("NonNative ", "NN ").replace(" disparity (FairEM permutation)", "")
        plot_data.append({
            "Label": label, "Value": r["Observed_Value"],
            "Significant": r["Significant_0.05"],
            "P_Value": r["P_Value"], "Source": "S1"
        })

    # S3: bootstrap disparities
    s3_rows = summary[summary["Source"] == "S3"]
    for _, r in s3_rows.iterrows():
        label = r["Finding"].replace("_", " ")
        if len(label) > 35:
            label = label[:32] + "..."
        plot_data.append({
            "Label": label, "Value": r["Observed_Value"],
            "Significant": r["Significant_0.05"],
            "P_Value": None, "Source": "S3"
        })

    # S5: key regression terms
    for feat in ["IsNonNative", "IsHard", "NonNative_x_Hard",
                 "NonNative_x_Medium", "HighEdu_x_Hard",
                 "ConfidenceNorm", "ClickCount"]:
        r_rows = summary[(summary["Source"] == "S5") &
                         (summary["Finding"].str.contains(feat))]
        if len(r_rows) > 0:
            r = r_rows.iloc[0]
            plot_data.append({
                "Label": f"Reg: {feat}", "Value": r["Observed_Value"],
                "Significant": r["Significant_0.05"],
                "P_Value": r["P_Value"], "Source": "S5"
            })

    if len(plot_data) == 0:
        print("  No data for visualization.")
        return

    pdf = pd.DataFrame(plot_data)

    fig, axes = plt.subplots(1, 2, figsize=(20, max(10, len(pdf) * 0.35)),
                             gridspec_kw={"width_ratios": [2, 1]})

    # Left panel: coefficient / disparity values
    ax1 = axes[0]
    colors = []
    for _, r in pdf.iterrows():
        if r["Significant"]:
            colors.append("#EF5350" if r["Value"] < 0 else "#43A047")
        else:
            colors.append("#BDBDBD")

    y_pos = np.arange(len(pdf))
    bars = ax1.barh(y_pos, pdf["Value"], color=colors, alpha=0.85, height=0.7)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(pdf["Label"], fontsize=8)
    ax1.axvline(0, color="black", linewidth=1)
    ax1.axvline(-0.1, color="orange", linewidth=0.8, linestyle=":",
                label="Fairness threshold (+/-0.1)")
    ax1.axvline(0.1, color="orange", linewidth=0.8, linestyle=":")
    ax1.set_xlabel("Effect Size (disparity / coefficient)")
    ax1.set_title("Effect Sizes with Significance\n"
                   "(colored = p<0.05 or CI excl. zero; gray = not significant)",
                   fontsize=11, fontweight="bold")
    ax1.legend(fontsize=8, loc="lower right")
    ax1.invert_yaxis()

    # Right panel: p-values (log scale)
    ax2 = axes[1]
    p_vals = []
    p_labels = []
    p_colors = []
    for _, r in pdf.iterrows():
        if r["P_Value"] is not None and r["P_Value"] > 0:
            p_vals.append(r["P_Value"])
            p_labels.append(r["Label"])
            p_colors.append("#EF5350" if r["Significant"] else "#BDBDBD")

    if len(p_vals) > 0:
        y_pos_p = np.arange(len(p_vals))
        # Plot -log10(p) so significant values are larger
        neg_log_p = [-np.log10(p) for p in p_vals]
        ax2.barh(y_pos_p, neg_log_p, color=p_colors, alpha=0.85, height=0.7)
        ax2.set_yticks(y_pos_p)
        ax2.set_yticklabels(p_labels, fontsize=8)
        # Significance threshold line
        ax2.axvline(-np.log10(0.05), color="red", linewidth=1.2, linestyle="--",
                    label="alpha = 0.05")
        ax2.axvline(-np.log10(0.01), color="darkred", linewidth=0.8, linestyle=":",
                    label="alpha = 0.01")
        ax2.set_xlabel("-log10(p-value)")
        ax2.set_title("Statistical Significance\n(right of red line = p<0.05)",
                       fontsize=11, fontweight="bold")
        ax2.legend(fontsize=8)
        ax2.invert_yaxis()

    # Add legend patches
    sig_patch = mpatches.Patch(color="#EF5350", alpha=0.85,
                               label="Significant (p < 0.05)")
    ns_patch = mpatches.Patch(color="#BDBDBD", alpha=0.85,
                              label="Not significant")
    pos_patch = mpatches.Patch(color="#43A047", alpha=0.85,
                               label="Significant positive")
    fig.legend(handles=[sig_patch, pos_patch, ns_patch],
               loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Phase 3 Statistical Tests: Significance Summary\n"
                 "Permutation (S1,S7) | Fisher+BH (S2,S4) | Bootstrap (S3) | "
                 "Logit z-test (S5) | Meta-analytic (S6)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.93])
    plt.savefig(os.path.join(RESULTS_DIR, "significance_summary.png"),
                dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> significance_summary.png")


# ===================================================================
# Main
# ===================================================================

def main():
    hdr("Phase 3 Statistical Significance Tests")
    print(f"  Permutations: {N_PERM:,}")
    print(f"  Bootstrap resamples: {N_BOOT:,}")
    print(f"  Random seed: {SEED}")

    df = load_and_prepare()

    s1 = s1_permutation_fairem(df)
    s2 = s2_fisher_per_cell(df)
    s3 = s3_bootstrap_ci(df)
    s4 = s4_intersection_tests(df)
    s5 = s5_regression_significance(df)
    s6 = s6_cross_version_consistency(df)
    s7 = s7_calibration_significance(df)

    summary = build_summary_table(s1, s2, s3, s4, s5, s6, s7)
    plot_significance_summary(summary)

    hdr("DONE -- All statistical tests complete")
    print(f"  Results directory: {os.path.abspath(RESULTS_DIR)}")
    print(f"  Files generated:")
    for f in sorted(os.listdir(RESULTS_DIR)):
        fpath = os.path.join(RESULTS_DIR, f)
        size = os.path.getsize(fpath)
        print(f"    {f:45s} ({size:,} bytes)")
    print(SEP)


if __name__ == "__main__":
    main()
