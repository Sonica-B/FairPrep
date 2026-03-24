"""
Re-run Phase 1 + Phase 2 on Cleaned Data
==========================================
1. Load raw data
2. Apply data cleaning (from src/data_cleaning.py)
3. Re-run Phase 1 experiments (Exp 1-9) on cleaned data
4. Re-run Phase 2 experiments (A1-A7) on cleaned data
5. Generate before-vs-after comparative analysis

Results saved to:
  results/cleaned_phase1/
  results/cleaned_phase2/
  results/cleaning_comparison/
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
from scipy.stats import chi2_contingency, fisher_exact, spearmanr, ttest_ind
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src.excel_data_loader import load_excel_data, get_annotator_demographics, group_distribution_summary
from src.data_cleaning import clean_data
from src.measures import (
    AP, SP, TPR, FPR, FNR, TNR, PPV, NPV, FDR, FOR,
    HIGHER_IS_BETTER, LOWER_IS_BETTER, calibration_gap,
)

# ---------------------------------------------------------------------------
EXCEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Export_and_Compiled.xlsx")
P1_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cleaned_phase1")
P2_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cleaned_phase2")
COMP_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "cleaning_comparison")
os.makedirs(P1_DIR, exist_ok=True)
os.makedirs(P2_DIR, exist_ok=True)
os.makedirs(COMP_DIR, exist_ok=True)

GROUP_COLUMNS = ["LinguisticGroup", "ExpertiseGroup", "ExperienceGroup", "AgeGroup"]
FAIRNESS_THRESHOLD = 0.1
HARD_QUESTIONS = [2, 5, 6]
EASY_QUESTIONS = [1, 3, 4, 7, 8]
SEP = "=" * 72

ALL_METRICS = [
    ("AP",  AP,  "higher"), ("SP",  SP,  "higher"),
    ("TPR", TPR, "higher"), ("TNR", TNR, "higher"),
    ("PPV", PPV, "higher"), ("NPV", NPV, "higher"),
    ("FPR", FPR, "lower"),  ("FNR", FNR, "lower"),
    ("FDR", FDR, "lower"),  ("FOR", FOR, "lower"),
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


# =========================================================================
# PHASE 1 EXPERIMENTS (on cleaned data)
# =========================================================================

def p1_exp1_distributions(df):
    """Exp 1: Group distributions."""
    demos = get_annotator_demographics(df)
    rows = []
    for gc in GROUP_COLUMNS:
        for gname, gdf in demos.groupby(gc):
            rows.append({"Partition": gc, "Group": gname, "N_Annotators": len(gdf),
                         "Pct": round(100 * len(gdf) / len(demos), 1)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(P1_DIR, "exp1_group_distributions.csv"), index=False)
    print("  Exp 1: Group distributions saved")
    return res

def p1_exp2_fairem_disparity(df):
    """Exp 2: Full 10-metric FairEM disparity."""
    overall = compute_all_metrics(df["SurveyAnswer"], df["ActualAnswer"])
    rows = []
    for gc in GROUP_COLUMNS:
        for gname, gdf in df.groupby(gc):
            TP, FP, TN, FN = confusion_counts(gdf["SurveyAnswer"], gdf["ActualAnswer"])
            row = {"Partition": gc, "Group": gname, "N": len(gdf), "TP": TP, "FP": FP, "TN": TN, "FN": FN}
            for mname, fn, direction in ALL_METRICS:
                val = fn(TP, FP, TN, FN)
                d = val - overall[mname]
                row[mname] = round(val, 4)
                row[f"{mname}_d"] = round(d, 4)
                row[f"{mname}_unfair"] = is_unfair(d, direction)
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(P1_DIR, "exp2_group_accuracy.csv"), index=False)

    # Print unfair flags
    for _, r in res.iterrows():
        flags = [m for m, _, d in ALL_METRICS if r[f"{m}_unfair"]]
        if flags:
            print(f"    *** UNFAIR: {r['Partition']}/{r['Group']}: {', '.join(flags)}")
    print("  Exp 2: FairEM disparity saved")
    return res

def p1_exp3_behavioral(df):
    """Exp 3: Behavioral profiles."""
    rows = []
    for gc in GROUP_COLUMNS:
        for gname, gdf in df.groupby(gc):
            row = {"Partition": gc, "Group": gname, "N": len(gdf)}
            for col in ["DecisionTime", "FirstClick", "LastClick", "ClickCount", "ConfidenceLevel"]:
                row[f"{col}_mean"] = round(gdf[col].mean(), 4)
                row[f"{col}_median"] = round(gdf[col].median(), 4)
                row[f"{col}_std"] = round(gdf[col].std(), 4)
            dtr = gdf["FirstClick"] / gdf["DecisionTime"].replace(0, np.nan)
            row["DecisionTimeFract_mean"] = round(dtr.mean(), 4)
            row["DecisionTimeFract_median"] = round(dtr.median(), 4)
            rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(P1_DIR, "exp3_behavioral_profiles.csv"), index=False)
    print("  Exp 3: Behavioral profiles saved")
    return res

def p1_exp4_majority(df):
    """Exp 4: Majority vote disagreement."""
    rows = []
    for gc in GROUP_COLUMNS:
        for gname, gdf in df.groupby(gc):
            disagree = (gdf["SurveyAnswer"] != gdf["Majority"]).mean()
            rows.append({"Partition": gc, "Group": gname, "N": len(gdf),
                         "DisagreementRate": round(disagree, 4)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(P1_DIR, "exp4_majority_disagreement.csv"), index=False)
    print("  Exp 4: Majority disagreement saved")
    return res

def p1_exp5_calibration(df):
    """Exp 5: Calibration gap."""
    rows = []
    for gc in GROUP_COLUMNS:
        for gname, gdf in df.groupby(gc):
            mean_conf = gdf["ConfidenceLevelNorm"].mean()
            mean_acc = gdf["Accuracy"].mean()
            gap = mean_conf - mean_acc
            rows.append({"Partition": gc, "Group": gname, "N": len(gdf),
                         "MeanConfidence": round(mean_conf, 4),
                         "MeanAccuracy": round(mean_acc, 4),
                         "CalibrationGap": round(gap, 4),
                         "IsOverconfident": gap > 0,
                         "CalibFairFlag": "[CALIB-UNFAIR]" if abs(gap) > 0.1 else "fair"})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(P1_DIR, "exp5_calibration_gap.csv"), index=False)
    for _, r in res.iterrows():
        if r["CalibFairFlag"] != "fair":
            print(f"    *** CALIB-UNFAIR: {r['Partition']}/{r['Group']} gap={r['CalibrationGap']:.4f}")
    print("  Exp 5: Calibration gap saved")
    return res

def p1_exp6_explanation(df):
    """Exp 6: Explanation rate."""
    rows = []
    for gc in GROUP_COLUMNS:
        for gname, gdf in df.groupby(gc):
            exp_rate = gdf["IsExp"].mean()
            rows.append({"Partition": gc, "Group": gname, "N": len(gdf),
                         "ExplanationRate": round(exp_rate, 4)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(P1_DIR, "exp6_explanation_rate.csv"), index=False)
    print("  Exp 6: Explanation rate saved")
    return res

def p1_exp7_per_question(df):
    """Exp 7: Per-question accuracy disparity."""
    rows = []
    for gc in GROUP_COLUMNS:
        for q in range(1, 9):
            qdf = df[df["QuestionNum"] == q]
            overall_acc = qdf["Accuracy"].mean()
            for gname, gdf in qdf.groupby(gc):
                g_acc = gdf["Accuracy"].mean()
                rows.append({"Partition": gc, "Group": gname, "QuestionNum": q,
                             "GroupAcc": round(g_acc, 4), "OverallAcc": round(overall_acc, 4),
                             "Disparity": round(g_acc - overall_acc, 4), "N": len(gdf)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(P1_DIR, "exp7_per_question_disparity.csv"), index=False)
    # Flag large disparities
    big = res[res["Disparity"].abs() > 0.15]
    for _, r in big.iterrows():
        print(f"    *** Q{r['QuestionNum']} {r['Partition']}/{r['Group']}: d={r['Disparity']:+.4f}")
    print("  Exp 7: Per-question disparity saved")
    return res

def p1_exp8_conditional(df):
    """Exp 8: Conditional fairness by difficulty."""
    tiers = {"Hard": HARD_QUESTIONS, "Easy": EASY_QUESTIONS}
    rows = []
    for tier, qs in tiers.items():
        dft = df[df["QuestionNum"].isin(qs)]
        TP_o, FP_o, TN_o, FN_o = confusion_counts(dft["SurveyAnswer"], dft["ActualAnswer"])
        oa = AP(TP_o, FP_o, TN_o, FN_o)
        ot = TPR(TP_o, FP_o, TN_o, FN_o)
        for gc in GROUP_COLUMNS:
            for gname, gdf in dft.groupby(gc):
                TP, FP, TN, FN = confusion_counts(gdf["SurveyAnswer"], gdf["ActualAnswer"])
                acc = AP(TP, FP, TN, FN)
                tpr = TPR(TP, FP, TN, FN)
                rows.append({"Tier": tier, "Partition": gc, "Group": gname, "N": len(gdf),
                             "Accuracy": round(acc, 4), "TPR": round(tpr, 4),
                             "Accuracy_Disparity": round(acc - oa, 4),
                             "TPR_Disparity": round(tpr - ot, 4),
                             "UNFAIR_ACC": abs(acc - oa) > 0.1,
                             "UNFAIR_TPR": abs(tpr - ot) > 0.1})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(P1_DIR, "exp8_conditional_fairness.csv"), index=False)
    unfair = res[res["UNFAIR_ACC"] | res["UNFAIR_TPR"]]
    for _, r in unfair.iterrows():
        print(f"    *** {r['Tier']} {r['Partition']}/{r['Group']}: Acc_d={r['Accuracy_Disparity']:+.4f}")
    print("  Exp 8: Conditional fairness saved")
    return res

def p1_exp9_bootstrap(df, n_boot=5000, seed=42):
    """Exp 9: Bootstrap CIs."""
    rng = np.random.default_rng(seed)
    rows = []
    all_preds = df["SurveyAnswer"].values
    all_labels = df["ActualAnswer"].values
    for gc in GROUP_COLUMNS:
        for gname, gdf in df.groupby(gc):
            preds = gdf["SurveyAnswer"].values
            labels = gdf["ActualAnswer"].values
            n = len(preds)
            acc_disps = []
            for _ in range(n_boot):
                ig = rng.integers(0, n, n)
                TP_g, FP_g, TN_g, FN_g = confusion_counts(preds[ig], labels[ig])
                ia = rng.integers(0, len(all_preds), len(all_preds))
                TP_a, FP_a, TN_a, FN_a = confusion_counts(all_preds[ia], all_labels[ia])
                acc_disps.append(AP(TP_g, FP_g, TN_g, FN_g) - AP(TP_a, FP_a, TN_a, FN_a))
            lo, hi = np.percentile(acc_disps, [2.5, 97.5])
            TP, FP, TN, FN = confusion_counts(preds, labels)
            all_TP, all_FP, all_TN, all_FN = confusion_counts(all_preds, all_labels)
            obs = AP(TP, FP, TN, FN) - AP(all_TP, all_FP, all_TN, all_FN)
            rows.append({"Partition": gc, "Group": gname, "N": n,
                         "Obs_AccDisparity": round(obs, 4),
                         "Acc_CI_lo": round(lo, 4), "Acc_CI_hi": round(hi, 4),
                         "Acc_Significant": (lo > 0 or hi < 0)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(P1_DIR, "exp9_bootstrap_ci.csv"), index=False)
    sig = res[res["Acc_Significant"]]
    for _, r in sig.iterrows():
        print(f"    *** SIG: {r['Partition']}/{r['Group']}: d={r['Obs_AccDisparity']:+.4f} CI=[{r['Acc_CI_lo']:+.4f},{r['Acc_CI_hi']:+.4f}]")
    print("  Exp 9: Bootstrap CIs saved")
    return res


# =========================================================================
# PHASE 2 EXPERIMENTS (on cleaned data)
# =========================================================================

def p2_A1_intersectional(df):
    """A1: Intersectional fairness."""
    overall = compute_all_metrics(df["SurveyAnswer"], df["ActualAnswer"])
    rows = []
    for gc1, gc2 in combinations(GROUP_COLUMNS, 2):
        df["_inter"] = df[gc1] + " x " + df[gc2]
        for combo, gdf in df.groupby("_inter"):
            m = compute_all_metrics(gdf["SurveyAnswer"], gdf["ActualAnswer"])
            row = {"Intersection": f"{gc1} x {gc2}", "Group": combo, "N": len(gdf),
                   "N_Annotators": gdf["ResponseId"].nunique()}
            unfair_list = []
            for mname, _, direction in ALL_METRICS:
                d = m[mname] - overall[mname]
                row[f"{mname}_d"] = round(d, 4)
                if is_unfair(d, direction):
                    unfair_list.append(f"{mname}({d:+.3f})")
            row["Unfair_Metrics"] = "; ".join(unfair_list)
            row["N_Unfair"] = len(unfair_list)
            rows.append(row)
        df.drop("_inter", axis=1, inplace=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(P2_DIR, "A1_intersectional.csv"), index=False)

    # Heatmap
    fig, axes = plt.subplots(2, 3, figsize=(22, 14))
    axes = axes.flatten()
    idx = 0
    for gc1, gc2 in combinations(GROUP_COLUMNS, 2):
        sub = res[res["Intersection"] == f"{gc1} x {gc2}"]
        g1s = sorted(df[gc1].unique())
        g2s = sorted(df[gc2].unique())
        pivot = pd.DataFrame(index=g1s, columns=g2s, dtype=float)
        for _, r in sub.iterrows():
            parts = r["Group"].split(" x ")
            if len(parts) == 2:
                pivot.loc[parts[0].strip(), parts[1].strip()] = r["AP_d"]
        pivot = pivot.astype(float)
        sns.heatmap(pivot, ax=axes[idx], annot=True, fmt=".3f", cmap="RdYlGn",
                    center=0, vmin=-0.15, vmax=0.15, linewidths=0.5)
        axes[idx].set_title(f"AP Disparity\n{gc1} x {gc2}", fontsize=9, fontweight="bold")
        idx += 1
    plt.suptitle("A1 -- Intersectional AP Disparities (CLEANED)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(P2_DIR, "A1_intersectional.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  A1: Intersectional fairness saved")
    return res

def p2_A2_leave_one_out(df):
    """A2: Leave-one-out annotator influence."""
    overall = compute_all_metrics(df["SurveyAnswer"], df["ActualAnswer"])
    rows = []
    for rid in df["ResponseId"].unique():
        dfw = df[df["ResponseId"] != rid]
        ann = df[df["ResponseId"] == rid].iloc[0]
        ann_acc = float(df[df["ResponseId"] == rid]["Accuracy"].mean())
        row = {"ResponseId": rid, "Annotator_Accuracy": round(ann_acc, 4)}
        overall_w = compute_all_metrics(dfw["SurveyAnswer"], dfw["ActualAnswer"])
        for gc in GROUP_COLUMNS:
            row[gc] = ann[gc]
            gname = ann[gc]
            grp_orig = df[df[gc] == gname]
            grp_w = dfw[dfw[gc] == gname]
            if len(grp_w) == 0:
                continue
            om = compute_all_metrics(grp_orig["SurveyAnswer"], grp_orig["ActualAnswer"])
            wm = compute_all_metrics(grp_w["SurveyAnswer"], grp_w["ActualAnswer"])
            row[f"{gc}_AP_shift"] = round((wm["AP"] - overall_w["AP"]) - (om["AP"] - overall["AP"]), 5)
        rows.append(row)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(P2_DIR, "A2_leave_one_out.csv"), index=False)
    print("  A2: Leave-one-out saved")
    return res

def p2_A3_ablation(df):
    """A3: Ablation study."""
    overall = compute_all_metrics(df["SurveyAnswer"], df["ActualAnswer"])
    baselines = {}
    for gc in GROUP_COLUMNS:
        mx = 0
        for gname, gdf in df.groupby(gc):
            m = compute_all_metrics(gdf["SurveyAnswer"], gdf["ActualAnswer"])
            for mname, _, direction in ALL_METRICS:
                d = m[mname] - overall[mname]
                if direction == "higher":
                    mx = max(mx, abs(min(d, 0)))
                else:
                    mx = max(mx, abs(max(d, 0)))
        baselines[gc] = mx

    rows = []
    rng = np.random.default_rng(42)
    for tgt in GROUP_COLUMNS:
        groups = df.groupby(tgt)
        min_sz = groups.size().min()
        bal = []
        for gn, gdf in groups:
            if len(gdf) > min_sz:
                bal.append(gdf.iloc[rng.choice(len(gdf), min_sz, replace=False)])
            else:
                bal.append(gdf)
        dfb = pd.concat(bal, ignore_index=True)
        for gc in GROUP_COLUMNS:
            if gc == tgt:
                continue
            ob = compute_all_metrics(dfb["SurveyAnswer"], dfb["ActualAnswer"])
            mx = 0
            for gn, gdf in dfb.groupby(gc):
                m = compute_all_metrics(gdf["SurveyAnswer"], gdf["ActualAnswer"])
                for mn, _, dr in ALL_METRICS:
                    d = m[mn] - ob[mn]
                    if dr == "higher":
                        mx = max(mx, abs(min(d, 0)))
                    else:
                        mx = max(mx, abs(max(d, 0)))
            change = mx - baselines[gc]
            rows.append({"Balanced_Partition": tgt, "Measured_Partition": gc,
                         "Baseline_MaxDisp": round(baselines[gc], 4),
                         "After_MaxDisp": round(mx, 4), "Change": round(change, 4),
                         "Interpretation": "ACTIVE (correlated)" if abs(change) >= 0.02 else "PASSIVE (independent)"})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(P2_DIR, "A3_ablation.csv"), index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    pivot = res.pivot(index="Balanced_Partition", columns="Measured_Partition", values="Change")
    sns.heatmap(pivot, annot=True, fmt="+.3f", cmap="RdBu_r", center=0, linewidths=0.5, ax=ax)
    ax.set_title("A3 -- Ablation: Change in Max Disparity (CLEANED)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(P2_DIR, "A3_ablation.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  A3: Ablation saved")
    return res

def p2_A4_statistical(df):
    """A4: Chi-square & Cramer's V."""
    rows = []
    for gc in GROUP_COLUMNS:
        ct = pd.crosstab(df[gc], df["Accuracy"])
        chi2, p, dof, _ = chi2_contingency(ct)
        n = len(df)
        k = min(ct.shape)
        v = np.sqrt(chi2 / (n * (k - 1))) if (n * (k - 1)) > 0 else 0
        rows.append({"Test": "Chi2_vs_Accuracy", "Variable": gc,
                     "Chi2": round(chi2, 4), "p_value": round(p, 4), "Cramers_V": round(v, 4),
                     "Significant": p < 0.05})
    for gc1, gc2 in combinations(GROUP_COLUMNS, 2):
        ct = pd.crosstab(df[gc1], df[gc2])
        chi2, p, dof, _ = chi2_contingency(ct)
        k = min(ct.shape)
        v = np.sqrt(chi2 / (len(df) * (k - 1))) if (len(df) * (k - 1)) > 0 else 0
        rows.append({"Test": "Chi2_Pairwise", "Variable": f"{gc1} x {gc2}",
                     "Chi2": round(chi2, 4), "p_value": round(p, 4), "Cramers_V": round(v, 4),
                     "Significant": p < 0.05})
    for col in ["Age", "Education", "EngProf", "Major"]:
        rho, p = spearmanr(df[col], df["Accuracy"])
        rows.append({"Test": "Spearman", "Variable": col,
                     "Chi2": round(rho, 4), "p_value": round(p, 4), "Cramers_V": round(abs(rho), 4),
                     "Significant": p < 0.05})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(P2_DIR, "A4_statistical_association.csv"), index=False)

    # Cramer's V heatmap
    labels = GROUP_COLUMNS + ["Accuracy"]
    n_l = len(labels)
    vm = np.zeros((n_l, n_l))
    for i, l1 in enumerate(labels):
        for j, l2 in enumerate(labels):
            if i == j:
                vm[i, j] = 1.0
                continue
            c1 = df[l1] if l1 != "Accuracy" else df["Accuracy"].astype(str)
            c2 = df[l2] if l2 != "Accuracy" else df["Accuracy"].astype(str)
            ct = pd.crosstab(c1, c2)
            chi2, p, dof, exp = chi2_contingency(ct)
            k = min(ct.shape)
            vm[i, j] = np.sqrt(chi2 / (len(df) * (k - 1))) if (len(df) * (k - 1)) > 0 else 0
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(pd.DataFrame(vm, index=labels, columns=labels),
                annot=True, fmt=".3f", cmap="YlOrRd", vmin=0, vmax=0.5, linewidths=0.5, ax=ax)
    ax.set_title("A4 -- Cramer's V (CLEANED)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(P2_DIR, "A4_statistical_association.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("  A4: Statistical association saved")
    return res

def p2_A5_regression(df):
    """A5: Logistic regression decomposition."""
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
    X["NonNative_x_HardQ"] = X["IsNonNative"] * X["IsHardQ"]
    X["HighEdu_x_HardQ"] = X["IsHighEdu"] * X["IsHardQ"]
    scaler = StandardScaler()
    Xs = X.copy()
    for col in ["ConfidenceNorm", "ClickCount"]:
        Xs[col] = scaler.fit_transform(Xs[[col]])
    model = LogisticRegression(max_iter=1000, random_state=42, penalty="l2", C=1.0)
    model.fit(Xs, y)
    coefs = pd.DataFrame({
        "Feature": X.columns, "Coefficient": model.coef_[0],
        "Odds_Ratio": np.exp(model.coef_[0]), "Abs_Coef": np.abs(model.coef_[0]),
    }).sort_values("Abs_Coef", ascending=False)
    coefs.to_csv(os.path.join(P2_DIR, "A5_regression.csv"), index=False)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = ["#E53935" if c < 0 else "#43A047" for c in coefs["Coefficient"]]
    ax.barh(range(len(coefs)), coefs["Coefficient"], color=colors, alpha=0.8)
    ax.set_yticks(range(len(coefs)))
    ax.set_yticklabels(coefs["Feature"])
    ax.axvline(0, color="black", linewidth=1)
    ax.axvline(-0.1, color="orange", linewidth=0.8, linestyle=":")
    ax.axvline(0.1, color="orange", linewidth=0.8, linestyle=":")
    ax.set_title("A5 -- Logistic Regression (CLEANED)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(P2_DIR, "A5_regression.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  A5: Regression saved (model acc={model.score(Xs, y):.4f})")
    for _, r in coefs.iterrows():
        tag = "ACTIVE" if r["Abs_Coef"] > 0.1 else "passive"
        print(f"    {r['Feature']:25s}: coef={r['Coefficient']:+.4f} OR={r['Odds_Ratio']:.4f} [{tag}]")
    return coefs

def p2_A6_question_interaction(df):
    """A6: Per-question x per-demographic interaction."""
    rows = []
    for gc in GROUP_COLUMNS:
        for q in range(1, 9):
            qdf = df[df["QuestionNum"] == q]
            oa = qdf["Accuracy"].mean()
            for gname, gdf in qdf.groupby(gc):
                ga = gdf["Accuracy"].mean()
                n = len(gdf)
                d = ga - oa
                other = qdf[qdf[gc] != gname]
                table = [[int(gdf["Accuracy"].sum()), n - int(gdf["Accuracy"].sum())],
                         [int(other["Accuracy"].sum()), len(other) - int(other["Accuracy"].sum())]]
                try:
                    odds, p = fisher_exact(table)
                except ValueError:
                    odds, p = 1.0, 1.0
                cls = "ACTIVE-UNFAIR" if abs(d) > 0.15 and p < 0.1 else \
                      "BORDERLINE" if abs(d) > 0.1 else \
                      "PASSIVE" if abs(d) > 0.05 else "FAIR"
                rows.append({"Partition": gc, "Group": gname, "QuestionNum": q,
                             "GroupAcc": round(ga, 4), "OverallQAcc": round(oa, 4),
                             "Disparity": round(d, 4), "N": n,
                             "Fisher_OR": round(odds, 4), "Fisher_p": round(p, 4),
                             "Classification": cls})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(P2_DIR, "A6_question_interaction.csv"), index=False)

    # Heatmap
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    axes = axes.flatten()
    for i, gc in enumerate(GROUP_COLUMNS):
        sub = res[res["Partition"] == gc]
        pivot = sub.pivot(index="QuestionNum", columns="Group", values="Disparity")
        annot_cls = sub.pivot(index="QuestionNum", columns="Group", values="Classification")
        annot_text = pd.DataFrame("", index=pivot.index, columns=pivot.columns)
        for q in annot_text.index:
            for g in annot_text.columns:
                d = pivot.loc[q, g]
                cls = annot_cls.loc[q, g] if pd.notna(annot_cls.loc[q, g]) else ""
                marker = "***" if cls == "ACTIVE-UNFAIR" else "**" if cls == "BORDERLINE" else ""
                annot_text.loc[q, g] = f"{d:+.2f}{marker}"
        sns.heatmap(pivot, ax=axes[i], annot=annot_text, fmt="", cmap="RdYlGn",
                    center=0, vmin=-0.4, vmax=0.4, linewidths=0.5)
        axes[i].set_title(f"{gc}", fontsize=10, fontweight="bold")
    plt.suptitle("A6 -- Question x Demographic Disparity (CLEANED)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(P2_DIR, "A6_question_interaction.png"), dpi=150, bbox_inches="tight")
    plt.close()

    active = res[res["Classification"] == "ACTIVE-UNFAIR"]
    for _, r in active.iterrows():
        print(f"    *** ACTIVE-UNFAIR: Q{r['QuestionNum']} x {r['Group']} ({r['Partition']}): "
              f"d={r['Disparity']:+.3f} p={r['Fisher_p']:.4f}")
    print("  A6: Question interaction saved")
    return res

def p2_A7_summary(df, a1, a2, a3, a4, a5, a6):
    """A7: Evidence synthesis."""
    classifications = []
    for gc in GROUP_COLUMNS:
        ev = {"Partition": gc, "Active": [], "Passive": []}
        # A1
        gc_int = a1[a1["Intersection"].str.contains(gc)]
        nu = gc_int["N_Unfair"].sum()
        if nu > 2:
            ev["Active"].append(f"A1: {nu} unfair intersections")
        else:
            ev["Passive"].append(f"A1: {nu} unfair intersections")
        # A2
        col = f"{gc}_AP_shift"
        if col in a2.columns:
            hi = (a2[col].abs() > 0.005).sum()
            if hi >= 3:
                ev["Active"].append(f"A2: {hi} high-influence annotators")
            else:
                ev["Passive"].append(f"A2: {hi} high-influence")
        # A3
        abl = a3[(a3["Balanced_Partition"] == gc) | (a3["Measured_Partition"] == gc)]
        na = abl["Interpretation"].str.contains("ACTIVE").sum()
        if na > 0:
            ev["Active"].append(f"A3: {na} active ablation")
        else:
            ev["Passive"].append("A3: no ablation effect")
        # A4
        chi_row = a4[(a4["Test"] == "Chi2_vs_Accuracy") & (a4["Variable"] == gc)]
        if len(chi_row) and chi_row.iloc[0]["Significant"]:
            ev["Active"].append(f"A4: Chi2 sig (p={chi_row.iloc[0]['p_value']:.4f})")
        else:
            ev["Passive"].append(f"A4: Chi2 ns")
        # A5
        feat_map = {"LinguisticGroup": "IsNonNative", "ExpertiseGroup": "IsSTEM",
                    "ExperienceGroup": "IsHighEdu", "AgeGroup": "IsOlder35"}
        feat = feat_map.get(gc)
        if feat:
            rr = a5[a5["Feature"] == feat]
            if len(rr) and rr.iloc[0]["Abs_Coef"] > 0.1:
                ev["Active"].append(f"A5: coef={rr.iloc[0]['Coefficient']:+.3f}")
            else:
                ev["Passive"].append(f"A5: coef={rr.iloc[0]['Coefficient']:+.3f}" if len(rr) else "A5: N/A")
        # A6
        gc_cells = a6[a6["Partition"] == gc]
        nau = (gc_cells["Classification"] == "ACTIVE-UNFAIR").sum()
        nb = (gc_cells["Classification"] == "BORDERLINE").sum()
        if nau >= 2:
            ev["Active"].append(f"A6: {nau} ACTIVE-UNFAIR + {nb} BORDERLINE")
        elif nau >= 1 or nb >= 2:
            ev["Active"].append(f"A6: {nau} ACTIVE + {nb} BORDERLINE")
        else:
            ev["Passive"].append(f"A6: {nau} active, {nb} borderline")

        na_count = len(ev["Active"])
        total = na_count + len(ev["Passive"])
        if na_count >= 4:
            cls = "STRONGLY ACTIVE"
        elif na_count >= 3:
            cls = "ACTIVE"
        elif na_count >= 2:
            cls = "MODERATELY ACTIVE"
        elif na_count >= 1:
            cls = "WEAKLY ACTIVE"
        else:
            cls = "PASSIVE"
        ev["Classification"] = cls
        ev["Score"] = f"{na_count}/{total}"
        classifications.append(ev)

    rows = []
    for c in classifications:
        rows.append({"Partition": c["Partition"], "Classification": c["Classification"],
                     "Active_Score": c["Score"],
                     "Active_Evidence": " | ".join(c["Active"]),
                     "Passive_Evidence": " | ".join(c["Passive"])})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(P2_DIR, "A7_classification_summary.csv"), index=False)

    # Viz
    fig, ax = plt.subplots(figsize=(12, 6))
    partitions = [c["Partition"] for c in classifications]
    ac = [len(c["Active"]) for c in classifications]
    pc = [len(c["Passive"]) for c in classifications]
    x = np.arange(len(partitions))
    w = 0.35
    ax.barh(x - w/2, ac, w, label="Active Evidence", color="#E53935", alpha=0.8)
    ax.barh(x + w/2, pc, w, label="Passive Evidence", color="#90A4AE", alpha=0.8)
    ax.set_yticks(x)
    ax.set_yticklabels(partitions)
    ax.set_title("A7 -- Bias Classification (CLEANED)", fontsize=11, fontweight="bold")
    ax.legend()
    for i, c in enumerate(classifications):
        ax.text(max(ac[i], pc[i]) + 0.3, i, c["Classification"],
                va="center", fontsize=9, fontweight="bold",
                color="#E53935" if "ACTIVE" in c["Classification"] else "#666")
    plt.tight_layout()
    plt.savefig(os.path.join(P2_DIR, "A7_classification_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print("\n  A7 Classification (CLEANED):")
    for c in classifications:
        print(f"    {c['Partition']:25s}: {c['Classification']:25s} ({c['Score']})")
    return res


# =========================================================================
# COMPARISON: Before vs After Cleaning
# =========================================================================

def generate_comparison(df_raw, df_clean, clean_report):
    """Generate before-vs-after comparison."""
    hdr("COMPARISON: Before vs After Data Cleaning")

    rows = []

    # Overall stats
    for label, d in [("Before", df_raw), ("After", df_clean)]:
        m = compute_all_metrics(d["SurveyAnswer"], d["ActualAnswer"])
        rows.append({"Stage": label, "Metric": "Overall_Accuracy", "Value": round(m["AP"], 4)})
        rows.append({"Stage": label, "Metric": "Overall_TPR", "Value": round(m["TPR"], 4)})
        rows.append({"Stage": label, "Metric": "N_Rows", "Value": len(d)})
        rows.append({"Stage": label, "Metric": "N_Annotators", "Value": d["ResponseId"].nunique()})

        for gc in GROUP_COLUMNS:
            overall_m = compute_all_metrics(d["SurveyAnswer"], d["ActualAnswer"])
            for gname, gdf in d.groupby(gc):
                gm = compute_all_metrics(gdf["SurveyAnswer"], gdf["ActualAnswer"])
                rows.append({"Stage": label, "Metric": f"{gc}_{gname}_AP",
                             "Value": round(gm["AP"], 4)})
                rows.append({"Stage": label, "Metric": f"{gc}_{gname}_AP_d",
                             "Value": round(gm["AP"] - overall_m["AP"], 4)})
                rows.append({"Stage": label, "Metric": f"{gc}_{gname}_TPR_d",
                             "Value": round(gm["TPR"] - overall_m["TPR"], 4)})
                rows.append({"Stage": label, "Metric": f"{gc}_{gname}_N",
                             "Value": len(gdf)})

        # Calibration
        for gc in GROUP_COLUMNS:
            for gname, gdf in d.groupby(gc):
                gap = gdf["ConfidenceLevelNorm"].mean() - gdf["Accuracy"].mean()
                rows.append({"Stage": label, "Metric": f"CalibGap_{gc}_{gname}",
                             "Value": round(gap, 4)})

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(os.path.join(COMP_DIR, "before_vs_after.csv"), index=False)

    # Pivot for readability
    pivot = comp_df.pivot(index="Metric", columns="Stage", values="Value")
    if "Before" in pivot.columns and "After" in pivot.columns:
        pivot["Delta"] = pivot["After"] - pivot["Before"]
    pivot.to_csv(os.path.join(COMP_DIR, "comparison_pivot.csv"))

    # Print key comparisons
    print(f"\n  {'Metric':45s} | {'Before':>8s} | {'After':>8s} | {'Delta':>8s}")
    print("  " + "-" * 75)
    for _, r in pivot.iterrows():
        name = r.name
        before = r.get("Before", np.nan)
        after = r.get("After", np.nan)
        delta = r.get("Delta", np.nan)
        if pd.notna(before) and pd.notna(after):
            print(f"  {name:45s} | {before:8.4f} | {after:8.4f} | {delta:+8.4f}")

    # Visualization: side-by-side disparity bars
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for i, gc in enumerate(GROUP_COLUMNS):
        ax = axes[i]
        groups = sorted(df_raw[gc].unique())
        before_d, after_d = [], []
        for gname in groups:
            # Before
            gdf_b = df_raw[df_raw[gc] == gname]
            om_b = compute_all_metrics(df_raw["SurveyAnswer"], df_raw["ActualAnswer"])
            gm_b = compute_all_metrics(gdf_b["SurveyAnswer"], gdf_b["ActualAnswer"])
            before_d.append(gm_b["AP"] - om_b["AP"])
            # After
            gdf_a = df_clean[df_clean[gc] == gname]
            if len(gdf_a) > 0:
                om_a = compute_all_metrics(df_clean["SurveyAnswer"], df_clean["ActualAnswer"])
                gm_a = compute_all_metrics(gdf_a["SurveyAnswer"], gdf_a["ActualAnswer"])
                after_d.append(gm_a["AP"] - om_a["AP"])
            else:
                after_d.append(0)
        x = np.arange(len(groups))
        w = 0.3
        ax.bar(x - w/2, before_d, w, label="Before Cleaning", color="#EF5350", alpha=0.7)
        ax.bar(x + w/2, after_d, w, label="After Cleaning", color="#66BB6A", alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(groups, rotation=15)
        ax.axhline(-0.1, color="red", linestyle=":", linewidth=0.8)
        ax.axhline(0.1, color="red", linestyle=":", linewidth=0.8)
        ax.axhline(0, color="black", linewidth=0.5)
        ax.set_title(gc, fontsize=10, fontweight="bold")
        ax.set_ylabel("AP Disparity")
        ax.legend(fontsize=8)
        # Annotate values
        for j in range(len(groups)):
            ax.text(j - w/2, before_d[j] + 0.003 * np.sign(before_d[j]),
                    f"{before_d[j]:+.3f}", ha="center", fontsize=7)
            ax.text(j + w/2, after_d[j] + 0.003 * np.sign(after_d[j]),
                    f"{after_d[j]:+.3f}", ha="center", fontsize=7)
    plt.suptitle("Before vs After Cleaning: AP Disparity per Group", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(COMP_DIR, "disparity_comparison.png"), dpi=150, bbox_inches="tight")
    plt.close()

    # Cleaning report viz
    fig, ax = plt.subplots(figsize=(10, 5))
    steps = [s["step"] for s in clean_report["steps"]]
    flagged = [s["rows_flagged"] for s in clean_report["steps"]]
    colors = ["#E53935" if f > 0 else "#90A4AE" for f in flagged]
    ax.barh(range(len(steps)), flagged, color=colors, alpha=0.8)
    ax.set_yticks(range(len(steps)))
    ax.set_yticklabels(steps, fontsize=9)
    ax.set_xlabel("Rows Flagged")
    ax.set_title(f"Data Cleaning Steps: {clean_report['total_rows_removed']} rows removed "
                 f"({clean_report['pct_removed']}%)", fontsize=11, fontweight="bold")
    for j, v in enumerate(flagged):
        if v > 0:
            ax.text(v + 0.5, j, str(v), va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(COMP_DIR, "cleaning_steps.png"), dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  -> Comparison saved to {os.path.abspath(COMP_DIR)}")


# =========================================================================
# MAIN
# =========================================================================

def main():
    hdr("FairPrep: Re-run Analysis on Cleaned Data")
    print(f"  Excel: {os.path.abspath(EXCEL_PATH)}")

    # Load raw
    print("\n  Loading raw data...")
    df_raw = load_excel_data(EXCEL_PATH)
    print(f"  Raw: {len(df_raw)} rows, {df_raw['ResponseId'].nunique()} annotators")

    # Clean
    hdr("DATA CLEANING")
    df_clean, clean_report = clean_data(df_raw)

    # Save cleaning report
    report_rows = []
    for s in clean_report["steps"]:
        report_rows.append(s)
    pd.DataFrame(report_rows).to_csv(os.path.join(COMP_DIR, "cleaning_report.csv"), index=False)

    # Phase 1 on cleaned data
    hdr("PHASE 1 EXPERIMENTS (CLEANED DATA)")
    print(f"  Working with {len(df_clean)} rows, {df_clean['ResponseId'].nunique()} annotators\n")
    p1_exp1_distributions(df_clean)
    p1_exp2_fairem_disparity(df_clean)
    p1_exp3_behavioral(df_clean)
    p1_exp4_majority(df_clean)
    p1_exp5_calibration(df_clean)
    p1_exp6_explanation(df_clean)
    p1_exp7_per_question(df_clean)
    p1_exp8_conditional(df_clean)
    p1_exp9_bootstrap(df_clean)

    # Phase 2 on cleaned data
    hdr("PHASE 2 EXPERIMENTS (CLEANED DATA)")
    a1 = p2_A1_intersectional(df_clean)
    a2 = p2_A2_leave_one_out(df_clean)
    a3 = p2_A3_ablation(df_clean)
    a4 = p2_A4_statistical(df_clean)
    a5 = p2_A5_regression(df_clean)
    a6 = p2_A6_question_interaction(df_clean)
    a7 = p2_A7_summary(df_clean, a1, a2, a3, a4, a5, a6)

    # Comparison
    generate_comparison(df_raw, df_clean, clean_report)

    hdr("ALL DONE")
    print(f"  Cleaned Phase 1: {os.path.abspath(P1_DIR)}")
    print(f"  Cleaned Phase 2: {os.path.abspath(P2_DIR)}")
    print(f"  Comparison:      {os.path.abspath(COMP_DIR)}")


if __name__ == "__main__":
    main()
