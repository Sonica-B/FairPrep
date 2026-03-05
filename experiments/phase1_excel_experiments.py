"""
Phase 1: Demographic Partitioning — Experiments on Export_and_Compiled.xlsx
=============================================================================
Problem Statement 2: Demographic Fairness Direction for TUS

Runs Phase 1 fairness experiments on the new annotation study data:
  Exp 1 — Group Distribution Analysis (all 60 annotator STEM classifications)
  Exp 2 — Full 10-Metric FairEM Disparity Table (all of FairEM's measures.py)
  Exp 3 — Behavioral Signal Profiles + Per-Annotator Accuracy Distributions
  Exp 4 — Majority Vote Disagreement per Group
  Exp 5 — Calibration Gap (flagged as fairness concern when gap > 0.1)
  Exp 6 — Explanation Rate per Group
  Exp 7 — Per-Question Accuracy Disparity (difficulty x demographic)

Key refinements vs initial run (informed by memory of TUNE results):
  - All 10 FairEM metrics (was: 4). Adds SP, TNR, NPV, FDR, FOR.
  - Correct one-sided FairEM fairness test: higher-is-better metrics are
    UNFAIR when group_metric - overall < -0.1 (disadvantaged group);
    lower-is-better metrics UNFAIR when group_metric - overall > +0.1.
    (Previous: abs(d) > 0.1 was two-sided and missed the direction.)
  - Per-annotator accuracy box-plots in Exp 3 (Annotators-with-Attitudes insight).
  - DecisionTimeFract (FirstClick/DecisionTime) added to Exp 3 behavioral profile.
  - Calibration gap > 0.1 flagged as supplementary UNFAIR signal in Exp 5.
  - Summary cross-references TUNE Phase 1 results to identify replicating signals.

Fairness assessment threshold: |disparity| = 0.1 (from FairEM is_fair_measure_specific)

Four demographic partitions (vs three in original TUNE analysis):
  LinguisticGroup  : Native vs Non-Native
  ExpertiseGroup   : STEM vs Non-STEM
  ExperienceGroup  : High-Edu vs Lower-Edu
  AgeGroup         : Young-18-34 vs Older-35plus

Usage:
    cd FairPrep
    python experiments/phase1_excel_experiments.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.excel_data_loader import (
    load_excel_data,
    get_annotator_demographics,
    group_distribution_summary,
)
from src.measures import (
    AP, SP, TPR, FPR, FNR, TNR, PPV, NPV, FDR, FOR,
    HIGHER_IS_BETTER, LOWER_IS_BETTER,
    calibration_gap,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EXCEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "Export_and_Compiled.xlsx"
)
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "phase1_excel")
os.makedirs(RESULTS_DIR, exist_ok=True)

GROUP_COLUMNS = ["LinguisticGroup", "ExpertiseGroup", "ExperienceGroup", "AgeGroup"]
FAIRNESS_THRESHOLD = 0.1
SEP = "=" * 72

# All 10 FairEM metrics in display order (matching FairEMRepro/measures.py)
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

# TUNE Phase 1 results for cross-dataset comparison in summary
# (from CONVERSATION_EXPORT.md / architecture.md)
TUNE_RESULTS = {
    "LinguisticGroup": {
        "Native":     {"AP": 0.630, "TPR": 0.663},
        "Non-Native": {"AP": 0.604, "TPR": 0.579},
    },
    "ExpertiseGroup": {
        "STEM":     {"AP": 0.608, "TPR": 0.593},
        "Non-STEM": {"AP": 0.661, "TPR": 0.750},   # UNFAIR (TPR_d=+0.138)
    },
    "ExperienceGroup": {
        "High-Edu":  {"AP": 0.590, "TPR": 0.528},
        "Lower-Edu": {"AP": 0.625, "TPR": 0.650},
    },
}


def hdr(title):
    print(f"\n{SEP}\n  {title}\n{SEP}")


# ---------------------------------------------------------------------------
# FairEM one-sided fairness test (matches FairEM is_fair_measure_specific)
# ---------------------------------------------------------------------------

def is_unfair(disparity: float, direction: str, threshold: float = FAIRNESS_THRESHOLD) -> bool:
    """
    One-sided FairEM fairness test.
    Higher-is-better: UNFAIR if disparity < -threshold  (group worse than overall)
    Lower-is-better : UNFAIR if disparity > +threshold  (group worse than overall)
    """
    if direction == "higher":
        return disparity < -threshold
    else:
        return disparity > threshold


def fairness_label(disparity: float, direction: str) -> str:
    return "UNFAIR" if is_unfair(disparity, direction) else "fair  "


# ---------------------------------------------------------------------------
# Confusion matrix helper
# ---------------------------------------------------------------------------

def confusion_counts(preds, labels):
    p, l = np.asarray(preds), np.asarray(labels)
    TP = int(np.sum((p == 1) & (l == 1)))
    FP = int(np.sum((p == 1) & (l == 0)))
    TN = int(np.sum((p == 0) & (l == 0)))
    FN = int(np.sum((p == 0) & (l == 1)))
    return TP, FP, TN, FN


# ===========================================================================
# Experiment 1: Group Distribution Analysis
# ===========================================================================

def experiment_1_distributions(df):
    hdr("Experiment 1: Demographic Group Distributions")

    annotators = get_annotator_demographics(df)
    print(f"\n  Total unique annotators : {len(annotators)}")
    print(f"  Total annotation rows   : {len(df)}")
    print(f"  Questions per annotator : {df['QuestionNum'].nunique()}")

    print("\n  Education breakdown:")
    for _, r in annotators.groupby("DQ2_text").size().reset_index(name="n").iterrows():
        print(f"    {r['DQ2_text']:50s}: {r['n']:3d}")

    print("\n  English Proficiency breakdown:")
    for _, r in annotators.groupby("DQ3_text").size().reset_index(name="n").iterrows():
        print(f"    {r['DQ3_text']:20s}: {r['n']:3d}")

    # Full STEM classification for all 60 annotators
    print("\n  STEM classification (all annotators):")
    for _, r in annotators.sort_values(["ExpertiseGroup", "DQ4_text"]).iterrows():
        label = "STEM    " if r["Major"] == 1 else "Non-STEM"
        print(f"    [{label}] {r['DQ4_text']}")

    summary = group_distribution_summary(df, GROUP_COLUMNS)
    results = []
    for gc in GROUP_COLUMNS:
        print(f"\n  --- {gc} ---")
        counts = summary[gc]
        total = sum(counts.values())
        for name, cnt in sorted(counts.items()):
            pct = cnt / total * 100
            n_rows = len(df[df[gc] == name])
            print(f"    {name:20s}: {cnt:3d} annotators ({pct:5.1f}%), {n_rows:5d} annotations")
            results.append({"Partition": gc, "Group": name,
                            "N_Annotators": cnt, "Pct_Annotators": round(pct, 1),
                            "N_Annotations": n_rows})

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "exp1_group_distributions.csv"), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    colors = ["#1E88E5", "#FF8F00"]
    for i, gc in enumerate(GROUP_COLUMNS):
        sub = results_df[results_df["Partition"] == gc]
        axes[i].bar(sub["Group"], sub["N_Annotators"], color=colors)
        axes[i].set_title(gc, fontsize=11, fontweight="bold")
        axes[i].set_ylabel("Annotators")
        for j, (_, row) in enumerate(sub.iterrows()):
            axes[i].text(j, row["N_Annotators"] + 0.3,
                         f'{row["Pct_Annotators"]:.0f}%', ha="center", fontsize=10)
    plt.suptitle("Exp 1 — Demographic Group Distributions", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp1_group_distributions.png"), dpi=150)
    plt.close()
    print("\n  -> exp1_group_distributions.csv + .png")
    return results_df


# ===========================================================================
# Experiment 2: Full 10-Metric FairEM Disparity Table
# ===========================================================================

def experiment_2_accuracy(df):
    hdr("Experiment 2: Full 10-Metric FairEM Disparity Table")
    print("  Fairness test: one-sided (FairEM is_fair_measure_specific)")
    print("  Higher-is-better (AP,SP,TPR,TNR,PPV,NPV): UNFAIR if d < -0.10")
    print("  Lower-is-better  (FPR,FNR,FDR,FOR):       UNFAIR if d > +0.10")

    all_TP, all_FP, all_TN, all_FN = confusion_counts(
        df["SurveyAnswer"], df["ActualAnswer"])
    overall_vals = {name: fn(all_TP, all_FP, all_TN, all_FN)
                    for name, fn, _ in ALL_METRICS}
    print(f"\n  Overall: " +
          "  ".join(f"{n}={overall_vals[n]:.4f}" for n, _, _ in ALL_METRICS))

    results = []
    unfair_flags = []

    for gc in GROUP_COLUMNS:
        print(f"\n  --- {gc} ---")
        for gname, gdf in df.groupby(gc):
            TP, FP, TN, FN = confusion_counts(gdf["SurveyAnswer"], gdf["ActualAnswer"])
            row = {"Partition": gc, "Group": gname, "N": len(gdf),
                   "TP": TP, "FP": FP, "TN": TN, "FN": FN}
            flags = []
            for mname, mfn, direction in ALL_METRICS:
                val = mfn(TP, FP, TN, FN)
                disp = val - overall_vals[mname]
                row[mname] = round(val, 4)
                row[f"{mname}_d"] = round(disp, 4)
                row[f"{mname}_unfair"] = is_unfair(disp, direction)
                if is_unfair(disp, direction):
                    flags.append(f"{mname}(d={disp:+.3f})")
            results.append(row)

            # Console summary (key metrics)
            ap_d = row["AP_d"]; tpr_d = row["TPR_d"]; fpr_d = row["FPR_d"]
            ap_flag = fairness_label(ap_d, "higher")
            tpr_flag = fairness_label(tpr_d, "higher")
            print(f"    {gname:20s}: AP={row['AP']:.4f}(d={ap_d:+.4f})[{ap_flag}] "
                  f"TPR={row['TPR']:.4f}(d={tpr_d:+.4f})[{tpr_flag}] "
                  f"n={len(gdf)}")
            if flags:
                print(f"      ** UNFAIR metrics: {', '.join(flags)}")
                unfair_flags.append((gc, gname, flags))

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "exp2_group_accuracy.csv"), index=False)

    # Heatmap of all 10 metric disparities per group
    n_gc = len(GROUP_COLUMNS)
    fig, axes = plt.subplots(1, n_gc, figsize=(5 * n_gc, 7))
    import seaborn as sns
    metric_names = [m for m, _, _ in ALL_METRICS]
    directions = {m: d for m, _, d in ALL_METRICS}
    for i, gc in enumerate(GROUP_COLUMNS):
        sub = results_df[results_df["Partition"] == gc]
        disp_cols = [f"{m}_d" for m in metric_names]
        pivot = sub.set_index("Group")[disp_cols]
        pivot.columns = metric_names
        sns.heatmap(pivot.T, ax=axes[i], annot=True, fmt=".3f",
                    cmap="RdYlGn", center=0, vmin=-0.25, vmax=0.25,
                    linewidths=0.5, cbar_kws={"shrink": 0.6})
        axes[i].set_title(gc, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Group")
        axes[i].set_ylabel("Metric")
    plt.suptitle("Exp 2 — All 10 FairEM Metric Disparities (group - overall)\n"
                 "Green=advantaged, Red=disadvantaged, threshold=±0.10",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp2_group_accuracy.png"), dpi=150,
                bbox_inches="tight")
    plt.close()

    print(f"\n  UNFAIR groups detected: {len(unfair_flags)}")
    for gc, gname, flags in unfair_flags:
        print(f"    {gc} / {gname}: {'; '.join(flags)}")
    print("\n  -> exp2_group_accuracy.csv + .png")
    return results_df, unfair_flags


# ===========================================================================
# Experiment 3: Behavioral Signal Profiles + Per-Annotator Accuracy
# ===========================================================================

def experiment_3_behavioral(df):
    hdr("Experiment 3: Behavioral Profiles + Per-Annotator Accuracy Distributions")

    # Compute DecisionTimeFract (hesitation ratio)
    df = df.copy()
    df["DecisionTimeFract"] = np.where(
        df["DecisionTime"] > 0,
        df["FirstClick"] / df["DecisionTime"],
        np.nan
    )

    behavioral_cols = ["DecisionTime", "FirstClick", "LastClick",
                       "ClickCount", "ConfidenceLevel", "DecisionTimeFract"]
    results = []
    for gc in GROUP_COLUMNS:
        print(f"\n  --- {gc} ---")
        for gname, gdf in df.groupby(gc):
            row = {"Partition": gc, "Group": gname, "N": len(gdf)}
            for col in behavioral_cols:
                if col in gdf.columns:
                    vals = gdf[col].dropna()
                    row[f"{col}_mean"]   = round(float(vals.mean()), 4)
                    row[f"{col}_median"] = round(float(vals.median()), 4)
                    row[f"{col}_std"]    = round(float(vals.std()), 4)
            results.append(row)
            print(f"    {gname:20s}: DecTime={row.get('DecisionTime_mean',0):.1f}s  "
                  f"Clicks={row.get('ClickCount_mean',0):.2f}  "
                  f"Conf={row.get('ConfidenceLevel_mean',0):.1f}  "
                  f"TimeFract={row.get('DecisionTimeFract_mean',0):.3f}")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "exp3_behavioral_profiles.csv"), index=False)

    # Plot 1: Behavioral box plots
    plot_cols = ["DecisionTime", "ClickCount", "ConfidenceLevel", "DecisionTimeFract"]
    fig, axes = plt.subplots(len(GROUP_COLUMNS), 4,
                             figsize=(18, 3.5 * len(GROUP_COLUMNS)))
    for i, gc in enumerate(GROUP_COLUMNS):
        for j, col in enumerate(plot_cols):
            ax = axes[i][j]
            data_to_plot, labels = [], []
            for gname in sorted(df[gc].unique()):
                vals = df[df[gc] == gname][col].dropna().values
                data_to_plot.append(vals)
                labels.append(gname)
            ax.boxplot(data_to_plot, tick_labels=labels)
            ax.set_title(f"{col}\n({gc})", fontsize=8)
            ax.set_ylabel(col, fontsize=7)
            ax.tick_params(axis="x", labelsize=7)
    plt.suptitle("Exp 3 — Behavioral Signal Profiles (incl. DecisionTimeFract)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3_behavioral_profiles.png"), dpi=150)
    plt.close()

    # Plot 2: Per-annotator accuracy distribution (Annotators-with-Attitudes insight)
    per_ann = df.groupby(["ResponseId"] + GROUP_COLUMNS)["Accuracy"].mean().reset_index()

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    colors = {"Native": "#1E88E5", "Non-Native": "#FF8F00",
              "STEM": "#43A047", "Non-STEM": "#E53935",
              "High-Edu": "#8E24AA", "Lower-Edu": "#FB8C00",
              "Young-18-34": "#00ACC1", "Older-35plus": "#6D4C41"}
    for i, gc in enumerate(GROUP_COLUMNS):
        ax = axes[i]
        groups = sorted(per_ann[gc].unique())
        data_to_plot = [per_ann[per_ann[gc] == g]["Accuracy"].values for g in groups]
        bp = ax.boxplot(data_to_plot, tick_labels=groups, patch_artist=True)
        for patch, g in zip(bp["boxes"], groups):
            patch.set_facecolor(colors.get(g, "#90A4AE"))
            patch.set_alpha(0.7)
        # Overlay individual points
        for k, (g, vals) in enumerate(zip(groups, data_to_plot)):
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(vals))
            ax.scatter(np.full(len(vals), k + 1) + jitter, vals,
                       alpha=0.6, s=25, color=colors.get(g, "#90A4AE"), zorder=3)
        ax.set_title(f"Per-Annotator Accuracy\n({gc})", fontsize=10, fontweight="bold")
        ax.set_ylabel("Individual Accuracy (8 questions)")
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(per_ann["Accuracy"].mean(), linestyle="--", color="gray",
                   linewidth=0.8, label="Overall mean")
        ax.legend(fontsize=8)
        # Annotate n
        for k, g in enumerate(groups):
            n = len(per_ann[per_ann[gc] == g])
            ax.text(k + 1, 1.06, f"n={n}", ha="center", fontsize=8, color="gray")
    plt.suptitle("Exp 3 — Per-Annotator Accuracy Distribution by Demographic Group",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp3b_annotator_accuracy_dist.png"), dpi=150)
    plt.close()

    print("\n  -> exp3_behavioral_profiles.csv + .png + exp3b_annotator_accuracy_dist.png")
    return results_df


# ===========================================================================
# Experiment 4: Majority Vote Disagreement
# ===========================================================================

def experiment_4_majority(df):
    hdr("Experiment 4: Majority Vote Disagreement per Group")

    results = []
    for gc in GROUP_COLUMNS:
        print(f"\n  --- {gc} ---")
        for gname, gdf in df.groupby(gc):
            total = len(gdf)
            disagree_maj = int(np.sum(gdf["SurveyAnswer"] != gdf["Majority"]))
            error_rate   = int(np.sum(gdf["SurveyAnswer"] != gdf["ActualAnswer"]))
            maj_error    = int(np.sum(gdf["Majority"] != gdf["ActualAnswer"]))
            results.append({
                "Partition": gc, "Group": gname, "N": total,
                "DisagreesWithMajority": disagree_maj,
                "MajDisagreeRate":     round(disagree_maj / total, 4) if total else 0,
                "AnnotatorErrorRate":  round(error_rate   / total, 4) if total else 0,
                "MajorityErrorRate":   round(maj_error    / total, 4) if total else 0,
            })
            print(f"    {gname:20s}: MajDisagree={disagree_maj}/{total} "
                  f"({disagree_maj/total*100:.1f}%)  "
                  f"AnnotatorErr={error_rate/total*100:.1f}%  "
                  f"MajorityErr={maj_error/total*100:.1f}%")

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "exp4_majority_disagreement.csv"), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    for i, gc in enumerate(GROUP_COLUMNS):
        sub = results_df[results_df["Partition"] == gc].reset_index(drop=True)
        x = np.arange(len(sub))
        w = 0.3
        axes[i].bar(x - w / 2, sub["MajDisagreeRate"],    w,
                    label="Disagrees w/ Majority", color="#E53935")
        axes[i].bar(x + w / 2, sub["AnnotatorErrorRate"], w,
                    label="Annotator Error Rate",  color="#1E88E5")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(sub["Group"], rotation=12, ha="right", fontsize=9)
        axes[i].set_title(gc, fontsize=10, fontweight="bold")
        axes[i].set_ylabel("Rate")
        axes[i].legend(fontsize=8)
        for j, (_, row) in enumerate(sub.iterrows()):
            axes[i].text(j - w/2, row["MajDisagreeRate"]    + 0.005,
                         f'{row["MajDisagreeRate"]:.3f}',    ha="center", fontsize=8)
            axes[i].text(j + w/2, row["AnnotatorErrorRate"] + 0.005,
                         f'{row["AnnotatorErrorRate"]:.3f}', ha="center", fontsize=8)
    plt.suptitle("Exp 4 — Majority Vote Disagreement per Group", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp4_majority_disagreement.png"), dpi=150)
    plt.close()
    print("\n  -> exp4_majority_disagreement.csv + .png")
    return results_df


# ===========================================================================
# Experiment 5: Calibration Gap — now flagged as fairness concern when > 0.1
# ===========================================================================

def experiment_5_calibration(df):
    hdr("Experiment 5: Calibration Gap per Group  (gap > 0.10 = supplementary UNFAIR)")
    print("  Note: gap = mean(ConfidenceNorm - Accuracy).  TUNE calibration gaps:")
    print("    Non-Native +0.182, High-Edu +0.209 (largest), STEM +0.103")
    print("  Checking if Export data replicates these calibration unfairness signals.")

    results = []
    calib_unfair = []
    for gc in GROUP_COLUMNS:
        print(f"\n  --- {gc} ---")
        for gname, gdf in df.groupby(gc):
            conf = gdf["ConfidenceLevelNorm"].values
            acc  = (gdf["SurveyAnswer"] == gdf["ActualAnswer"]).astype(float).values
            gap  = calibration_gap(conf, acc)
            mean_conf = float(np.nanmean(conf))
            mean_acc  = float(np.mean(acc))

            unfair = abs(gap) > FAIRNESS_THRESHOLD
            flag = "[CALIB-UNFAIR]" if unfair else ""
            direction = "OVERCONFIDENT" if gap > 0 else "underconfident"
            print(f"    {gname:20s}: Conf={mean_conf:.3f}  Acc={mean_acc:.3f}  "
                  f"Gap={gap:+.4f} [{direction}] {flag}")
            results.append({
                "Partition": gc, "Group": gname, "N": len(gdf),
                "MeanConfidence": round(mean_conf, 4),
                "MeanAccuracy":   round(mean_acc, 4),
                "CalibrationGap": round(gap, 4),
                "IsOverconfident": gap > 0,
                "CalibFairFlag":  flag.strip() if flag else "fair",
            })
            if unfair:
                calib_unfair.append((gc, gname, gap))

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "exp5_calibration_gap.csv"), index=False)

    if calib_unfair:
        print(f"\n  Calibration UNFAIR signals ({len(calib_unfair)} groups, |gap| > 0.1):")
        for gc, gname, gap in calib_unfair:
            print(f"    ** {gname} ({gc}): gap={gap:+.4f}")
    else:
        print("\n  No calibration gaps exceed 0.10 threshold.")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    for i, gc in enumerate(GROUP_COLUMNS):
        sub = results_df[results_df["Partition"] == gc].reset_index(drop=True)
        x = np.arange(len(sub))
        w = 0.25
        axes[i].bar(x - w, sub["MeanConfidence"], w, label="Confidence", color="#42A5F5")
        axes[i].bar(x,     sub["MeanAccuracy"],   w, label="Accuracy",   color="#66BB6A")
        axes[i].bar(x + w, sub["CalibrationGap"], w, label="Gap",        color="#EF5350")
        axes[i].axhline(0, color="black", linewidth=0.6, linestyle="--")
        axes[i].axhline( FAIRNESS_THRESHOLD, color="orange", linewidth=0.8,
                         linestyle=":", label=f"+{FAIRNESS_THRESHOLD} threshold")
        axes[i].axhline(-FAIRNESS_THRESHOLD, color="orange", linewidth=0.8, linestyle=":")
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(sub["Group"], rotation=12, ha="right", fontsize=9)
        axes[i].set_title(gc, fontsize=10, fontweight="bold")
        axes[i].set_ylabel("Value (0-1 scale)")
        axes[i].legend(fontsize=7)
        for j, (_, row) in enumerate(sub.iterrows()):
            color = "red" if abs(row["CalibrationGap"]) > FAIRNESS_THRESHOLD else "black"
            axes[i].text(j + w, row["CalibrationGap"] + 0.005,
                         f'{row["CalibrationGap"]:+.3f}', ha="center",
                         fontsize=8, color=color, fontweight="bold" if color == "red" else "normal")
    plt.suptitle("Exp 5 — Calibration Gap (Confidence vs Accuracy)\n"
                 "Orange dotted = ±0.10 fairness threshold; red labels = CALIB-UNFAIR",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp5_calibration_gap.png"), dpi=150)
    plt.close()
    print("\n  -> exp5_calibration_gap.csv + .png")
    return results_df, calib_unfair


# ===========================================================================
# Experiment 6: Explanation Rate per Group
# ===========================================================================

def experiment_6_explanation_rate(df):
    hdr("Experiment 6: Explanation Rate per Group")

    results = []
    for gc in GROUP_COLUMNS:
        print(f"\n  --- {gc} ---")
        for gname, gdf in df.groupby(gc):
            n_exp  = int(gdf["IsExp"].sum())
            n_total = len(gdf)
            rate   = n_exp / n_total if n_total else 0
            acc_exp = float(gdf[gdf["IsExp"] == 1]["Accuracy"].mean()) if n_exp else np.nan
            acc_no  = float(gdf[gdf["IsExp"] == 0]["Accuracy"].mean()) if (n_total - n_exp) > 0 else np.nan
            print(f"    {gname:20s}: ExpRate={rate:.3f} ({n_exp}/{n_total})  "
                  f"Acc|Exp={acc_exp:.3f}  "
                  f"Acc|NoExp={'N/A' if np.isnan(acc_no) else f'{acc_no:.3f}'}")
            results.append({
                "Partition": gc, "Group": gname, "N": n_total,
                "N_Explained": n_exp, "ExplanationRate": round(rate, 4),
                "Acc_WithExplanation":    round(acc_exp, 4) if not np.isnan(acc_exp) else None,
                "Acc_WithoutExplanation": round(acc_no,  4) if not np.isnan(acc_no)  else None,
            })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "exp6_explanation_rate.csv"), index=False)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.flatten()
    colors_two = ["#AB47BC", "#FF7043"]
    for i, gc in enumerate(GROUP_COLUMNS):
        sub = results_df[results_df["Partition"] == gc].reset_index(drop=True)
        axes[i].bar(sub["Group"], sub["ExplanationRate"],
                    color=colors_two[:len(sub)])
        for j, (_, row) in enumerate(sub.iterrows()):
            axes[i].text(j, row["ExplanationRate"] + 0.002,
                         f'{row["ExplanationRate"]:.3f}', ha="center", fontsize=10)
        axes[i].set_title(gc, fontsize=10, fontweight="bold")
        axes[i].set_ylabel("Explanation Rate")
        axes[i].set_ylim(0, 1.05)
        axes[i].tick_params(axis="x", labelsize=9)
    plt.suptitle("Exp 6 — Explanation Rate per Group", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp6_explanation_rate.png"), dpi=150)
    plt.close()
    print("\n  -> exp6_explanation_rate.csv + .png")
    return results_df


# ===========================================================================
# Experiment 7: Per-Question Accuracy Disparity
# ===========================================================================

def experiment_7_per_question(df):
    hdr("Experiment 7: Per-Question Accuracy Disparity (Difficulty x Demographics)")

    results = []
    q_acc = df.groupby("QuestionNum")["Accuracy"].mean().rename("OverallAcc")

    for gc in GROUP_COLUMNS:
        for gname, gdf in df.groupby(gc):
            per_q = gdf.groupby("QuestionNum")["Accuracy"].mean()
            for q in range(1, 9):
                g_acc = per_q.get(q, np.nan)
                ov_acc = q_acc.get(q, np.nan)
                results.append({
                    "Partition": gc, "Group": gname, "QuestionNum": q,
                    "GroupAcc": round(g_acc, 4), "OverallAcc": round(ov_acc, 4),
                    "Disparity": round(g_acc - ov_acc, 4),
                    "N": int(gdf[gdf["QuestionNum"] == q].shape[0]),
                })

    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(RESULTS_DIR, "exp7_per_question_disparity.csv"), index=False)

    print("\n  Per-question overall accuracy:")
    for q in range(1, 9):
        ov = q_acc.get(q, np.nan)
        tier = "HARD" if ov < 0.65 else "easy"
        print(f"    Q{q}: {ov:.3f}  [{tier}]")

    print("\n  Largest group-level disparities by question:")
    for gc in GROUP_COLUMNS:
        sub = results_df[results_df["Partition"] == gc]
        max_row = sub.loc[sub["Disparity"].abs().idxmax()]
        print(f"    {gc:25s}: Q{int(max_row['QuestionNum'])}  "
              f"{max_row['Group']}  d={max_row['Disparity']:+.4f}")

    import seaborn as sns
    fig, axes = plt.subplots(1, len(GROUP_COLUMNS),
                             figsize=(5 * len(GROUP_COLUMNS), 7))
    for i, gc in enumerate(GROUP_COLUMNS):
        sub = results_df[results_df["Partition"] == gc]
        pivot = sub.pivot(index="QuestionNum", columns="Group", values="GroupAcc")
        sns.heatmap(pivot, ax=axes[i], annot=True, fmt=".2f",
                    cmap="RdYlGn", vmin=0.2, vmax=0.9, linewidths=0.5,
                    cbar_kws={"shrink": 0.7})
        axes[i].set_title(gc, fontsize=10, fontweight="bold")
        axes[i].set_xlabel("Group")
        axes[i].set_ylabel("Question")
    plt.suptitle("Exp 7 — Per-Question Accuracy by Demographic Group",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "exp7_per_question_disparity.png"), dpi=150)
    plt.close()
    print("\n  -> exp7_per_question_disparity.csv + .png")
    return results_df


# ===========================================================================
# Summary Report — with TUNE vs Export cross-dataset comparison
# ===========================================================================

def generate_summary(exp1, exp2_df, exp2_unfair, exp3, exp4, exp5_df, exp5_calib,
                     exp6, exp7, df):
    hdr("PHASE 1 SUMMARY: Demographic Fairness — Export_and_Compiled.xlsx")

    print("\n1. GROUP SIZES:")
    for _, r in exp1.iterrows():
        print(f"   {r['Partition']:25s} | {r['Group']:22s} | "
              f"{r['N_Annotators']:3.0f} annotators ({r['Pct_Annotators']:.0f}%)")

    # Count unfair flags per metric across all groups
    unfair_metric_counts = {}
    metric_names = [m for m, _, _ in ALL_METRICS]
    for m in metric_names:
        col = f"{m}_unfair"
        if col in exp2_df.columns:
            unfair_metric_counts[m] = int(exp2_df[col].sum())

    print("\n2. FULL 10-METRIC FAIRNESS ASSESSMENT (one-sided FairEM test):")
    for gc in GROUP_COLUMNS:
        sub = exp2_df[exp2_df["Partition"] == gc]
        for _, r in sub.iterrows():
            unfair_ms = [m for m in metric_names if r.get(f"{m}_unfair", False)]
            flag_str = f"UNFAIR[{','.join(unfair_ms)}]" if unfair_ms else "FAIR (all 10 metrics)"
            print(f"   {r['Partition']:25s} | {r['Group']:22s} | "
                  f"AP={r['AP']:.4f}  TPR={r['TPR']:.4f}  FPR={r['FPR']:.4f} | {flag_str}")

    print("\n3. CALIBRATION GAP FAIRNESS (supplementary, |gap| > 0.10):")
    for _, r in exp5_df.iterrows():
        flag = r.get("CalibFairFlag", "fair")
        print(f"   {r['Partition']:25s} | {r['Group']:22s} | "
              f"gap={r['CalibrationGap']:+.4f} [{flag}]")

    print("\n4. CROSS-DATASET COMPARISON — Export vs TUNE (3 common partitions):")
    print(f"   {'Partition':25s} | {'Group':20s} | {'Export AP':>10s} | {'TUNE AP':>8s} | "
          f"{'Export TPR':>10s} | {'TUNE TPR':>9s} | Replicates?")
    print("   " + "-" * 100)
    for gc, tune_groups in TUNE_RESULTS.items():
        sub = exp2_df[exp2_df["Partition"] == gc]
        for gname in sorted(tune_groups.keys()):
            row = sub[sub["Group"] == gname]
            if row.empty:
                continue
            row = row.iloc[0]
            tune_ap  = tune_groups[gname]["AP"]
            tune_tpr = tune_groups[gname]["TPR"]
            exp_ap   = row["AP"]
            exp_tpr  = row["TPR"]
            # Same direction of disparity?
            replicate = ""
            overall_ap  = df["Accuracy"].mean()
            overall_tpr = float((df["SurveyAnswer"] == df["ActualAnswer"]).mean())
            same_ap_dir  = (exp_ap - overall_ap) * (tune_ap - 0.614) > 0
            same_tpr_dir = (exp_tpr - overall_tpr) * (tune_tpr - 0.614) > 0
            replicate = "Yes(AP+TPR)" if same_ap_dir and same_tpr_dir else \
                        "Partial" if (same_ap_dir or same_tpr_dir) else "No"
            print(f"   {gc:25s} | {gname:20s} | {exp_ap:10.4f} | {tune_ap:8.4f} | "
                  f"{exp_tpr:10.4f} | {tune_tpr:9.4f} | {replicate}")

    print("\n5. KEY FINDINGS:")
    max_acc_row = exp2_df.loc[exp2_df["AP_d"].abs().idxmax()]
    max_tpr_row = exp2_df.loc[exp2_df["TPR_d"].abs().idxmax()]
    max_dis_row = exp4.loc[exp4["MajDisagreeRate"].idxmax()]
    max_gap_row = exp5_df.loc[exp5_df["CalibrationGap"].abs().idxmax()]
    print(f"   - Largest AP Disparity  : {max_acc_row['Group']} ({max_acc_row['Partition']}) "
          f"= {max_acc_row['AP_d']:+.4f}")
    print(f"   - Largest TPR Disparity : {max_tpr_row['Group']} ({max_tpr_row['Partition']}) "
          f"= {max_tpr_row['TPR_d']:+.4f}")
    print(f"   - Highest Maj Disagree  : {max_dis_row['Group']} ({max_dis_row['Partition']}) "
          f"= {max_dis_row['MajDisagreeRate']:.4f}")
    print(f"   - Largest Calib Gap     : {max_gap_row['Group']} ({max_gap_row['Partition']}) "
          f"= {max_gap_row['CalibrationGap']:+.4f}")

    if exp2_unfair:
        print(f"\n   Standard FairEM UNFAIR: {len(exp2_unfair)} group(s)")
        for gc, gname, flags in exp2_unfair:
            print(f"     *** {gname} ({gc}): {'; '.join(flags)}")
    else:
        print("\n   Standard FairEM: ALL groups FAIR across all 10 metrics.")

    if exp5_calib:
        print(f"\n   Calibration UNFAIR (supplementary): {len(exp5_calib)} group(s)")
        for gc, gname, gap in exp5_calib:
            print(f"     ** {gname} ({gc}): gap={gap:+.4f}")

    # Write summary file
    lines = [
        "Phase 1 (Export_and_Compiled.xlsx) — Summary Report",
        "=" * 55,
        f"Annotations: {len(df)} | Annotators: {df['ResponseId'].nunique()} "
        f"| Questions: {df['QuestionNum'].nunique()}",
        "",
        "10-Metric FairEM Assessment (one-sided, threshold=0.10):",
    ]
    for _, r in exp2_df.iterrows():
        unfair_ms = [m for m in metric_names if r.get(f"{m}_unfair", False)]
        lines.append(f"  {r['Partition']} / {r['Group']}: "
                     f"AP={r['AP']:.4f}(d={r['AP_d']:+.4f}) "
                     f"TPR={r['TPR']:.4f}(d={r['TPR_d']:+.4f}) "
                     f"| {'UNFAIR: '+','.join(unfair_ms) if unfair_ms else 'FAIR'}")
    lines += ["", "Calibration Gap (|gap|>0.10 flagged):"]
    for _, r in exp5_df.iterrows():
        lines.append(f"  {r['Partition']} / {r['Group']}: gap={r['CalibrationGap']:+.4f} "
                     f"[{r['CalibFairFlag']}]")

    with open(os.path.join(RESULTS_DIR, "phase1_summary.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"\n  -> phase1_summary.txt saved.")


# ===========================================================================
# Main
# ===========================================================================

def main():
    hdr("FairPrep Phase 1 (Refined): Demographic Experiments — Export_and_Compiled.xlsx")
    print(f"\n  Excel path : {os.path.abspath(EXCEL_PATH)}")
    print(f"  Results dir: {os.path.abspath(RESULTS_DIR)}")
    print(f"\n  FairEM metrics: all 10  (was: 4 in initial run)")
    print(f"  Fairness test : one-sided (FairEM is_fair_measure_specific)")

    print("\nLoading data...")
    df = load_excel_data(EXCEL_PATH)
    print(f"  Loaded {len(df)} rows | {df['ResponseId'].nunique()} annotators "
          f"| {df['QuestionNum'].nunique()} questions")

    exp1          = experiment_1_distributions(df)
    exp2_df, exp2_unfair = experiment_2_accuracy(df)
    exp3          = experiment_3_behavioral(df)
    exp4          = experiment_4_majority(df)
    exp5_df, exp5_calib = experiment_5_calibration(df)
    exp6          = experiment_6_explanation_rate(df)
    exp7          = experiment_7_per_question(df)

    generate_summary(exp1, exp2_df, exp2_unfair, exp3, exp4,
                     exp5_df, exp5_calib, exp6, exp7, df)

    print(f"\n{SEP}")
    print(f"  All results saved to: {os.path.abspath(RESULTS_DIR)}")
    print(SEP)


if __name__ == "__main__":
    main()
