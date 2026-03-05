"""
Phase 1 + Synthetic TUS: Demographic Partitioning with Synthetic Data
=======================================================================
Extends the Phase 1 experiments by generating synthetic TUS datasets
from fair_entity_matching/synthetic dataset generator/ source data:

  - FacultyMatch (csranking.csv) → Gender as sensitive attribute
  - NoFlyCompas (compas-scores-raw.csv) → Ethnicity as sensitive attribute

Uses the same perturbation methods from the original generators but
adapted to table-level union search pairs. Then runs FairEM metrics
on the synthetic annotator decisions and compares with TUNE results.

Usage:
    python experiments/phase1_synthetic_tus.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.demographic_partitioning import (
    load_tune_data,
    apply_all_partitions,
    partition_linguistic,
    partition_expertise,
    partition_experience,
)
from src.measures import AP, TPR, FPR, PPV, calibration_gap, wasserstein_score_bias
from src.synthetic_tus_generator import (
    generate_faculty_tus_pairs,
    generate_compas_tus_pairs,
    simulate_annotator_decisions,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE = os.path.join(os.path.dirname(__file__), "..", "..")
FACULTY_CSV = os.path.join(BASE, "fair_entity_matching", "synthetic dataset generator",
                           "FacultyMatch", "csranking.csv")
COMPAS_CSV = os.path.join(BASE, "fair_entity_matching", "synthetic dataset generator",
                          "NoFlyCompas", "compas-scores-raw.csv")
TUNE_CSV = os.path.join(BASE, "TUNE_Benchmark", "data", "Feature_Engineered.csv")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "phase1_synthetic")
os.makedirs(RESULTS_DIR, exist_ok=True)

GROUP_COLUMNS = ["LinguisticGroup", "ExpertiseGroup", "ExperienceGroup"]
SEP = "=" * 70


def print_header(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ===========================================================================
# Step 1: Generate synthetic datasets
# ===========================================================================
def step1_generate(faculty_csv, compas_csv):
    print_header("Step 1: Generating Synthetic TUS Datasets")

    print(f"\n  Source: {faculty_csv}")
    faculty_pairs = generate_faculty_tus_pairs(faculty_csv, n_tables=30, table_size=20)
    print(f"  Generated {len(faculty_pairs)} FacultyMatch table-pairs")
    print(f"    Unionable: {(faculty_pairs['IsUnionable'] == 1).sum()}")
    print(f"    Non-unionable: {(faculty_pairs['IsUnionable'] == 0).sum()}")
    print(f"    Perturbation levels: {faculty_pairs['PerturbationLevel'].value_counts().to_dict()}")

    print(f"\n  Source: {compas_csv}")
    compas_pairs = generate_compas_tus_pairs(compas_csv, n_tables=30, table_size=50)
    print(f"  Generated {len(compas_pairs)} Compas table-pairs")
    print(f"    Unionable: {(compas_pairs['IsUnionable'] == 1).sum()}")
    print(f"    Non-unionable: {(compas_pairs['IsUnionable'] == 0).sum()}")
    print(f"    Compositions: {compas_pairs['Composition'].value_counts().to_dict()}")

    faculty_pairs.to_csv(os.path.join(RESULTS_DIR, "faculty_tus_pairs.csv"), index=False)
    compas_pairs.to_csv(os.path.join(RESULTS_DIR, "compas_tus_pairs.csv"), index=False)

    return faculty_pairs, compas_pairs


# ===========================================================================
# Step 2: Simulate annotator decisions
# ===========================================================================
def step2_simulate(faculty_pairs, compas_pairs):
    print_header("Step 2: Simulating Annotator Decisions")

    faculty_ann = simulate_annotator_decisions(faculty_pairs, n_annotators=30, seed=42)
    compas_ann = simulate_annotator_decisions(compas_pairs, n_annotators=30, seed=43)

    print(f"  FacultyMatch annotations: {len(faculty_ann)} rows, "
          f"{faculty_ann['ByWho'].nunique()} annotators")
    print(f"  Compas annotations: {len(compas_ann)} rows, "
          f"{compas_ann['ByWho'].nunique()} annotators")

    # Apply demographic partitions (same as TUNE Phase 1)
    for df in [faculty_ann, compas_ann]:
        df['LinguisticGroup'] = np.where(df['EngProf'] >= 4, 'Native', 'Non-Native')
        df['ExpertiseGroup'] = np.where(df['Major'] >= 4, 'STEM', 'Non-STEM')
        df['ExperienceGroup'] = np.where(df['Education'] >= 3, 'High-Edu', 'Lower-Edu')

    synthetic_all = pd.concat([faculty_ann, compas_ann], ignore_index=True)
    synthetic_all.to_csv(os.path.join(RESULTS_DIR, "synthetic_annotations.csv"), index=False)

    return faculty_ann, compas_ann, synthetic_all


# ===========================================================================
# Step 3: FairEM-style fairness audit on synthetic data
# ===========================================================================
def step3_fairness_audit(synthetic_df, label):
    print_header(f"Step 3: FairEM Metrics — {label}")

    results = []
    all_preds = synthetic_df["SurveyAnswer"].values
    all_labels = synthetic_df["ActualAnswer"].values
    overall_TP = int(np.sum((all_preds == 1) & (all_labels == 1)))
    overall_FP = int(np.sum((all_preds == 1) & (all_labels == 0)))
    overall_TN = int(np.sum((all_preds == 0) & (all_labels == 0)))
    overall_FN = int(np.sum((all_preds == 0) & (all_labels == 1)))
    overall_acc = AP(overall_TP, overall_FP, overall_TN, overall_FN)
    overall_tpr = TPR(overall_TP, overall_FP, overall_TN, overall_FN)
    overall_fpr = FPR(overall_TP, overall_FP, overall_TN, overall_FN)
    overall_ppv = PPV(overall_TP, overall_FP, overall_TN, overall_FN)

    print(f"  Overall: Acc={overall_acc:.4f}  TPR={overall_tpr:.4f}  "
          f"FPR={overall_fpr:.4f}  PPV={overall_ppv:.4f}  [n={len(synthetic_df)}]")

    for group_col in GROUP_COLUMNS:
        print(f"\n  --- {group_col} ---")
        for group_name, group_df in synthetic_df.groupby(group_col):
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

            acc_d = acc - overall_acc
            tpr_d = tpr - overall_tpr
            fpr_d = fpr - overall_fpr

            fair_flag = "FAIR" if abs(tpr_d) < 0.1 else "UNFAIR"

            print(f"  {group_name:15s}: Acc={acc:.4f} (d={acc_d:+.4f})  "
                  f"TPR={tpr:.4f} (d={tpr_d:+.4f}) [{fair_flag}]  "
                  f"FPR={fpr:.4f}  [n={len(group_df)}]")

            # Calibration gap
            conf = group_df["ConfidenceLevel"].values
            is_correct = (preds == labels).astype(float)
            cal_gap = calibration_gap(conf, is_correct)

            results.append({
                "Source": label,
                "Partition": group_col,
                "Group": group_name,
                "N": len(group_df),
                "TP": TP, "FP": FP, "TN": TN, "FN": FN,
                "Accuracy": round(acc, 4),
                "TPR": round(tpr, 4),
                "FPR": round(fpr, 4),
                "PPV": round(ppv, 4),
                "Accuracy_Disparity": round(acc_d, 4),
                "TPR_Disparity": round(tpr_d, 4),
                "FPR_Disparity": round(fpr_d, 4),
                "CalibrationGap": round(cal_gap, 4),
                "TPR_Fair": abs(tpr_d) < 0.1,
            })

    return pd.DataFrame(results)


# ===========================================================================
# Step 4: Fairness by perturbation level and demographic composition
# ===========================================================================
def step4_perturbation_analysis(synthetic_df, label):
    print_header(f"Step 4: Fairness x Perturbation Level — {label}")

    results = []
    for perturb, p_df in synthetic_df.groupby("PerturbationLevel"):
        for group_col in ["LinguisticGroup"]:  # Focus on linguistic for clarity
            for group_name, g_df in p_df.groupby(group_col):
                preds = g_df["SurveyAnswer"].values
                labels = g_df["ActualAnswer"].values
                TP = int(np.sum((preds == 1) & (labels == 1)))
                FP = int(np.sum((preds == 1) & (labels == 0)))
                TN = int(np.sum((preds == 0) & (labels == 0)))
                FN = int(np.sum((preds == 0) & (labels == 1)))
                acc = AP(TP, FP, TN, FN)
                tpr = TPR(TP, FP, TN, FN)

                results.append({
                    "Source": label,
                    "PerturbationLevel": perturb,
                    "Group": group_name,
                    "N": len(g_df),
                    "Accuracy": round(acc, 4),
                    "TPR": round(tpr, 4),
                })

                print(f"  Perturb={perturb:6s} | {group_name:12s}: "
                      f"Acc={acc:.4f}  TPR={tpr:.4f}  [n={len(g_df)}]")

    return pd.DataFrame(results)


# ===========================================================================
# Step 5: Compare TUNE vs Synthetic
# ===========================================================================
def step5_comparison(tune_results, synthetic_results):
    print_header("Step 5: TUNE vs Synthetic — Comparative Analysis")

    combined = pd.concat([tune_results, synthetic_results], ignore_index=True)
    combined.to_csv(os.path.join(RESULTS_DIR, "combined_fairness_comparison.csv"), index=False)

    print("\n  --- TPR Disparity Comparison ---")
    print(f"  {'Source':25s} {'Partition':20s} {'Group':15s} {'TPR_Disp':>10s} {'Fair?':>6s}")
    print("  " + "-" * 80)
    for _, row in combined.iterrows():
        fair = "YES" if row["TPR_Fair"] else "NO"
        print(f"  {row['Source']:25s} {row['Partition']:20s} {row['Group']:15s} "
              f"{row['TPR_Disparity']:+10.4f} {fair:>6s}")

    # Visualization: side-by-side TPR disparity
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    sources = combined["Source"].unique()
    colors = {"TUNE": "#1E88E5", "Synthetic-Faculty": "#43A047",
              "Synthetic-Compas": "#E53935", "Synthetic-All": "#FF9800"}

    for i, group_col in enumerate(GROUP_COLUMNS):
        ax = axes[i]
        sub = combined[combined["Partition"] == group_col]
        groups = sub["Group"].unique()
        x = np.arange(len(groups))
        width = 0.8 / len(sources)

        for j, src in enumerate(sources):
            src_data = sub[sub["Source"] == src]
            vals = [src_data[src_data["Group"] == g]["TPR_Disparity"].values[0]
                    if len(src_data[src_data["Group"] == g]) > 0 else 0
                    for g in groups]
            bars = ax.bar(x + j * width, vals, width,
                          label=src, color=colors.get(src, "#999999"), alpha=0.85)
            for k, v in enumerate(vals):
                ax.text(x[k] + j * width, v + 0.002 * np.sign(v),
                        f'{v:+.3f}', ha='center', fontsize=7, rotation=45)

        ax.set_title(f"TPR Disparity\n({group_col.replace('Group', '')})")
        ax.set_xticks(x + width * (len(sources) - 1) / 2)
        ax.set_xticklabels(groups)
        ax.set_ylabel("TPR Disparity from Overall")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.axhline(y=0.1, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.axhline(y=-0.1, color="red", linewidth=0.5, linestyle="--", alpha=0.5)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "tune_vs_synthetic_tpr_disparity.png"), dpi=150)
    plt.close()
    print("\n  -> Saved tune_vs_synthetic_tpr_disparity.png")

    # Calibration gap comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, group_col in enumerate(GROUP_COLUMNS):
        ax = axes[i]
        sub = combined[combined["Partition"] == group_col]
        groups = sub["Group"].unique()
        x = np.arange(len(groups))
        width = 0.8 / len(sources)

        for j, src in enumerate(sources):
            src_data = sub[sub["Source"] == src]
            vals = [src_data[src_data["Group"] == g]["CalibrationGap"].values[0]
                    if len(src_data[src_data["Group"] == g]) > 0 else 0
                    for g in groups]
            ax.bar(x + j * width, vals, width,
                   label=src, color=colors.get(src, "#999999"), alpha=0.85)
            for k, v in enumerate(vals):
                ax.text(x[k] + j * width, v + 0.005, f'{v:.3f}',
                        ha='center', fontsize=7, rotation=45)

        ax.set_title(f"Calibration Gap\n({group_col.replace('Group', '')})")
        ax.set_xticks(x + width * (len(sources) - 1) / 2)
        ax.set_xticklabels(groups)
        ax.set_ylabel("Calibration Gap (Conf - Acc)")
        ax.axhline(y=0, color="black", linewidth=0.5)
        ax.legend(fontsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "tune_vs_synthetic_calibration.png"), dpi=150)
    plt.close()
    print("  -> Saved tune_vs_synthetic_calibration.png")

    return combined


# ===========================================================================
# Main
# ===========================================================================
def main():
    print_header("FairPrep Phase 1 + Synthetic TUS Experiments")

    # --- Generate synthetic data ---
    faculty_pairs, compas_pairs = step1_generate(FACULTY_CSV, COMPAS_CSV)
    faculty_ann, compas_ann, synthetic_all = step2_simulate(faculty_pairs, compas_pairs)

    # --- FairEM metrics on synthetic data ---
    faculty_results = step3_fairness_audit(faculty_ann, "Synthetic-Faculty")
    compas_results = step3_fairness_audit(compas_ann, "Synthetic-Compas")

    # --- Perturbation-level analysis ---
    faculty_perturb = step4_perturbation_analysis(faculty_ann, "Synthetic-Faculty")
    compas_perturb = step4_perturbation_analysis(compas_ann, "Synthetic-Compas")
    perturb_all = pd.concat([faculty_perturb, compas_perturb], ignore_index=True)
    perturb_all.to_csv(os.path.join(RESULTS_DIR, "perturbation_analysis.csv"), index=False)

    # --- Load TUNE data for comparison ---
    print_header("Loading TUNE data for comparison...")
    tune_df = load_tune_data(TUNE_CSV)
    tune_df = apply_all_partitions(tune_df)

    # Run same FairEM metrics on TUNE
    tune_results = []
    all_preds = tune_df["SurveyAnswer"].values
    all_labels = tune_df["ActualAnswer"].values
    o_TP = int(np.sum((all_preds == 1) & (all_labels == 1)))
    o_FP = int(np.sum((all_preds == 1) & (all_labels == 0)))
    o_TN = int(np.sum((all_preds == 0) & (all_labels == 0)))
    o_FN = int(np.sum((all_preds == 0) & (all_labels == 1)))
    o_acc = AP(o_TP, o_FP, o_TN, o_FN)
    o_tpr = TPR(o_TP, o_FP, o_TN, o_FN)
    o_fpr = FPR(o_TP, o_FP, o_TN, o_FN)
    o_ppv = PPV(o_TP, o_FP, o_TN, o_FN)

    for group_col in GROUP_COLUMNS:
        for group_name, group_df in tune_df.groupby(group_col):
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
            conf = group_df["ConfidenceLevel"].values
            is_correct = (preds == labels).astype(float)

            tune_results.append({
                "Source": "TUNE",
                "Partition": group_col,
                "Group": group_name,
                "N": len(group_df),
                "TP": TP, "FP": FP, "TN": TN, "FN": FN,
                "Accuracy": round(acc, 4),
                "TPR": round(tpr, 4),
                "FPR": round(fpr, 4),
                "PPV": round(ppv, 4),
                "Accuracy_Disparity": round(acc - o_acc, 4),
                "TPR_Disparity": round(tpr - o_tpr, 4),
                "FPR_Disparity": round(fpr - o_fpr, 4),
                "CalibrationGap": round(calibration_gap(conf, is_correct), 4),
                "TPR_Fair": abs(tpr - o_tpr) < 0.1,
            })

    tune_results_df = pd.DataFrame(tune_results)
    synthetic_combined = pd.concat([faculty_results, compas_results], ignore_index=True)

    # --- Comparative analysis ---
    combined = step5_comparison(tune_results_df, synthetic_combined)

    print(f"\n{SEP}")
    print(f"  All results saved to: {os.path.abspath(RESULTS_DIR)}")
    print(SEP)


if __name__ == "__main__":
    main()
