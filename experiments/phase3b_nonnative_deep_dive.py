"""
Phase 3b: Deep Non-Native Bias Root Cause Analysis
====================================================
10 experiments (D1-D10) that exhaustively permute Non-Native with every
available feature to pinpoint exactly how linguistic background creates bias.

Uses Nina's performance-based difficulty (second survey) on cleaned data.

Usage:
    cd FairPrep
    python experiments/phase3b_nonnative_deep_dive.py
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr

from src.excel_data_loader import load_excel_data

EXCEL_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Export_and_Compiled.xlsx")
NINA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "Fairness", "Fairness",
                          "question_difficulty_summary_second_survey.csv")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "phase3b_nonnative_deep")
os.makedirs(RESULTS, exist_ok=True)

SEP = "=" * 72
def hdr(t): print(f"\n{SEP}\n  {t}\n{SEP}")

# ── Load & Clean ─────────────────────────────────────────────────────
def load():
    df = load_excel_data(EXCEL_PATH)
    q1, q3 = df["DecisionTime"].quantile(0.25), df["DecisionTime"].quantile(0.75)
    df = df[df["DecisionTime"] <= q3 + 3 * (q3 - q1)]
    df = df[~((df["DecisionTime"] < 3) & (df["ClickCount"] <= 1))]
    acc = df.groupby("ResponseId")["Accuracy"].mean()
    df = df[~df["ResponseId"].isin(acc[acc <= 0.25].index)]
    for rid in list(df["ResponseId"].unique()):
        a = df[df["ResponseId"] == rid]["SurveyAnswer"].values
        if len(set(a)) == 1 and len(a) >= 8:
            df = df[df["ResponseId"] != rid]
    df["ConfidenceLevel"] = df["ConfidenceLevel"].clip(5, 95)
    df["ConfidenceLevelNorm"] = df["ConfidenceLevel"] / 100.0

    nina = pd.read_csv(NINA_PATH)[["SurveyVersion", "QuestionNum", "difficulty_performance"]]
    nina = nina.rename(columns={"difficulty_performance": "DiffPerf"})
    df = df.merge(nina, on=["SurveyVersion", "QuestionNum"], how="left")
    df["ExplLen"] = df["Explanations"].fillna("").str.len()
    return df


# ══════════════════════════════════════════════════════════════════════
# D1: EngProf Granularity (3-level)
# ══════════════════════════════════════════════════════════════════════
def d1_engprof_granularity(df):
    hdr("D1: EngProf 3-Level Granularity (Proficient / Fluent / Native)")
    prof_map = {3: "Proficient", 4: "Fluent", 5: "Native"}
    df["EngProfLabel"] = df["EngProf"].map(prof_map)

    rows = []
    for tier in ["Hard", "Medium", "Easy"]:
        t = df[df["DiffPerf"] == tier]
        ov = t["Accuracy"].mean()
        for lbl in ["Proficient", "Fluent", "Native"]:
            s = t[t["EngProfLabel"] == lbl]
            if len(s) == 0: continue
            acc = s["Accuracy"].mean()
            rows.append({"Tier": tier, "EngProf": lbl, "N": len(s),
                         "Accuracy": round(acc, 4), "Disparity": round(acc - ov, 4),
                         "N_Ann": s["ResponseId"].nunique()})
            print(f"  {tier:6s} {lbl:12s}: acc={acc:.3f}(d={acc-ov:+.3f}) n={len(s)} ann={s['ResponseId'].nunique()}")
    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(RESULTS, "D1_engprof_granularity.csv"), index=False)

    fig, ax = plt.subplots(figsize=(10, 6))
    for lbl, color in [("Proficient", "#EF5350"), ("Fluent", "#FFA726"), ("Native", "#1E88E5")]:
        s = rdf[rdf["EngProf"] == lbl]
        ax.plot(s["Tier"], s["Accuracy"], "o-", label=f"{lbl} (n_ann={s['N_Ann'].iloc[0]})",
                color=color, linewidth=2, markersize=8)
        for _, r in s.iterrows():
            ax.annotate(f"{r['Accuracy']:.1%}", (r["Tier"], r["Accuracy"]),
                        textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
    ax.set_ylabel("Accuracy"); ax.set_xlabel("Difficulty Tier (Performance-Based)")
    ax.set_title("D1 -- EngProf 3-Level: Proficient vs Fluent vs Native by Difficulty", fontweight="bold")
    ax.legend(); ax.set_ylim(0, 1.05)
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "D1_engprof_granularity.png"), dpi=150); plt.close()
    return rdf


# ══════════════════════════════════════════════════════════════════════
# D2: Per-Annotator Profiles (all 11 Non-Native)
# ══════════════════════════════════════════════════════════════════════
def d2_per_annotator(df):
    hdr("D2: Per Non-Native Annotator Profile")
    nn = df[df["LinguisticGroup"] == "Non-Native"]
    rows = []
    for rid in sorted(nn["ResponseId"].unique()):
        a = nn[nn["ResponseId"] == rid]
        info = a.iloc[0]
        profile = {
            "Annotator": rid[:20], "Version": int(info["SurveyVersion"]),
            "Age": info["DQ1_text"], "Edu": info["DQ2_text"],
            "EngProf": info["DQ3_text"], "Major": info["DQ4_text"][:25],
            "AgeGroup": info["AgeGroup"], "EduGroup": info["ExperienceGroup"],
            "STEMGroup": info["ExpertiseGroup"],
            "Overall_Acc": round(a["Accuracy"].mean(), 3),
            "Hard_Acc": round(a[a["DiffPerf"] == "Hard"]["Accuracy"].mean(), 3)
                if len(a[a["DiffPerf"] == "Hard"]) > 0 else None,
            "Med_Acc": round(a[a["DiffPerf"] == "Medium"]["Accuracy"].mean(), 3)
                if len(a[a["DiffPerf"] == "Medium"]) > 0 else None,
            "Easy_Acc": round(a[a["DiffPerf"] == "Easy"]["Accuracy"].mean(), 3)
                if len(a[a["DiffPerf"] == "Easy"]) > 0 else None,
            "Yes_Rate": round((a["SurveyAnswer"] == 1).mean(), 3),
            "Hard_Yes_Rate": round((a[a["DiffPerf"] == "Hard"]["SurveyAnswer"] == 1).mean(), 3)
                if len(a[a["DiffPerf"] == "Hard"]) > 0 else None,
            "Avg_Confidence": round(a["ConfidenceLevelNorm"].mean(), 3),
            "Avg_DecTime": round(a["DecisionTime"].mean(), 1),
            "Avg_ExplLen": round(a["ExplLen"].mean(), 0),
        }
        rows.append(profile)
    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(RESULTS, "D2_per_annotator.csv"), index=False)

    print(f"\n  {'Annotator':20s} V  {'Age':5s} {'Edu':7s} {'STEM':7s} Overall Hard   Med  Easy  Yes%H Conf")
    for _, r in rdf.iterrows():
        h = f"{r['Hard_Acc']:.0%}" if r["Hard_Acc"] is not None else "N/A"
        m = f"{r['Med_Acc']:.0%}" if r["Med_Acc"] is not None else "N/A"
        e = f"{r['Easy_Acc']:.0%}" if r["Easy_Acc"] is not None else "N/A"
        hy = f"{r['Hard_Yes_Rate']:.0%}" if r["Hard_Yes_Rate"] is not None else "N/A"
        print(f"  {r['Annotator']:20s} {r['Version']}  {r['AgeGroup'][:5]:5s} {r['EduGroup'][:7]:7s} "
              f"{r['STEMGroup'][:7]:7s} {r['Overall_Acc']:.0%}    {h:4s}  {m:4s} {e:4s}  {hy:4s} {r['Avg_Confidence']:.0%}")

    # Heatmap: annotator x tier
    fig, ax = plt.subplots(figsize=(10, 6))
    heat_data = rdf.set_index("Annotator")[["Hard_Acc", "Med_Acc", "Easy_Acc", "Overall_Acc"]]
    heat_data.columns = ["Hard", "Medium", "Easy", "Overall"]
    sns.heatmap(heat_data, annot=True, fmt=".0%", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.7})
    ax.set_title("D2 -- Per Non-Native Annotator Accuracy by Difficulty Tier", fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "D2_per_annotator.png"), dpi=150); plt.close()
    return rdf


# ══════════════════════════════════════════════════════════════════════
# D3: Answer Pattern Analysis (Yes/No x Difficulty x ActualAnswer)
# ══════════════════════════════════════════════════════════════════════
def d3_answer_pattern(df):
    hdr("D3: Answer Pattern -- False Positive vs False Negative Decomposition")
    rows = []
    for grp in ["Non-Native", "Native"]:
        g = df[df["LinguisticGroup"] == grp]
        for tier in ["Hard", "Medium", "Easy"]:
            t = g[g["DiffPerf"] == tier]
            if len(t) == 0: continue
            tp = int(((t["SurveyAnswer"] == 1) & (t["ActualAnswer"] == 1)).sum())
            fp = int(((t["SurveyAnswer"] == 1) & (t["ActualAnswer"] == 0)).sum())
            tn = int(((t["SurveyAnswer"] == 0) & (t["ActualAnswer"] == 0)).sum())
            fn = int(((t["SurveyAnswer"] == 0) & (t["ActualAnswer"] == 1)).sum())
            total = tp + fp + tn + fn
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            fnr = fn / (tp + fn) if (tp + fn) > 0 else 0
            tnr = tn / (fp + tn) if (fp + tn) > 0 else 0
            yes_rate = (tp + fp) / total if total > 0 else 0
            actual_yes = (tp + fn) / total if total > 0 else 0
            rows.append({"Group": grp, "Tier": tier, "TP": tp, "FP": fp, "TN": tn, "FN": fn,
                         "N": total, "TPR": round(tpr, 4), "FPR": round(fpr, 4),
                         "FNR": round(fnr, 4), "TNR": round(tnr, 4),
                         "Yes_Rate": round(yes_rate, 4), "Actual_Yes_Rate": round(actual_yes, 4),
                         "Yes_Bias": round(yes_rate - actual_yes, 4)})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(RESULTS, "D3_answer_pattern.csv"), index=False)

    print(f"\n  {'Group':12s} {'Tier':6s} | TP  FP  TN  FN | TPR    FNR    FPR    TNR   | Yes%  ActY% Bias")
    for _, r in rdf.iterrows():
        print(f"  {r['Group']:12s} {r['Tier']:6s} | {r['TP']:3d} {r['FP']:3d} {r['TN']:3d} {r['FN']:3d} | "
              f"{r['TPR']:.3f}  {r['FNR']:.3f}  {r['FPR']:.3f}  {r['TNR']:.3f} | "
              f"{r['Yes_Rate']:.3f} {r['Actual_Yes_Rate']:.3f} {r['Yes_Bias']:+.3f}")

    # Visualization: stacked error bars
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, metric in enumerate(["TPR", "FNR", "FPR"]):
        ax = axes[i]
        for grp, color in [("Non-Native", "#FF8F00"), ("Native", "#1E88E5")]:
            s = rdf[rdf["Group"] == grp]
            ax.bar([f"{t}\n({grp[:3]})" for t in s["Tier"]], s[metric], color=color,
                   alpha=0.8, label=grp, width=0.35,
                   align="edge" if grp == "Non-Native" else "center")
        labels = {"TPR": "True Positive Rate\n(detecting unionable)", "FNR": "False Negative Rate\n(missing unionable)",
                  "FPR": "False Positive Rate\n(false alarm)"}
        ax.set_title(labels.get(metric, metric), fontweight="bold")
        ax.set_ylabel(metric); ax.legend(fontsize=8)
    # Re-do with grouped bars properly
    plt.close()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    tiers = ["Hard", "Medium", "Easy"]
    for ax_i, (metric, title) in enumerate([("TPR", "True Positive Rate\n(correctly identify unionable)"),
                                             ("FNR", "False Negative Rate\n(miss unionable = say No when Yes)"),
                                             ("FPR", "False Positive Rate\n(false alarm = say Yes when No)"),
                                             ("Yes_Bias", "Yes-Rate Bias\n(tendency to say Yes minus actual Yes rate)")]):
        ax = axes.flat[ax_i]
        x = np.arange(len(tiers))
        w = 0.35
        nn_vals = [rdf[(rdf["Group"] == "Non-Native") & (rdf["Tier"] == t)][metric].values[0]
                   if len(rdf[(rdf["Group"] == "Non-Native") & (rdf["Tier"] == t)]) > 0 else 0 for t in tiers]
        nat_vals = [rdf[(rdf["Group"] == "Native") & (rdf["Tier"] == t)][metric].values[0]
                    if len(rdf[(rdf["Group"] == "Native") & (rdf["Tier"] == t)]) > 0 else 0 for t in tiers]
        ax.bar(x - w/2, nn_vals, w, label="Non-Native", color="#FF8F00", alpha=0.85)
        ax.bar(x + w/2, nat_vals, w, label="Native", color="#1E88E5", alpha=0.85)
        for j in range(len(tiers)):
            ax.text(x[j] - w/2, nn_vals[j] + 0.02, f"{nn_vals[j]:.2f}", ha="center", fontsize=8, fontweight="bold")
            ax.text(x[j] + w/2, nat_vals[j] + 0.02, f"{nat_vals[j]:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(tiers)
        ax.set_title(title, fontsize=10, fontweight="bold"); ax.legend(fontsize=8)
        if metric == "Yes_Bias": ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    plt.suptitle("D3 -- Error Decomposition: Non-Native vs Native by Difficulty", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "D3_answer_pattern.png"), dpi=150); plt.close()
    return rdf


# ══════════════════════════════════════════════════════════════════════
# D4: Confidence Calibration Curves
# ══════════════════════════════════════════════════════════════════════
def d4_calibration(df):
    hdr("D4: Confidence Calibration Curves by Difficulty")
    bins = [0, 0.4, 0.6, 0.7, 0.8, 0.9, 1.01]
    bin_labels = ["<40%", "40-60%", "60-70%", "70-80%", "80-90%", "90-100%"]
    rows = []
    for grp in ["Non-Native", "Native"]:
        g = df[df["LinguisticGroup"] == grp]
        for tier in ["Hard", "Medium", "Easy"]:
            t = g[g["DiffPerf"] == tier]
            if len(t) == 0: continue
            t = t.copy()
            t["ConfBin"] = pd.cut(t["ConfidenceLevelNorm"], bins=bins, labels=bin_labels, right=False)
            for bl in bin_labels:
                b = t[t["ConfBin"] == bl]
                if len(b) == 0: continue
                rows.append({"Group": grp, "Tier": tier, "ConfBin": bl,
                             "N": len(b), "MeanConf": round(b["ConfidenceLevelNorm"].mean(), 3),
                             "MeanAcc": round(b["Accuracy"].mean(), 3),
                             "Gap": round(b["ConfidenceLevelNorm"].mean() - b["Accuracy"].mean(), 3)})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(RESULTS, "D4_calibration.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, tier in enumerate(["Hard", "Medium", "Easy"]):
        ax = axes[i]
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, label="Perfect calibration")
        for grp, color, marker in [("Non-Native", "#FF8F00", "o"), ("Native", "#1E88E5", "s")]:
            s = rdf[(rdf["Group"] == grp) & (rdf["Tier"] == tier)]
            if len(s) > 0:
                ax.plot(s["MeanConf"], s["MeanAcc"], f"{marker}-", color=color, label=grp,
                        linewidth=2, markersize=7, alpha=0.85)
        ax.set_xlabel("Mean Confidence"); ax.set_ylabel("Mean Accuracy")
        ax.set_title(f"{tier} Questions", fontweight="bold")
        ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    plt.suptitle("D4 -- Calibration Curves: Confidence vs Actual Accuracy\n(Points above diagonal = overconfident)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "D4_calibration.png"), dpi=150); plt.close()
    return rdf


# ══════════════════════════════════════════════════════════════════════
# D5: Decision Time on Correct vs Incorrect
# ══════════════════════════════════════════════════════════════════════
def d5_time_pattern(df):
    hdr("D5: Decision Time -- Correct vs Incorrect by Group & Difficulty")
    rows = []
    for grp in ["Non-Native", "Native"]:
        g = df[df["LinguisticGroup"] == grp]
        for tier in ["Hard", "Medium", "Easy"]:
            t = g[g["DiffPerf"] == tier]
            for correct in [0, 1]:
                s = t[t["Accuracy"] == correct]
                if len(s) == 0: continue
                rows.append({"Group": grp, "Tier": tier, "Correct": bool(correct),
                             "N": len(s), "MeanTime": round(s["DecisionTime"].mean(), 1),
                             "MedianTime": round(s["DecisionTime"].median(), 1),
                             "MeanClicks": round(s["ClickCount"].mean(), 2),
                             "MeanConf": round(s["ConfidenceLevelNorm"].mean(), 3)})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(RESULTS, "D5_time_pattern.csv"), index=False)

    print(f"\n  {'Group':12s} {'Tier':6s} {'Correct':7s} | N   Time   Clicks Conf")
    for _, r in rdf.iterrows():
        print(f"  {r['Group']:12s} {r['Tier']:6s} {'Yes' if r['Correct'] else 'No ':3s}     | "
              f"{r['N']:3d} {r['MeanTime']:6.1f}s {r['MeanClicks']:5.2f}  {r['MeanConf']:.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, tier in enumerate(["Hard", "Medium", "Easy"]):
        ax = axes[i]
        sub = rdf[rdf["Tier"] == tier]
        x = np.arange(2)
        w = 0.35
        for j, (grp, color) in enumerate([("Non-Native", "#FF8F00"), ("Native", "#1E88E5")]):
            gs = sub[sub["Group"] == grp]
            correct_time = gs[gs["Correct"] == True]["MeanTime"].values
            incorrect_time = gs[gs["Correct"] == False]["MeanTime"].values
            vals = [incorrect_time[0] if len(incorrect_time) > 0 else 0,
                    correct_time[0] if len(correct_time) > 0 else 0]
            bars = ax.bar(x + (j - 0.5) * w, vals, w, label=grp, color=color, alpha=0.85)
            for k, v in enumerate(vals):
                ax.text(x[k] + (j - 0.5) * w, v + 0.5, f"{v:.0f}s", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(["Incorrect", "Correct"])
        ax.set_title(f"{tier} Questions", fontweight="bold"); ax.set_ylabel("Decision Time (s)")
        ax.legend(fontsize=8)
    plt.suptitle("D5 -- Decision Time: Do Non-Native Spend More/Less Time on Errors?", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "D5_time_pattern.png"), dpi=150); plt.close()
    return rdf


# ══════════════════════════════════════════════════════════════════════
# D6: Explanation Length Analysis
# ══════════════════════════════════════════════════════════════════════
def d6_explanation_length(df):
    hdr("D6: Explanation Length -- Reasoning Depth by Group, Difficulty, Correctness")
    rows = []
    for grp in ["Non-Native", "Native"]:
        g = df[df["LinguisticGroup"] == grp]
        for tier in ["Hard", "Medium", "Easy"]:
            t = g[g["DiffPerf"] == tier]
            for correct in [0, 1]:
                s = t[t["Accuracy"] == correct]
                if len(s) == 0: continue
                el = s["ExplLen"]
                rows.append({"Group": grp, "Tier": tier, "Correct": bool(correct),
                             "N": len(s), "MeanLen": round(el.mean(), 1),
                             "MedianLen": round(el.median(), 1),
                             "StdLen": round(el.std(), 1)})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(RESULTS, "D6_explanation_length.csv"), index=False)

    print(f"\n  {'Group':12s} {'Tier':6s} {'Correct':7s} | N   MeanLen MedianLen")
    for _, r in rdf.iterrows():
        print(f"  {r['Group']:12s} {r['Tier']:6s} {'Yes' if r['Correct'] else 'No ':3s}     | "
              f"{r['N']:3d} {r['MeanLen']:7.1f}  {r['MedianLen']:8.1f}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for i, tier in enumerate(["Hard", "Medium", "Easy"]):
        ax = axes[i]
        sub = rdf[rdf["Tier"] == tier]
        x = np.arange(2); w = 0.35
        for j, (grp, color) in enumerate([("Non-Native", "#FF8F00"), ("Native", "#1E88E5")]):
            gs = sub[sub["Group"] == grp]
            inc = gs[gs["Correct"] == False]["MeanLen"].values
            cor = gs[gs["Correct"] == True]["MeanLen"].values
            vals = [inc[0] if len(inc) > 0 else 0, cor[0] if len(cor) > 0 else 0]
            ax.bar(x + (j - 0.5) * w, vals, w, label=grp, color=color, alpha=0.85)
            for k, v in enumerate(vals):
                ax.text(x[k] + (j - 0.5) * w, v + 2, f"{v:.0f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(["Incorrect", "Correct"])
        ax.set_title(f"{tier} Questions", fontweight="bold"); ax.set_ylabel("Mean Explanation Length (chars)")
        ax.legend(fontsize=8)
    plt.suptitle("D6 -- Explanation Length: Do Non-Native Write Less When Wrong?", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "D6_explanation_length.png"), dpi=150); plt.close()
    return rdf


# ══════════════════════════════════════════════════════════════════════
# D7: Per Hard-Question Deep Drill
# ══════════════════════════════════════════════════════════════════════
def d7_per_question_drill(df):
    hdr("D7: Per Hard-Question Drill -- NN vs Native on Each Item")
    rows = []
    for v in sorted(df["SurveyVersion"].unique()):
        for q in range(1, 9):
            cell = df[(df["SurveyVersion"] == v) & (df["QuestionNum"] == q)]
            if len(cell) == 0: continue
            diff = cell["DiffPerf"].iloc[0]
            if diff != "Hard": continue
            actual = int(cell["ActualAnswer"].iloc[0])
            nn = cell[cell["LinguisticGroup"] == "Non-Native"]
            nat = cell[cell["LinguisticGroup"] == "Native"]
            nn_acc = nn["Accuracy"].mean() if len(nn) > 0 else None
            nat_acc = nat["Accuracy"].mean() if len(nat) > 0 else None
            nn_yes = (nn["SurveyAnswer"] == 1).mean() if len(nn) > 0 else None
            nat_yes = (nat["SurveyAnswer"] == 1).mean() if len(nat) > 0 else None
            nn_conf = nn["ConfidenceLevelNorm"].mean() if len(nn) > 0 else None
            nn_time = nn["DecisionTime"].mean() if len(nn) > 0 else None
            nn_explen = nn["ExplLen"].mean() if len(nn) > 0 else None
            rows.append({"Version": v, "Question": q, "Actual": "Yes" if actual else "No",
                         "NN_N": len(nn), "NN_Acc": round(nn_acc, 3) if nn_acc is not None else None,
                         "NN_YesRate": round(nn_yes, 3) if nn_yes is not None else None,
                         "Nat_N": len(nat), "Nat_Acc": round(nat_acc, 3) if nat_acc is not None else None,
                         "Nat_YesRate": round(nat_yes, 3) if nat_yes is not None else None,
                         "NN_Conf": round(nn_conf, 3) if nn_conf is not None else None,
                         "NN_Time": round(nn_time, 1) if nn_time is not None else None,
                         "NN_ExplLen": round(nn_explen, 0) if nn_explen is not None else None})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(RESULTS, "D7_per_question_drill.csv"), index=False)

    print(f"\n  {'V.Q':5s} Actual | NN_Acc NN_Yes Nat_Acc Nat_Yes | NN_Conf NN_Time NN_Expl")
    for _, r in rdf.iterrows():
        print(f"  V{r['Version']}Q{r['Question']} {r['Actual']:3s}    | "
              f"{r['NN_Acc']:.0%}    {r['NN_YesRate']:.0%}    {r['Nat_Acc']:.0%}     {r['Nat_YesRate']:.0%}    | "
              f"{r['NN_Conf']:.0%}     {r['NN_Time']:.0f}s    {r['NN_ExplLen']:.0f}ch")

    fig, ax = plt.subplots(figsize=(14, 6))
    labels = [f"V{r['Version']}Q{r['Question']}\n(Ans={r['Actual']})" for _, r in rdf.iterrows()]
    x = np.arange(len(rdf)); w = 0.35
    ax.bar(x - w/2, rdf["NN_Acc"].fillna(0), w, label="Non-Native", color="#FF8F00", alpha=0.85)
    ax.bar(x + w/2, rdf["Nat_Acc"].fillna(0), w, label="Native", color="#1E88E5", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.5)
    ax.set_ylabel("Accuracy"); ax.set_title("D7 -- Per Hard-Question: NN vs Native Accuracy", fontweight="bold")
    ax.legend()
    for j in range(len(rdf)):
        nn_v = rdf.iloc[j]["NN_Acc"]
        if nn_v is not None:
            ax.text(x[j] - w/2, nn_v + 0.02, f"{nn_v:.0%}", ha="center", fontsize=8, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "D7_per_question_drill.png"), dpi=150); plt.close()
    return rdf


# ══════════════════════════════════════════════════════════════════════
# D8: Temporal Pattern (Q1-Q8 sequence)
# ══════════════════════════════════════════════════════════════════════
def d8_temporal(df):
    hdr("D8: Temporal Pattern -- Accuracy by Question Sequence Position")
    rows = []
    for grp in ["Non-Native", "Native"]:
        g = df[df["LinguisticGroup"] == grp]
        for q in range(1, 9):
            s = g[g["QuestionNum"] == q]
            if len(s) == 0: continue
            rows.append({"Group": grp, "QuestionPos": q,
                         "Accuracy": round(s["Accuracy"].mean(), 4), "N": len(s),
                         "MeanConf": round(s["ConfidenceLevelNorm"].mean(), 3),
                         "MeanTime": round(s["DecisionTime"].mean(), 1)})
    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(RESULTS, "D8_temporal.csv"), index=False)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, (col, ylabel) in enumerate([("Accuracy", "Accuracy"), ("MeanConf", "Confidence"), ("MeanTime", "Time (s)")]):
        ax = axes[i]
        for grp, color in [("Non-Native", "#FF8F00"), ("Native", "#1E88E5")]:
            s = rdf[rdf["Group"] == grp]
            ax.plot(s["QuestionPos"], s[col], "o-", color=color, label=grp, linewidth=2, markersize=6)
        ax.set_xlabel("Question Position (1-8)"); ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} Over Survey Sequence", fontweight="bold"); ax.legend(fontsize=8)
        ax.set_xticks(range(1, 9))
    plt.suptitle("D8 -- Temporal Pattern: Does Non-Native Performance Degrade Over Survey?", fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "D8_temporal.png"), dpi=150); plt.close()
    return rdf


# ══════════════════════════════════════════════════════════════════════
# D9: Exhaustive Pairwise Permutation Matrix
# ══════════════════════════════════════════════════════════════════════
def d9_permutation_matrix(df):
    hdr("D9: Exhaustive Non-Native Pairwise & Triple Permutations")

    # Define all possible demographic values
    dims = {
        "AgeGroup": sorted(df["AgeGroup"].unique()),
        "ExperienceGroup": sorted(df["ExperienceGroup"].unique()),
        "ExpertiseGroup": sorted(df["ExpertiseGroup"].unique()),
    }

    nn = df[df["LinguisticGroup"] == "Non-Native"]
    nat = df[df["LinguisticGroup"] == "Native"]

    rows = []
    # 2-way: NN x each demographic value x each difficulty tier
    for dim, vals in dims.items():
        for val in vals:
            for tier in ["Hard", "Medium", "Easy", "All"]:
                nn_sub = nn[nn[dim] == val]
                nat_sub = nat[nat[dim] == val]
                if tier != "All":
                    nn_sub = nn_sub[nn_sub["DiffPerf"] == tier]
                    nat_sub = nat_sub[nat_sub["DiffPerf"] == tier]
                if len(nn_sub) == 0: continue
                nn_acc = nn_sub["Accuracy"].mean()
                nat_acc = nat_sub["Accuracy"].mean() if len(nat_sub) > 0 else None
                gap = nn_acc - nat_acc if nat_acc is not None else None
                # Fisher
                p = None
                if len(nat_sub) > 0:
                    table = [[int(nat_sub["Accuracy"].sum()), len(nat_sub) - int(nat_sub["Accuracy"].sum())],
                             [int(nn_sub["Accuracy"].sum()), len(nn_sub) - int(nn_sub["Accuracy"].sum())]]
                    try: _, p = fisher_exact(table)
                    except: p = 1.0
                rows.append({"Cross1": dim, "Value1": val, "Cross2": "", "Value2": "",
                             "Tier": tier, "NN_N": len(nn_sub), "NN_Acc": round(nn_acc, 4),
                             "Nat_N": len(nat_sub), "Nat_Acc": round(nat_acc, 4) if nat_acc else None,
                             "Gap": round(gap, 4) if gap is not None else None,
                             "Fisher_p": round(p, 4) if p is not None else None})

    # 3-way: NN x dim1_val x dim2_val on Hard
    dim_keys = list(dims.keys())
    from itertools import combinations
    for d1, d2 in combinations(dim_keys, 2):
        for v1 in dims[d1]:
            for v2 in dims[d2]:
                nn_sub = nn[(nn[d1] == v1) & (nn[d2] == v2) & (nn["DiffPerf"] == "Hard")]
                nat_sub = nat[(nat[d1] == v1) & (nat[d2] == v2) & (nat["DiffPerf"] == "Hard")]
                if len(nn_sub) == 0: continue
                nn_acc = nn_sub["Accuracy"].mean()
                nat_acc = nat_sub["Accuracy"].mean() if len(nat_sub) > 0 else None
                gap = nn_acc - nat_acc if nat_acc is not None else None
                p = None
                if len(nat_sub) > 0 and len(nn_sub) > 0:
                    table = [[int(nat_sub["Accuracy"].sum()), len(nat_sub) - int(nat_sub["Accuracy"].sum())],
                             [int(nn_sub["Accuracy"].sum()), len(nn_sub) - int(nn_sub["Accuracy"].sum())]]
                    try: _, p = fisher_exact(table)
                    except: p = 1.0
                rows.append({"Cross1": d1, "Value1": v1, "Cross2": d2, "Value2": v2,
                             "Tier": "Hard", "NN_N": len(nn_sub), "NN_Acc": round(nn_acc, 4),
                             "Nat_N": len(nat_sub), "Nat_Acc": round(nat_acc, 4) if nat_acc else None,
                             "Gap": round(gap, 4) if gap is not None else None,
                             "Fisher_p": round(p, 4) if p is not None else None})

    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(RESULTS, "D9_permutation_matrix.csv"), index=False)

    # Print worst gaps
    sig = rdf[(rdf["Gap"].notna()) & (rdf["Gap"] < -0.1)].sort_values("Gap")
    print(f"\n  All NN subgroups with gap < -0.10 (sorted by gap):")
    for _, r in sig.head(20).iterrows():
        cross = f"NN+{r['Value1']}" + (f"+{r['Value2']}" if r["Value2"] else "")
        sig_str = f"p={r['Fisher_p']:.3f}" if r["Fisher_p"] is not None else ""
        print(f"  {cross:35s} [{r['Tier']:6s}]: NN={r['NN_Acc']:.1%}(n={r['NN_N']}) "
              f"Nat={r['Nat_Acc']:.1%}(n={r['Nat_N']}) gap={r['Gap']:+.3f} {sig_str}")

    # Heatmap: 2-way gaps on Hard
    two_way = rdf[(rdf["Cross2"] == "") & (rdf["Tier"] == "Hard") & (rdf["Gap"].notna())]
    fig, ax = plt.subplots(figsize=(10, 5))
    pivot_data = two_way.pivot(index="Cross1", columns="Value1", values="Gap")
    sns.heatmap(pivot_data, annot=True, fmt="+.2f", cmap="RdYlGn", center=0,
                linewidths=0.5, ax=ax, vmin=-0.5, vmax=0.2)
    ax.set_title("D9 -- Non-Native vs Native Gap on HARD Questions\nby Intersecting Demographic",
                 fontweight="bold")
    plt.tight_layout(); plt.savefig(os.path.join(RESULTS, "D9_permutation_matrix.png"), dpi=150); plt.close()
    return rdf


# ══════════════════════════════════════════════════════════════════════
# D10: Summary Dashboard
# ══════════════════════════════════════════════════════════════════════
def d10_dashboard(df, d1, d2, d3, d5, d6, d8, d9):
    hdr("D10: Summary Dashboard")

    fig = plt.figure(figsize=(22, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    # Panel 1: EngProf 3-level (D1)
    ax1 = fig.add_subplot(gs[0, 0])
    for lbl, color in [("Proficient", "#EF5350"), ("Fluent", "#FFA726"), ("Native", "#1E88E5")]:
        s = d1[d1["EngProf"] == lbl]
        ax1.plot(s["Tier"], s["Accuracy"], "o-", color=color, label=lbl, linewidth=2)
    ax1.set_title("D1: EngProf 3-Level", fontweight="bold", fontsize=9)
    ax1.set_ylabel("Accuracy"); ax1.legend(fontsize=7); ax1.set_ylim(0, 1)

    # Panel 2: FNR comparison (D3)
    ax2 = fig.add_subplot(gs[0, 1])
    tiers = ["Hard", "Medium", "Easy"]; x = np.arange(3); w = 0.35
    nn_fnr = [d3[(d3["Group"] == "Non-Native") & (d3["Tier"] == t)]["FNR"].values[0]
              if len(d3[(d3["Group"] == "Non-Native") & (d3["Tier"] == t)]) > 0 else 0 for t in tiers]
    nat_fnr = [d3[(d3["Group"] == "Native") & (d3["Tier"] == t)]["FNR"].values[0]
               if len(d3[(d3["Group"] == "Native") & (d3["Tier"] == t)]) > 0 else 0 for t in tiers]
    ax2.bar(x - w/2, nn_fnr, w, label="NN", color="#FF8F00"); ax2.bar(x + w/2, nat_fnr, w, label="Nat", color="#1E88E5")
    ax2.set_xticks(x); ax2.set_xticklabels(tiers)
    ax2.set_title("D3: False Negative Rate\n(miss unionable)", fontweight="bold", fontsize=9)
    ax2.legend(fontsize=7)

    # Panel 3: Temporal (D8)
    ax3 = fig.add_subplot(gs[0, 2])
    for grp, color in [("Non-Native", "#FF8F00"), ("Native", "#1E88E5")]:
        s = d8[d8["Group"] == grp]
        ax3.plot(s["QuestionPos"], s["Accuracy"], "o-", color=color, label=grp, linewidth=2)
    ax3.set_title("D8: Accuracy Over Survey Sequence", fontweight="bold", fontsize=9)
    ax3.set_xlabel("Q Position"); ax3.legend(fontsize=7); ax3.set_xticks(range(1, 9))

    # Panel 4: Per-annotator heatmap (D2)
    ax4 = fig.add_subplot(gs[1, :2])
    heat = d2.set_index("Annotator")[["Hard_Acc", "Med_Acc", "Easy_Acc"]].rename(
        columns={"Hard_Acc": "Hard", "Med_Acc": "Medium", "Easy_Acc": "Easy"})
    sns.heatmap(heat, annot=True, fmt=".0%", cmap="RdYlGn", vmin=0, vmax=1,
                linewidths=0.5, ax=ax4, cbar_kws={"shrink": 0.5})
    ax4.set_title("D2: Per Non-Native Annotator Accuracy by Tier", fontweight="bold", fontsize=9)

    # Panel 5: Explanation length (D6)
    ax5 = fig.add_subplot(gs[1, 2])
    for grp, color in [("Non-Native", "#FF8F00"), ("Native", "#1E88E5")]:
        s = d6[(d6["Group"] == grp) & (d6["Correct"] == False)]
        ax5.bar([f"{t[:1]}\n{grp[:3]}" for t in s["Tier"]], s["MeanLen"], color=color, alpha=0.8)
    ax5.set_title("D6: Expl Length (Incorrect Only)", fontweight="bold", fontsize=9)
    ax5.set_ylabel("Chars")

    # Panel 6: Permutation gaps (D9) - worst subgroups
    ax6 = fig.add_subplot(gs[2, :])
    worst = d9[(d9["Gap"].notna()) & (d9["Gap"] < -0.05)].sort_values("Gap").head(15)
    labels = [f"NN+{r['Value1']}" + (f"+{r['Value2']}" if r['Value2'] else "") + f"\n[{r['Tier'][:1]}]"
              for _, r in worst.iterrows()]
    colors = ["#EF5350" if (r["Fisher_p"] is not None and r["Fisher_p"] < 0.05) else "#FFA726"
              for _, r in worst.iterrows()]
    ax6.barh(range(len(worst)), worst["Gap"].values, color=colors, alpha=0.85)
    ax6.set_yticks(range(len(worst))); ax6.set_yticklabels(labels, fontsize=7)
    ax6.set_xlabel("NN - Native Accuracy Gap")
    ax6.axvline(0, color="black", linewidth=0.5); ax6.axvline(-0.1, color="red", linestyle=":", linewidth=0.8)
    ax6.set_title("D9: Worst Non-Native Subgroups (red = p<0.05, orange = p>=0.05)", fontweight="bold", fontsize=9)

    plt.suptitle("D10 -- Non-Native Bias Root Cause Dashboard", fontsize=14, fontweight="bold")
    plt.savefig(os.path.join(RESULTS, "D10_dashboard.png"), dpi=150, bbox_inches="tight"); plt.close()
    print("  -> D10_dashboard.png")


# ══════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════
def main():
    hdr("Phase 3b: Deep Non-Native Bias Root Cause Analysis")
    df = load()
    print(f"  {len(df)} rows, {df['ResponseId'].nunique()} annotators")
    print(f"  Non-Native: {(df['LinguisticGroup']=='Non-Native').sum()} rows, "
          f"{df[df['LinguisticGroup']=='Non-Native']['ResponseId'].nunique()} annotators")

    d1 = d1_engprof_granularity(df)
    d2 = d2_per_annotator(df)
    d3 = d3_answer_pattern(df)
    d4 = d4_calibration(df)
    d5 = d5_time_pattern(df)
    d6 = d6_explanation_length(df)
    d7 = d7_per_question_drill(df)
    d8 = d8_temporal(df)
    d9 = d9_permutation_matrix(df)
    d10_dashboard(df, d1, d2, d3, d5, d6, d8, d9)

    print(f"\n{SEP}\n  All results: {os.path.abspath(RESULTS)}\n{SEP}")

if __name__ == "__main__":
    main()
