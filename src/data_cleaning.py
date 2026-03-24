"""
Data Cleaning Module for Export_and_Compiled.xlsx
==================================================
Applies cleaning steps motivated by Phase 1 experiment findings:

  1. Decision-time outlier removal (Exp 3 finding: Non-Native std=304s)
     - Rows with DecisionTime > 3x IQR above Q3 are flagged/removed
     - These represent annotators who likely walked away or lost focus

  2. Speed-clicker removal (Exp 3 + Exp 6 finding)
     - Rows where DecisionTime < 3s AND ClickCount <= 1 are flagged
     - These represent zero-effort, likely random annotations

  3. Low-accuracy annotator flagging (Exp 7 + Exp 9)
     - Annotators with overall accuracy <= 0.25 (below random for binary task)
     - These are either adversarial or did not understand the task

  4. Confidence outlier capping (Exp 5 finding: calibration gap)
     - ConfidenceLevel values of 0 or 100 (exact extremes) are suspicious
     - Cap at [5, 95] to reduce calibration noise from slider-boundary effects

  5. Duplicate/identical response patterns (data integrity)
     - Flag annotators who gave the same answer to ALL 8 questions
     - These may indicate straight-lining (all-Yes or all-No)

Each step is independently toggleable. The module returns the cleaned
DataFrame plus a cleaning report DataFrame.

Usage:
    from src.data_cleaning import clean_data
    df_cleaned, report = clean_data(df_raw)
"""

import numpy as np
import pandas as pd


def _flag_decision_time_outliers(df, iqr_multiplier=3.0):
    """Flag rows where DecisionTime is an extreme outlier (> Q3 + mult*IQR)."""
    q1 = df["DecisionTime"].quantile(0.25)
    q3 = df["DecisionTime"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + iqr_multiplier * iqr
    lower = max(0, q1 - iqr_multiplier * iqr)
    flag = (df["DecisionTime"] > upper) | (df["DecisionTime"] < lower)
    return flag, {"upper_bound": round(upper, 2), "lower_bound": round(lower, 2),
                  "Q1": round(q1, 2), "Q3": round(q3, 2), "IQR": round(iqr, 2)}


def _flag_speed_clickers(df, min_time=3.0, max_clicks=1):
    """Flag rows where annotator spent < min_time seconds AND clicked <= max_clicks."""
    flag = (df["DecisionTime"] < min_time) & (df["ClickCount"] <= max_clicks)
    return flag


def _flag_low_accuracy_annotators(df, min_accuracy=0.25):
    """Flag all rows from annotators with overall accuracy <= threshold."""
    ann_acc = df.groupby("ResponseId")["Accuracy"].mean()
    bad_annotators = ann_acc[ann_acc <= min_accuracy].index
    flag = df["ResponseId"].isin(bad_annotators)
    return flag, bad_annotators.tolist()


def _cap_confidence(df, low=5, high=95):
    """Cap ConfidenceLevel at [low, high] to remove slider-boundary noise."""
    original = df["ConfidenceLevel"].copy()
    capped = df["ConfidenceLevel"].clip(low, high)
    n_changed = (original != capped).sum()
    return capped, n_changed


def _flag_straightliners(df):
    """Flag annotators who gave the same answer to ALL 8 questions."""
    ann_answers = df.groupby("ResponseId")["SurveyAnswer"].agg(["mean", "count"])
    # Straightliners: all 0s (mean=0) or all 1s (mean=1)
    straightliners = ann_answers[(ann_answers["mean"] == 0) | (ann_answers["mean"] == 1)]
    # Only flag if they answered all 8 questions
    straightliners = straightliners[straightliners["count"] >= 8]
    flag = df["ResponseId"].isin(straightliners.index)
    return flag, straightliners.index.tolist()


def clean_data(df,
               remove_time_outliers=True,
               remove_speed_clickers=True,
               remove_low_accuracy=True,
               cap_confidence=True,
               remove_straightliners=True,
               iqr_multiplier=3.0,
               min_decision_time=3.0,
               min_accuracy=0.25,
               confidence_range=(5, 95),
               verbose=True):
    """
    Apply all cleaning steps and return (cleaned_df, report_df).

    Parameters
    ----------
    df : pd.DataFrame
        Raw data from load_excel_data()
    remove_time_outliers : bool
        Remove extreme DecisionTime outliers
    remove_speed_clickers : bool
        Remove rows with suspiciously fast + low-click responses
    remove_low_accuracy : bool
        Remove annotators with accuracy <= min_accuracy
    cap_confidence : bool
        Cap ConfidenceLevel at confidence_range
    remove_straightliners : bool
        Remove annotators who answered all questions identically
    verbose : bool
        Print cleaning report to stdout

    Returns
    -------
    df_cleaned : pd.DataFrame
        Cleaned data
    report : dict
        Cleaning statistics
    """
    df = df.copy()
    n_original = len(df)
    n_annotators_original = df["ResponseId"].nunique()
    report = {
        "original_rows": n_original,
        "original_annotators": n_annotators_original,
        "steps": [],
    }

    # Track cumulative removal mask
    remove_mask = pd.Series(False, index=df.index)

    # Step 1: Decision time outliers
    if remove_time_outliers:
        flag, bounds = _flag_decision_time_outliers(df, iqr_multiplier)
        n_flagged = flag.sum()
        remove_mask |= flag
        step = {
            "step": "DecisionTime outlier removal",
            "method": f"IQR x{iqr_multiplier} (upper={bounds['upper_bound']}s)",
            "rows_flagged": int(n_flagged),
            "annotators_affected": int(df.loc[flag, "ResponseId"].nunique()) if n_flagged > 0 else 0,
        }
        report["steps"].append(step)
        if verbose:
            print(f"  [Clean 1] DecisionTime outliers: {n_flagged} rows flagged "
                  f"(upper bound={bounds['upper_bound']}s)")

    # Step 2: Speed clickers
    if remove_speed_clickers:
        flag = _flag_speed_clickers(df, min_decision_time)
        n_flagged = flag.sum()
        remove_mask |= flag
        step = {
            "step": "Speed-clicker removal",
            "method": f"DecisionTime < {min_decision_time}s AND ClickCount <= 1",
            "rows_flagged": int(n_flagged),
            "annotators_affected": int(df.loc[flag, "ResponseId"].nunique()) if n_flagged > 0 else 0,
        }
        report["steps"].append(step)
        if verbose:
            print(f"  [Clean 2] Speed clickers: {n_flagged} rows flagged "
                  f"(< {min_decision_time}s + <= 1 click)")

    # Step 3: Low-accuracy annotators
    if remove_low_accuracy:
        flag, bad_ids = _flag_low_accuracy_annotators(df, min_accuracy)
        n_flagged = flag.sum()
        remove_mask |= flag
        step = {
            "step": "Low-accuracy annotator removal",
            "method": f"Overall accuracy <= {min_accuracy}",
            "rows_flagged": int(n_flagged),
            "annotators_affected": len(bad_ids),
            "annotator_ids": bad_ids,
        }
        report["steps"].append(step)
        if verbose:
            print(f"  [Clean 3] Low-accuracy annotators (<= {min_accuracy}): "
                  f"{n_flagged} rows ({len(bad_ids)} annotators)")

    # Step 4: Straightliners
    if remove_straightliners:
        flag, sl_ids = _flag_straightliners(df)
        n_flagged = flag.sum()
        remove_mask |= flag
        step = {
            "step": "Straightliner removal",
            "method": "Same answer to all 8 questions",
            "rows_flagged": int(n_flagged),
            "annotators_affected": len(sl_ids),
            "annotator_ids": sl_ids,
        }
        report["steps"].append(step)
        if verbose:
            print(f"  [Clean 4] Straightliners: {n_flagged} rows ({len(sl_ids)} annotators)")

    # Apply removal
    df_cleaned = df[~remove_mask].copy()

    # Step 5: Confidence capping (applied to surviving rows)
    if cap_confidence:
        lo, hi = confidence_range
        capped, n_changed = _cap_confidence(df_cleaned, lo, hi)
        df_cleaned["ConfidenceLevel"] = capped
        df_cleaned["ConfidenceLevelNorm"] = capped / 100.0
        step = {
            "step": "Confidence capping",
            "method": f"Clipped to [{lo}, {hi}]",
            "rows_flagged": int(n_changed),
            "annotators_affected": 0,  # no rows removed
        }
        report["steps"].append(step)
        if verbose:
            print(f"  [Clean 5] Confidence capping [{lo}, {hi}]: {n_changed} values adjusted")

    # Recompute Majority vote after cleaning
    majority = (
        df_cleaned.groupby("QuestionNum")["SurveyAnswer"]
        .apply(lambda x: 1 if x.mean() >= 0.5 else 0)
        .reset_index()
    )
    majority.columns = ["QuestionNum", "Majority"]
    df_cleaned = df_cleaned.drop("Majority", axis=1, errors="ignore")
    df_cleaned = df_cleaned.merge(majority, on="QuestionNum", how="left")

    # Final stats
    n_final = len(df_cleaned)
    n_annotators_final = df_cleaned["ResponseId"].nunique()
    total_removed = n_original - n_final
    report["final_rows"] = n_final
    report["final_annotators"] = n_annotators_final
    report["total_rows_removed"] = total_removed
    report["total_annotators_removed"] = n_annotators_original - n_annotators_final
    report["pct_removed"] = round(100.0 * total_removed / n_original, 2)

    if verbose:
        print(f"\n  === Cleaning Summary ===")
        print(f"  Before: {n_original} rows, {n_annotators_original} annotators")
        print(f"  After:  {n_final} rows, {n_annotators_final} annotators")
        print(f"  Removed: {total_removed} rows ({report['pct_removed']}%), "
              f"{n_annotators_original - n_annotators_final} annotators")

    return df_cleaned, report
