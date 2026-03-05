"""
Excel Data Loader for Export_and_Compiled.xlsx
================================================
Loads the Qualtrics-exported survey data from the new annotation study
(Export_and_Compiled.xlsx) and returns a long-format DataFrame that is
structurally compatible with the Phase 1 experiment pipeline.

Data structure:
  - V1/V2/V3/V4 sheets: one row per respondent, with per-question answer,
    timing, click-count, and confidence columns.
  - Compiled sheet: one row per (annotator × question), includes IsExp flag.
  - Qualtrics sheet: duplicate of V1-V4 respondent data merged.

Demographic encoding:
  - Age      : 18-24=1, 25-34=2, 35-44=3, 45-54=4, 55+=5
  - Education: HS=1, Associate/Other=2, Bachelor=3, Master=4, Doctoral=5
  - EngProf  : Proficient=3, Fluent=4, Native speaker=5
  - Major    : 0=Non-STEM, 1=STEM (text classification)

Partitions applied:
  - LinguisticGroup : Native (EngProf=5) vs Non-Native
  - ExpertiseGroup  : STEM (Major=1) vs Non-STEM
  - ExperienceGroup : High-Edu (Education>=4) vs Lower-Edu
  - AgeGroup        : Young-18-34 (Age<=2) vs Older-35plus
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# STEM keyword lists (lower-case matching)
# ---------------------------------------------------------------------------
_STEM_INCLUDES = [
    "computer", "computing", "software", "data science", "data ",
    "mathematics", "math", " math", "statistics", "statistical",
    "engineering", "physics", "chemistry", "biology", "biochemistry",
    "bioinformatics", "biomed", "biomedical", "genetics", "genomics",
    "microbiology", "geology", "geoscience", "earth science",
    "information technology", "information science", "information tech",
    " it ", "it manager", " ee", "electrical", "mechanical", "civil",
    "aerospace", "environmental science", "science", "technology",
    "veterinary", "physical science", "general science",
]

_NON_STEM_EXCLUDES = [
    "business", "finance", "accounting", "economics", "management",
    "marketing", "undeclared", "education", "law", "psychology",
    "sociology", "history", "english", "arts", "liberal",
    "communication", "environmental management",
]


def _classify_stem(major_text: str) -> int:
    """Return 1 if the major is STEM, 0 otherwise."""
    if not isinstance(major_text, str) or not major_text.strip():
        return 0
    t = " " + major_text.lower().strip() + " "

    # Non-STEM overrides (checked first to handle ambiguous cases like
    # "Business Management", "Environmental Management")
    for kw in _NON_STEM_EXCLUDES:
        if kw in t:
            return 0

    for kw in _STEM_INCLUDES:
        if kw in t:
            return 1

    return 0


def _encode_age(text: str) -> int:
    _map = {"18 - 24": 1, "25 - 34": 2, "35 - 44": 3, "45 - 54": 4, "55 & above": 5}
    return _map.get(str(text).strip(), 3)


def _encode_education(edu_text: str, other_text=None) -> int:
    t = str(edu_text).lower().strip()
    if "doctoral" in t or "phd" in t:
        return 5
    if "master" in t:
        return 4
    if "bachelor" in t:
        return 3
    # "Other (please specify)" — check the free-text field
    if "other" in t or "associate" in t:
        ot = str(other_text).lower() if other_text and str(other_text) != "nan" else ""
        return 2 if "associate" in ot else 2
    if "high school" in t:
        return 1
    return 3  # default to Bachelor's


def _encode_engprof(text: str) -> int:
    _map = {"proficient": 3, "fluent": 4, "native speaker": 5}
    return _map.get(str(text).lower().strip(), 4)


def _to_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return np.nan


def _to_int(val):
    try:
        return int(val)
    except (ValueError, TypeError):
        return np.nan


def _yes_no_to_int(val):
    return 1 if str(val).strip().lower() == "yes" else 0


# ---------------------------------------------------------------------------
# Offsets for joining V-sheet rows to Compiled rows
# ---------------------------------------------------------------------------
# Compiled is sorted by (SurveyVersion, QuestionNum).
# Within each (version, question) block, rows are in the same order as
# the corresponding V-sheet respondents.
_VERSION_SIZES = {1: 17, 2: 15, 3: 15, 4: 13}
_VERSION_OFFSETS = {1: 0, 2: 136, 3: 256, 4: 376}


def _compiled_row_index(version: int, question: int, respondent_idx: int) -> int:
    """Return the 0-based row index in the Compiled sheet."""
    size = _VERSION_SIZES[version]
    return _VERSION_OFFSETS[version] + size * (question - 1) + respondent_idx


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_excel_data(excel_path: str) -> pd.DataFrame:
    """
    Load Export_and_Compiled.xlsx and return a long-format DataFrame.

    Returns one row per (annotator × question) with columns:
        ResponseId, SurveyVersion, QuestionNum,
        SurveyAnswer (0/1), ActualAnswer (0/1),
        FirstClick, LastClick, DecisionTime, ClickCount,
        ConfidenceLevel (0-100), ConfidenceLevelNorm (0-1),
        IsExp (0/1), Accuracy (0/1),
        Age (1-5), Education (1-5), EngProf (3/4/5), Major (0/1),
        DQ1_text, DQ2_text, DQ3_text, DQ4_text,
        LinguisticGroup, ExpertiseGroup, ExperienceGroup, AgeGroup,
        Majority (0/1, computed per question across all annotators),
        ByWho  (alias for ResponseId, for pipeline compatibility)
    """
    xl = pd.ExcelFile(excel_path)

    # Load Compiled sheet for IsExp / Accuracy (pre-computed)
    df_compiled = xl.parse("Compiled")
    # df_compiled rows are indexed 0..479 as described above

    records = []

    for v in [1, 2, 3, 4]:
        df_v = xl.parse(f"V{v}")
        data = df_v.iloc[2:].reset_index(drop=True)  # skip 2 metadata rows

        for resp_idx, row in data.iterrows():
            resp_id = row.get("ResponseId", f"V{v}_{resp_idx}")
            dq1 = row.get("DQ1", "")
            dq2 = row.get("DQ2", "")
            dq2_other = row.get("DQ2_6_TEXT", None)
            dq3 = row.get("DQ3", "")
            dq4 = row.get("DQ4", "")

            age_num = _encode_age(dq1)
            edu_num = _encode_education(dq2, dq2_other)
            eng_num = _encode_engprof(dq3)
            stem_num = _classify_stem(str(dq4))

            for q in range(1, 9):
                compiled_idx = _compiled_row_index(v, q, resp_idx)
                compiled_row = df_compiled.iloc[compiled_idx]

                is_exp = int(compiled_row.get("IsExp", 1))
                accuracy = int(compiled_row.get("Accuracy", 0))

                records.append({
                    "ResponseId": resp_id,
                    "SurveyVersion": v,
                    "QuestionNum": q,
                    "SurveyAnswer": _yes_no_to_int(row.get(f"{v}MS{q}")),
                    "ActualAnswer": _yes_no_to_int(row.get(f"{v}MS{q}_Ans")),
                    "FirstClick": _to_float(row.get(f"{v}DT{q}_First Click")),
                    "LastClick": _to_float(row.get(f"{v}DT{q}_Last Click")),
                    "DecisionTime": _to_float(row.get(f"{v}DT{q}_Page Submit")),
                    "ClickCount": _to_int(row.get(f"{v}DT{q}_Click Count")),
                    "ConfidenceLevel": _to_float(row.get(f"{v}CL{q}_1")),
                    "IsExp": is_exp,
                    "Accuracy": accuracy,
                    "Age": age_num,
                    "Education": edu_num,
                    "EngProf": eng_num,
                    "Major": stem_num,
                    "DQ1_text": str(dq1),
                    "DQ2_text": str(dq2),
                    "DQ3_text": str(dq3),
                    "DQ4_text": str(dq4),
                })

    df = pd.DataFrame(records)

    # Normalize ConfidenceLevel to [0, 1] for calibration gap calculation
    df["ConfidenceLevelNorm"] = df["ConfidenceLevel"] / 100.0

    # ByWho alias for pipeline compatibility
    df["ByWho"] = df["ResponseId"]

    # Majority vote per question (most common SurveyAnswer across all annotators)
    majority = (
        df.groupby("QuestionNum")["SurveyAnswer"]
        .apply(lambda x: 1 if x.mean() >= 0.5 else 0)
        .reset_index()
    )
    majority.columns = ["QuestionNum", "Majority"]
    df = df.merge(majority, on="QuestionNum", how="left")

    # Demographic partitions
    df["LinguisticGroup"] = np.where(df["EngProf"] == 5, "Native", "Non-Native")
    df["ExpertiseGroup"] = np.where(df["Major"] == 1, "STEM", "Non-STEM")
    df["ExperienceGroup"] = np.where(df["Education"] >= 4, "High-Edu", "Lower-Edu")
    df["AgeGroup"] = np.where(df["Age"] <= 2, "Young-18-34", "Older-35plus")

    return df


def get_annotator_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """Return one row per unique annotator with their demographic profile."""
    cols = [
        "ResponseId", "SurveyVersion",
        "Age", "Education", "EngProf", "Major",
        "DQ1_text", "DQ2_text", "DQ3_text", "DQ4_text",
        "LinguisticGroup", "ExpertiseGroup", "ExperienceGroup", "AgeGroup",
    ]
    return df[cols].drop_duplicates(subset=["ResponseId"]).reset_index(drop=True)


def group_distribution_summary(df: pd.DataFrame, group_cols: list) -> dict:
    """Return {group_col: {group_name: n_annotators}} for each partition."""
    summary = {}
    for gc in group_cols:
        counts = df.groupby(gc)["ResponseId"].nunique()
        summary[gc] = counts.to_dict()
    return summary
