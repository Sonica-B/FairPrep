"""
Excel Data Loader for Export_and_Compiled.xlsx
================================================
Loads survey data using ONLY the Compiled and Qualtrics sheets.

Data sources:
  - Compiled sheet (480 rows x 12 cols): one row per (annotator x question),
    ordered by (SurveyVersion, QuestionNum). Contains SurveyAnswer, ActualAnswer,
    timing, confidence, IsExp, Explanations, Accuracy.
  - Qualtrics sheet (62 rows x 258 cols): rows 0-1 are metadata headers,
    rows 2-61 are the 60 respondents. Contains ResponseId, demographics
    (DQ1-DQ4), and per-version answer columns to determine which version
    each respondent answered.

Linking: Within each (SurveyVersion, QuestionNum) block in Compiled, rows
are in the same order as Qualtrics respondents for that version (sorted by
their Qualtrics row order).

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
    if "other" in t or "associate" in t:
        return 2
    if "high school" in t:
        return 1
    return 3  # default to Bachelor's


def _encode_engprof(text: str) -> int:
    _map = {"proficient": 3, "fluent": 4, "native speaker": 5}
    return _map.get(str(text).lower().strip(), 4)


def _yes_no_to_int(val) -> int:
    return 1 if str(val).strip().lower() == "yes" else 0


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_excel_data(excel_path: str) -> pd.DataFrame:
    """
    Load Export_and_Compiled.xlsx using only Compiled + Qualtrics sheets.

    Returns one row per (annotator x question) with columns:
        ResponseId, SurveyVersion, QuestionNum,
        SurveyAnswer (0/1), ActualAnswer (0/1),
        FirstClick, LastClick, DecisionTime, ClickCount,
        ConfidenceLevel (0-100), ConfidenceLevelNorm (0-1),
        IsExp (0/1), Accuracy (0/1), Explanations,
        Age (1-5), Education (1-5), EngProf (3/4/5), Major (0/1),
        DQ1_text, DQ2_text, DQ3_text, DQ4_text,
        LinguisticGroup, ExpertiseGroup, ExperienceGroup, AgeGroup,
        Majority (0/1, computed per question across all annotators),
        ByWho  (alias for ResponseId, for pipeline compatibility)
    """
    xl = pd.ExcelFile(excel_path)

    # ------------------------------------------------------------------
    # 1. Parse Qualtrics sheet for demographics + ResponseId
    # ------------------------------------------------------------------
    df_q = xl.parse("Qualtrics")
    # Rows 0-1 are metadata headers; actual data starts at row 2
    data_q = df_q.iloc[2:].reset_index(drop=True)

    # Determine which SurveyVersion each respondent answered and extract demographics
    respondent_info = []
    for i in range(len(data_q)):
        row = data_q.iloc[i]
        rid = row["ResponseId"]

        # Find which version this respondent answered
        version = None
        for v in [1, 2, 3, 4]:
            if pd.notna(row.get(f"{v}MS1")):
                version = v
                break
        if version is None:
            continue

        dq1 = row.get("DQ1", "")
        dq2 = row.get("DQ2", "")
        dq2_other = row.get("DQ2_6_TEXT", None)
        dq3 = row.get("DQ3", "")
        dq4 = row.get("DQ4", "")

        respondent_info.append({
            "ResponseId": rid,
            "SurveyVersion": version,
            "qualtrics_order": i,
            "Age": _encode_age(dq1),
            "Education": _encode_education(dq2, dq2_other),
            "EngProf": _encode_engprof(dq3),
            "Major": _classify_stem(str(dq4)),
            "DQ1_text": str(dq1),
            "DQ2_text": str(dq2),
            "DQ3_text": str(dq3),
            "DQ4_text": str(dq4),
        })

    df_resp = pd.DataFrame(respondent_info)

    # Build ordered lists of ResponseIds per version (by Qualtrics row order)
    version_respondents = {}
    for v in [1, 2, 3, 4]:
        v_df = df_resp[df_resp["SurveyVersion"] == v].sort_values("qualtrics_order")
        version_respondents[v] = v_df["ResponseId"].tolist()

    # ------------------------------------------------------------------
    # 2. Parse Compiled sheet (480 rows, already one row per annotator x question)
    # ------------------------------------------------------------------
    df_compiled = xl.parse("Compiled")

    # ------------------------------------------------------------------
    # 3. Assign ResponseId to each Compiled row
    #    Compiled is ordered by (SurveyVersion, QuestionNum).
    #    Within each (version, question) block, rows are in the same
    #    order as the Qualtrics respondents for that version.
    # ------------------------------------------------------------------
    response_ids = []
    for v in [1, 2, 3, 4]:
        rid_list = version_respondents[v]
        for q in range(1, 9):
            response_ids.extend(rid_list)

    df_compiled["ResponseId"] = response_ids

    # ------------------------------------------------------------------
    # 4. Convert SurveyAnswer/ActualAnswer from Yes/No strings to 0/1
    # ------------------------------------------------------------------
    df_compiled["SurveyAnswer"] = df_compiled["SurveyAnswer"].apply(_yes_no_to_int)
    df_compiled["ActualAnswer"] = df_compiled["ActualAnswer"].apply(_yes_no_to_int)

    # ------------------------------------------------------------------
    # 5. Merge demographics from Qualtrics onto Compiled
    # ------------------------------------------------------------------
    demo_cols = [
        "ResponseId", "Age", "Education", "EngProf", "Major",
        "DQ1_text", "DQ2_text", "DQ3_text", "DQ4_text",
    ]
    df = df_compiled.merge(df_resp[demo_cols], on="ResponseId", how="left")

    # ------------------------------------------------------------------
    # 6. Derived columns
    # ------------------------------------------------------------------
    df["ConfidenceLevelNorm"] = df["ConfidenceLevel"] / 100.0
    df["ByWho"] = df["ResponseId"]

    # Majority vote per question
    majority = (
        df.groupby("QuestionNum")["SurveyAnswer"]
        .apply(lambda x: 1 if x.mean() >= 0.5 else 0)
        .reset_index()
    )
    majority.columns = ["QuestionNum", "Majority"]
    df = df.merge(majority, on="QuestionNum", how="left")

    # ------------------------------------------------------------------
    # 7. Demographic partitions
    # ------------------------------------------------------------------
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
