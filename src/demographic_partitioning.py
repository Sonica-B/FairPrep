"""
Phase 1: Demographic Partitioning (The "Who")
==============================================
Partitions the 58 expert annotators in the TUNE dataset into demographic
subgroups based on available metadata:

  a. Linguistic: Native English Speakers vs Non-Native Speakers
  b. Domain Expertise: STEM Majors vs Non-STEM/Humanities
  c. Experience: High Education (PhD) vs Lower (Undergrad/Masters)

Adapted from FairEM (Shahbazi et al., VLDB 2023) workload partitioning
for entity matching, re-targeted to annotator demographics in TUS.
"""

import pandas as pd
import numpy as np
from collections import defaultdict


# ---------------------------------------------------------------------------
# Column mapping from Feature_Engineered.csv
# ---------------------------------------------------------------------------
# Age        : ordinal (1=18-24, 2=25-34, 3=35-44, …)
# Education  : ordinal (1=HS, 2=Bachelor, 3=Master, 4=PhD, 5=Other)
# EngProf    : English proficiency 1-5  (5=Native)
# Major      : 1-5  where STEM ~ 4,5 and non-STEM ~ 1,2,3
# ---------------------------------------------------------------------------

DEMOGRAPHIC_COLUMNS = ["Age", "Education", "EngProf", "Major"]


def load_tune_data(csv_path: str) -> pd.DataFrame:
    """Load Feature_Engineered.csv and return a cleaned DataFrame."""
    df = pd.read_csv(csv_path)
    # Ensure key columns are numeric
    for col in DEMOGRAPHIC_COLUMNS + ["SurveyAnswer", "ActualAnswer", "Majority"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


# ===== Partitioning functions =====

def partition_linguistic(df: pd.DataFrame, threshold: int = 5):
    """
    Linguistic grouping.
    Group A (Native)    : EngProf >= threshold (5 = native)
    Group B (Non-Native): EngProf <  threshold
    """
    df = df.copy()
    df["LinguisticGroup"] = np.where(
        df["EngProf"] >= threshold, "Native", "Non-Native"
    )
    return df


def partition_expertise(df: pd.DataFrame, stem_threshold: int = 4):
    """
    Domain Expertise grouping.
    Group A (STEM)     : Major >= stem_threshold
    Group B (Non-STEM) : Major <  stem_threshold
    """
    df = df.copy()
    df["ExpertiseGroup"] = np.where(
        df["Major"] >= stem_threshold, "STEM", "Non-STEM"
    )
    return df


def partition_experience(df: pd.DataFrame, high_edu_threshold: int = 3):
    """
    Experience/Education grouping.
    Group A (High-Edu)  : Education >= high_edu_threshold (Master/PhD)
    Group B (Lower-Edu) : Education <  high_edu_threshold
    """
    df = df.copy()
    df["ExperienceGroup"] = np.where(
        df["Education"] >= high_edu_threshold, "High-Edu", "Lower-Edu"
    )
    return df


def apply_all_partitions(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all three demographic partitions to the dataframe."""
    df = partition_linguistic(df)
    df = partition_expertise(df)
    df = partition_experience(df)
    return df


def get_annotator_demographics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract unique annotator-level demographics.
    Each annotator is identified by `ByWho` column.
    Returns a DataFrame of unique annotators with their demographic groups.
    """
    df = apply_all_partitions(df)
    # Extract annotator ID from ByWho (format: "surveyversion_annotatornum")
    annotator_cols = [
        "ByWho", "Age", "Education", "EngProf", "Major",
        "LinguisticGroup", "ExpertiseGroup", "ExperienceGroup"
    ]
    annotators = df[annotator_cols].drop_duplicates(subset=["ByWho"])
    return annotators.reset_index(drop=True)


def group_distribution_summary(df: pd.DataFrame) -> dict:
    """
    Print and return the size of each demographic subgroup partition.
    """
    df = apply_all_partitions(df)
    summary = {}
    for group_col in ["LinguisticGroup", "ExpertiseGroup", "ExperienceGroup"]:
        counts = df.groupby(group_col)["ByWho"].nunique()
        summary[group_col] = counts.to_dict()
    return summary


def get_group_decisions(df: pd.DataFrame, group_col: str) -> dict:
    """
    For a given group column, return per-group aggregated decisions.
    Returns dict of {group_name: DataFrame of rows belonging to that group}.
    """
    df = apply_all_partitions(df)
    groups = {}
    for name, group_df in df.groupby(group_col):
        groups[name] = group_df.reset_index(drop=True)
    return groups
