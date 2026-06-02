# TUNE 2.0 Dataset

This directory contains the TUNE 2.0 (Table UNionability with human Evaluation, version 2.0) dataset used throughout the FairPrep fairness analysis pipeline.

## What is TUNE 2.0

TUNE 2.0 is a human-centered benchmark of table-unionability judgments collected via the Prolific platform. It comprises 480 individual judgments from 60 participants across 26 table-pair presentations derived from 13 underlying table pairs in the UGEN benchmark suite (7 unionable, 6 non-unionable), each shown in two schema-visibility conditions (with column headers and without). Participants were randomly assigned to one of four survey versions (V1–V4), each containing 8 unionability questions.

TUNE 2.0 extends the original TUNE benchmark by capturing not only the binary unionable/non-unionable response but also a rich set of behavioral indicators for each judgment (self-reported confidence, decision time, click engagement, free-text explanations). It additionally introduces the `IsExp` column tracking whether a participant provided an explanation (96% provision rate in the cleaned corpus).

## Files

### `Export_and_Compiled.xlsx` (146 KB)
The primary dataset. A multi-sheet Excel workbook containing:

- **Compiled** sheet (480 rows × 12 columns): one row per (annotator × question), ordered by (SurveyVersion, QuestionNum). Contains `SurveyAnswer` (Yes/No), `ActualAnswer` (ground-truth Yes/No), `FirstClick`, `LastClick`, `DecisionTime`, `ClickCount`, `ConfidenceLevel` (0–100 slider), `IsExp`, `Explanations`, `Accuracy`.
- **Qualtrics** sheet (62 rows × 258 columns): rows 0–1 are metadata headers, rows 2–61 are the 60 respondents. Contains `ResponseId`, demographics (`DQ1`=Age, `DQ2`=Education, `DQ3`=English Proficiency, `DQ4`=Major), and per-version answer columns identifying which survey version each respondent took.
- Auxiliary sheets V1–V4 are version-specific layouts not used by the analysis pipeline.

The loader in `src/excel_data_loader.py` reads only the Compiled and Qualtrics sheets, links them positionally within each (SurveyVersion, QuestionNum) block, and produces a single normalized DataFrame with demographic partitions applied.

### `Export_and_Compiled_TUNE2_0_second_dataset.csv` (63 KB)
A pre-flattened CSV version of the Compiled sheet, useful for quick inspection or import into tools that do not read Excel directly. Schema matches the Compiled sheet of the Excel file.

### `question_difficulty_summary_second_survey.csv` (2 KB)
Nina's performance-based difficulty classification per (SurveyVersion, QuestionNum) cell. Contains the empirical accuracy proportion for each cell along with two difficulty labels:

- `difficulty_disagreement`: entropy-based classification using response distribution.
- `difficulty_performance`: accuracy-based classification used throughout the FairPrep analysis. Cells are partitioned into Hard (accuracy ≤ 0.54), Medium (0.60 ≤ accuracy ≤ 0.80), and Easy (accuracy ≥ 0.85), yielding 10/11/11 cells across the 32 (version, question) pairs.

The `difficulty_performance` column is the primary difficulty label referenced throughout the analysis and the paper.

## Demographic Encoding

Encoded in `src/excel_data_loader.py` from the free-text Qualtrics responses:

| Field | Source | Encoding | Partition derived |
|---|---|---|---|
| Age | DQ1 | 18–24 → 1, 25–34 → 2, 35–44 → 3, 45–54 → 4, 55+ → 5 | AgeGroup: Young-18-34 (Age ≤ 2) vs Older-35plus |
| Education | DQ2 | High School → 1, Associate/Other → 2, Bachelor → 3, Master → 4, Doctoral → 5 | ExperienceGroup: High-Edu (Education ≥ 4) vs Lower-Edu |
| English Proficiency | DQ3 | Proficient → 3, Fluent → 4, Native speaker → 5 | LinguisticGroup: Native (EngProf = 5) vs Non-Native |
| Major | DQ4 | STEM keyword classifier → 0 / 1 | ExpertiseGroup: STEM (1) vs Non-STEM (0) |

## Group Distributions (Cleaned Corpus, n = 56 annotators, 427 judgments)

| Partition | Group A | Group B |
|---|---|---|
| LinguisticGroup | Native (n = 45) | Non-Native (n = 11) |
| ExpertiseGroup | STEM (n = 39) | Non-STEM (n = 17) |
| ExperienceGroup | High-Edu (n = 20) | Lower-Edu (n = 36) |
| AgeGroup | Young-18-34 (n = 21) | Older-35plus (n = 35) |

The cleaned corpus is produced by applying `src/data_cleaning.py` to the raw dataset; see that module for the four removal criteria and one capping step.

## Usage

```python
from src.excel_data_loader import load_excel_data
from src.data_cleaning import clean_data
import pandas as pd

# Load raw data
df = load_excel_data("data/Export_and_Compiled.xlsx")

# Apply the cleaning pipeline (5 steps, all toggleable)
df_cleaned, report = clean_data(df)

# Merge in the performance-based difficulty labels
difficulty = pd.read_csv("data/question_difficulty_summary_second_survey.csv")
df_cleaned = df_cleaned.merge(
    difficulty[["SurveyVersion", "QuestionNum", "difficulty_performance"]],
    on=["SurveyVersion", "QuestionNum"], how="left",
)
```

## Provenance

- **Collection platform:** Prolific
- **Source benchmark for table pairs:** UGEN
- **Original TUNE benchmark (version 1.0):** Marimuthu, Klimenkova, Shraga. *Humans, ML, and LMs in Union: A Human-Centered Exploration of Table Unionability.* HILDA 2025.
- **Difficulty classification:** Generated by Nina Klimenkova; see `Questions_difficulty.ipynb` (not in this directory; available in the source Fairness archive).
