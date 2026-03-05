"""
Synthetic TUS Dataset Generator
=================================
Adapted from fair_entity_matching/synthetic dataset generator/
(FacultyMatch and NoFlyCompas generators by Shahbazi et al.)

Instead of generating entity-pair matching datasets, this generator
creates table-level union search datasets with:
  - Known unionability ground truth
  - Controlled demographic composition per table
  - Perturbations from the original generators (character-level noise)
    applied at the schema/table level to simulate structural difficulty
  - FairEM-compatible output format for fairness auditing

The key adaptation:
  Original generators: row-level entity pairs with sensitive attributes
  This generator:      table-level pairs where the *content demographics*
                       become the sensitive attribute for fairness testing

This lets us test: "Do TUS systems/annotators perform worse on tables
whose content is dominated by a particular demographic group?"
"""

import os
import random
import string
import itertools
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Perturbation functions — directly from fair_entity_matching generators
# ---------------------------------------------------------------------------

def randomly_change_n_char(word, value):
    """From fair_entity_matching/synthetic dataset generator/FacultyMatch/faculty_match_for_em.py"""
    length = len(word)
    if length == 0 or value <= 0:
        return word
    value = min(value, length)
    word = list(word)
    k = random.sample(range(0, length), value)
    for index in k:
        word[index] = random.choice(string.ascii_lowercase)
    return "".join(word)


def randomly_add_n_char(word, value):
    """From fair_entity_matching/synthetic dataset generator/FacultyMatch/faculty_match_for_em.py"""
    length = len(word)
    if length == 0 or value <= 0:
        return word
    word = list(word)
    k = random.sample(range(0, length), min(value, length))
    for index in sorted(k, reverse=True):
        word.insert(index, random.choice(string.ascii_lowercase))
    return "".join(word)


def randomly_remove_n_char(word, value):
    """From fair_entity_matching/synthetic dataset generator/FacultyMatch/faculty_match_for_em.py"""
    length = len(word)
    if length <= value or value <= 0:
        return word
    word = list(word)
    k = random.sample(range(0, length), value)
    k.sort(reverse=True)
    for index in k:
        word.pop(index)
    return "".join(word)


def random_perturbation(word, value):
    """From fair_entity_matching/synthetic dataset generator/ — randomly choose perturbation type."""
    if not isinstance(word, str) or len(word) == 0:
        return str(word)
    choice = random.choice([1, 2, 3])
    if choice == 1:
        return randomly_change_n_char(word, value)
    elif choice == 2:
        return randomly_add_n_char(word, value)
    else:
        return randomly_remove_n_char(word, value)


# ---------------------------------------------------------------------------
# Table-level perturbations (new — extends character-level to schema-level)
# ---------------------------------------------------------------------------

def perturb_column_names(columns, n_perturb=1):
    """Apply character perturbation to column names (simulates schema noise)."""
    cols = list(columns)
    indices = random.sample(range(len(cols)), min(n_perturb, len(cols)))
    for i in indices:
        cols[i] = random_perturbation(cols[i], 1)
    return cols


def shuffle_rows(df):
    """Shuffle row order (simulates entropy/noise condition)."""
    return df.sample(frac=1, random_state=random.randint(0, 9999)).reset_index(drop=True)


def drop_random_rows(df, frac=0.1):
    """Remove a fraction of rows (simulates partial overlap)."""
    n_drop = max(1, int(len(df) * frac))
    return df.drop(df.sample(n=n_drop).index).reset_index(drop=True)


# ---------------------------------------------------------------------------
# FacultyMatch-based TUS Generator
# ---------------------------------------------------------------------------

def generate_faculty_tus_pairs(csranking_path, n_tables=30, table_size=20, seed=42):
    """
    Generate synthetic TUS table-pairs from csranking.csv.

    Creates tables by sampling subsets of faculty records.
    Unionable pairs share the same schema; non-unionable pairs have
    different schemas or perturbed column names.

    Each table has a demographic composition (% male, % female).
    The sensitive attribute for fairness testing is the majority
    demographic of the table-pair.

    Returns a DataFrame in a format compatible with Phase 1 experiments.
    """
    random.seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(csranking_path)
    df = df.drop(df[df['scholarid'] == 'NOSCHOLARPAGE'].index)
    df = df.drop_duplicates(subset=['name']).reset_index(drop=True)

    # Core columns for TUS
    schema_A = ['name', 'institution', 'region', 'countryabbrv', 'Gender']
    schema_B = ['name', 'homepage', 'scholarid']  # Different schema

    pairs = []
    pair_id = 0

    for _ in range(n_tables):
        # Sample two tables from the same domain
        if len(df) < table_size * 2:
            table_size = len(df) // 3

        sample_a = df.sample(n=table_size, replace=True)
        sample_b_union = df.sample(n=table_size, replace=True)
        sample_b_nonunion = df.sample(n=table_size, replace=True)

        # Demographics of each table
        gender_a = sample_a['Gender'].value_counts(normalize=True).to_dict()
        gender_b_union = sample_b_union['Gender'].value_counts(normalize=True).to_dict()
        gender_b_nonunion = sample_b_nonunion['Gender'].value_counts(normalize=True).to_dict()

        # --- Unionable pair (same schema) ---
        table_left = sample_a[schema_A].copy()
        table_right = sample_b_union[schema_A].copy()

        # Apply perturbations to right table (from original generator approach)
        for _, row in table_right.iterrows():
            if isinstance(row.get('name', ''), str) and len(row['name']) > 2:
                table_right.at[row.name, 'name'] = random_perturbation(row['name'], 1)

        # Determine dominant demographic
        male_frac = (gender_a.get('male', 0) + gender_b_union.get('male', 0)) / 2
        female_frac = (gender_a.get('female', 0) + gender_b_union.get('female', 0)) / 2

        pairs.append({
            'PairID': pair_id,
            'TableA_Schema': str(schema_A),
            'TableB_Schema': str(schema_A),
            'IsUnionable': 1,
            'DemographicAttr': 'Gender',
            'MaleFraction': round(male_frac, 3),
            'FemaleFraction': round(female_frac, 3),
            'DominantGroup': 'male' if male_frac > female_frac else 'female',
            'PerturbationLevel': 'low',
            'SchemaMatch': True,
            'N_Rows_A': len(table_left),
            'N_Rows_B': len(table_right),
        })
        pair_id += 1

        # --- Non-unionable pair (different schema) ---
        table_left_2 = sample_a[schema_A].copy()
        table_right_2 = sample_b_nonunion[schema_B].copy()

        pairs.append({
            'PairID': pair_id,
            'TableA_Schema': str(schema_A),
            'TableB_Schema': str(schema_B),
            'IsUnionable': 0,
            'DemographicAttr': 'Gender',
            'MaleFraction': round(male_frac, 3),
            'FemaleFraction': round(female_frac, 3),
            'DominantGroup': 'male' if male_frac > female_frac else 'female',
            'PerturbationLevel': 'none',
            'SchemaMatch': False,
            'N_Rows_A': len(table_left_2),
            'N_Rows_B': len(table_right_2),
        })
        pair_id += 1

        # --- Unionable pair with HIGH perturbation (harder) ---
        table_left_3 = sample_a[schema_A].copy()
        table_right_3 = sample_b_union[schema_A].copy()
        table_right_3.columns = perturb_column_names(table_right_3.columns, n_perturb=2)
        table_right_3 = shuffle_rows(table_right_3)

        pairs.append({
            'PairID': pair_id,
            'TableA_Schema': str(schema_A),
            'TableB_Schema': str(list(table_right_3.columns)),
            'IsUnionable': 1,
            'DemographicAttr': 'Gender',
            'MaleFraction': round(male_frac, 3),
            'FemaleFraction': round(female_frac, 3),
            'DominantGroup': 'male' if male_frac > female_frac else 'female',
            'PerturbationLevel': 'high',
            'SchemaMatch': False,
            'N_Rows_A': len(table_left_3),
            'N_Rows_B': len(table_right_3),
        })
        pair_id += 1

    return pd.DataFrame(pairs)


# ---------------------------------------------------------------------------
# Compas-based TUS Generator
# ---------------------------------------------------------------------------

def generate_compas_tus_pairs(compas_path, n_tables=30, table_size=50, seed=42):
    """
    Generate synthetic TUS table-pairs from compas-scores-raw.csv.

    Adapts the NoFlyCompas approach:
    - Original: 80% Caucasian / 20% African-American population split
    - Here: creates table-pairs where demographic composition varies

    Sensitive attribute: Ethnic_Code_Text
    """
    random.seed(seed)
    np.random.seed(seed)

    df = pd.read_csv(compas_path)
    df['FullName'] = df[['FirstName', 'LastName']].agg(' '.join, axis=1)
    df = df[['Person_ID', 'FullName', 'Ethnic_Code_Text', 'Sex_Code_Text',
             'Agency_Text', 'MaritalStatus', 'Language']].drop_duplicates(subset=['Person_ID'])

    # Filter to the two groups (same as original generator)
    df = df[(df['Ethnic_Code_Text'] == 'Caucasian') |
            (df['Ethnic_Code_Text'] == 'African-American')]

    caucasian = df[df['Ethnic_Code_Text'] == 'Caucasian']
    african_american = df[df['Ethnic_Code_Text'] == 'African-American']

    schema_A = ['FullName', 'Ethnic_Code_Text', 'Sex_Code_Text', 'MaritalStatus']
    schema_B = ['FullName', 'Agency_Text', 'Language']  # Different schema

    pairs = []
    pair_id = 0

    compositions = [
        ('balanced', 0.5, 0.5),
        ('majority_caucasian', 0.8, 0.2),   # Mirrors original generator's 80/20 split
        ('majority_aa', 0.2, 0.8),
    ]

    for comp_name, cauc_frac, aa_frac in compositions:
        for _ in range(n_tables // 3):
            n_cauc = max(1, int(table_size * cauc_frac))
            n_aa = max(1, int(table_size * aa_frac))

            if len(caucasian) < n_cauc or len(african_american) < n_aa:
                continue

            sample_cauc_a = caucasian.sample(n=n_cauc, replace=True)
            sample_aa_a = african_american.sample(n=n_aa, replace=True)
            table_a = pd.concat([sample_cauc_a, sample_aa_a]).sample(frac=1)

            sample_cauc_b = caucasian.sample(n=n_cauc, replace=True)
            sample_aa_b = african_american.sample(n=n_aa, replace=True)
            table_b = pd.concat([sample_cauc_b, sample_aa_b]).sample(frac=1)

            # Apply name perturbation (1 char, same as original Compas generator)
            table_b_perturbed = table_b.copy()
            table_b_perturbed['FullName'] = table_b_perturbed['FullName'].apply(
                lambda x: random_perturbation(x, 1)
            )

            # --- Unionable pair ---
            pairs.append({
                'PairID': pair_id,
                'TableA_Schema': str(schema_A),
                'TableB_Schema': str(schema_A),
                'IsUnionable': 1,
                'DemographicAttr': 'Ethnicity',
                'CaucasianFraction': round(cauc_frac, 3),
                'AfricanAmericanFraction': round(aa_frac, 3),
                'DominantGroup': 'Caucasian' if cauc_frac > aa_frac else 'African-American',
                'Composition': comp_name,
                'PerturbationLevel': 'low',
                'SchemaMatch': True,
                'N_Rows_A': len(table_a),
                'N_Rows_B': len(table_b),
            })
            pair_id += 1

            # --- Non-unionable pair ---
            table_b_diff = table_b[schema_B[:min(len(schema_B), len(table_b.columns))]].copy() \
                if all(c in table_b.columns for c in schema_B) else table_b.iloc[:, :3].copy()

            pairs.append({
                'PairID': pair_id,
                'TableA_Schema': str(schema_A),
                'TableB_Schema': str(schema_B),
                'IsUnionable': 0,
                'DemographicAttr': 'Ethnicity',
                'CaucasianFraction': round(cauc_frac, 3),
                'AfricanAmericanFraction': round(aa_frac, 3),
                'DominantGroup': 'Caucasian' if cauc_frac > aa_frac else 'African-American',
                'Composition': comp_name,
                'PerturbationLevel': 'none',
                'SchemaMatch': False,
                'N_Rows_A': len(table_a),
                'N_Rows_B': len(table_b_diff),
            })
            pair_id += 1

            # --- Unionable but heavily perturbed ---
            table_b_hard = table_b_perturbed[schema_A].copy()
            table_b_hard.columns = perturb_column_names(table_b_hard.columns, n_perturb=2)
            table_b_hard = shuffle_rows(table_b_hard)

            pairs.append({
                'PairID': pair_id,
                'TableA_Schema': str(schema_A),
                'TableB_Schema': str(list(table_b_hard.columns)),
                'IsUnionable': 1,
                'DemographicAttr': 'Ethnicity',
                'CaucasianFraction': round(cauc_frac, 3),
                'AfricanAmericanFraction': round(aa_frac, 3),
                'DominantGroup': 'Caucasian' if cauc_frac > aa_frac else 'African-American',
                'Composition': comp_name,
                'PerturbationLevel': 'high',
                'SchemaMatch': False,
                'N_Rows_A': len(table_a),
                'N_Rows_B': len(table_b_hard),
            })
            pair_id += 1

    return pd.DataFrame(pairs)


# ---------------------------------------------------------------------------
# Simulate annotator decisions on synthetic pairs
# ---------------------------------------------------------------------------

def simulate_annotator_decisions(
    pairs_df,
    n_annotators=30,
    native_frac=0.4,
    stem_frac=0.85,
    seed=42
):
    """
    Simulate annotator decisions on synthetic table pairs.

    Models the key PS2 hypothesis: annotator accuracy varies by
    demographic group AND by the demographic composition of the tables.

    Native speakers → slightly higher accuracy on text-heavy tables
    STEM experts → slightly higher accuracy on schema-matching tasks
    Non-native speakers → more clicks, longer decision times, lower accuracy
    """
    random.seed(seed)
    np.random.seed(seed)

    n_native = int(n_annotators * native_frac)
    n_nonnative = n_annotators - n_native
    n_stem = int(n_annotators * stem_frac)
    n_nonstem = n_annotators - n_stem

    annotators = []
    for i in range(n_annotators):
        annotators.append({
            'AnnotatorID': f'syn_{i}',
            'EngProf': random.choice([4, 5]) if i < n_native else random.choice([1, 2, 3]),
            'Major': random.choice([4, 5]) if i < n_stem else random.choice([1, 2, 3]),
            'Education': random.choice([1, 2, 3, 4]),
            'Age': random.choice([1, 2, 3]),
        })

    rows = []
    for _, pair in pairs_df.iterrows():
        ground_truth = pair['IsUnionable']
        perturb_level = pair['PerturbationLevel']
        schema_match = pair['SchemaMatch']

        for ann in annotators:
            # Base accuracy depends on task difficulty
            if schema_match and perturb_level == 'low':
                base_acc = 0.80  # Easy: schemas match, low perturbation
            elif not schema_match and perturb_level == 'none':
                base_acc = 0.85  # Easy: clearly different schemas
            else:
                base_acc = 0.55  # Hard: perturbed schemas, ambiguous

            # Demographic modifiers (models PS2 hypothesis)
            is_native = ann['EngProf'] >= 4
            is_stem = ann['Major'] >= 4

            if is_native:
                base_acc += 0.05  # Native speakers slightly better
            else:
                base_acc -= 0.03  # Non-native slightly worse

            if is_stem and schema_match:
                base_acc += 0.03  # STEM better on schema tasks
            elif not is_stem and not schema_match:
                base_acc += 0.02  # Non-STEM slightly better on ambiguous

            base_acc = np.clip(base_acc, 0.3, 0.95)

            # Simulate decision
            is_correct = random.random() < base_acc
            if is_correct:
                decision = ground_truth
            else:
                decision = 1 - ground_truth

            # Simulate behavioral signals
            if is_native:
                dec_time = np.random.exponential(0.005)
                clicks = np.random.poisson(0.5)
                confidence = np.clip(np.random.normal(0.75, 0.1), 0.1, 1.0)
            else:
                dec_time = np.random.exponential(0.010)  # Slower
                clicks = np.random.poisson(1.5)          # More clicks
                confidence = np.clip(np.random.normal(0.80, 0.12), 0.1, 1.0)  # Higher but miscalibrated

            rows.append({
                'PairID': pair['PairID'],
                'ByWho': ann['AnnotatorID'],
                'SurveyAnswer': decision,
                'ActualAnswer': ground_truth,
                'ConfidenceLevel': round(confidence, 4),
                'DecisionTime': round(dec_time, 6),
                'ClickCount': clicks,
                'Age': ann['Age'],
                'Education': ann['Education'],
                'EngProf': ann['EngProf'],
                'Major': ann['Major'],
                'PerturbationLevel': perturb_level,
                'DominantGroup': pair['DominantGroup'],
                'DemographicAttr': pair['DemographicAttr'],
                'Source': 'FacultyMatch' if pair['DemographicAttr'] == 'Gender' else 'Compas',
            })

    result = pd.DataFrame(rows)

    # Compute majority vote per pair
    majority = result.groupby('PairID')['SurveyAnswer'].mean().round().astype(int)
    result['Majority'] = result['PairID'].map(majority)

    return result
