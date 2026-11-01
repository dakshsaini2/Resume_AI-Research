"""
error_analysis.py

Experiment 6: Error Analysis + Statistical Significance
=========================================================

Research goals
--------------
1. Analyze prediction errors of the trained Hybrid model.
2. Compare Hybrid against Structured ATS.
3. Identify difficult job groups.
4. Analyze errors across score ranges.
5. Analyze over-prediction and under-prediction.
6. Perform paired statistical tests using fold-level results.
7. Generate research-ready CSV outputs and plots.

Inputs
------
models/best_resume_model.pkl
models/scaler.pkl
dataset/Resume_Data_For_Ranking.csv
dataset/processed/ranking_v2_fold_results.csv

Outputs
-------
dataset/processed/error_analysis/
    prediction_errors.csv
    error_by_score_range.csv
    error_by_job_group.csv
    model_comparison_statistics.csv
    statistical_tests.csv
    error_summary.csv
    error_distribution.png
    error_by_score_range.png
    model_metric_comparison.png
"""

# ==========================================================
# IMPORTS
# ==========================================================

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from scipy.stats import (
    ttest_rel,
    wilcoxon,
)

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

warnings.filterwarnings("ignore")


# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

APP_DIR = BASE_DIR / "app"

DATASET_FILE = (
    BASE_DIR
    / "dataset"
    / "Resume_Data_For_Ranking.csv"
)

MODEL_FILE = (
    BASE_DIR
    / "models"
    / "best_resume_model.pkl"
)

SCALER_FILE = (
    BASE_DIR
    / "models"
    / "scaler.pkl"
)

PROCESSED_DIR = (
    BASE_DIR
    / "dataset"
    / "processed"
)

OUTPUT_DIR = (
    PROCESSED_DIR
    / "error_analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# IMPORT PROJECT MODULES
# ==========================================================

if str(APP_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(APP_DIR)
    )


from preprocessing import (
    preprocess_resume,
    preprocess_job,
)

from utils import (
    extract_skills,
    extract_degree,
    extract_experience,
    extract_certifications,
)

from config import (
    DEGREE_LEVELS,
)


# ==========================================================
# CONFIGURATION
# ==========================================================

RANDOM_STATE = 42

N_ESTIMATORS = 300

MIN_SAMPLES_LEAF = 2

MIN_SAMPLES_SPLIT = 2

TOP_ERROR_COUNT = 100


# ==========================================================
# HELPER
# ==========================================================

def section(title):

    print()

    print("=" * 70)

    print(title)

    print("=" * 70)


# ==========================================================
# LOAD DATASET
# ==========================================================

section(
    "LOADING DATASET"
)


if not DATASET_FILE.exists():

    raise FileNotFoundError(
        f"\nDataset not found:\n"
        f"{DATASET_FILE}"
    )


df = pd.read_csv(
    DATASET_FILE
)


print()

print(
    "Dataset shape:",
    df.shape
)


# ==========================================================
# NORMALIZE COLUMNS
# ==========================================================

df.columns = (
    df.columns
    .str.strip()
    .str.lower()
)


# ==========================================================
# REQUIRED COLUMNS
# ==========================================================

required_columns = [

    "career_objective",
    "skills",
    "degree_names",
    "major_field_of_studies",
    "professional_company_names",
    "positions",
    "certification_skills",
    "job_position_name",
    "skills_required",
    "educational_requirements",
    "experiencere_requirement",
    "matched_score",

]


missing = [

    column
    for column in required_columns
    if column not in df.columns

]


if missing:

    raise ValueError(

        "Missing columns:\n"

        + "\n".join(missing)

    )


# ==========================================================
# CLEAN DATA
# ==========================================================

for column in required_columns:

    df[column] = (
        df[column]
        .fillna("")
        .astype(str)
    )


df["matched_score"] = pd.to_numeric(
    df["matched_score"],
    errors="coerce"
)


df = df.dropna(
    subset=["matched_score"]
).copy()


df["matched_score"] = (
    df["matched_score"]
    .clip(0, 1)
)


# ==========================================================
# TEXT CONSTRUCTION
# ==========================================================

section(
    "BUILDING TEXT REPRESENTATIONS"
)


def combine_columns(
    row,
    columns
):

    values = []

    for column in columns:

        value = str(
            row[column]
        ).strip()

        if value:

            values.append(
                value
            )

    return " ".join(values)


candidate_columns = [

    "career_objective",
    "skills",
    "degree_names",
    "major_field_of_studies",
    "professional_company_names",
    "positions",
    "certification_skills",

]


job_columns = [

    "job_position_name",
    "skills_required",
    "educational_requirements",
    "experiencere_requirement",

]


df["candidate_text"] = df.apply(

    lambda row:
    combine_columns(
        row,
        candidate_columns
    ),

    axis=1

)


df["job_text"] = df.apply(

    lambda row:
    combine_columns(
        row,
        job_columns
    ),

    axis=1

)


# ==========================================================
# JOB GROUP
# ==========================================================

df["job_group"] = (

    df["job_text"]

    .str.lower()

    .str.replace(
        r"\s+",
        " ",
        regex=True
    )

    .str.strip()

)


# ==========================================================
# PREPROCESSING
# ==========================================================

section(
    "PREPROCESSING TEXT"
)


df["candidate_clean"] = (

    df["candidate_text"]

    .apply(

        lambda text:

        preprocess_resume(
            text
        )[
            "processed_text"
        ]

    )

)


df["job_clean"] = (

    df["job_text"]

    .apply(

        lambda text:

        preprocess_job(
            text
        )[
            "processed_text"
        ]

    )

)


# ==========================================================
# SKILL FEATURES
# ==========================================================

section(
    "BUILDING STRUCTURED FEATURES"
)


print(
    "Extracting candidate skills..."
)


df["candidate_skill_list"] = (

    df["candidate_text"]

    .apply(
        extract_skills
    )

)


print(
    "Extracting required skills..."
)


df["required_skill_list"] = (

    df["job_text"]

    .apply(
        extract_skills
    )

)


df["candidate_skill_set"] = (

    df["candidate_skill_list"]

    .apply(
        set
    )

)


df["required_skill_set"] = (

    df["required_skill_list"]

    .apply(
        set
    )

)


df["candidate_skill_count"] = (

    df["candidate_skill_list"]

    .apply(
        len
    )

)


df["required_skill_count"] = (

    df["required_skill_list"]

    .apply(
        len
    )

)


df["skill_overlap_count"] = df.apply(

    lambda row:

    len(
        row["candidate_skill_set"]
        &
        row["required_skill_set"]
    ),

    axis=1

)


def skill_score(row):

    candidate = row[
        "candidate_skill_set"
    ]

    required = row[
        "required_skill_set"
    ]

    if not required:

        return 1.0

    return (

        len(
            candidate &
            required
        )

        /

        len(required)

    )


df["skill_match_score"] = df.apply(
    skill_score,
    axis=1
)


# ==========================================================
# EDUCATION
# ==========================================================

def education_score(row):

    candidate = (
        extract_degree(
            row["candidate_text"]
        )
        .strip()
        .lower()
    )

    required = (
        extract_degree(
            row["job_text"]
        )
        .strip()
        .lower()
    )

    if (
        required == "not specified"
        or not required
    ):

        return 1.0

    if (
        candidate == "not specified"
        or not candidate
    ):

        return 0.0

    candidate_level = (
        DEGREE_LEVELS.get(
            candidate,
            2
        )
    )

    required_level = (
        DEGREE_LEVELS.get(
            required,
            2
        )
    )

    if candidate_level >= required_level:

        return 1.0

    return min(
        candidate_level /
        required_level,
        1.0
    )


def education_exact(row):

    candidate = (
        extract_degree(
            row["candidate_text"]
        )
        .strip()
        .lower()
    )

    required = (
        extract_degree(
            row["job_text"]
        )
        .strip()
        .lower()
    )

    if (
        not candidate
        or not required
        or candidate == "not specified"
        or required == "not specified"
    ):

        return 0

    candidate_level = (
        DEGREE_LEVELS.get(
            candidate,
            2
        )
    )

    required_level = (
        DEGREE_LEVELS.get(
            required,
            2
        )
    )

    if candidate_level >= required_level:

        return int(
            candidate == required
        )

    return 0


df["education_match_score"] = df.apply(
    education_score,
    axis=1
)


df["education_exact_match"] = df.apply(
    education_exact,
    axis=1
)


# ==========================================================
# EXPERIENCE
# ==========================================================

def experience_score(row):

    candidate = extract_experience(
        row["candidate_text"]
    )

    required = extract_experience(
        row["job_text"]
    )

    if required <= 0:

        return 1.0

    if candidate <= 0:

        return 0.0

    return min(
        candidate / required,
        1.0
    )


df["experience_match_score"] = df.apply(
    experience_score,
    axis=1
)


# ==========================================================
# CERTIFICATION
# ==========================================================

def certification_score(row):

    candidate = set(
        extract_certifications(
            row["candidate_text"]
        )
    )

    required = set(
        extract_certifications(
            row["job_text"]
        )
    )

    if not required:

        return 1.0

    return (
        len(
            candidate &
            required
        )
        /
        len(required)
    )


df["certification_match_score"] = df.apply(
    certification_score,
    axis=1
)


# ==========================================================
# TEXT FEATURES
# ==========================================================

df["resume_length"] = (
    df["candidate_clean"]
    .str.len()
)

df["job_length"] = (
    df["job_clean"]
    .str.len()
)

df["resume_word_count"] = (
    df["candidate_clean"]
    .str.split()
    .str.len()
)

df["job_word_count"] = (
    df["job_clean"]
    .str.split()
    .str.len()
)


# ==========================================================
# TF-IDF
# ==========================================================

section(
    "BUILDING TF-IDF FEATURES"
)

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)


tfidf = TfidfVectorizer(

    max_features=5000,

    ngram_range=(1, 2),

    stop_words="english",

)


all_text = pd.concat(

    [
        df["candidate_clean"],
        df["job_clean"],
    ]

)


tfidf.fit(
    all_text
)


candidate_vectors = tfidf.transform(
    df["candidate_clean"]
)


job_vectors = tfidf.transform(
    df["job_clean"]
)


tfidf_scores = []


for i in range(len(df)):

    score = cosine_similarity(

        candidate_vectors[i],

        job_vectors[i]

    )[0][0]

    tfidf_scores.append(
        score
    )


df["tfidf_similarity"] = (
    tfidf_scores
)


# ==========================================================
# SEMANTIC SIMILARITY
# ==========================================================

section(
    "LOADING SBERT"
)


from sentence_transformers import (
    SentenceTransformer
)


sentence_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


print()

print(
    "Generating candidate embeddings..."
)


candidate_embeddings = (
    sentence_model.encode(

        df["candidate_clean"].tolist(),

        convert_to_numpy=True,

        show_progress_bar=True,

    )
)


print()

print(
    "Generating job embeddings..."
)


job_embeddings = (
    sentence_model.encode(

        df["job_clean"].tolist(),

        convert_to_numpy=True,

        show_progress_bar=True,

    )
)


print()

print(
    "Calculating semantic similarity..."
)


semantic_scores = []


for i in range(len(df)):

    score = cosine_similarity(

        candidate_embeddings[i]
        .reshape(1, -1),

        job_embeddings[i]
        .reshape(1, -1)

    )[0][0]

    semantic_scores.append(
        score
    )


df["semantic_similarity"] = (
    semantic_scores
)


# ==========================================================
# FEATURES
# ==========================================================

HYBRID_FEATURES = [

    "tfidf_similarity",

    "semantic_similarity",

    "skill_match_score",

    "education_match_score",

    "experience_match_score",

    "certification_match_score",

    "resume_length",

    "job_length",

    "resume_word_count",

    "job_word_count",

    "candidate_skill_count",

    "required_skill_count",

    "skill_overlap_count",

    "education_exact_match",

]


STRUCTURED_FEATURES = [

    "skill_match_score",

    "education_match_score",

    "experience_match_score",

    "certification_match_score",

    "resume_length",

    "job_length",

    "resume_word_count",

    "job_word_count",

    "candidate_skill_count",

    "required_skill_count",

    "skill_overlap_count",

    "education_exact_match",

]


# ==========================================================
# TRAIN/TEST SPLIT
# ==========================================================

section(
    "RESEARCH SPLIT"
)


from sklearn.model_selection import (
    GroupShuffleSplit
)


splitter = GroupShuffleSplit(

    n_splits=1,

    test_size=0.20,

    random_state=RANDOM_STATE,

)


train_indices, test_indices = next(

    splitter.split(

        df,

        df["matched_score"],

        groups=df["job_group"]

    )

)


print()

print(
    "Training samples:",
    len(train_indices)
)


print(
    "Testing samples:",
    len(test_indices)
)


print(
    "Training groups:",
    df.iloc[
        train_indices
    ]["job_group"].nunique()
)


print(
    "Testing groups:",
    df.iloc[
        test_indices
    ]["job_group"].nunique()
)


# ==========================================================
# TRAIN HYBRID MODEL
# ==========================================================

section(
    "TRAINING HYBRID MODEL"
)


X_hybrid = (

    df[HYBRID_FEATURES]

    .replace(
        [np.inf, -np.inf],
        np.nan
    )

    .fillna(0)

)


y = df[
    "matched_score"
]


hybrid_model = RandomForestRegressor(

    n_estimators=N_ESTIMATORS,

    max_depth=None,

    min_samples_split=MIN_SAMPLES_SPLIT,

    min_samples_leaf=MIN_SAMPLES_LEAF,

    random_state=RANDOM_STATE,

    n_jobs=-1,

)


hybrid_model.fit(

    X_hybrid.iloc[
        train_indices
    ],

    y.iloc[
        train_indices
    ]

)


hybrid_predictions = hybrid_model.predict(
    X_hybrid
)


hybrid_predictions = np.clip(
    hybrid_predictions,
    0,
    1
)


# ==========================================================
# TRAIN STRUCTURED ATS
# ==========================================================

section(
    "TRAINING STRUCTURED ATS MODEL"
)


X_structured = (

    df[STRUCTURED_FEATURES]

    .replace(
        [np.inf, -np.inf],
        np.nan
    )

    .fillna(0)

)


structured_model = RandomForestRegressor(

    n_estimators=N_ESTIMATORS,

    max_depth=None,

    min_samples_split=MIN_SAMPLES_SPLIT,

    min_samples_leaf=MIN_SAMPLES_LEAF,

    random_state=RANDOM_STATE,

    n_jobs=-1,

)


structured_model.fit(

    X_structured.iloc[
        train_indices
    ],

    y.iloc[
        train_indices
    ]

)


structured_predictions = (
    structured_model.predict(
        X_structured
    )
)


structured_predictions = np.clip(
    structured_predictions,
    0,
    1
)


# ==========================================================
# CREATE TEST DATAFRAME
# ==========================================================

test_df = df.iloc[
    test_indices
].copy()


test_df["hybrid_prediction"] = (
    hybrid_predictions[
        test_indices
    ]
)


test_df["structured_prediction"] = (
    structured_predictions[
        test_indices
    ]
)


test_df["hybrid_error"] = (
    test_df["hybrid_prediction"]
    -
    test_df["matched_score"]
)


test_df["hybrid_absolute_error"] = (
    test_df["hybrid_error"]
    .abs()
)


test_df["structured_error"] = (
    test_df["structured_prediction"]
    -
    test_df["matched_score"]
)


test_df["structured_absolute_error"] = (
    test_df["structured_error"]
    .abs()
)


# ==========================================================
# ERROR TYPE
# ==========================================================

def error_type(error):

    if error > 0:

        return "Over-prediction"

    if error < 0:

        return "Under-prediction"

    return "Exact"


test_df["hybrid_error_type"] = (
    test_df["hybrid_error"]
    .apply(error_type)
)


# ==========================================================
# ERROR MAGNITUDE
# ==========================================================

def error_category(error):

    error = abs(error)

    if error < 0.05:

        return "Very Low (<0.05)"

    if error < 0.10:

        return "Low (0.05-0.10)"

    if error < 0.20:

        return "Moderate (0.10-0.20)"

    if error < 0.30:

        return "High (0.20-0.30)"

    return "Very High (>=0.30)"


test_df["hybrid_error_category"] = (
    test_df["hybrid_absolute_error"]
    .apply(error_category)
)


# ==========================================================
# SCORE RANGE
# ==========================================================

def score_range(score):

    if score < 0.20:

        return "0.00-0.19"

    if score < 0.40:

        return "0.20-0.39"

    if score < 0.60:

        return "0.40-0.59"

    if score < 0.80:

        return "0.60-0.79"

    return "0.80-1.00"


test_df["target_score_range"] = (
    test_df["matched_score"]
    .apply(score_range)
)


# ==========================================================
# ERROR STATISTICS
# ==========================================================

section(
    "HYBRID ERROR STATISTICS"
)


test_mae = mean_absolute_error(

    test_df["matched_score"],

    test_df["hybrid_prediction"]

)


test_rmse = np.sqrt(

    mean_squared_error(

        test_df["matched_score"],

        test_df["hybrid_prediction"]

    )

)


test_r2 = r2_score(

    test_df["matched_score"],

    test_df["hybrid_prediction"]

)


print()

print(
    f"MAE  : {test_mae:.4f}"
)


print(
    f"RMSE : {test_rmse:.4f}"
)


print(
    f"R²   : {test_r2:.4f}"
)


# ==========================================================
# OVER / UNDER PREDICTION
# ==========================================================

section(
    "OVER-PREDICTION / UNDER-PREDICTION"
)


over_prediction_count = (

    test_df["hybrid_error"]
    > 0

).sum()


under_prediction_count = (

    test_df["hybrid_error"]
    < 0

).sum()


exact_count = (

    test_df["hybrid_error"]
    == 0

).sum()


total = len(test_df)


print()

print(
    "Over-predictions:",
    over_prediction_count,
    f"({over_prediction_count / total:.2%})"
)


print(
    "Under-predictions:",
    under_prediction_count,
    f"({under_prediction_count / total:.2%})"
)


print(
    "Exact predictions:",
    exact_count,
    f"({exact_count / total:.2%})"
)


# ==========================================================
# ERROR BY SCORE RANGE
# ==========================================================

section(
    "ERROR BY TARGET SCORE RANGE"
)


error_by_score = (

    test_df

    .groupby(
        "target_score_range"
    )

    .agg(

        Samples=(
            "matched_score",
            "count"
        ),

        Mean_Absolute_Error=(
            "hybrid_absolute_error",
            "mean"
        ),

        Median_Absolute_Error=(
            "hybrid_absolute_error",
            "median"
        ),

        Mean_Error=(
            "hybrid_error",
            "mean"
        ),

        RMSE=(
            "hybrid_error",
            lambda x:
            np.sqrt(
                np.mean(
                    x ** 2
                )
            )
        ),

    )

    .reset_index()

)


print()

print(
    error_by_score.to_string(
        index=False
    )
)


# ==========================================================
# ERROR BY JOB GROUP
# ==========================================================

section(
    "ERROR BY JOB GROUP"
)


error_by_job = (

    test_df

    .groupby(
        "job_group"
    )

    .agg(

        Samples=(
            "matched_score",
            "count"
        ),

        MAE=(
            "hybrid_absolute_error",
            "mean"
        ),

        Mean_Error=(
            "hybrid_error",
            "mean"
        ),

        RMSE=(
            "hybrid_error",
            lambda x:
            np.sqrt(
                np.mean(
                    x ** 2
                )
            )
        ),

        Mean_Target=(
            "matched_score",
            "mean"
        ),

    )

    .reset_index()

)


error_by_job = (

    error_by_job

    .sort_values(
        "MAE",
        ascending=False
    )

)


print()

print(
    error_by_job.head(
        15
    ).to_string(
        index=False
    )
)


# ==========================================================
# WORST PREDICTIONS
# ==========================================================

section(
    "TOP ERROR CASES"
)


worst_errors = (

    test_df

    .sort_values(
        "hybrid_absolute_error",
        ascending=False
    )

    .head(
        TOP_ERROR_COUNT
    )

    [

        [

            "job_group",

            "matched_score",

            "hybrid_prediction",

            "hybrid_error",

            "hybrid_absolute_error",

            "hybrid_error_type",

            "hybrid_error_category",

            "tfidf_similarity",

            "semantic_similarity",

            "skill_match_score",

            "education_match_score",

            "experience_match_score",

            "certification_match_score",

            "candidate_skill_count",

            "required_skill_count",

        ]

    ]

)


# ==========================================================
# MODEL COMPARISON
# ==========================================================

section(
    "MODEL ERROR COMPARISON"
)


hybrid_test_y = (
    test_df["matched_score"]
)

hybrid_test_pred = (
    test_df["hybrid_prediction"]
)

structured_test_pred = (
    test_df["structured_prediction"]
)


comparison = [

    {

        "Model":
        "Hybrid",

        "MAE":
        mean_absolute_error(
            hybrid_test_y,
            hybrid_test_pred
        ),

        "RMSE":
        np.sqrt(
            mean_squared_error(
                hybrid_test_y,
                hybrid_test_pred
            )
        ),

        "R2":
        r2_score(
            hybrid_test_y,
            hybrid_test_pred
        ),

    },

    {

        "Model":
        "Structured ATS",

        "MAE":
        mean_absolute_error(
            hybrid_test_y,
            structured_test_pred
        ),

        "RMSE":
        np.sqrt(
            mean_squared_error(
                hybrid_test_y,
                structured_test_pred
            )
        ),

        "R2":
        r2_score(
            hybrid_test_y,
            structured_test_pred
        ),

    },

]


comparison_df = pd.DataFrame(
    comparison
)


print()

print(
    comparison_df.to_string(
        index=False
    )
)


# ==========================================================
# STATISTICAL SIGNIFICANCE
# ==========================================================
#
# Paired comparison is performed on identical test samples.
#
# Null hypothesis:
# There is no difference between the absolute prediction
# errors of the two models.
#
# ==========================================================

section(
    "STATISTICAL SIGNIFICANCE"
)


hybrid_abs_error = (

    hybrid_test_y
    -
    hybrid_test_pred
).abs()


structured_abs_error = (

    hybrid_test_y
    -
    structured_test_pred
).abs()


# ----------------------------------------------------------
# Paired t-test
# ----------------------------------------------------------

t_stat, t_p = ttest_rel(

    hybrid_abs_error,

    structured_abs_error

)


# ----------------------------------------------------------
# Wilcoxon signed-rank test
# ----------------------------------------------------------

try:

    w_stat, w_p = wilcoxon(

        hybrid_abs_error,

        structured_abs_error

    )

except ValueError:

    w_stat = np.nan

    w_p = np.nan


statistical_result = pd.DataFrame(

    [

        {

            "Comparison":
            "Hybrid vs Structured ATS",

            "Metric":
            "Absolute Error",

            "Paired_t_statistic":
            t_stat,

            "Paired_t_p_value":
            t_p,

            "Wilcoxon_statistic":
            w_stat,

            "Wilcoxon_p_value":
            w_p,

        }

    ]

)


print()

print(
    statistical_result.to_string(
        index=False
    )
)


# ==========================================================
# INTERPRETATION
# ==========================================================

section(
    "STATISTICAL INTERPRETATION"
)


alpha = 0.05


if t_p < alpha:

    print()

    print(
        "Paired t-test:"
    )

    print(
        "Statistically significant "
        "difference detected "
        f"(p={t_p:.6f})."
    )

else:

    print()

    print(
        "Paired t-test:"
    )

    print(
        "No statistically significant "
        "difference detected "
        f"(p={t_p:.6f})."
    )


if not np.isnan(w_p):

    if w_p < alpha:

        print()

        print(
            "Wilcoxon test:"
        )

        print(
            "Statistically significant "
            "difference detected "
            f"(p={w_p:.6f})."
        )

    else:

        print()

        print(
            "Wilcoxon test:"
        )

        print(
            "No statistically significant "
            "difference detected "
            f"(p={w_p:.6f})."
        )


# ==========================================================
# SAVE CSV FILES
# ==========================================================

section(
    "SAVING ANALYSIS RESULTS"
)


prediction_errors_file = (

    OUTPUT_DIR
    / "prediction_errors.csv"

)


test_df.to_csv(

    prediction_errors_file,

    index=False

)


error_by_score_file = (

    OUTPUT_DIR
    / "error_by_score_range.csv"

)


error_by_score.to_csv(

    error_by_score_file,

    index=False

)


error_by_job_file = (

    OUTPUT_DIR
    / "error_by_job_group.csv"

)


error_by_job.to_csv(

    error_by_job_file,

    index=False

)


worst_errors_file = (

    OUTPUT_DIR
    / "worst_prediction_cases.csv"

)


worst_errors.to_csv(

    worst_errors_file,

    index=False

)


comparison_file = (

    OUTPUT_DIR
    / "model_comparison_statistics.csv"

)


comparison_df.to_csv(

    comparison_file,

    index=False

)


statistical_file = (

    OUTPUT_DIR
    / "statistical_tests.csv"

)


statistical_result.to_csv(

    statistical_file,

    index=False

)


summary_file = (

    OUTPUT_DIR
    / "error_summary.csv"

)


summary = pd.DataFrame(

    [

        {

            "Test_Samples":
            len(test_df),

            "MAE":
            test_mae,

            "RMSE":
            test_rmse,

            "R2":
            test_r2,

            "Over_Prediction_Rate":
            over_prediction_count / total,

            "Under_Prediction_Rate":
            under_prediction_count / total,

            "Mean_Absolute_Error":
            test_df[
                "hybrid_absolute_error"
            ].mean(),

            "Median_Absolute_Error":
            test_df[
                "hybrid_absolute_error"
            ].median(),

            "Maximum_Absolute_Error":
            test_df[
                "hybrid_absolute_error"
            ].max(),

            "Paired_t_p_value":
            t_p,

            "Wilcoxon_p_value":
            w_p,

        }

    ]

)


summary.to_csv(

    summary_file,

    index=False

)


# ==========================================================
# PLOT 1 — ERROR DISTRIBUTION
# ==========================================================

section(
    "GENERATING ERROR DISTRIBUTION PLOT"
)


plt.figure(
    figsize=(10, 6)
)


plt.hist(

    test_df[
        "hybrid_error"
    ],

    bins=40,

)


plt.axvline(

    0,

    linestyle="--"

)


plt.xlabel(
    "Prediction Error"
)


plt.ylabel(
    "Number of Samples"
)


plt.title(
    "Hybrid Model Prediction Error Distribution"
)


plt.tight_layout()


error_distribution_file = (

    OUTPUT_DIR
    / "error_distribution.png"

)


plt.savefig(

    error_distribution_file,

    dpi=300

)


plt.close()


# ==========================================================
# PLOT 2 — ERROR BY SCORE RANGE
# ==========================================================

plt.figure(
    figsize=(10, 6)
)


plot_data = error_by_score.copy()


plt.bar(

    plot_data[
        "target_score_range"
    ],

    plot_data[
        "Mean_Absolute_Error"
    ]

)


plt.xlabel(
    "Ground-Truth Match Score Range"
)


plt.ylabel(
    "Mean Absolute Error"
)


plt.title(
    "Hybrid Model Error Across Match Score Ranges"
)


plt.xticks(
    rotation=30,
    ha="right"
)


plt.tight_layout()


error_score_plot_file = (

    OUTPUT_DIR
    / "error_by_score_range.png"

)


plt.savefig(

    error_score_plot_file,

    dpi=300

)


plt.close()


# ==========================================================
# PLOT 3 — MODEL COMPARISON
# ==========================================================

plt.figure(
    figsize=(8, 6)
)


x = np.arange(

    len(
        comparison_df
    )

)


width = 0.35


plt.bar(

    x - width / 2,

    comparison_df[
        "MAE"
    ],

    width,

    label="MAE"

)


plt.bar(

    x + width / 2,

    comparison_df[
        "RMSE"
    ],

    width,

    label="RMSE"

)


plt.xticks(

    x,

    comparison_df[
        "Model"
    ]

)


plt.ylabel(
    "Error"
)


plt.title(
    "Prediction Error Comparison"
)


plt.legend()


plt.tight_layout()


model_comparison_plot_file = (

    OUTPUT_DIR
    / "model_metric_comparison.png"

)


plt.savefig(

    model_comparison_plot_file,

    dpi=300

)


plt.close()


# ==========================================================
# FINAL OUTPUT
# ==========================================================

section(
    "EXPERIMENT 6 COMPLETE"
)


print()

print(
    "Outputs generated:"
)


print()

print(
    prediction_errors_file
)


print(
    error_by_score_file
)


print(
    error_by_job_file
)


print(
    worst_errors_file
)


print(
    comparison_file
)


print(
    statistical_file
)


print(
    summary_file
)


print(
    error_distribution_file
)


print(
    error_score_plot_file
)


print(
    model_comparison_plot_file
)


print()

print("=" * 70)

print(
    "ERROR ANALYSIS FINISHED SUCCESSFULLY"
)

print("=" * 70)