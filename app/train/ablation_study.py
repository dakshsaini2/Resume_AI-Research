"""
ablation_study.py

Experiment 2: Ablation Study
----------------------------

Purpose:
Determine the contribution of individual components
of the proposed hybrid resume screening framework.

Experiments:

1. Full Hybrid
2. Hybrid - TF-IDF
3. Hybrid - SBERT
4. Hybrid - Skills
5. Hybrid - Education
6. Hybrid - Experience
7. Hybrid - Certification

All experiments use:

- Same dataset
- Same 80/20 train-test split
- random_state = 42
- Same Random Forest configuration
- Same evaluation metrics

Metrics:
- MAE
- RMSE
- R²

Results:
dataset/processed/ablation_results.csv
"""


# ==========================================================
# IMPORTS
# ==========================================================

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

from sentence_transformers import SentenceTransformer

from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import train_test_split

from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings("ignore")


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

np.random.seed(
    RANDOM_STATE
)


# ==========================================================
# PROJECT PATHS
# ==========================================================

# File:
#
# Resume_AI_Research/
# └── app/
#     └── train/
#         └── ablation_study.py
#
# parents[0] = train
# parents[1] = app
# parents[2] = Resume_AI_Research

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


PROCESSED_DIR = (
    BASE_DIR
    / "dataset"
    / "processed"
)


PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# APP IMPORT PATH
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
    DEGREE_LEVELS
)


# ==========================================================
# CONFIGURATION
# ==========================================================

SBERT_MODEL = (
    "all-MiniLM-L6-v2"
)


TARGET_COLUMN = (
    "matched_score"
)


# ==========================================================
# RANDOM FOREST CONFIGURATION
# ==========================================================

# Same configuration used in Experiment 1.

RF_CONFIG = {

    "n_estimators": 300,

    "max_depth": None,

    "min_samples_split": 2,

    "min_samples_leaf": 2,

    "random_state": RANDOM_STATE,

    "n_jobs": -1,
}


# ==========================================================
# FEATURE GROUPS
# ==========================================================

TFIDF_FEATURES = [

    "tfidf_similarity",

]


SBERT_FEATURES = [

    "semantic_similarity",

]


SKILL_FEATURES = [

    "skill_match_score",

    "candidate_skill_count",

    "required_skill_count",

    "skill_overlap_count",

]


EDUCATION_FEATURES = [

    "education_match_score",

    "education_exact_match",

]


EXPERIENCE_FEATURES = [

    "experience_match_score",

]


CERTIFICATION_FEATURES = [

    "certification_match_score",

]


TEXT_STATISTICS = [

    "resume_length",

    "job_length",

    "resume_word_count",

    "job_word_count",

]


# ==========================================================
# COMPLETE HYBRID
# ==========================================================

HYBRID_FEATURES = (

    TFIDF_FEATURES

    +

    SBERT_FEATURES

    +

    SKILL_FEATURES

    +

    EDUCATION_FEATURES

    +

    EXPERIENCE_FEATURES

    +

    CERTIFICATION_FEATURES

    +

    TEXT_STATISTICS

)


# ==========================================================
# LOAD DATASET
# ==========================================================

print()

print("=" * 70)

print(
    "LOADING DATASET"
)

print("=" * 70)


if not DATASET_FILE.exists():

    raise FileNotFoundError(

        f"Dataset not found:\n"
        f"{DATASET_FILE}"

    )


df = pd.read_csv(
    DATASET_FILE
)


print(
    "Dataset shape:",
    df.shape
)


# ==========================================================
# NORMALIZE COLUMN NAMES
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


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    raise ValueError(

        "Missing required dataset columns:\n"

        +

        "\n".join(
            missing_columns
        )

    )


# ==========================================================
# HANDLE MISSING VALUES
# ==========================================================

for column in required_columns:

    df[column] = (

        df[column]

        .fillna("")

        .astype(str)

    )


# ==========================================================
# TARGET
# ==========================================================

df[
    TARGET_COLUMN
] = pd.to_numeric(

    df[
        TARGET_COLUMN
    ],

    errors="coerce"

)


df = df.dropna(

    subset=[
        TARGET_COLUMN
    ]

).copy()


df[
    TARGET_COLUMN
] = (

    df[
        TARGET_COLUMN
    ]

    .clip(
        0,
        1
    )

)


print()

print(
    "Number of samples:",
    len(df)
)


print(
    "Target range:",
    df[
        TARGET_COLUMN
    ].min(),

    "to",

    df[
        TARGET_COLUMN
    ].max()
)


# ==========================================================
# TEXT CONSTRUCTION
# ==========================================================

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


    return " ".join(
        values
    )


# ==========================================================
# CANDIDATE TEXT
# ==========================================================

candidate_columns = [

    "career_objective",

    "skills",

    "degree_names",

    "major_field_of_studies",

    "professional_company_names",

    "positions",

    "certification_skills",

]


# ==========================================================
# JOB TEXT
# ==========================================================

job_columns = [

    "job_position_name",

    "skills_required",

    "educational_requirements",

    "experiencere_requirement",

]


df[
    "candidate_text"
] = df.apply(

    lambda row:

    combine_columns(
        row,
        candidate_columns
    ),

    axis=1

)


df[
    "job_text"
] = df.apply(

    lambda row:

    combine_columns(
        row,
        job_columns
    ),

    axis=1

)


# ==========================================================
# PREPROCESSING
# ==========================================================

print()

print("=" * 70)

print(
    "PREPROCESSING TEXT"
)

print("=" * 70)


df[
    "candidate_clean"
] = (

    df[
        "candidate_text"
    ]

    .apply(

        lambda text:

        preprocess_resume(
            text
        )[
            "processed_text"
        ]

    )

)


df[
    "job_clean"
] = (

    df[
        "job_text"
    ]

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

print()
print("=" * 70)
print("BUILDING SKILL FEATURES")
print("=" * 70)


# Extract skills ONCE for every candidate.
print("Extracting candidate skills...")

df["candidate_skill_list"] = df[
    "candidate_text"
].apply(
    extract_skills
)


# Extract required skills ONCE for every job.
print("Extracting required skills...")

df["required_skill_list"] = df[
    "job_text"
].apply(
    extract_skills
)


# Convert to sets once.
print("Building skill sets...")

df["candidate_skill_set"] = df[
    "candidate_skill_list"
].apply(
    lambda x: set(x)
)


df["required_skill_set"] = df[
    "required_skill_list"
].apply(
    lambda x: set(x)
)


# ==========================================================
# SKILL COUNTS
# ==========================================================

df["candidate_skill_count"] = df[
    "candidate_skill_list"
].apply(
    len
)


df["required_skill_count"] = df[
    "required_skill_list"
].apply(
    len
)


# ==========================================================
# SKILL OVERLAP
# ==========================================================

df["skill_overlap_count"] = df.apply(

    lambda row:

    len(
        row["candidate_skill_set"]
        &
        row["required_skill_set"]
    ),

    axis=1
)


# ==========================================================
# SKILL MATCH SCORE
# ==========================================================

def calculate_skill_match(row):

    required = row[
        "required_skill_set"
    ]

    candidate = row[
        "candidate_skill_set"
    ]

    if not required:

        return 1.0

    return (

        len(
            candidate & required
        )
        /
        len(required)

    )


df["skill_match_score"] = df.apply(

    calculate_skill_match,

    axis=1

)

# ==========================================================
# EDUCATION FEATURES
# ==========================================================

def education_score(row):

    candidate = extract_degree(

        row[
            "candidate_text"
        ]

    )


    required = extract_degree(

        row[
            "job_text"
        ]

    )


    candidate = (

        candidate
        .strip()
        .lower()

    )


    required = (

        required
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


    return round(

        candidate_level
        /
        required_level,

        2

    )


def education_exact(row):

    candidate = (

        extract_degree(

            row[
                "candidate_text"
            ]

        )

        .strip()
        .lower()

    )


    required = (

        extract_degree(

            row[
                "job_text"
            ]

        )

        .strip()
        .lower()

    )


    if (

        candidate != "not specified"

        and

        required != "not specified"

        and

        candidate

        and

        required

    ):

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


df[
    "education_match_score"
] = df.apply(

    education_score,

    axis=1

)


df[
    "education_exact_match"
] = df.apply(

    education_exact,

    axis=1

)


# ==========================================================
# EXPERIENCE
# ==========================================================

def experience_score(row):

    candidate = extract_experience(

        row[
            "candidate_text"
        ]

    )


    required = extract_experience(

        row[
            "job_text"
        ]

    )


    if required <= 0:

        return 1.0


    if candidate <= 0:

        return 0.0


    return min(

        candidate
        /
        required,

        1.0

    )


df[
    "experience_match_score"
] = df.apply(

    experience_score,

    axis=1

)


# ==========================================================
# CERTIFICATION
# ==========================================================

def certification_score(row):

    candidate = set(

        extract_certifications(

            row[
                "candidate_text"
            ]

        )

    )


    required = set(

        extract_certifications(

            row[
                "job_text"
            ]

        )

    )


    if not required:

        return 1.0


    return (

        len(
            candidate & required
        )

        /

        len(
            required
        )

    )


df[
    "certification_match_score"
] = df.apply(

    certification_score,

    axis=1

)


# ==========================================================
# TEXT STATISTICS
# ==========================================================

df[
    "resume_length"
] = (

    df[
        "candidate_clean"
    ]

    .str.len()

)


df[
    "job_length"
] = (

    df[
        "job_clean"
    ]

    .str.len()

)


df[
    "resume_word_count"
] = (

    df[
        "candidate_clean"
    ]

    .str.split()
    .str.len()

)


df[
    "job_word_count"
] = (

    df[
        "job_clean"
    ]

    .str.split()
    .str.len()

)


# ==========================================================
# FIXED RESEARCH SPLIT
# ==========================================================

indices = np.arange(
    len(df)
)


train_indices, test_indices = (

    train_test_split(

        indices,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE

    )

)


print()

print("=" * 70)

print(
    "RESEARCH SPLIT"
)

print("=" * 70)


print(
    "Training samples:",
    len(train_indices)
)


print(
    "Test samples:",
    len(test_indices)
)


# ==========================================================
# TF-IDF
# ==========================================================

print()

print("=" * 70)

print(
    "BUILDING TF-IDF"
)

print("=" * 70)


tfidf = TfidfVectorizer(

    max_features=5000,

    ngram_range=(1, 2),

    stop_words="english",

)


# Fit ONLY on training text.

tfidf.fit(

    pd.concat(

        [

            df.iloc[
                train_indices
            ][
                "candidate_clean"
            ],

            df.iloc[
                train_indices
            ][
                "job_clean"
            ],

        ]

    )

)


candidate_vectors = tfidf.transform(

    df[
        "candidate_clean"
    ]

)


job_vectors = tfidf.transform(

    df[
        "job_clean"
    ]

)


tfidf_scores = np.array([

    cosine_similarity(

        candidate_vectors[i],

        job_vectors[i]

    )[0][0]

    for i in range(
        len(df)
    )

])


df[
    "tfidf_similarity"
] = tfidf_scores


# ==========================================================
# SBERT
# ==========================================================

print()

print("=" * 70)

print(
    "LOADING SBERT"
)

print("=" * 70)


sentence_model = SentenceTransformer(

    SBERT_MODEL

)


print()

print(
    "Generating resume embeddings..."
)


candidate_embeddings = (

    sentence_model.encode(

        df[
            "candidate_clean"
        ].tolist(),

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

        df[
            "job_clean"
        ].tolist(),

        convert_to_numpy=True,

        show_progress_bar=True,

    )

)


semantic_scores = np.array([

    cosine_similarity(

        candidate_embeddings[i]
        .reshape(1, -1),

        job_embeddings[i]
        .reshape(1, -1),

    )[0][0]

    for i in range(
        len(df)
    )

])


df[
    "semantic_similarity"
] = semantic_scores


# ==========================================================
# FEATURE GROUPS
# ==========================================================

# Complete proposed model.

HYBRID_FEATURES = (

    TFIDF_FEATURES

    +

    SBERT_FEATURES

    +

    SKILL_FEATURES

    +

    EDUCATION_FEATURES

    +

    EXPERIENCE_FEATURES

    +

    CERTIFICATION_FEATURES

    +

    TEXT_STATISTICS

)


# ==========================================================
# ABLATION DEFINITIONS
# ==========================================================

ABLATION_EXPERIMENTS = {

    "Full Hybrid":
        HYBRID_FEATURES,

    "Hybrid - TF-IDF":
        [
            feature
            for feature in HYBRID_FEATURES
            if feature
            not in TFIDF_FEATURES
        ],

    "Hybrid - SBERT":
        [
            feature
            for feature in HYBRID_FEATURES
            if feature
            not in SBERT_FEATURES
        ],

    "Hybrid - Skills":
        [
            feature
            for feature in HYBRID_FEATURES
            if feature
            not in SKILL_FEATURES
        ],

    "Hybrid - Education":
        [
            feature
            for feature in HYBRID_FEATURES
            if feature
            not in EDUCATION_FEATURES
        ],

    "Hybrid - Experience":
        [
            feature
            for feature in HYBRID_FEATURES
            if feature
            not in EXPERIENCE_FEATURES
        ],

    "Hybrid - Certification":
        [
            feature
            for feature in HYBRID_FEATURES
            if feature
            not in CERTIFICATION_FEATURES
        ],

}


# ==========================================================
# VALIDATE FEATURES
# ==========================================================

missing_features = []

for experiment_name, features in ABLATION_EXPERIMENTS.items():

    for feature in features:

        if feature not in df.columns:

            missing_features.append(
                feature
            )


missing_features = sorted(
    set(missing_features)
)


if missing_features:

    raise ValueError(

        "Missing engineered features:\n"

        +

        "\n".join(
            missing_features
        )

    )


# ==========================================================
# TRAINING FUNCTION
# ==========================================================

def train_and_evaluate(

    experiment_name,

    feature_columns,

):

    print()

    print("=" * 70)

    print(
        f"TRAINING: {experiment_name}"
    )

    print("=" * 70)


    print(
        "Number of features:",
        len(feature_columns)
    )


    X = df[
        feature_columns
    ].copy()


    y = df[
        TARGET_COLUMN
    ].copy()


    X = (

        X

        .replace(
            [np.inf, -np.inf],
            np.nan
        )

        .fillna(0)

    )


    X_train = X.iloc[
        train_indices
    ]


    X_test = X.iloc[
        test_indices
    ]


    y_train = y.iloc[
        train_indices
    ]


    y_test = y.iloc[
        test_indices
    ]


    model = RandomForestRegressor(

        **RF_CONFIG

    )


    model.fit(

        X_train,

        y_train

    )


    predictions = model.predict(

        X_test

    )


    predictions = np.clip(

        predictions,

        0,

        1

    )


    mae = mean_absolute_error(

        y_test,

        predictions

    )


    rmse = np.sqrt(

        mean_squared_error(

            y_test,

            predictions

        )

    )


    r2 = r2_score(

        y_test,

        predictions

    )


    print()

    print(
        f"MAE  : {mae:.4f}"
    )


    print(
        f"RMSE : {rmse:.4f}"
    )


    print(
        f"R²   : {r2:.4f}"
    )


    return {

        "Experiment":
            experiment_name,

        "MAE":
            mae,

        "RMSE":
            rmse,

        "R2":
            r2,

        "Number_of_features":
            len(feature_columns),

    }


# ==========================================================
# RUN ABLATION EXPERIMENTS
# ==========================================================

results = []


for experiment_name, feature_columns in (

    ABLATION_EXPERIMENTS.items()

):

    result = train_and_evaluate(

        experiment_name,

        feature_columns

    )

    results.append(
        result
    )


# ==========================================================
# RESULTS TABLE
# ==========================================================

results_df = pd.DataFrame(
    results
)


# Calculate performance change
# relative to the Full Hybrid model.

hybrid_row = results_df[
    results_df[
        "Experiment"
    ]
    == "Full Hybrid"
].iloc[0]


hybrid_mae = hybrid_row[
    "MAE"
]

hybrid_rmse = hybrid_row[
    "RMSE"
]

hybrid_r2 = hybrid_row[
    "R2"
]


results_df[
    "MAE_change_vs_hybrid"
] = (

    results_df[
        "MAE"
    ]

    -
    hybrid_mae

)


results_df[
    "RMSE_change_vs_hybrid"
] = (

    results_df[
        "RMSE"
    ]

    -
    hybrid_rmse

)


results_df[
    "R2_change_vs_hybrid"
] = (

    results_df[
        "R2"
    ]

    -
    hybrid_r2

)


# ==========================================================
# DISPLAY RESULTS
# ==========================================================

print()

print("=" * 70)

print(
    "ABLATION RESULTS"
)

print("=" * 70)


print(

    results_df.to_string(

        index=False

    )

)


# ==========================================================
# SAVE RESULTS
# ==========================================================

results_file = (

    PROCESSED_DIR

    / "ablation_results.csv"

)


results_df.to_csv(

    results_file,

    index=False

)


# ==========================================================
# SAVE FEATURE DEFINITIONS
# ==========================================================

feature_file = (

    PROCESSED_DIR

    / "ablation_feature_definitions.csv"

)


feature_rows = []


for experiment_name, features in (

    ABLATION_EXPERIMENTS.items()

):

    for feature in features:

        feature_rows.append({

            "Experiment":
                experiment_name,

            "Feature":
                feature,

        })


pd.DataFrame(

    feature_rows

).to_csv(

    feature_file,

    index=False

)


# ==========================================================
# COMPLETE
# ==========================================================

print()

print("=" * 70)

print(
    "EXPERIMENT 2 COMPLETE"
)

print("=" * 70)


print()

print(
    "Ablation results saved to:"
)

print(
    results_file
)


print()

print(
    "Feature definitions saved to:"
)

print(
    feature_file
)