"""
robustness_evaluation.py

Experiment 3: Robustness Evaluation
-----------------------------------

Purpose:
Evaluate whether the proposed hybrid resume screening
framework remains stable across multiple train/test splits.

Models:
1. Hybrid
2. Hybrid - TF-IDF
3. Structured ATS
4. TF-IDF Only
5. SBERT Only

Evaluation:
- 5 random seeds
- Same 80/20 train-test ratio
- MAE
- RMSE
- R²
- Mean
- Standard deviation

Outputs:
dataset/processed/robustness_results.csv
dataset/processed/robustness_summary.csv
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

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics.pairwise import (
    cosine_similarity
)


warnings.filterwarnings("ignore")


# ==========================================================
# PROJECT CONFIGURATION
# ==========================================================

SBERT_MODEL = "all-MiniLM-L6-v2"

TARGET_COLUMN = "matched_score"

TEST_SIZE = 0.20


# ==========================================================
# ROBUSTNESS SEEDS
# ==========================================================

SEEDS = [
    42,
    7,
    21,
    35,
    100,
]


# ==========================================================
# RANDOM FOREST CONFIGURATION
# ==========================================================

RF_CONFIG = {

    "n_estimators": 300,

    "max_depth": None,

    "min_samples_split": 2,

    "min_samples_leaf": 2,

    "n_jobs": -1,

}


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
# IMPORT APP MODULES
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


from config import DEGREE_LEVELS


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
# FULL HYBRID FEATURE SET
# ==========================================================

HYBRID_FEATURES = (

    TFIDF_FEATURES

    + SBERT_FEATURES

    + SKILL_FEATURES

    + EDUCATION_FEATURES

    + EXPERIENCE_FEATURES

    + CERTIFICATION_FEATURES

    + TEXT_STATISTICS

)


# ==========================================================
# MODEL DEFINITIONS
# ==========================================================

MODEL_FEATURES = {

    "Hybrid":

        HYBRID_FEATURES,


    "Hybrid - TF-IDF":

        [

            feature

            for feature in HYBRID_FEATURES

            if feature not in TFIDF_FEATURES

        ],


    "Structured ATS":

        (

            SKILL_FEATURES

            + EDUCATION_FEATURES

            + EXPERIENCE_FEATURES

            + CERTIFICATION_FEATURES

            + TEXT_STATISTICS

        ),


    "TF-IDF Only":

        TFIDF_FEATURES,


    "SBERT Only":

        SBERT_FEATURES,

}


# ==========================================================
# LOAD DATASET
# ==========================================================

print()

print("=" * 70)

print("LOADING DATASET")

print("=" * 70)


if not DATASET_FILE.exists():

    raise FileNotFoundError(

        f"\nDataset not found:\n"
        f"{DATASET_FILE}\n"

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

        "\nMissing required dataset columns:\n"

        + "\n".join(

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

df[TARGET_COLUMN] = pd.to_numeric(

    df[TARGET_COLUMN],

    errors="coerce"

)


df = df.dropna(

    subset=[

        TARGET_COLUMN

    ]

).copy()


df[TARGET_COLUMN] = (

    df[TARGET_COLUMN]

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

    df[TARGET_COLUMN].min(),

    "to",

    df[TARGET_COLUMN].max()

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
# CANDIDATE COLUMNS
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
# JOB COLUMNS
# ==========================================================

job_columns = [

    "job_position_name",

    "skills_required",

    "educational_requirements",

    "experiencere_requirement",

]


# ==========================================================
# BUILD TEXT
# ==========================================================

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
# PREPROCESSING
# ==========================================================

print()

print("=" * 70)

print("PREPROCESSING TEXT")

print("=" * 70)


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

print()

print("=" * 70)

print("BUILDING SKILL FEATURES")

print("=" * 70)


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


print(

    "Building skill sets..."

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


def calculate_skill_match(

    row

):

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

            candidate & required

        )

        /

        len(

            required

        )

    )


df["skill_match_score"] = df.apply(

    calculate_skill_match,

    axis=1

)


# ==========================================================
# EDUCATION FEATURES
# ==========================================================

print()

print(

    "Building education features..."

)


def education_score(

    row

):

    candidate = extract_degree(

        row["candidate_text"]

    )


    required = extract_degree(

        row["job_text"]

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


def education_exact(

    row

):

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


df["education_match_score"] = df.apply(

    education_score,

    axis=1

)


df["education_exact_match"] = df.apply(

    education_exact,

    axis=1

)


# ==========================================================
# EXPERIENCE FEATURES
# ==========================================================

print()

print(

    "Building experience features..."

)


def experience_score(

    row

):

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

        candidate

        /

        required,

        1.0

    )


df["experience_match_score"] = df.apply(

    experience_score,

    axis=1

)


# ==========================================================
# CERTIFICATION FEATURES
# ==========================================================

print()

print(

    "Building certification features..."

)


def certification_score(

    row

):

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

            candidate & required

        )

        /

        len(

            required

        )

    )


df["certification_match_score"] = df.apply(

    certification_score,

    axis=1

)


# ==========================================================
# TEXT STATISTICS
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
# LOAD SBERT
# ==========================================================

print()

print("=" * 70)

print("LOADING SBERT")

print("=" * 70)


sentence_model = SentenceTransformer(

    SBERT_MODEL

)


# ==========================================================
# SBERT EMBEDDINGS
# ==========================================================

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


# ==========================================================
# SBERT SIMILARITY
# ==========================================================

semantic_scores = np.array(

    [

        cosine_similarity(

            candidate_embeddings[i]

            .reshape(

                1,

                -1

            ),

            job_embeddings[i]

            .reshape(

                1,

                -1

            ),

        )[0][0]

        for i in range(

            len(df)

        )

    ]

)


df[

    "semantic_similarity"

] = semantic_scores


# ==========================================================
# TF-IDF FUNCTION
# ==========================================================

def build_tfidf_similarity(

    train_indices

):

    vectorizer = TfidfVectorizer(

        max_features=5000,

        ngram_range=(1, 2),

        stop_words="english",

    )


    # IMPORTANT:
    # Fit TF-IDF ONLY on the training portion
    # of the current robustness split.

    training_text = pd.concat(

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


    vectorizer.fit(

        training_text

    )


    candidate_vectors = (

        vectorizer.transform(

            df[

                "candidate_clean"

            ]

        )

    )


    job_vectors = (

        vectorizer.transform(

            df[

                "job_clean"

            ]

        )

    )


    scores = np.array(

        [

            cosine_similarity(

                candidate_vectors[i],

                job_vectors[i]

            )[0][0]

            for i in range(

                len(df)

            )

        ]

    )


    return scores


# ==========================================================
# MODEL EVALUATION
# ==========================================================

def evaluate_model(

    model_name,

    feature_columns,

    train_indices,

    test_indices,

    seed,

):

    X = df[

        feature_columns

    ].copy()


    y = df[

        TARGET_COLUMN

    ].copy()


    # ------------------------------------------------------
    # Numerical safety
    # ------------------------------------------------------

    X = (

        X

        .replace(

            [np.inf, -np.inf],

            np.nan

        )

        .fillna(0)

    )


    # ------------------------------------------------------
    # Train / test data
    # ------------------------------------------------------

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


    # ------------------------------------------------------
    # RANDOM FOREST
    #
    # IMPORTANT:
    # The model uses the CURRENT robustness seed.
    # ------------------------------------------------------

    model = RandomForestRegressor(

        n_estimators=RF_CONFIG[

            "n_estimators"

        ],

        max_depth=RF_CONFIG[

            "max_depth"

        ],

        min_samples_split=RF_CONFIG[

            "min_samples_split"

        ],

        min_samples_leaf=RF_CONFIG[

            "min_samples_leaf"

        ],

        random_state=seed,

        n_jobs=RF_CONFIG[

            "n_jobs"

        ],

    )


    # ------------------------------------------------------
    # TRAIN
    # ------------------------------------------------------

    model.fit(

        X_train,

        y_train

    )


    # ------------------------------------------------------
    # PREDICT
    # ------------------------------------------------------

    predictions = model.predict(

        X_test

    )


    # Keep predictions within valid score range.

    predictions = np.clip(

        predictions,

        0,

        1

    )


    # ------------------------------------------------------
    # METRICS
    # ------------------------------------------------------

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


    return {

        "Seed":
            seed,

        "Model":
            model_name,

        "MAE":
            mae,

        "RMSE":
            rmse,

        "R2":
            r2,

    }


# ==========================================================
# START EXPERIMENT
# ==========================================================

print()

print("=" * 70)

print(

    "STARTING ROBUSTNESS EVALUATION"

)

print("=" * 70)


all_results = []


# ==========================================================
# MULTI-SEED EVALUATION
# ==========================================================

for seed in SEEDS:

    print()

    print("=" * 70)

    print(

        f"RANDOM SEED: {seed}"

    )

    print("=" * 70)


    # ------------------------------------------------------
    # CREATE SPLIT
    # ------------------------------------------------------

    indices = np.arange(

        len(df)

    )


    train_indices, test_indices = (

        train_test_split(

            indices,

            test_size=TEST_SIZE,

            random_state=seed,

        )

    )


    print(

        "Training:",

        len(train_indices)

    )


    print(

        "Testing:",

        len(test_indices)

    )


    # ------------------------------------------------------
    # BUILD TF-IDF FOR CURRENT SEED
    # ------------------------------------------------------

    print(

        "Building seed-specific TF-IDF..."

    )


    df[

        "tfidf_similarity"

    ] = build_tfidf_similarity(

        train_indices

    )


    # ------------------------------------------------------
    # EVALUATE EACH MODEL
    # ------------------------------------------------------

    for (

        model_name,

        feature_columns

    ) in MODEL_FEATURES.items():

        print()

        print(

            f"Evaluating {model_name}..."

        )


        metrics = evaluate_model(

            model_name,

            feature_columns,

            train_indices,

            test_indices,

            seed,

        )


        all_results.append(

            metrics

        )


        print(

            f"MAE={metrics['MAE']:.4f} | "

            f"RMSE={metrics['RMSE']:.4f} | "

            f"R²={metrics['R2']:.4f}"

        )


# ==========================================================
# RAW RESULTS
# ==========================================================

results_df = pd.DataFrame(

    all_results

)


results_df = results_df[

    [

        "Seed",

        "Model",

        "MAE",

        "RMSE",

        "R2",

    ]

]


# ==========================================================
# SUMMARY STATISTICS
# ==========================================================

summary_df = (

    results_df

    .groupby(

        "Model"

    )

    .agg(

        MAE_mean=(

            "MAE",

            "mean"

        ),

        MAE_std=(

            "MAE",

            "std"

        ),

        RMSE_mean=(

            "RMSE",

            "mean"

        ),

        RMSE_std=(

            "RMSE",

            "std"

        ),

        R2_mean=(

            "R2",

            "mean"

        ),

        R2_std=(

            "R2",

            "std"

        ),

    )

    .reset_index()

)


# ==========================================================
# SORT BY MEAN MAE
# ==========================================================

summary_df = (

    summary_df

    .sort_values(

        by="MAE_mean",

        ascending=True

    )

    .reset_index(

        drop=True

    )

)


# ==========================================================
# DISPLAY RAW RESULTS
# ==========================================================

print()

print("=" * 70)

print(

    "ROBUSTNESS RAW RESULTS"

)

print("=" * 70)


print(

    results_df.to_string(

        index=False

    )

)


# ==========================================================
# DISPLAY SUMMARY
# ==========================================================

print()

print("=" * 70)

print(

    "ROBUSTNESS SUMMARY"

)

print("=" * 70)


print(

    summary_df.to_string(

        index=False

    )

)


# ==========================================================
# SAVE RAW RESULTS
# ==========================================================

raw_results_file = (

    PROCESSED_DIR

    / "robustness_results.csv"

)


results_df.to_csv(

    raw_results_file,

    index=False

)


# ==========================================================
# SAVE SUMMARY
# ==========================================================

summary_file = (

    PROCESSED_DIR

    / "robustness_summary.csv"

)


summary_df.to_csv(

    summary_file,

    index=False

)


# ==========================================================
# FINAL MESSAGE
# ==========================================================

print()

print("=" * 70)

print(

    "EXPERIMENT 3 COMPLETE"

)

print("=" * 70)


print()

print(

    "Raw results saved to:"

)

print(

    raw_results_file

)


print()

print(

    "Summary saved to:"

)

print(

    summary_file

)