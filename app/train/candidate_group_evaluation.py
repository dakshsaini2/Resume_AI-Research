"""
candidate_group_evaluation.py

Experiment 8: Candidate-Level Grouped Evaluation
==================================================

Purpose
-------
Evaluate whether the resume-job matching system generalizes
to completely unseen candidates.

Dataset structure discovered in Experiment 7:
    9544 samples
    344 unique resume representations
    28 unique job groups

Evaluation strategy:
    GroupShuffleSplit / GroupKFold by candidate resume

Important:
    The same candidate/resume representation must NEVER appear
    in both training and testing.

Models:
    1. Hybrid
    2. Structured ATS
    3. TF-IDF Only
    4. SBERT Only

Metrics:
    Regression:
        MAE
        RMSE
        R²

    Ranking:
        NDCG@1
        NDCG@3
        NDCG@5
        NDCG@10
        MRR
        Precision@3
        Recall@3

Outputs:
    dataset/processed/candidate_group_evaluation/
        candidate_group_results.csv
        candidate_group_summary.csv
        candidate_group_statistics.csv
        candidate_leakage_check.csv
"""

# ==========================================================
# IMPORTS
# ==========================================================

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")


from sklearn.model_selection import GroupKFold

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sentence_transformers import SentenceTransformer


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

OUTPUT_DIR = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "candidate_group_evaluation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# CONFIGURATION
# ==========================================================

RANDOM_STATE = 42

N_SPLITS = 5

N_ESTIMATORS = 300

MIN_SAMPLES_LEAF = 2

MIN_SAMPLES_SPLIT = 2


# ==========================================================
# HELPER
# ==========================================================

def section(title):

    print()

    print("=" * 70)

    print(title)

    print("=" * 70)


# ==========================================================
# LOAD PROJECT PREPROCESSING
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
# LOAD DATASET
# ==========================================================

section(
    "LOADING DATASET"
)


if not DATASET_FILE.exists():

    raise FileNotFoundError(

        f"Dataset not found:\n"
        f"{DATASET_FILE}"

    )


df = pd.read_csv(
    DATASET_FILE
)


df.columns = (

    df.columns
    .str.strip()
    .str.lower()

)


print()

print(
    "Dataset shape:",
    df.shape
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

        "Missing required columns:\n"

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

    subset=[
        "matched_score"
    ]

).reset_index(
    drop=True
)


df["matched_score"] = (

    df["matched_score"]

    .clip(
        0,
        1
    )

)


# ==========================================================
# BUILD TEXT REPRESENTATIONS
# ==========================================================

section(
    "BUILDING TEXT REPRESENTATIONS"
)


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
# NORMALIZED RESUME GROUP
# ==========================================================

def normalize_text(text):

    return (

        " ".join(

            str(text)
            .lower()
            .split()

        )

    )


df["candidate_group"] = (

    df["candidate_text"]

    .apply(
        normalize_text
    )

)


df["job_group"] = (

    df["job_text"]

    .apply(
        normalize_text
    )

)


# ==========================================================
# VERIFY CANDIDATE GROUPS
# ==========================================================

section(
    "CANDIDATE GROUP ANALYSIS"
)


candidate_counts = (

    df["candidate_group"]

    .value_counts()

)


print()

print(
    "Unique candidates:",
    len(candidate_counts)
)


print()

print(
    "Candidates per candidate group:"
)


print(
    candidate_counts.describe()
    .to_string()
)


# ==========================================================
# TEXT PREPROCESSING
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
    "BUILDING SKILL FEATURES"
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


def skill_match(row):

    candidate = (
        row["candidate_skill_set"]
    )

    required = (
        row["required_skill_set"]
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


df["skill_match_score"] = df.apply(

    skill_match,

    axis=1

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


# ==========================================================
# EDUCATION FEATURES
# ==========================================================

section(
    "BUILDING EDUCATION FEATURES"
)


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

        not required

        or

        required == "not specified"

    ):

        return 1.0

    if (

        not candidate

        or

        candidate == "not specified"

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

        candidate_level
        /
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

        or

        not required

        or

        candidate == "not specified"

        or

        required == "not specified"

    ):

        return 0

    return int(
        candidate == required
    )


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

section(
    "BUILDING EXPERIENCE FEATURES"
)


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


df["experience_match_score"] = df.apply(

    experience_score,

    axis=1

)


# ==========================================================
# CERTIFICATION FEATURES
# ==========================================================

section(
    "BUILDING CERTIFICATION FEATURES"
)


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
# LENGTH FEATURES
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
    "BUILDING TF-IDF"
)


tfidf = TfidfVectorizer(

    max_features=5000,

    ngram_range=(1, 2),

    stop_words="english",

)


# Important:
# TF-IDF is fitted separately inside each fold.
# This prevents test-candidate information from entering
# the vectorizer vocabulary/weights.

# ==========================================================
# SBERT
# ==========================================================

section(
    "LOADING SBERT"
)


sentence_model = SentenceTransformer(

    "all-MiniLM-L6-v2"

)


print()

print(
    "Generating candidate embeddings..."
)


candidate_embeddings = sentence_model.encode(

    df["candidate_clean"].tolist(),

    convert_to_numpy=True,

    show_progress_bar=True,

)


print()

print(
    "Generating job embeddings..."
)


job_embeddings = sentence_model.encode(

    df["job_clean"].tolist(),

    convert_to_numpy=True,

    show_progress_bar=True,

)


print()

print(
    "Calculating semantic similarity..."
)


semantic_similarity = np.array(

    [

        cosine_similarity(

            candidate_embeddings[i]
            .reshape(1, -1),

            job_embeddings[i]
            .reshape(1, -1)

        )[0][0]

        for i in range(len(df))

    ]

)


df["semantic_similarity"] = (
    semantic_similarity
)


# ==========================================================
# FEATURE DEFINITIONS
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
# NDCG
# ==========================================================

def dcg_at_k(
    relevance,
    k
):

    relevance = np.asarray(
        relevance
    )[:k]

    if len(relevance) == 0:

        return 0.0

    discounts = np.log2(

        np.arange(
            2,
            len(relevance) + 2
        )

    )

    return np.sum(

        (
            2 ** relevance
            - 1
        )

        /

        discounts

    )


def ndcg_at_k(
    y_true,
    y_pred,
    k
):

    order = np.argsort(
        y_pred
    )[::-1]

    predicted_relevance = (
        y_true[order]
    )

    ideal_relevance = np.sort(
        y_true
    )[::-1]

    dcg = dcg_at_k(
        predicted_relevance,
        k
    )

    idcg = dcg_at_k(
        ideal_relevance,
        k
    )

    if idcg == 0:

        return 0.0

    return dcg / idcg


# ==========================================================
# MRR
# ==========================================================

def reciprocal_rank(
    y_true,
    y_pred
):

    order = np.argsort(
        y_pred
    )[::-1]

    relevance = (
        y_true[order]
    )

    # Use the top candidate as the relevant threshold
    # based on the actual candidate relevance distribution.

    threshold = np.percentile(
        y_true,
        75
    )

    relevant_positions = np.where(

        relevance >= threshold

    )[0]

    if len(relevant_positions) == 0:

        return 0.0

    return 1.0 / (
        relevant_positions[0] + 1
    )


# ==========================================================
# PRECISION / RECALL
# ==========================================================

def precision_recall_at_k(

    y_true,

    y_pred,

    k

):

    order = np.argsort(
        y_pred
    )[::-1]

    top_indices = order[:k]

    threshold = np.percentile(
        y_true,
        75
    )

    relevant = (
        y_true >= threshold
    )

    retrieved_relevant = (

        relevant[
            top_indices
        ]

        .sum()

    )

    precision = (

        retrieved_relevant
        /
        min(
            k,
            len(y_true)
        )

    )

    total_relevant = (
        relevant.sum()
    )

    if total_relevant == 0:

        recall = 0.0

    else:

        recall = (

            retrieved_relevant
            /
            total_relevant

        )

    return precision, recall


# ==========================================================
# MODEL EVALUATION
# ==========================================================

def evaluate_regression(

    model,

    X_test,

    y_test

):

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

    return (

        predictions,

        mae,

        rmse,

        r2

    )


# ==========================================================
# GROUP SPLIT
# ==========================================================

section(
    "CANDIDATE-LEVEL GROUP SPLIT"
)


groups = df[
    "candidate_group"
]


print()

print(
    "Unique candidate groups:",
    groups.nunique()
)


if groups.nunique() < N_SPLITS:

    raise ValueError(

        f"Only {groups.nunique()} "
        f"candidate groups available. "
        f"Need at least {N_SPLITS}."

    )


group_kfold = GroupKFold(

    n_splits=N_SPLITS

)


# ==========================================================
# RESULTS
# ==========================================================

fold_results = []

group_statistics = []

leakage_results = []


# ==========================================================
# CROSS VALIDATION
# ==========================================================

section(
    "STARTING CANDIDATE-LEVEL CROSS VALIDATION"
)


for fold, (

    train_idx,

    test_idx

) in enumerate(

    group_kfold.split(

        df,

        df["matched_score"],

        groups=groups

    ),

    start=1

):

    print()

    print(
        "=" * 70
    )

    print(
        f"FOLD {fold}/{N_SPLITS}"
    )

    print(
        "=" * 70
    )


    train_df = df.iloc[
        train_idx
    ]

    test_df = df.iloc[
        test_idx
    ]


    train_groups = set(

        train_df[
            "candidate_group"
        ]

    )


    test_groups = set(

        test_df[
            "candidate_group"
        ]

    )


    overlap = (

        train_groups
        &
        test_groups

    )


    print()

    print(
        "Training samples:",
        len(train_df)
    )


    print(
        "Testing samples:",
        len(test_df)
    )


    print(
        "Training candidates:",
        len(train_groups)
    )


    print(
        "Testing candidates:",
        len(test_groups)
    )


    print(
        "Candidate overlap:",
        len(overlap)
    )


    if overlap:

        raise RuntimeError(

            "Candidate leakage detected "
            f"in fold {fold}."

        )


    leakage_results.append(

        {

            "Fold":
            fold,

            "Training_Candidates":
            len(train_groups),

            "Testing_Candidates":
            len(test_groups),

            "Overlapping_Candidates":
            len(overlap),

        }

    )


    # ======================================================
    # TF-IDF — FIT TRAIN ONLY
    # ======================================================

    print()

    print(
        "Fitting TF-IDF on training candidates..."
    )


    fold_tfidf = TfidfVectorizer(

        max_features=5000,

        ngram_range=(1, 2),

        stop_words="english",

    )


    train_all_text = pd.concat(

        [

            train_df[
                "candidate_clean"
            ],

            train_df[
                "job_clean"
            ],

        ]

    )


    fold_tfidf.fit(
        train_all_text
    )


    train_candidate_tfidf = (
        fold_tfidf.transform(
            train_df[
                "candidate_clean"
            ]
        )
    )


    train_job_tfidf = (
        fold_tfidf.transform(
            train_df[
                "job_clean"
            ]
        )
    )


    test_candidate_tfidf = (
        fold_tfidf.transform(
            test_df[
                "candidate_clean"
            ]
        )
    )


    test_job_tfidf = (
        fold_tfidf.transform(
            test_df[
                "job_clean"
            ]
        )
    )


    train_tfidf_scores = np.array(

        [

            cosine_similarity(

                train_candidate_tfidf[i],

                train_job_tfidf[i]

            )[0][0]

            for i in range(
                len(train_df)
            )

        ]

    )


    test_tfidf_scores = np.array(

        [

            cosine_similarity(

                test_candidate_tfidf[i],

                test_job_tfidf[i]

            )[0][0]

            for i in range(
                len(test_df)
            )

        ]

    )


    # ======================================================
    # FEATURE MATRICES
    # ======================================================

    train_features = train_df[
        STRUCTURED_FEATURES
    ].copy()


    test_features = test_df[
        STRUCTURED_FEATURES
    ].copy()


    train_features[
        "tfidf_similarity"
    ] = train_tfidf_scores


    test_features[
        "tfidf_similarity"
    ] = test_tfidf_scores


    train_features[
        "semantic_similarity"
    ] = train_df[
        "semantic_similarity"
    ].values


    test_features[
        "semantic_similarity"
    ] = test_df[
        "semantic_similarity"
    ].values


    hybrid_train = train_features[
        HYBRID_FEATURES
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)


    hybrid_test = test_features[
        HYBRID_FEATURES
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)


    structured_train = train_features[
        STRUCTURED_FEATURES
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)


    structured_test = test_features[
        STRUCTURED_FEATURES
    ].replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(0)


    y_train = train_df[
        "matched_score"
    ].values


    y_test = test_df[
        "matched_score"
    ].values


    # ======================================================
    # HYBRID
    # ======================================================

    print()

    print(
        "Training Hybrid..."
    )


    hybrid_model = RandomForestRegressor(

        n_estimators=N_ESTIMATORS,

        max_depth=None,

        min_samples_split=MIN_SAMPLES_SPLIT,

        min_samples_leaf=MIN_SAMPLES_LEAF,

        random_state=RANDOM_STATE,

        n_jobs=-1,

    )


    hybrid_model.fit(

        hybrid_train,

        y_train

    )


    hybrid_pred, hybrid_mae, hybrid_rmse, hybrid_r2 = (

        evaluate_regression(

            hybrid_model,

            hybrid_test,

            y_test

        )

    )


    # ======================================================
    # STRUCTURED ATS
    # ======================================================

    print(
        "Training Structured ATS..."
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

        structured_train,

        y_train

    )


    structured_pred, structured_mae, structured_rmse, structured_r2 = (

        evaluate_regression(

            structured_model,

            structured_test,

            y_test

        )

    )


    # ======================================================
    # TF-IDF ONLY
    # ======================================================

    print(
        "Training TF-IDF Only..."
    )


    tfidf_model = RandomForestRegressor(

        n_estimators=N_ESTIMATORS,

        random_state=RANDOM_STATE,

        n_jobs=-1,

    )


    tfidf_model.fit(

        train_tfidf_scores.reshape(-1, 1),

        y_train

    )


    tfidf_pred, tfidf_mae, tfidf_rmse, tfidf_r2 = (

        evaluate_regression(

            tfidf_model,

            test_tfidf_scores.reshape(-1, 1),

            y_test

        )

    )


    # ======================================================
    # SBERT ONLY
    # ======================================================

    print(
        "Training SBERT Only..."
    )


    sbert_model = RandomForestRegressor(

        n_estimators=N_ESTIMATORS,

        random_state=RANDOM_STATE,

        n_jobs=-1,

    )


    train_semantic = train_df[
        "semantic_similarity"
    ].values.reshape(-1, 1)


    test_semantic = test_df[
        "semantic_similarity"
    ].values.reshape(-1, 1)


    sbert_model.fit(

        train_semantic,

        y_train

    )


    sbert_pred, sbert_mae, sbert_rmse, sbert_r2 = (

        evaluate_regression(

            sbert_model,

            test_semantic,

            y_test

        )

    )


    # ======================================================
    # REGRESSION RESULTS
    # ======================================================

    models = [

        (

            "Hybrid",

            hybrid_pred,

            hybrid_mae,

            hybrid_rmse,

            hybrid_r2

        ),

        (

            "Structured ATS",

            structured_pred,

            structured_mae,

            structured_rmse,

            structured_r2

        ),

        (

            "TF-IDF Only",

            tfidf_pred,

            tfidf_mae,

            tfidf_rmse,

            tfidf_r2

        ),

        (

            "SBERT Only",

            sbert_pred,

            sbert_mae,

            sbert_rmse,

            sbert_r2

        ),

    ]


    # ======================================================
    # RANKING BY JOB
    # ======================================================

    print()

    print(
        "Evaluating candidate ranking within jobs..."
    )


    test_temp = test_df.copy()


    test_temp[
        "Hybrid"
    ] = hybrid_pred


    test_temp[
        "Structured ATS"
    ] = structured_pred


    test_temp[
        "TF-IDF Only"
    ] = tfidf_pred


    test_temp[
        "SBERT Only"
    ] = sbert_pred


    for (

        model_name,

        predictions,

        mae,

        rmse,

        r2

    ) in models:


        ndcg_values = {

            1: [],

            3: [],

            5: [],

            10: [],

        }


        mrr_values = []


        precision_values = {

            3: [],

        }


        recall_values = {

            3: [],

        }


        evaluated_groups = 0


        for job_group, group in (

            test_temp.groupby(
                "job_group"
            )

        ):

            if len(group) < 2:

                continue


            evaluated_groups += 1


            true_scores = (
                group[
                    "matched_score"
                ].values
            )


            pred_scores = (
                group[
                    model_name
                ].values
            )


            for k in ndcg_values:

                ndcg_values[k].append(

                    ndcg_at_k(

                        true_scores,

                        pred_scores,

                        k

                    )

                )


            mrr_values.append(

                reciprocal_rank(

                    true_scores,

                    pred_scores

                )

            )


            precision, recall = (

                precision_recall_at_k(

                    true_scores,

                    pred_scores,

                    3

                )

            )


            precision_values[
                3
            ].append(
                precision
            )


            recall_values[
                3
            ].append(
                recall
            )


        ranking_row = {

            "Fold":
            fold,

            "Model":
            model_name,

            "MAE":
            mae,

            "RMSE":
            rmse,

            "R2":
            r2,

            "NDCG@1":
            np.mean(
                ndcg_values[1]
            ),

            "NDCG@3":
            np.mean(
                ndcg_values[3]
            ),

            "NDCG@5":
            np.mean(
                ndcg_values[5]
            ),

            "NDCG@10":
            np.mean(
                ndcg_values[10]
            ),

            "MRR":
            np.mean(
                mrr_values
            ),

            "Precision@3":
            np.mean(
                precision_values[3]
            ),

            "Recall@3":
            np.mean(
                recall_values[3]
            ),

            "Evaluated_Job_Groups":
            evaluated_groups,

            "Training_Candidates":
            len(train_df),

            "Testing_Candidates":
            len(test_df),

            "Training_Unique_Candidates":
            len(train_groups),

            "Testing_Unique_Candidates":
            len(test_groups),

        }


        fold_results.append(
            ranking_row
        )


        print()

        print(
            model_name
        )

        print(
            f"MAE      : {mae:.4f}"
        )

        print(
            f"RMSE     : {rmse:.4f}"
        )

        print(
            f"R²       : {r2:.4f}"
        )

        print(
            f"NDCG@1   : "
            f"{ranking_row['NDCG@1']:.4f}"
        )

        print(
            f"NDCG@3   : "
            f"{ranking_row['NDCG@3']:.4f}"
        )

        print(
            f"NDCG@5   : "
            f"{ranking_row['NDCG@5']:.4f}"
        )

        print(
            f"NDCG@10  : "
            f"{ranking_row['NDCG@10']:.4f}"
        )

        print(
            f"MRR      : "
            f"{ranking_row['MRR']:.4f}"
        )

        print(
            f"Precision@3: "
            f"{ranking_row['Precision@3']:.4f}"
        )

        print(
            f"Recall@3  : "
            f"{ranking_row['Recall@3']:.4f}"
        )


# ==========================================================
# RESULTS DATAFRAME
# ==========================================================

section(
    "CANDIDATE-LEVEL RAW RESULTS"
)


results_df = pd.DataFrame(
    fold_results
)


print()

print(
    results_df.to_string(
        index=False
    )
)


# ==========================================================
# SUMMARY
# ==========================================================

section(
    "CANDIDATE-LEVEL SUMMARY"
)


metric_columns = [

    "MAE",

    "RMSE",

    "R2",

    "NDCG@1",

    "NDCG@3",

    "NDCG@5",

    "NDCG@10",

    "MRR",

    "Precision@3",

    "Recall@3",

]


summary_rows = []


for model_name in [

    "Hybrid",

    "Structured ATS",

    "TF-IDF Only",

    "SBERT Only",

]:

    model_results = results_df[
        results_df["Model"]
        ==
        model_name
    ]


    row = {

        "Model":
        model_name,

    }


    for metric in metric_columns:

        row[
            metric + "_mean"
        ] = model_results[
            metric
        ].mean()


        row[
            metric + "_std"
        ] = model_results[
            metric
        ].std()


    row[
        "Evaluated_Folds"
    ] = len(
        model_results
    )


    summary_rows.append(
        row
    )


summary_df = pd.DataFrame(
    summary_rows
)


print()

print(
    summary_df.to_string(
        index=False
    )
)


# ==========================================================
# BEST MODELS
# ==========================================================

section(
    "BEST CANDIDATE-LEVEL MODELS"
)


best_ndcg5 = summary_df.loc[

    summary_df[
        "NDCG@5_mean"
    ].idxmax()

]


best_ndcg10 = summary_df.loc[

    summary_df[
        "NDCG@10_mean"
    ].idxmax()

]


best_mae = summary_df.loc[

    summary_df[
        "MAE_mean"
    ].idxmin()

]


best_mrr = summary_df.loc[

    summary_df[
        "MRR_mean"
    ].idxmax()

]


print()

print(
    "Best NDCG@5:",
    best_ndcg5["Model"],
    f"({best_ndcg5['NDCG@5_mean']:.4f})"
)


print(
    "Best NDCG@10:",
    best_ndcg10["Model"],
    f"({best_ndcg10['NDCG@10_mean']:.4f})"
)


print(
    "Best MAE:",
    best_mae["Model"],
    f"({best_mae['MAE_mean']:.4f})"
)


print(
    "Best MRR:",
    best_mrr["Model"],
    f"({best_mrr['MRR_mean']:.4f})"
)


# ==========================================================
# LEAKAGE SUMMARY
# ==========================================================

section(
    "CANDIDATE LEAKAGE VALIDATION"
)


leakage_df = pd.DataFrame(
    leakage_results
)


print()

print(
    leakage_df.to_string(
        index=False
    )
)


total_overlap = (

    leakage_df[
        "Overlapping_Candidates"
    ]
    .sum()

)


if total_overlap == 0:

    print()

    print(
        "PASS: ZERO candidate leakage "
        "across all folds."
    )

else:

    raise RuntimeError(

        "Candidate leakage detected."

    )


# ==========================================================
# SAVE RESULTS
# ==========================================================

section(
    "SAVING RESULTS"
)


raw_results_file = (

    OUTPUT_DIR
    /
    "candidate_group_results.csv"

)


results_df.to_csv(

    raw_results_file,

    index=False

)


summary_file = (

    OUTPUT_DIR
    /
    "candidate_group_summary.csv"

)


summary_df.to_csv(

    summary_file,

    index=False

)


leakage_file = (

    OUTPUT_DIR
    /
    "candidate_group_statistics.csv"

)


leakage_df.to_csv(

    leakage_file,

    index=False

)


# ==========================================================
# FINAL
# ==========================================================

section(
    "EXPERIMENT 8 COMPLETE"
)


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


print()

print(
    "Leakage validation saved to:"
)


print(
    leakage_file
)


print()

print("=" * 70)

print(
    "CANDIDATE-LEVEL EVALUATION FINISHED SUCCESSFULLY"
)

print("=" * 70)