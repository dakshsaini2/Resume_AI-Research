"""
ranking_experiment_v2.py

Experiment 5 (Final): Candidate Ranking Evaluation
===================================================

Research objective
------------------
Evaluate the ability of the resume screening framework to rank
candidates for the same job.

Important methodological decisions
----------------------------------
1. Job groups are kept intact using GroupKFold.
2. TF-IDF is fitted only on each training fold.
3. Models are trained only on training candidates.
4. Ranking is evaluated only on unseen job groups.
5. Continuous matched_score is used for NDCG.
6. Binary relevance is used for MRR / Precision / Recall.
7. Results are reported as mean +/- standard deviation.

Models
------
1. Hybrid
2. Structured ATS
3. TF-IDF Only
4. SBERT Only

Metrics
-------
NDCG@1
NDCG@3
NDCG@5
NDCG@10
NDCG@20

MRR

Precision@3
Precision@5
Precision@10
Precision@20

Recall@3
Recall@5
Recall@10
Recall@20

Outputs
-------
dataset/processed/ranking_v2_results.csv
dataset/processed/ranking_v2_summary.csv
dataset/processed/ranking_v2_fold_results.csv
dataset/processed/ranking_v2_group_statistics.csv
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

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import GroupKFold

from sklearn.metrics import ndcg_score

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

N_SPLITS = 5

MIN_CANDIDATES_PER_JOB = 5

RELEVANCE_THRESHOLD = 0.70

SBERT_MODEL = "all-MiniLM-L6-v2"


# ==========================================================
# RANDOM FOREST
# ==========================================================

RF_CONFIG = {

    "n_estimators": 300,

    "max_depth": None,

    "min_samples_split": 2,

    "min_samples_leaf": 2,

    "n_jobs": -1,

}


# ==========================================================
# TOP-K VALUES
# ==========================================================

K_VALUES = [

    1,
    3,
    5,
    10,
    20,

]


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


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    raise ValueError(

        "\nMissing required columns:\n"

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

df["matched_score"] = pd.to_numeric(

    df["matched_score"],

    errors="coerce"

)


df = df.dropna(

    subset=[

        "matched_score"

    ]

).copy()


df["matched_score"] = (

    df["matched_score"]

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

    df["matched_score"].min(),

    "to",

    df["matched_score"].max()

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


print()

print("=" * 70)

print("BUILDING TEXT REPRESENTATIONS")

print("=" * 70)


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
#
# Same job description = same ranking group.
#
# We normalize whitespace and lowercase the complete job text.
#
# ==========================================================

print()

print("=" * 70)

print("BUILDING JOB GROUPS")

print("=" * 70)


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


group_counts = (

    df["job_group"]

    .value_counts()

)


valid_groups = group_counts[

    group_counts >= MIN_CANDIDATES_PER_JOB

].index


df = df[

    df["job_group"].isin(

        valid_groups

    )

].copy()


df = df.reset_index(

    drop=True

)


print()

print(

    "Unique job groups:",

    df["job_group"].nunique()

)


print(

    "Eligible samples:",

    len(df)

)


print()

print(

    "Candidate distribution per job:"

)


print(

    df["job_group"]

    .value_counts()

    .describe()

)


# ==========================================================
# PREPROCESS TEXT
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

        lambda x:

        set(x)

    )

)


df["required_skill_set"] = (

    df["required_skill_list"]

    .apply(

        lambda x:

        set(x)

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
# EDUCATION
# ==========================================================

print()

print(

    "Building education features..."

)


def calculate_education_score(

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


def calculate_education_exact(

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

    calculate_education_score,

    axis=1

)


df["education_exact_match"] = df.apply(

    calculate_education_exact,

    axis=1

)


# ==========================================================
# EXPERIENCE
# ==========================================================

print()

print(

    "Building experience features..."

)


def calculate_experience_score(

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

    calculate_experience_score,

    axis=1

)


# ==========================================================
# CERTIFICATION
# ==========================================================

print()

print(

    "Building certification features..."

)


def calculate_certification_score(

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

    calculate_certification_score,

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


# ==========================================================
# SEMANTIC SIMILARITY
# ==========================================================

print()

print(

    "Calculating semantic similarity..."

)


semantic_scores = []


for i in range(

    len(df)

):

    score = cosine_similarity(

        candidate_embeddings[i]

        .reshape(

            1,

            -1

        ),

        job_embeddings[i]

        .reshape(

            1,

            -1

        )

    )[0][0]


    semantic_scores.append(

        score

    )


df["semantic_similarity"] = (

    semantic_scores

)


# ==========================================================
# FEATURE DEFINITIONS
# ==========================================================

TFIDF_FEATURES = [

    "tfidf_similarity"

]


SBERT_FEATURES = [

    "semantic_similarity"

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

    "experience_match_score"

]


CERTIFICATION_FEATURES = [

    "certification_match_score"

]


TEXT_FEATURES = [

    "resume_length",

    "job_length",

    "resume_word_count",

    "job_word_count",

]


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

    TEXT_FEATURES

)


STRUCTURED_FEATURES = (

    SKILL_FEATURES

    +

    EDUCATION_FEATURES

    +

    EXPERIENCE_FEATURES

    +

    CERTIFICATION_FEATURES

    +

    TEXT_FEATURES

)


# ==========================================================
# MODEL TRAINING
# ==========================================================

def train_random_forest(

    X_train,

    y_train,

    seed

):

    model = RandomForestRegressor(

        n_estimators=

        RF_CONFIG[

            "n_estimators"

        ],

        max_depth=

        RF_CONFIG[

            "max_depth"

        ],

        min_samples_split=

        RF_CONFIG[

            "min_samples_split"

        ],

        min_samples_leaf=

        RF_CONFIG[

            "min_samples_leaf"

        ],

        random_state=seed,

        n_jobs=

        RF_CONFIG[

            "n_jobs"

        ],

    )


    model.fit(

        X_train,

        y_train

    )


    return model


# ==========================================================
# MRR
# ==========================================================

def calculate_mrr(

    relevance,

    predictions

):

    ranking = np.argsort(

        -predictions

    )


    sorted_relevance = relevance[

        ranking

    ]


    relevant_positions = np.where(

        sorted_relevance > 0

    )[0]


    if len(

        relevant_positions

    ) == 0:

        return 0.0


    return (

        1.0

        /

        (

            relevant_positions[0]

            + 1

        )

    )


# ==========================================================
# PRECISION@K
# ==========================================================

def calculate_precision_at_k(

    relevance,

    predictions,

    k

):

    k = min(

        k,

        len(predictions)

    )


    ranking = np.argsort(

        -predictions

    )[:k]


    return (

        np.sum(

            relevance[

                ranking

            ]

        )

        /

        k

    )


# ==========================================================
# RECALL@K
# ==========================================================

def calculate_recall_at_k(

    relevance,

    predictions,

    k

):

    total_relevant = np.sum(

        relevance

    )


    if total_relevant == 0:

        return np.nan


    k = min(

        k,

        len(predictions)

    )


    ranking = np.argsort(

        -predictions

    )[:k]


    retrieved_relevant = np.sum(

        relevance[

            ranking

        ]

    )


    return (

        retrieved_relevant

        /

        total_relevant

    )


# ==========================================================
# EVALUATE ONE MODEL ON ONE FOLD
# ==========================================================

def evaluate_model_on_fold(

    predictions,

    test_indices

):

    test_data = df.iloc[

        test_indices

    ]


    metrics = {

        "NDCG@1": [],

        "NDCG@3": [],

        "NDCG@5": [],

        "NDCG@10": [],

        "NDCG@20": [],

        "MRR": [],

        "Precision@3": [],

        "Precision@5": [],

        "Precision@10": [],

        "Precision@20": [],

        "Recall@3": [],

        "Recall@5": [],

        "Recall@10": [],

        "Recall@20": [],

    }


    group_sizes = []


    evaluated_groups = 0


    for group_name, group in test_data.groupby(

        "job_group"

    ):

        if len(group) < MIN_CANDIDATES_PER_JOB:

            continue


        group_indices = group.index.to_numpy()


        group_predictions = predictions[

            group_indices

        ]


        relevance_continuous = group[

            "matched_score"

        ].to_numpy()


        relevance_binary = (

            relevance_continuous

            >=

            RELEVANCE_THRESHOLD

        ).astype(int)


        # --------------------------------------------------
        # NDCG
        # --------------------------------------------------

        for k in K_VALUES:

            k_actual = min(

                k,

                len(group)

            )


            value = ndcg_score(

                relevance_continuous.reshape(

                    1,

                    -1

                ),

                group_predictions.reshape(

                    1,

                    -1

                ),

                k=k_actual

            )


            metrics[

                f"NDCG@{k}"

            ].append(

                value

            )


        # --------------------------------------------------
        # MRR
        # --------------------------------------------------

        metrics["MRR"].append(

            calculate_mrr(

                relevance_binary,

                group_predictions

            )

        )


        # --------------------------------------------------
        # Precision / Recall
        # --------------------------------------------------

        for k in [

            3,

            5,

            10,

            20,

        ]:

            precision = (

                calculate_precision_at_k(

                    relevance_binary,

                    group_predictions,

                    k

                )

            )


            recall = (

                calculate_recall_at_k(

                    relevance_binary,

                    group_predictions,

                    k

                )

            )


            metrics[

                f"Precision@{k}"

            ].append(

                precision

            )


            if not np.isnan(

                recall

            ):

                metrics[

                    f"Recall@{k}"

                ].append(

                    recall

                )


        group_sizes.append(

            len(group)

        )


        evaluated_groups += 1


    results = {}


    for metric_name, values in metrics.items():

        if values:

            results[metric_name] = np.mean(

                values

            )

        else:

            results[metric_name] = np.nan


    results[

        "Evaluated_Groups"

    ] = evaluated_groups


    results[

        "Average_Candidates_Per_Group"

    ] = (

        np.mean(

            group_sizes

        )

        if group_sizes

        else np.nan

    )


    return results


# ==========================================================
# GROUP CROSS VALIDATION
# ==========================================================

print()

print("=" * 70)

print("STARTING 5-FOLD GROUP CROSS-VALIDATION")

print("=" * 70)


groups = df["job_group"].to_numpy()


group_kfold = GroupKFold(

    n_splits=N_SPLITS

)


all_fold_results = []


# ==========================================================
# FOLD LOOP
# ==========================================================

for fold_number, (

    train_indices,

    test_indices

) in enumerate(

    group_kfold.split(

        df,

        df["matched_score"],

        groups

    ),

    start=1

):

    print()

    print("=" * 70)

    print(

        f"FOLD {fold_number}/{N_SPLITS}"

    )

    print("=" * 70)


    print()

    print(

        "Training samples:",

        len(train_indices)

    )


    print(

        "Testing samples:",

        len(test_indices)

    )


    print()

    print(

        "Training groups:",

        len(

            np.unique(

                groups[

                    train_indices

                ]

            )

        )

    )


    print(

        "Testing groups:",

        len(

            np.unique(

                groups[

                    test_indices

                ]

            )

        )

    )


    # ======================================================
    # TF-IDF FIT ONLY ON TRAINING DATA
    # ======================================================

    print()

    print(

        "Building fold-specific TF-IDF..."

    )


    tfidf_vectorizer = TfidfVectorizer(

        max_features=5000,

        ngram_range=(1, 2),

        stop_words="english",

    )


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


    tfidf_vectorizer.fit(

        training_text

    )


    candidate_vectors = (

        tfidf_vectorizer.transform(

            df["candidate_clean"]

        )

    )


    job_vectors = (

        tfidf_vectorizer.transform(

            df["job_clean"]

        )

    )


    fold_tfidf_scores = np.array(

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


    df["fold_tfidf_similarity"] = (

        fold_tfidf_scores

    )


    # ======================================================
    # FEATURE MATRICES
    # ======================================================

    hybrid_columns = [

        feature

        for feature in HYBRID_FEATURES

        if feature != "tfidf_similarity"

    ]


    hybrid_columns = [

        "fold_tfidf_similarity"

        if feature == "tfidf_similarity"

        else feature

        for feature in HYBRID_FEATURES

    ]


    structured_columns = STRUCTURED_FEATURES


    X_hybrid = (

        df[

            hybrid_columns

        ]

        .replace(

            [np.inf, -np.inf],

            np.nan

        )

        .fillna(0)

    )


    X_structured = (

        df[

            structured_columns

        ]

        .replace(

            [np.inf, -np.inf],

            np.nan

        )

        .fillna(0)

    )


    y = df[

        "matched_score"

    ]


    # ======================================================
    # TRAIN HYBRID
    # ======================================================

    print()

    print(

        "Training Hybrid..."

    )


    hybrid_model = train_random_forest(

        X_hybrid.iloc[

            train_indices

        ],

        y.iloc[

            train_indices

        ],

        RANDOM_STATE + fold_number

    )


    hybrid_predictions = (

        hybrid_model.predict(

            X_hybrid

        )

    )


    hybrid_predictions = np.clip(

        hybrid_predictions,

        0,

        1

    )


    # ======================================================
    # TRAIN STRUCTURED
    # ======================================================

    print()

    print(

        "Training Structured ATS..."

    )


    structured_model = train_random_forest(

        X_structured.iloc[

            train_indices

        ],

        y.iloc[

            train_indices

        ],

        RANDOM_STATE + fold_number

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


    # ======================================================
    # DIRECT BASELINES
    # ======================================================

    tfidf_predictions = (

        df["fold_tfidf_similarity"]

        .to_numpy()

    )


    sbert_predictions = (

        df["semantic_similarity"]

        .to_numpy()

    )


    predictions_dict = {

        "Hybrid":

            hybrid_predictions,

        "Structured ATS":

            structured_predictions,

        "TF-IDF Only":

            tfidf_predictions,

        "SBERT Only":

            sbert_predictions,

    }


    # ======================================================
    # EVALUATE MODELS
    # ======================================================

    for model_name, predictions in predictions_dict.items():

        print()

        print(

            f"Evaluating {model_name}..."

        )


        fold_metrics = evaluate_model_on_fold(

            predictions,

            test_indices

        )


        row = {

            "Fold":

                fold_number,

            "Model":

                model_name,

        }


        row.update(

            fold_metrics

        )


        all_fold_results.append(

            row

        )


        print(

            f"NDCG@5="

            f"{fold_metrics['NDCG@5']:.4f} | "

            f"MRR="

            f"{fold_metrics['MRR']:.4f}"

        )


# ==========================================================
# RAW FOLD RESULTS
# ==========================================================

print()

print("=" * 70)

print("RAW 5-FOLD RANKING RESULTS")

print("=" * 70)


fold_results_df = pd.DataFrame(

    all_fold_results

)


print()

print(

    fold_results_df.to_string(

        index=False

    )

)


# ==========================================================
# SAVE RAW RESULTS
# ==========================================================

fold_results_file = (

    PROCESSED_DIR

    / "ranking_v2_fold_results.csv"

)


fold_results_df.to_csv(

    fold_results_file,

    index=False

)


# ==========================================================
# SUMMARY STATISTICS
# ==========================================================

metric_columns = [

    "NDCG@1",

    "NDCG@3",

    "NDCG@5",

    "NDCG@10",

    "NDCG@20",

    "MRR",

    "Precision@3",

    "Precision@5",

    "Precision@10",

    "Precision@20",

    "Recall@3",

    "Recall@5",

    "Recall@10",

    "Recall@20",

]


summary_rows = []


for model_name in (

    fold_results_df[

        "Model"

    ].unique()

):

    model_data = fold_results_df[

        fold_results_df[

            "Model"

        ]

        ==

        model_name

    ]


    row = {

        "Model":

            model_name,

    }


    for metric in metric_columns:

        row[

            f"{metric}_mean"

        ] = model_data[

            metric

        ].mean()


        row[

            f"{metric}_std"

        ] = model_data[

            metric

        ].std(

            ddof=1

        )


    row[

        "Mean_Evaluated_Groups"

    ] = model_data[

        "Evaluated_Groups"

    ].mean()


    row[

        "Average_Candidates_Per_Group"

    ] = model_data[

        "Average_Candidates_Per_Group"

    ].mean()


    summary_rows.append(

        row

    )


summary_df = pd.DataFrame(

    summary_rows

)


# ==========================================================
# SORT BY NDCG@5
# ==========================================================

summary_df = (

    summary_df

    .sort_values(

        by="NDCG@5_mean",

        ascending=False

    )

    .reset_index(

        drop=True

    )

)


# ==========================================================
# SAVE SUMMARY
# ==========================================================

summary_file = (

    PROCESSED_DIR

    / "ranking_v2_summary.csv"

)


summary_df.to_csv(

    summary_file,

    index=False

)


# ==========================================================
# GROUP STATISTICS
# ==========================================================

test_group_statistics = (

    df["job_group"]

    .value_counts()

    .reset_index()

)


test_group_statistics.columns = [

    "Job_Group",

    "Candidate_Count",

]


group_statistics_file = (

    PROCESSED_DIR

    / "ranking_v2_group_statistics.csv"

)


test_group_statistics.to_csv(

    group_statistics_file,

    index=False

)


# ==========================================================
# FINAL SUMMARY DISPLAY
# ==========================================================

print()

print("=" * 70)

print("FINAL 5-FOLD RANKING SUMMARY")

print("=" * 70)


display_columns = [

    "Model",

    "NDCG@1_mean",

    "NDCG@1_std",

    "NDCG@3_mean",

    "NDCG@3_std",

    "NDCG@5_mean",

    "NDCG@5_std",

    "NDCG@10_mean",

    "NDCG@10_std",

    "NDCG@20_mean",

    "NDCG@20_std",

    "MRR_mean",

    "MRR_std",

    "Precision@10_mean",

    "Precision@10_std",

    "Recall@10_mean",

    "Recall@10_std",

]


print()

print(

    summary_df[

        display_columns

    ].to_string(

        index=False

    )

)


# ==========================================================
# BEST MODEL
# ==========================================================

best_model = summary_df.iloc[0]


print()

print("=" * 70)

print("BEST RANKING MODEL")

print("=" * 70)


print()

print(

    "Model:",

    best_model["Model"]

)


print(

    "NDCG@5:",

    f"{best_model['NDCG@5_mean']:.4f}",

    "+/-",

    f"{best_model['NDCG@5_std']:.4f}"

)


print(

    "NDCG@10:",

    f"{best_model['NDCG@10_mean']:.4f}",

    "+/-",

    f"{best_model['NDCG@10_std']:.4f}"

)


print(

    "MRR:",

    f"{best_model['MRR_mean']:.4f}",

    "+/-",

    f"{best_model['MRR_std']:.4f}"

)


# ==========================================================
# OUTPUT FILES
# ==========================================================

print()

print("=" * 70)

print("EXPERIMENT 5 V2 COMPLETE")

print("=" * 70)


print()

print(

    "Fold results saved to:"

)

print(

    fold_results_file

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

    "Group statistics saved to:"

)

print(

    group_statistics_file

)


print()

print("=" * 70)

print("FINAL RANKING EXPERIMENT FINISHED")

print("=" * 70)