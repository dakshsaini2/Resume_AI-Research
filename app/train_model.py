"""
train_model.py

Train the ATS Resume Screening ML model.

This script:
1. Loads the training dataset
2. Builds TF-IDF features
3. Builds SBERT semantic features
4. Builds ATS matching features
5. Trains a RandomForestRegressor
6. Evaluates the model
7. Prints feature importance
8. Saves the trained model, scaler and TF-IDF vectorizer
"""

from pathlib import Path
import re
import warnings

import joblib
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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


warnings.filterwarnings("ignore")


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

DATASET_FILE = (
    DATASET_DIR /
    "Resume_Data_For_Ranking.csv"
)


# ==========================================================
# MODEL CONFIGURATION
# ==========================================================

SBERT_MODEL = "all-MiniLM-L6-v2"


FEATURE_COLUMNS = [

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


# ==========================================================
# LOAD DATASET
# ==========================================================

print()
print("=" * 70)
print("LOADING DATASET")
print("=" * 70)

if not DATASET_FILE.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATASET_FILE}"
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


print()
print("Dataset columns:")

for column in df.columns:

    print(
        " -",
        column
    )


# ==========================================================
# REQUIRED DATASET COLUMNS
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
        + "\n".join(missing_columns)
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
    subset=["matched_score"]
).copy()


# Keep target between 0 and 1
df["matched_score"] = (
    df["matched_score"]
    .clip(0, 1)
)


print()
print(
    "Number of training samples:",
    len(df)
)

print(
    "Target minimum:",
    df["matched_score"].min()
)

print(
    "Target maximum:",
    df["matched_score"].max()
)


# ==========================================================
# TEXT PREPARATION
# ==========================================================

def clean_text(text):

    text = str(text).lower()

    text = re.sub(
        r"http\S+|www\S+",
        " ",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    text = re.sub(
        r"[^a-z0-9+#.\-/ ]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


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

            values.append(value)

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


df["candidate_clean"] = (
    df["candidate_text"]
    .apply(clean_text)
)


df["job_clean"] = (
    df["job_text"]
    .apply(clean_text)
)


# ==========================================================
# SKILL NORMALIZATION
# ==========================================================

SKILL_SYNONYMS = {

    "github": "git",

    "gitlab": "git",

    "mysql": "sql",

    "postgresql": "sql",

    "sqlite": "sql",

    "react.js": "react",

    "reactjs": "react",

    "node.js": "node",

    "nodejs": "node",

}


def normalize_skill(skill):

    skill = str(
        skill
    ).strip().lower()

    skill = SKILL_SYNONYMS.get(
        skill,
        skill
    )

    return skill


def extract_skills(value):

    if not value:

        return set()

    text = str(value)

    text = (
        text
        .replace("[", "")
        .replace("]", "")
        .replace("'", "")
        .replace('"', "")
    )

    parts = re.split(
        r",|;|\n|\|",
        text
    )

    skills = set()

    for part in parts:

        skill = normalize_skill(
            part
        )

        if skill:

            skills.add(skill)

    return skills


# ==========================================================
# SKILL FEATURES
# ==========================================================

def skill_overlap(row):

    candidate = extract_skills(
        row["skills"]
    )

    required = extract_skills(
        row["skills_required"]
    )

    return len(
        candidate & required
    )


def skill_score(row):

    candidate = extract_skills(
        row["skills"]
    )

    required = extract_skills(
        row["skills_required"]
    )

    if not required:

        return 1.0

    return (
        len(candidate & required)
        /
        len(required)
    )


df["candidate_skill_count"] = (
    df["skills"]
    .apply(
        lambda x:
        len(
            extract_skills(x)
        )
    )
)


df["required_skill_count"] = (
    df["skills_required"]
    .apply(
        lambda x:
        len(
            extract_skills(x)
        )
    )
)


df["skill_overlap_count"] = (
    df.apply(
        skill_overlap,
        axis=1
    )
)


df["skill_match_score"] = (
    df.apply(
        skill_score,
        axis=1
    )
)


# ==========================================================
# EDUCATION
# ==========================================================

DEGREE_LEVELS = {

    "high school": 1,

    "diploma": 2,

    "b.sc": 3,

    "bsc": 3,

    "b.tech": 3,

    "btech": 3,

    "b.e": 3,

    "be": 3,

    "bachelor": 3,

    "m.sc": 4,

    "msc": 4,

    "m.tech": 4,

    "mtech": 4,

    "mba": 4,

    "master": 4,

    "phd": 5,

    "doctorate": 5,

}


def get_degree_level(text):

    text = str(
        text
    ).lower()

    level = 0

    for degree, degree_level in DEGREE_LEVELS.items():

        if degree in text:

            level = max(
                level,
                degree_level
            )

    return level


def education_score(row):

    candidate = get_degree_level(
        row["degree_names"]
    )

    required = get_degree_level(
        row["educational_requirements"]
    )

    if required == 0:

        return 1.0

    return min(
        candidate / required,
        1.0
    )


def education_exact(row):

    candidate = get_degree_level(
        row["degree_names"]
    )

    required = get_degree_level(
        row["educational_requirements"]
    )

    if (
        candidate > 0
        and required > 0
        and candidate >= required
    ):

        return 1

    return 0


df["education_match_score"] = (
    df.apply(
        education_score,
        axis=1
    )
)


df["education_exact_match"] = (
    df.apply(
        education_exact,
        axis=1
    )
)


# ==========================================================
# EXPERIENCE
# ==========================================================

def extract_years(text):

    text = str(
        text
    ).lower()

    patterns = [

        r"(\d+(?:\.\d+)?)\s*\+?\s*years?",

        r"(\d+(?:\.\d+)?)\s*\+?\s*yrs?",

    ]

    values = []

    for pattern in patterns:

        matches = re.findall(
            pattern,
            text
        )

        for value in matches:

            try:

                values.append(
                    float(value)
                )

            except ValueError:

                pass

    if not values:

        return 0.0

    return max(values)


def experience_score(row):

    candidate = extract_years(
        row["positions"]
    )

    required = extract_years(
        row["experiencere_requirement"]
    )

    if required <= 0:

        return 1.0

    return min(
        candidate / required,
        1.0
    )


df["experience_match_score"] = (
    df.apply(
        experience_score,
        axis=1
    )
)


# ==========================================================
# CERTIFICATIONS
# ==========================================================

def extract_certifications(text):

    if not text:

        return set()

    parts = re.split(
        r",|;|\n|\|",
        str(text).lower()
    )

    return {
        part.strip()
        for part in parts
        if part.strip()
    }


def certification_score(row):

    candidate = extract_certifications(
        row["certification_skills"]
    )

    # Dataset does not provide a reliable
    # required-certification column.
    #
    # Therefore we use a neutral value
    # instead of incorrectly comparing
    # certifications against job skills.

    if not candidate:

        return 0.0

    return 0.5


df["certification_match_score"] = (
    df.apply(
        certification_score,
        axis=1
    )
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
# TF-IDF
# ==========================================================

print()
print("=" * 70)
print("BUILDING TF-IDF")
print("=" * 70)


tfidf = TfidfVectorizer(

    max_features=5000,

    ngram_range=(1, 2),

    stop_words="english",

)


tfidf.fit(
    pd.concat(
        [
            df["candidate_clean"],
            df["job_clean"],
        ]
    )
)


candidate_vectors = (
    tfidf.transform(
        df["candidate_clean"]
    )
)


job_vectors = (
    tfidf.transform(
        df["job_clean"]
    )
)


tfidf_scores = []


for i in range(
    len(df)
):

    similarity = cosine_similarity(

        candidate_vectors[i],

        job_vectors[i],

    )[0][0]

    tfidf_scores.append(
        float(similarity)
    )


df["tfidf_similarity"] = (
    tfidf_scores
)


# ==========================================================
# SBERT
# ==========================================================

print()
print("=" * 70)
print("LOADING SBERT")
print("=" * 70)


sentence_model = SentenceTransformer(
    SBERT_MODEL
)


print()
print("Generating resume embeddings...")


candidate_embeddings = (
    sentence_model.encode(

        df["candidate_clean"].tolist(),

        convert_to_numpy=True,

        show_progress_bar=True,

    )
)


print()
print("Generating job embeddings...")


job_embeddings = (
    sentence_model.encode(

        df["job_clean"].tolist(),

        convert_to_numpy=True,

        show_progress_bar=True,

    )
)


semantic_scores = []


for i in range(
    len(df)
):

    similarity = cosine_similarity(

        candidate_embeddings[i]
        .reshape(1, -1),

        job_embeddings[i]
        .reshape(1, -1),

    )[0][0]

    semantic_scores.append(
        float(similarity)
    )


df["semantic_similarity"] = (
    semantic_scores
)


# ==========================================================
# BUILD FINAL FEATURE MATRIX
# ==========================================================

print()
print("=" * 70)
print("BUILDING FINAL FEATURE MATRIX")
print("=" * 70)


X = df[
    FEATURE_COLUMNS
].copy()


y = df[
    "matched_score"
].copy()


X = X.replace(
    [np.inf, -np.inf],
    np.nan
)


X = X.fillna(0)


print()
print(
    "Feature matrix:",
    X.shape
)


print(
    "Target:",
    y.shape
)


# ==========================================================
# FEATURE VALIDATION
# ==========================================================

print()
print("=" * 70)
print("FEATURE VALIDATION")
print("=" * 70)


for feature in FEATURE_COLUMNS:

    unique_values = X[
        feature
    ].nunique()

    print(
        f"{feature:30s}"
        f" unique={unique_values}"
    )


print()
print("Feature statistics:")

print(
    X.describe().T[
        [
            "mean",
            "std",
            "min",
            "max",
        ]
    ]
)


# ==========================================================
# TRAIN / TEST SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = (
    train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

    )
)


# ==========================================================
# SCALING
# ==========================================================

scaler = StandardScaler()


X_train_scaled = (
    scaler.fit_transform(
        X_train
    )
)


X_test_scaled = (
    scaler.transform(
        X_test
    )
)


# ==========================================================
# RANDOM FOREST
# ==========================================================

print()
print("=" * 70)
print("TRAINING RANDOM FOREST")
print("=" * 70)


model = RandomForestRegressor(

    random_state=42,

    n_jobs=-1,

)


param_grid = {

    "n_estimators": [
        200,
        300,
    ],

    "max_depth": [
        10,
        20,
        None,
    ],

    "min_samples_split": [
        2,
        5,
    ],

    "min_samples_leaf": [
        1,
        2,
    ],

}


grid_search = GridSearchCV(

    estimator=model,

    param_grid=param_grid,

    cv=5,

    scoring="neg_mean_absolute_error",

    n_jobs=-1,

    verbose=1,

)


grid_search.fit(

    X_train_scaled,

    y_train,

)


best_model = (
    grid_search
    .best_estimator_
)


print()
print(
    "Best parameters:"
)

print(
    grid_search.best_params_
)


# ==========================================================
# PREDICTIONS
# ==========================================================

predictions = (
    best_model.predict(
        X_test_scaled
    )
)


predictions = np.clip(
    predictions,
    0,
    1
)


# ==========================================================
# MODEL EVALUATION
# ==========================================================

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
print("=" * 70)
print("MODEL EVALUATION")
print("=" * 70)


print(
    f"MAE  : {mae:.4f}"
)


print(
    f"RMSE : {rmse:.4f}"
)


print(
    f"R²   : {r2:.4f}"
)


# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

importance_df = pd.DataFrame({

    "Feature":
    FEATURE_COLUMNS,

    "Importance":
    best_model.feature_importances_,

})


importance_df = (
    importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print()
print("=" * 70)
print("FEATURE IMPORTANCE")
print("=" * 70)


print(
    importance_df.to_string(
        index=False
    )
)


# ==========================================================
# SAVE MODEL
# ==========================================================

print()
print("=" * 70)
print("SAVING MODEL FILES")
print("=" * 70)


model_file = (
    MODEL_DIR /
    "best_resume_model.pkl"
)


scaler_file = (
    MODEL_DIR /
    "scaler.pkl"
)


tfidf_file = (
    MODEL_DIR /
    "tfidf_vectorizer.pkl"
)


joblib.dump(
    best_model,
    model_file
)


joblib.dump(
    scaler,
    scaler_file
)


joblib.dump(
    tfidf,
    tfidf_file
)


print(
    "Saved:",
    model_file
)


print(
    "Saved:",
    scaler_file
)


print(
    "Saved:",
    tfidf_file
)


# ==========================================================
# SAVE TRAINING RESULTS
# ==========================================================

processed_dir = (
    DATASET_DIR /
    "processed"
)


processed_dir.mkdir(
    parents=True,
    exist_ok=True
)


importance_df.to_csv(

    processed_dir /
    "feature_importance.csv",

    index=False,

)


pd.DataFrame({

    "Metric": [
        "MAE",
        "RMSE",
        "R2",
    ],

    "Value": [
        mae,
        rmse,
        r2,
    ],

}).to_csv(

    processed_dir /
    "evaluation_metrics.csv",

    index=False,

)


print()
print("=" * 70)
print("TRAINING COMPLETE")
print("=" * 70)