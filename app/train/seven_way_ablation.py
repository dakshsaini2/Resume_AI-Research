"""
seven_way_ablation.py

Experiment: Seven-Way Feature-Family Combination Ablation
==========================================================

Purpose
-------
Determine whether lexical similarity (TF-IDF), semantic similarity
(SBERT), and structured ATS information provide complementary signals
for resume-job matching by systematically evaluating all possible
combinations of the three feature families.

This directly addresses Section 4.1 of the research roadmap:
"Complete Seven-Way Ablation — CRITICAL"

Configurations
--------------
1. TF-IDF Only               (Lexical)
2. SBERT Only                 (Semantic)
3. Structured ATS Only        (Structured)
4. TF-IDF + SBERT             (Lexical + Semantic)
5. TF-IDF + Structured ATS    (Lexical + Structured)
6. SBERT + Structured ATS     (Semantic + Structured)
7. Full Hybrid                (Lexical + Semantic + Structured)

Design decisions
----------------
- Text statistics (resume_length, job_length, resume_word_count,
  job_word_count) are included in ALL configurations as auxiliary
  meta-features. This ensures the comparison isolates the three
  representation families rather than conflating the contribution
  of text-length signals.

- The official job-group split (GroupShuffleSplit, 22 train / 6 test,
  random_state=42) is used — matching the protocol freeze.

- TF-IDF is fitted ONLY on training data.

- Same RandomForestRegressor configuration across all experiments.

Evaluation
----------
Regression: MAE, RMSE, R²
Ranking:    NDCG@5, NDCG@10, MRR (per test job group, then averaged)

Outputs
-------
dataset/processed/seven_way_ablation/
    seven_way_results.csv
    seven_way_ranking.csv
    seven_way_feature_definitions.csv
    seven_way_summary.txt
    seven_way_ablation_chart.png
"""


# ==========================================================
# IMPORTS
# ==========================================================

from pathlib import Path
import sys
import warnings
import time

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sentence_transformers import SentenceTransformer

from sklearn.ensemble import RandomForestRegressor

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    ndcg_score,
)

from sklearn.metrics.pairwise import cosine_similarity

from sklearn.model_selection import GroupShuffleSplit


warnings.filterwarnings("ignore")


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

RANDOM_STATE = 42

np.random.seed(RANDOM_STATE)


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
    / "seven_way_ablation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================
# APP IMPORT PATH
# ==========================================================

if str(APP_DIR) not in sys.path:

    sys.path.insert(
        0,
        str(APP_DIR),
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

SBERT_MODEL_NAME = "all-MiniLM-L6-v2"

TARGET_COLUMN = "matched_score"

TEST_SIZE = 0.20


# ==========================================================
# RANDOM FOREST — IDENTICAL TO ALL OTHER EXPERIMENTS
# ==========================================================

RF_CONFIG = {
    "n_estimators": 300,
    "max_depth": None,
    "min_samples_split": 2,
    "min_samples_leaf": 2,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}


# ==========================================================
# FEATURE FAMILY DEFINITIONS
# ==========================================================

# Lexical family
TFIDF_FEATURES = [
    "tfidf_similarity",
]

# Semantic family
SBERT_FEATURES = [
    "semantic_similarity",
]

# Structured ATS family
STRUCTURED_ATS_FEATURES = [
    "skill_match_score",
    "education_match_score",
    "experience_match_score",
    "certification_match_score",
    "candidate_skill_count",
    "required_skill_count",
    "skill_overlap_count",
    "education_exact_match",
]

# Auxiliary text statistics — included in ALL configurations
TEXT_STATISTICS = [
    "resume_length",
    "job_length",
    "resume_word_count",
    "job_word_count",
]


# ==========================================================
# SEVEN-WAY CONFIGURATION DEFINITIONS
# ==========================================================

SEVEN_WAY_CONFIGS = {

    "TF-IDF Only": {
        "features": TFIDF_FEATURES + TEXT_STATISTICS,
        "family": "Lexical",
    },

    "SBERT Only": {
        "features": SBERT_FEATURES + TEXT_STATISTICS,
        "family": "Semantic",
    },

    "Structured ATS Only": {
        "features": STRUCTURED_ATS_FEATURES + TEXT_STATISTICS,
        "family": "Structured",
    },

    "TF-IDF + SBERT": {
        "features": TFIDF_FEATURES + SBERT_FEATURES + TEXT_STATISTICS,
        "family": "Lexical + Semantic",
    },

    "TF-IDF + Structured ATS": {
        "features": TFIDF_FEATURES + STRUCTURED_ATS_FEATURES + TEXT_STATISTICS,
        "family": "Lexical + Structured",
    },

    "SBERT + Structured ATS": {
        "features": SBERT_FEATURES + STRUCTURED_ATS_FEATURES + TEXT_STATISTICS,
        "family": "Semantic + Structured",
    },

    "Full Hybrid": {
        "features": (
            TFIDF_FEATURES
            + SBERT_FEATURES
            + STRUCTURED_ATS_FEATURES
            + TEXT_STATISTICS
        ),
        "family": "Lexical + Semantic + Structured",
    },

}


# ==========================================================
# HELPER
# ==========================================================

def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def combine_columns(row, columns):
    values = []
    for column in columns:
        value = str(row[column]).strip()
        if value:
            values.append(value)
    return " ".join(values)


# ==========================================================
# LOAD DATASET
# ==========================================================

section("LOADING DATASET")

if not DATASET_FILE.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATASET_FILE}"
    )

df = pd.read_csv(DATASET_FILE)

print(f"\nDataset shape: {df.shape}")


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

df[TARGET_COLUMN] = pd.to_numeric(
    df[TARGET_COLUMN],
    errors="coerce",
)

df = df.dropna(
    subset=[TARGET_COLUMN]
).copy()

df[TARGET_COLUMN] = (
    df[TARGET_COLUMN]
    .clip(0, 1)
)

print(f"Number of samples: {len(df)}")
print(
    f"Target range: {df[TARGET_COLUMN].min():.4f} "
    f"to {df[TARGET_COLUMN].max():.4f}"
)


# ==========================================================
# TEXT CONSTRUCTION
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

job_columns = [
    "job_position_name",
    "skills_required",
    "educational_requirements",
    "experiencere_requirement",
]

df["candidate_text"] = df.apply(
    lambda row: combine_columns(row, candidate_columns),
    axis=1,
)

df["job_text"] = df.apply(
    lambda row: combine_columns(row, job_columns),
    axis=1,
)


# ==========================================================
# PREPROCESSING
# ==========================================================

section("PREPROCESSING TEXT")

df["candidate_clean"] = (
    df["candidate_text"]
    .apply(
        lambda text: preprocess_resume(text)["processed_text"]
    )
)

df["job_clean"] = (
    df["job_text"]
    .apply(
        lambda text: preprocess_job(text)["processed_text"]
    )
)

print("Preprocessing complete.")


# ==========================================================
# STRUCTURED ATS FEATURES
# ==========================================================

section("BUILDING STRUCTURED ATS FEATURES")


# ----------------------------------------------------------
# SKILLS
# ----------------------------------------------------------

print("Extracting skills...")

df["candidate_skill_list"] = df["candidate_text"].apply(extract_skills)
df["required_skill_list"] = df["job_text"].apply(extract_skills)

df["candidate_skill_set"] = df["candidate_skill_list"].apply(set)
df["required_skill_set"] = df["required_skill_list"].apply(set)

df["candidate_skill_count"] = df["candidate_skill_list"].apply(len)
df["required_skill_count"] = df["required_skill_list"].apply(len)

df["skill_overlap_count"] = df.apply(
    lambda row: len(
        row["candidate_skill_set"]
        & row["required_skill_set"]
    ),
    axis=1,
)


def calculate_skill_match(row):
    required = row["required_skill_set"]
    candidate = row["candidate_skill_set"]
    if not required:
        return 1.0
    return len(candidate & required) / len(required)


df["skill_match_score"] = df.apply(
    calculate_skill_match,
    axis=1,
)


# ----------------------------------------------------------
# EDUCATION
# ----------------------------------------------------------

print("Computing education features...")


def education_score(row):
    candidate = extract_degree(row["candidate_text"]).strip().lower()
    required = extract_degree(row["job_text"]).strip().lower()

    if required == "not specified" or not required:
        return 1.0
    if candidate == "not specified" or not candidate:
        return 0.0

    candidate_level = DEGREE_LEVELS.get(candidate, 2)
    required_level = DEGREE_LEVELS.get(required, 2)

    if candidate_level >= required_level:
        return 1.0

    return round(candidate_level / required_level, 2)


def education_exact(row):
    candidate = extract_degree(row["candidate_text"]).strip().lower()
    required = extract_degree(row["job_text"]).strip().lower()

    if (
        candidate != "not specified"
        and required != "not specified"
        and candidate
        and required
    ):
        candidate_level = DEGREE_LEVELS.get(candidate, 2)
        required_level = DEGREE_LEVELS.get(required, 2)

        if candidate_level >= required_level:
            return int(candidate == required)

    return 0


df["education_match_score"] = df.apply(education_score, axis=1)
df["education_exact_match"] = df.apply(education_exact, axis=1)


# ----------------------------------------------------------
# EXPERIENCE
# ----------------------------------------------------------

print("Computing experience features...")


def experience_score(row):
    candidate = extract_experience(row["candidate_text"])
    required = extract_experience(row["job_text"])

    if required <= 0:
        return 1.0
    if candidate <= 0:
        return 0.0

    return min(candidate / required, 1.0)


df["experience_match_score"] = df.apply(experience_score, axis=1)


# ----------------------------------------------------------
# CERTIFICATION
# ----------------------------------------------------------

print("Computing certification features...")


def certification_score(row):
    candidate = set(extract_certifications(row["candidate_text"]))
    required = set(extract_certifications(row["job_text"]))

    if not required:
        return 1.0

    return len(candidate & required) / len(required)


df["certification_match_score"] = df.apply(certification_score, axis=1)


# ----------------------------------------------------------
# TEXT STATISTICS
# ----------------------------------------------------------

print("Computing text statistics...")

df["resume_length"] = df["candidate_clean"].str.len()
df["job_length"] = df["job_clean"].str.len()
df["resume_word_count"] = df["candidate_clean"].str.split().str.len()
df["job_word_count"] = df["job_clean"].str.split().str.len()


# ==========================================================
# OFFICIAL JOB-GROUP SPLIT
# ==========================================================

section("OFFICIAL JOB-GROUP SPLIT")

# Use job_position_name as the group key,
# matching the protocol freeze definition.

df["job_group"] = (
    df["job_position_name"]
    .str.strip()
    .str.lower()
)

group_split = GroupShuffleSplit(
    n_splits=1,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
)

train_indices, test_indices = next(
    group_split.split(
        df,
        groups=df["job_group"],
    )
)

train_groups = sorted(
    df.iloc[train_indices]["job_group"].unique()
)

test_groups = sorted(
    df.iloc[test_indices]["job_group"].unique()
)

print(f"\nTraining groups: {len(train_groups)}")
print(f"Testing groups:  {len(test_groups)}")
print(f"Training samples: {len(train_indices)}")
print(f"Testing samples:  {len(test_indices)}")

# Verify no overlap
group_overlap = set(train_groups) & set(test_groups)
assert len(group_overlap) == 0, (
    f"LEAKAGE: {len(group_overlap)} groups in both train and test!"
)
print("Group overlap check: PASS (0 overlap)")


# ==========================================================
# TF-IDF (fitted on training data ONLY)
# ==========================================================

section("BUILDING TF-IDF FEATURES")

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    stop_words="english",
)

# Fit ONLY on training text.
tfidf.fit(
    pd.concat([
        df.iloc[train_indices]["candidate_clean"],
        df.iloc[train_indices]["job_clean"],
    ])
)

candidate_vectors = tfidf.transform(df["candidate_clean"])
job_vectors = tfidf.transform(df["job_clean"])

tfidf_scores = np.array([
    cosine_similarity(
        candidate_vectors[i],
        job_vectors[i],
    )[0][0]
    for i in range(len(df))
])

df["tfidf_similarity"] = tfidf_scores

print("TF-IDF features computed.")


# ==========================================================
# SBERT EMBEDDINGS
# ==========================================================

section("BUILDING SBERT FEATURES")

sentence_model = SentenceTransformer(SBERT_MODEL_NAME)

print("Generating resume embeddings...")

candidate_embeddings = sentence_model.encode(
    df["candidate_clean"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True,
)

print("Generating job embeddings...")

job_embeddings = sentence_model.encode(
    df["job_clean"].tolist(),
    convert_to_numpy=True,
    show_progress_bar=True,
)

semantic_scores = np.array([
    cosine_similarity(
        candidate_embeddings[i].reshape(1, -1),
        job_embeddings[i].reshape(1, -1),
    )[0][0]
    for i in range(len(df))
])

df["semantic_similarity"] = semantic_scores

print("SBERT features computed.")


# ==========================================================
# VALIDATE ALL FEATURES EXIST
# ==========================================================

all_features = set()

for config in SEVEN_WAY_CONFIGS.values():
    all_features.update(config["features"])

missing_features = [
    f for f in sorted(all_features)
    if f not in df.columns
]

if missing_features:
    raise ValueError(
        "Missing engineered features:\n"
        + "\n".join(missing_features)
    )

print(f"\nAll {len(all_features)} features validated.")


# ==========================================================
# RANKING METRICS
# ==========================================================

def compute_ranking_metrics(
    df_group,
    actual_col,
    predicted_col,
):
    """
    Compute NDCG@5, NDCG@10, and MRR for a single job group.
    """

    actual = df_group[actual_col].values
    predicted = df_group[predicted_col].values

    n = len(actual)

    if n < 2:
        return None

    # NDCG (continuous relevance)
    ndcg_5 = ndcg_score(
        [actual],
        [predicted],
        k=min(5, n),
    )

    ndcg_10 = ndcg_score(
        [actual],
        [predicted],
        k=min(10, n),
    )

    # MRR (binary: top-25% of actual scores are "relevant")
    threshold = np.percentile(actual, 75)
    relevant = actual >= threshold

    # Rank by predicted (descending)
    ranked_indices = np.argsort(-predicted)
    ranked_relevant = relevant[ranked_indices]

    mrr = 0.0
    for i, is_relevant in enumerate(ranked_relevant):
        if is_relevant:
            mrr = 1.0 / (i + 1)
            break

    return {
        "NDCG@5": ndcg_5,
        "NDCG@10": ndcg_10,
        "MRR": mrr,
    }


# ==========================================================
# TRAIN-AND-EVALUATE FUNCTION
# ==========================================================

def train_and_evaluate(
    experiment_name,
    feature_columns,
    family,
):

    print()
    print("-" * 60)
    print(f"  TRAINING: {experiment_name}")
    print(f"  Family: {family}")
    print(f"  Features: {len(feature_columns)}")
    print("-" * 60)

    # Prepare data
    X = (
        df[feature_columns]
        .copy()
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    y = df[TARGET_COLUMN].copy()

    X_train = X.iloc[train_indices]
    X_test = X.iloc[test_indices]

    y_train = y.iloc[train_indices]
    y_test = y.iloc[test_indices]

    # Train
    model = RandomForestRegressor(**RF_CONFIG)
    model.fit(X_train, y_train)

    # Predict
    predictions = np.clip(
        model.predict(X_test),
        0,
        1,
    )

    # --- Regression metrics ---
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    print(f"  MAE  : {mae:.6f}")
    print(f"  RMSE : {rmse:.6f}")
    print(f"  R²   : {r2:.6f}")

    # --- Ranking metrics (per test job group) ---
    test_df = df.iloc[test_indices].copy()
    test_df["predicted"] = predictions

    ranking_results = []

    for group_name in test_groups:

        group_df = test_df[
            test_df["job_group"] == group_name
        ]

        if len(group_df) < 2:
            continue

        group_ranking = compute_ranking_metrics(
            group_df,
            TARGET_COLUMN,
            "predicted",
        )

        if group_ranking is not None:
            group_ranking["group"] = group_name
            ranking_results.append(group_ranking)

    # Average ranking metrics across groups
    if ranking_results:

        ranking_df = pd.DataFrame(ranking_results)

        ndcg5_mean = ranking_df["NDCG@5"].mean()
        ndcg10_mean = ranking_df["NDCG@10"].mean()
        mrr_mean = ranking_df["MRR"].mean()

        ndcg5_std = ranking_df["NDCG@5"].std()
        ndcg10_std = ranking_df["NDCG@10"].std()
        mrr_std = ranking_df["MRR"].std()

    else:

        ndcg5_mean = ndcg10_mean = mrr_mean = 0.0
        ndcg5_std = ndcg10_std = mrr_std = 0.0

    print(f"  NDCG@5 : {ndcg5_mean:.6f} ± {ndcg5_std:.6f}")
    print(f"  NDCG@10: {ndcg10_mean:.6f} ± {ndcg10_std:.6f}")
    print(f"  MRR    : {mrr_mean:.6f} ± {mrr_std:.6f}")

    return {
        "Configuration": experiment_name,
        "Family": family,
        "Num_Features": len(feature_columns),
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "NDCG@5_mean": ndcg5_mean,
        "NDCG@5_std": ndcg5_std,
        "NDCG@10_mean": ndcg10_mean,
        "NDCG@10_std": ndcg10_std,
        "MRR_mean": mrr_mean,
        "MRR_std": mrr_std,
    }


# ==========================================================
# RUN ALL SEVEN EXPERIMENTS
# ==========================================================

section("RUNNING SEVEN-WAY ABLATION")

start_time = time.time()

results = []

for experiment_name, config in SEVEN_WAY_CONFIGS.items():

    result = train_and_evaluate(
        experiment_name,
        config["features"],
        config["family"],
    )

    results.append(result)


elapsed = time.time() - start_time


# ==========================================================
# RESULTS TABLE
# ==========================================================

section("SEVEN-WAY ABLATION RESULTS")

results_df = pd.DataFrame(results)

# Add performance relative to Full Hybrid
hybrid_row = results_df[
    results_df["Configuration"] == "Full Hybrid"
].iloc[0]

results_df["MAE_vs_Hybrid"] = results_df["MAE"] - hybrid_row["MAE"]
results_df["RMSE_vs_Hybrid"] = results_df["RMSE"] - hybrid_row["RMSE"]
results_df["R2_vs_Hybrid"] = results_df["R2"] - hybrid_row["R2"]

# Sort by R² descending
results_df = results_df.sort_values(
    "R2",
    ascending=False,
).reset_index(drop=True)

print()
print(
    results_df[
        [
            "Configuration",
            "Family",
            "Num_Features",
            "MAE",
            "RMSE",
            "R2",
            "NDCG@5_mean",
            "MRR_mean",
        ]
    ].to_string(index=False)
)


# ==========================================================
# SAVE RESULTS
# ==========================================================

section("SAVING RESULTS")

# Main results
results_file = OUTPUT_DIR / "seven_way_results.csv"

results_df.to_csv(
    results_file,
    index=False,
)

print(f"Results: {results_file}")


# Ranking detail
ranking_file = OUTPUT_DIR / "seven_way_ranking.csv"

results_df[
    [
        "Configuration",
        "NDCG@5_mean",
        "NDCG@5_std",
        "NDCG@10_mean",
        "NDCG@10_std",
        "MRR_mean",
        "MRR_std",
    ]
].to_csv(
    ranking_file,
    index=False,
)

print(f"Ranking: {ranking_file}")


# Feature definitions
feature_rows = []

for experiment_name, config in SEVEN_WAY_CONFIGS.items():

    for feature in config["features"]:

        feature_rows.append({
            "Configuration": experiment_name,
            "Family": config["family"],
            "Feature": feature,
        })


feature_def_file = OUTPUT_DIR / "seven_way_feature_definitions.csv"

pd.DataFrame(feature_rows).to_csv(
    feature_def_file,
    index=False,
)

print(f"Feature defs: {feature_def_file}")


# ==========================================================
# GENERATE COMPARISON CHART
# ==========================================================

section("GENERATING CHART")

fig, axes = plt.subplots(1, 3, figsize=(18, 7))

# Sort for display
display_df = results_df.sort_values("R2", ascending=True)

configs = display_df["Configuration"].values
families = display_df["Family"].values

# Color by number of families
family_colors = {
    1: "#EF4444",    # Single family — red
    2: "#F59E0B",    # Pairwise — amber
    3: "#10B981",    # Full hybrid — green
}

def count_families(family_str):
    return family_str.count("+") + 1

colors = [
    family_colors[count_families(f)]
    for f in families
]


# MAE (lower is better)
axes[0].barh(configs, display_df["MAE"], color=colors, edgecolor="white")
axes[0].set_xlabel("MAE (lower is better)", fontsize=11)
axes[0].set_title("Mean Absolute Error", fontsize=13, fontweight="bold")
for i, v in enumerate(display_df["MAE"]):
    axes[0].text(v + 0.001, i, f"{v:.4f}", va="center", fontsize=9)

# R² (higher is better)
axes[1].barh(configs, display_df["R2"], color=colors, edgecolor="white")
axes[1].set_xlabel("R² (higher is better)", fontsize=11)
axes[1].set_title("R² Score", fontsize=13, fontweight="bold")
for i, v in enumerate(display_df["R2"]):
    axes[1].text(
        max(v + 0.01, 0.01) if v >= 0 else v - 0.06,
        i,
        f"{v:.4f}",
        va="center",
        fontsize=9,
    )

# NDCG@5 (higher is better)
axes[2].barh(
    configs,
    display_df["NDCG@5_mean"],
    color=colors,
    edgecolor="white",
)
axes[2].set_xlabel("NDCG@5 (higher is better)", fontsize=11)
axes[2].set_title("NDCG@5 (Ranking)", fontsize=13, fontweight="bold")
for i, v in enumerate(display_df["NDCG@5_mean"]):
    axes[2].text(v + 0.002, i, f"{v:.4f}", va="center", fontsize=9)


# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#EF4444", label="Single family"),
    Patch(facecolor="#F59E0B", label="Pairwise combination"),
    Patch(facecolor="#10B981", label="Full hybrid (all three)"),
]

fig.legend(
    handles=legend_elements,
    loc="lower center",
    ncol=3,
    fontsize=11,
    frameon=True,
    fancybox=True,
)

plt.suptitle(
    "Seven-Way Feature-Family Ablation Study",
    fontsize=16,
    fontweight="bold",
    y=1.01,
)

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)

chart_file = OUTPUT_DIR / "seven_way_ablation_chart.png"

fig.savefig(
    chart_file,
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
)

plt.close()

print(f"Chart: {chart_file}")


# ==========================================================
# SUMMARY REPORT
# ==========================================================

section("SUMMARY REPORT")

best_mae = results_df.loc[results_df["MAE"].idxmin()]
best_r2 = results_df.loc[results_df["R2"].idxmax()]
best_ndcg = results_df.loc[results_df["NDCG@5_mean"].idxmax()]

summary_lines = [
    "=" * 70,
    "SEVEN-WAY FEATURE-FAMILY ABLATION — SUMMARY",
    "=" * 70,
    "",
    "--- Protocol ---",
    f"Split: Official job-group split (GroupShuffleSplit)",
    f"Train groups: {len(train_groups)}, Test groups: {len(test_groups)}",
    f"Train samples: {len(train_indices)}, Test samples: {len(test_indices)}",
    f"Random state: {RANDOM_STATE}",
    f"TF-IDF fitted on training data only: Yes",
    f"RF config: {RF_CONFIG}",
    "",
    "--- Best Models ---",
    f"Best MAE:    {best_mae['Configuration']} ({best_mae['MAE']:.6f})",
    f"Best R²:     {best_r2['Configuration']} ({best_r2['R2']:.6f})",
    f"Best NDCG@5: {best_ndcg['Configuration']} ({best_ndcg['NDCG@5_mean']:.6f})",
    "",
    "--- Full Results ---",
]

for _, row in results_df.iterrows():
    summary_lines.append(
        f"  {row['Configuration']:30s}  "
        f"MAE={row['MAE']:.4f}  "
        f"RMSE={row['RMSE']:.4f}  "
        f"R²={row['R2']:.4f}  "
        f"NDCG@5={row['NDCG@5_mean']:.4f}  "
        f"MRR={row['MRR_mean']:.4f}"
    )

summary_lines.extend([
    "",
    "--- Complementarity Analysis ---",
])

# Check if pairwise > max of its components
for pair_name in [
    "TF-IDF + SBERT",
    "TF-IDF + Structured ATS",
    "SBERT + Structured ATS",
]:

    pair_r2 = results_df[
        results_df["Configuration"] == pair_name
    ]["R2"].values[0]

    # Find component names
    if pair_name == "TF-IDF + SBERT":
        comp_names = ["TF-IDF Only", "SBERT Only"]
    elif pair_name == "TF-IDF + Structured ATS":
        comp_names = ["TF-IDF Only", "Structured ATS Only"]
    else:
        comp_names = ["SBERT Only", "Structured ATS Only"]

    comp_r2s = [
        results_df[
            results_df["Configuration"] == c
        ]["R2"].values[0]
        for c in comp_names
    ]

    max_comp = max(comp_r2s)
    improvement = pair_r2 - max_comp

    if improvement > 0:
        summary_lines.append(
            f"  {pair_name}: R²={pair_r2:.4f} > "
            f"max({comp_names[0]}, {comp_names[1]})={max_comp:.4f} "
            f"-> COMPLEMENTARY (+{improvement:.4f})"
        )
    else:
        summary_lines.append(
            f"  {pair_name}: R²={pair_r2:.4f} <= "
            f"max({comp_names[0]}, {comp_names[1]})={max_comp:.4f} "
            f"-> No complementarity ({improvement:.4f})"
        )

# Check Full Hybrid vs best pairwise
hybrid_r2 = results_df[
    results_df["Configuration"] == "Full Hybrid"
]["R2"].values[0]

pairwise_configs = [
    "TF-IDF + SBERT",
    "TF-IDF + Structured ATS",
    "SBERT + Structured ATS",
]

best_pair = max(
    results_df[
        results_df["Configuration"].isin(pairwise_configs)
    ]["R2"]
)

full_improvement = hybrid_r2 - best_pair

summary_lines.append(
    f"\n  Full Hybrid R²={hybrid_r2:.4f} vs "
    f"Best Pairwise R²={best_pair:.4f} "
    f"-> {'COMPLEMENTARY' if full_improvement > 0 else 'No improvement'} "
    f"({'+' if full_improvement > 0 else ''}{full_improvement:.4f})"
)

summary_lines.extend([
    "",
    f"--- Timing ---",
    f"Total elapsed: {elapsed:.1f} seconds",
    "",
    "=" * 70,
    "END OF SEVEN-WAY ABLATION REPORT",
    "=" * 70,
])

summary_text = "\n".join(summary_lines)
print(summary_text)

summary_file = OUTPUT_DIR / "seven_way_summary.txt"

with open(summary_file, "w", encoding="utf-8") as f:
    f.write(summary_text)

print(f"\nSummary: {summary_file}")


# ==========================================================
# COMPLETE
# ==========================================================

section("EXPERIMENT COMPLETE")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
print(f"Total elapsed: {elapsed:.1f} seconds")
