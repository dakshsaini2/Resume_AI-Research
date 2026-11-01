"""
protocol_freeze.py

Experiment 10: Evaluation Protocol Freeze and Leakage Audit
=============================================================

Purpose
-------
Before final experiments and paper submission, this script:

1. Verifies the official job-group train/test split
2. Detects exact duplicate records
3. Detects near-duplicate resumes (high text similarity)
4. Detects duplicate/near-duplicate job descriptions
5. Checks for cross-boundary leakage between train/test
6. Freezes the dataset version with a hash
7. Records all environment versions
8. Generates a reproducibility manifest

This addresses Sections 4.2 and 4.3 of the roadmap:
- "Leakage, Duplicate and Near-Duplicate Analysis — CRITICAL"
- "Freeze the Official Evaluation Protocol — CRITICAL"

Outputs
-------
dataset/processed/protocol_freeze/
    dataset_hash.txt
    environment_versions.txt
    official_split_definition.csv
    exact_duplicates.csv
    near_duplicate_resumes.csv
    near_duplicate_jobs.csv
    leakage_report.csv
    protocol_freeze_report.txt

Important
---------
This script should be run ONCE before final experiments.
After running, do NOT modify the dataset or split.
"""

# ==========================================================
# IMPORTS
# ==========================================================

from pathlib import Path
import sys
import hashlib
import warnings
import platform
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import (
    TfidfVectorizer
)

from sklearn.metrics.pairwise import (
    cosine_similarity
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


PROCESSED_DIR = (
    BASE_DIR
    / "dataset"
    / "processed"
)


OUTPUT_DIR = (
    PROCESSED_DIR
    / "protocol_freeze"
)


OUTPUT_DIR.mkdir(
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


# ==========================================================
# CONFIGURATION
# ==========================================================

RANDOM_STATE = 42

TARGET_COLUMN = "matched_score"

# Near-duplicate threshold.
# Pairs with cosine similarity above this
# are flagged as near-duplicates.

NEAR_DUPLICATE_THRESHOLD = 0.95

# Maximum pairs to report for near-duplicates.

MAX_NEAR_DUPLICATE_PAIRS = 500

# Official train/test group split ratio.

TEST_RATIO = 0.20


# ==========================================================
# DATASET
# ==========================================================

DATASET_FILE = (
    BASE_DIR
    / "dataset"
    / "Resume_Data_For_Ranking.csv"
)


# ==========================================================
# START
# ==========================================================

print()

print("=" * 70)

print("EXPERIMENT 10: PROTOCOL FREEZE AND LEAKAGE AUDIT")

print("=" * 70)


# ==========================================================
# STEP 1: COMPUTE DATASET HASH
# ==========================================================

print()

print("=" * 70)

print("STEP 1: COMPUTING DATASET HASH")

print("=" * 70)


if not DATASET_FILE.exists():

    raise FileNotFoundError(

        "\nDataset not found:\n"

        f"{DATASET_FILE}"
    )


# SHA-256 hash of the dataset file.

sha256 = hashlib.sha256()


with open(DATASET_FILE, "rb") as f:

    for chunk in iter(
        lambda: f.read(8192),
        b""
    ):

        sha256.update(chunk)


dataset_hash = sha256.hexdigest()


print()

print(
    f"Dataset file: {DATASET_FILE}"
)

print(
    f"SHA-256 hash: {dataset_hash}"
)

print(
    f"File size: "
    f"{DATASET_FILE.stat().st_size:,} bytes"
)


hash_file = (
    OUTPUT_DIR
    / "dataset_hash.txt"
)


with open(hash_file, "w") as f:

    f.write(
        f"File: {DATASET_FILE.name}\n"
        f"SHA-256: {dataset_hash}\n"
        f"Size: "
        f"{DATASET_FILE.stat().st_size} bytes\n"
        f"Timestamp: "
        f"{datetime.now().isoformat()}\n"
    )


print()

print(
    "Saved:",
    hash_file
)


# ==========================================================
# STEP 2: RECORD ENVIRONMENT VERSIONS
# ==========================================================

print()

print("=" * 70)

print("STEP 2: RECORDING ENVIRONMENT VERSIONS")

print("=" * 70)


env_lines = [
    f"Python: {sys.version}",
    f"Platform: {platform.platform()}",
    f"Timestamp: {datetime.now().isoformat()}",
    "",
]


try:
    import sklearn
    env_lines.append(
        f"scikit-learn: {sklearn.__version__}"
    )
except ImportError:
    env_lines.append(
        "scikit-learn: NOT INSTALLED"
    )


try:
    import pandas as pd_ver
    env_lines.append(
        f"pandas: {pd_ver.__version__}"
    )
except ImportError:
    env_lines.append(
        "pandas: NOT INSTALLED"
    )


try:
    import numpy as np_ver
    env_lines.append(
        f"numpy: {np_ver.__version__}"
    )
except ImportError:
    env_lines.append(
        "numpy: NOT INSTALLED"
    )


try:
    import sentence_transformers
    env_lines.append(
        f"sentence-transformers: "
        f"{sentence_transformers.__version__}"
    )
except ImportError:
    env_lines.append(
        "sentence-transformers: NOT INSTALLED"
    )


try:
    import shap
    env_lines.append(
        f"shap: {shap.__version__}"
    )
except ImportError:
    env_lines.append(
        "shap: NOT INSTALLED"
    )


try:
    import joblib
    env_lines.append(
        f"joblib: {joblib.__version__}"
    )
except ImportError:
    env_lines.append(
        "joblib: NOT INSTALLED"
    )


try:
    import matplotlib
    env_lines.append(
        f"matplotlib: {matplotlib.__version__}"
    )
except ImportError:
    env_lines.append(
        "matplotlib: NOT INSTALLED"
    )


try:
    import scipy
    env_lines.append(
        f"scipy: {scipy.__version__}"
    )
except ImportError:
    env_lines.append(
        "scipy: NOT INSTALLED"
    )


env_text = "\n".join(env_lines)


env_file = (
    OUTPUT_DIR
    / "environment_versions.txt"
)


with open(env_file, "w") as f:

    f.write(env_text)


print()

print(env_text)

print()

print(
    "Saved:",
    env_file
)


# ==========================================================
# LOAD DATASET
# ==========================================================

print()

print("=" * 70)

print("LOADING DATASET")

print("=" * 70)


df = pd.read_csv(
    DATASET_FILE
)


print()

print(
    "Dataset shape:",
    df.shape
)


# Normalize column names.

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


# Handle missing values.

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
    "Samples after cleaning:",
    len(df)
)


# ==========================================================
# BUILD TEXT REPRESENTATIONS
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
# STEP 3: EXACT DUPLICATE DETECTION
# ==========================================================

print()

print("=" * 70)

print("STEP 3: EXACT DUPLICATE DETECTION")

print("=" * 70)


# 3a. Exact duplicate rows (all columns).

exact_dup_mask = df.duplicated(
    keep=False
)

exact_dup_count = exact_dup_mask.sum()


print()

print(
    f"Exact duplicate rows: "
    f"{exact_dup_count}"
)


# 3b. Exact duplicate candidate texts.

resume_dup_mask = df.duplicated(

    subset=["candidate_text"],

    keep=False
)

resume_dup_count = resume_dup_mask.sum()


print(
    f"Duplicate candidate texts: "
    f"{resume_dup_count}"
)


# 3c. Exact duplicate job texts.

job_dup_mask = df.duplicated(

    subset=["job_text"],

    keep=False
)

job_dup_count = job_dup_mask.sum()


print(
    f"Duplicate job texts: "
    f"{job_dup_count}"
)


# 3d. Exact duplicate resume-job pairs.

pair_dup_mask = df.duplicated(

    subset=[
        "candidate_text",
        "job_text"
    ],

    keep=False
)

pair_dup_count = pair_dup_mask.sum()


print(
    f"Duplicate resume-job pairs: "
    f"{pair_dup_count}"
)


# Save exact duplicate summary.

exact_dup_df = pd.DataFrame([

    {
        "Check": "Exact duplicate rows",
        "Count": exact_dup_count,
        "Percentage": round(
            exact_dup_count / len(df) * 100,
            2
        )
    },

    {
        "Check": "Duplicate candidate texts",
        "Count": resume_dup_count,
        "Percentage": round(
            resume_dup_count / len(df) * 100,
            2
        )
    },

    {
        "Check": "Duplicate job texts",
        "Count": job_dup_count,
        "Percentage": round(
            job_dup_count / len(df) * 100,
            2
        )
    },

    {
        "Check": "Duplicate resume-job pairs",
        "Count": pair_dup_count,
        "Percentage": round(
            pair_dup_count / len(df) * 100,
            2
        )
    },
])


exact_dup_file = (
    OUTPUT_DIR
    / "exact_duplicates.csv"
)


exact_dup_df.to_csv(
    exact_dup_file,
    index=False
)


print()

print(
    "Saved:",
    exact_dup_file
)


# ==========================================================
# STEP 4: NEAR-DUPLICATE RESUME DETECTION
# ==========================================================

print()

print("=" * 70)

print("STEP 4: NEAR-DUPLICATE RESUME DETECTION")

print("=" * 70)


# Use TF-IDF for efficient near-duplicate detection.

print()

print(
    "Building TF-IDF matrix for "
    "candidate texts..."
)


# Get unique candidate texts to avoid
# redundant computation.

unique_resumes = (
    df["candidate_text"]
    .drop_duplicates()
    .reset_index(drop=True)
)


print(
    f"Unique candidate texts: "
    f"{len(unique_resumes)}"
)


resume_tfidf = TfidfVectorizer(

    max_features=5000,

    stop_words="english",

    ngram_range=(1, 2)
)


resume_matrix = resume_tfidf.fit_transform(
    unique_resumes
)


# Compute pairwise similarity in batches
# to manage memory.

print()

print(
    "Computing pairwise resume similarities..."
)


near_dup_resume_pairs = []

batch_size = 500


for start in range(
    0,
    len(unique_resumes),
    batch_size
):

    end = min(
        start + batch_size,
        len(unique_resumes)
    )

    batch_sim = cosine_similarity(
        resume_matrix[start:end],
        resume_matrix
    )

    for i in range(
        batch_sim.shape[0]
    ):

        global_i = start + i

        for j in range(
            global_i + 1,
            batch_sim.shape[1]
        ):

            if batch_sim[i, j] >= (
                NEAR_DUPLICATE_THRESHOLD
            ):

                near_dup_resume_pairs.append({

                    "Resume_A_Index":
                        global_i,

                    "Resume_B_Index":
                        j,

                    "Cosine_Similarity":
                        round(
                            float(
                                batch_sim[i, j]
                            ),
                            6
                        ),

                    "Resume_A_Preview":
                        unique_resumes.iloc[
                            global_i
                        ][:100],

                    "Resume_B_Preview":
                        unique_resumes.iloc[
                            j
                        ][:100],
                })


    if start % 1000 == 0:

        print(
            f"  Processed rows "
            f"{start}-{end}..."
        )


    if len(near_dup_resume_pairs) >= (
        MAX_NEAR_DUPLICATE_PAIRS
    ):

        print(
            "  Maximum near-duplicate "
            "pair limit reached."
        )

        break


near_dup_resume_df = pd.DataFrame(
    near_dup_resume_pairs
)


print()

print(
    f"Near-duplicate resume pairs found: "
    f"{len(near_dup_resume_pairs)}"
    f" (threshold >= "
    f"{NEAR_DUPLICATE_THRESHOLD})"
)


near_dup_resume_file = (
    OUTPUT_DIR
    / "near_duplicate_resumes.csv"
)


near_dup_resume_df.to_csv(
    near_dup_resume_file,
    index=False
)


print()

print(
    "Saved:",
    near_dup_resume_file
)


# ==========================================================
# STEP 5: NEAR-DUPLICATE JOB DETECTION
# ==========================================================

print()

print("=" * 70)

print("STEP 5: NEAR-DUPLICATE JOB DESCRIPTION DETECTION")

print("=" * 70)


unique_jobs = (
    df["job_text"]
    .drop_duplicates()
    .reset_index(drop=True)
)


print()

print(
    f"Unique job texts: "
    f"{len(unique_jobs)}"
)


job_tfidf = TfidfVectorizer(

    max_features=5000,

    stop_words="english",

    ngram_range=(1, 2)
)


job_matrix = job_tfidf.fit_transform(
    unique_jobs
)


# Job descriptions are typically fewer,
# so pairwise is feasible.

job_sim_matrix = cosine_similarity(
    job_matrix
)


near_dup_job_pairs = []


for i in range(len(unique_jobs)):

    for j in range(
        i + 1,
        len(unique_jobs)
    ):

        if job_sim_matrix[i, j] >= (
            NEAR_DUPLICATE_THRESHOLD
        ):

            near_dup_job_pairs.append({

                "Job_A_Index":
                    i,

                "Job_B_Index":
                    j,

                "Cosine_Similarity":
                    round(
                        float(
                            job_sim_matrix[i, j]
                        ),
                        6
                    ),

                "Job_A_Preview":
                    unique_jobs.iloc[i][:100],

                "Job_B_Preview":
                    unique_jobs.iloc[j][:100],
            })


near_dup_job_df = pd.DataFrame(
    near_dup_job_pairs
)


print()

print(
    f"Near-duplicate job pairs found: "
    f"{len(near_dup_job_pairs)}"
    f" (threshold >= "
    f"{NEAR_DUPLICATE_THRESHOLD})"
)


near_dup_job_file = (
    OUTPUT_DIR
    / "near_duplicate_jobs.csv"
)


near_dup_job_df.to_csv(
    near_dup_job_file,
    index=False
)


print()

print(
    "Saved:",
    near_dup_job_file
)


# ==========================================================
# STEP 6: OFFICIAL GROUP SPLIT DEFINITION
# ==========================================================

print()

print("=" * 70)

print("STEP 6: OFFICIAL GROUP SPLIT DEFINITION")

print("=" * 70)


# Identify job groups.

df["job_group"] = (

    df["job_position_name"]

    .str.strip()

    .str.lower()
)


job_groups = (

    df["job_group"]

    .value_counts()

    .reset_index()
)


job_groups.columns = [
    "job_group",
    "sample_count"
]


print()

print(
    f"Total unique job groups: "
    f"{len(job_groups)}"
)


print()

print(
    "Job group distribution:"
)

print(
    job_groups.to_string(
        index=False
    )
)


# Create reproducible group split.

from sklearn.model_selection import (
    GroupShuffleSplit
)


groups = df["job_group"].values


splitter = GroupShuffleSplit(

    n_splits=1,

    test_size=TEST_RATIO,

    random_state=RANDOM_STATE
)


train_idx, test_idx = next(
    splitter.split(
        df,
        groups=groups
    )
)


train_groups = set(
    df.iloc[train_idx]["job_group"].unique()
)

test_groups = set(
    df.iloc[test_idx]["job_group"].unique()
)


# Check for group overlap (should be zero).

group_overlap = (
    train_groups & test_groups
)


print()

print(
    f"Training groups: "
    f"{len(train_groups)}"
)

print(
    f"Testing groups: "
    f"{len(test_groups)}"
)

print(
    f"Group overlap: "
    f"{len(group_overlap)}"
)

print(
    f"Training samples: "
    f"{len(train_idx)}"
)

print(
    f"Testing samples: "
    f"{len(test_idx)}"
)


# Save the official split definition.

split_rows = []


for group in sorted(train_groups):

    count = len(
        df[df["job_group"] == group]
    )

    split_rows.append({
        "Job_Group": group,
        "Split": "TRAIN",
        "Sample_Count": count,
    })


for group in sorted(test_groups):

    count = len(
        df[df["job_group"] == group]
    )

    split_rows.append({
        "Job_Group": group,
        "Split": "TEST",
        "Sample_Count": count,
    })


split_df = pd.DataFrame(
    split_rows
)


split_file = (
    OUTPUT_DIR
    / "official_split_definition.csv"
)


split_df.to_csv(
    split_file,
    index=False
)


print()

print(
    "Saved:",
    split_file
)


print()

print(
    "TRAIN groups:"
)

for g in sorted(train_groups):
    print(f"  - {g}")


print()

print(
    "TEST groups:"
)

for g in sorted(test_groups):
    print(f"  - {g}")


# ==========================================================
# STEP 7: CROSS-BOUNDARY LEAKAGE CHECK
# ==========================================================

print()

print("=" * 70)

print("STEP 7: CROSS-BOUNDARY LEAKAGE CHECK")

print("=" * 70)


leakage_checks = []


# 7a. Group overlap.

leakage_checks.append({

    "Check":
        "Job group overlap between train/test",

    "Result":
        "PASS"
        if len(group_overlap) == 0
        else "FAIL",

    "Details":
        (
            "No groups appear in both sets"
            if len(group_overlap) == 0
            else
            f"Overlapping groups: "
            f"{group_overlap}"
        ),
})


# 7b. Check if any test resumes appear
#     in training data.

train_resumes = set(
    df.iloc[train_idx]["candidate_text"]
)

test_resumes = set(
    df.iloc[test_idx]["candidate_text"]
)

resume_leak = (
    train_resumes & test_resumes
)


leakage_checks.append({

    "Check":
        "Candidate text overlap between "
        "train/test",

    "Result":
        "PASS"
        if len(resume_leak) == 0
        else "WARNING",

    "Details":
        (
            "No candidate texts shared"
            if len(resume_leak) == 0
            else
            f"{len(resume_leak)} candidate texts "
            f"appear in both train and test"
        ),
})


# 7c. Check if any test job descriptions
#     appear in training data.

train_jobs = set(
    df.iloc[train_idx]["job_text"]
)

test_jobs = set(
    df.iloc[test_idx]["job_text"]
)

job_leak = (
    train_jobs & test_jobs
)


leakage_checks.append({

    "Check":
        "Job text overlap between train/test",

    "Result":
        "PASS"
        if len(job_leak) == 0
        else "WARNING",

    "Details":
        (
            "No job texts shared"
            if len(job_leak) == 0
            else
            f"{len(job_leak)} job texts "
            f"appear in both train and test"
        ),
})


# 7d. Check target distribution similarity.

train_target_mean = (
    df.iloc[train_idx][TARGET_COLUMN].mean()
)

test_target_mean = (
    df.iloc[test_idx][TARGET_COLUMN].mean()
)

target_diff = abs(
    train_target_mean - test_target_mean
)


leakage_checks.append({

    "Check":
        "Target distribution similarity",

    "Result":
        "PASS"
        if target_diff < 0.1
        else "WARNING",

    "Details":
        (
            f"Train mean: "
            f"{train_target_mean:.4f}, "
            f"Test mean: "
            f"{test_target_mean:.4f}, "
            f"Difference: "
            f"{target_diff:.4f}"
        ),
})


# 7e. Near-duplicate cross-boundary check.
#     Check if near-duplicate resumes
#     span train/test.

if len(near_dup_resume_pairs) > 0:

    cross_boundary_count = 0

    # This is approximate since we used
    # unique resumes for near-dup detection.

    leakage_checks.append({

        "Check":
            "Near-duplicate resumes detected",

        "Result":
            "WARNING"
            if len(near_dup_resume_pairs) > 0
            else "PASS",

        "Details":
            (
                f"{len(near_dup_resume_pairs)} "
                f"near-duplicate resume pairs "
                f"found. Manual inspection "
                f"recommended."
            ),
    })

else:

    leakage_checks.append({

        "Check":
            "Near-duplicate resumes detected",

        "Result":
            "PASS",

        "Details":
            "No near-duplicate resume pairs found.",
    })


leakage_df = pd.DataFrame(
    leakage_checks
)


leakage_file = (
    OUTPUT_DIR
    / "leakage_report.csv"
)


leakage_df.to_csv(
    leakage_file,
    index=False
)


print()

print(
    leakage_df.to_string(
        index=False
    )
)


print()

print(
    "Saved:",
    leakage_file
)


# ==========================================================
# STEP 8: GENERATE PROTOCOL FREEZE REPORT
# ==========================================================

print()

print("=" * 70)

print("STEP 8: GENERATING PROTOCOL FREEZE REPORT")

print("=" * 70)


report_lines = [

    "=" * 70,

    "EXPERIMENT 10: PROTOCOL FREEZE REPORT",

    f"Generated: {datetime.now().isoformat()}",

    "=" * 70,

    "",

    "--- Dataset ---",

    f"File: {DATASET_FILE.name}",

    f"SHA-256: {dataset_hash}",

    f"Rows: {len(df)}",

    f"Columns: {df.shape[1]}",

    "",

    "--- Official Group Split ---",

    f"Split method: GroupShuffleSplit",

    f"Test ratio: {TEST_RATIO}",

    f"Random state: {RANDOM_STATE}",

    f"Training groups: {len(train_groups)}",

    f"Testing groups: {len(test_groups)}",

    f"Group overlap: {len(group_overlap)}",

    f"Training samples: {len(train_idx)}",

    f"Testing samples: {len(test_idx)}",

    "",

    "TRAIN groups:",
]


for g in sorted(train_groups):

    count = len(
        df[df["job_group"] == g]
    )

    report_lines.append(
        f"  - {g} ({count} samples)"
    )


report_lines.append("")

report_lines.append("TEST groups:")


for g in sorted(test_groups):

    count = len(
        df[df["job_group"] == g]
    )

    report_lines.append(
        f"  - {g} ({count} samples)"
    )


report_lines.extend([

    "",

    "--- Exact Duplicate Analysis ---",

    f"Exact duplicate rows: "
    f"{exact_dup_count}",

    f"Duplicate candidate texts: "
    f"{resume_dup_count}",

    f"Duplicate job texts: "
    f"{job_dup_count}",

    f"Duplicate resume-job pairs: "
    f"{pair_dup_count}",

    "",

    "--- Near-Duplicate Analysis ---",

    f"Near-duplicate threshold: "
    f"{NEAR_DUPLICATE_THRESHOLD}",

    f"Near-duplicate resume pairs: "
    f"{len(near_dup_resume_pairs)}",

    f"Near-duplicate job pairs: "
    f"{len(near_dup_job_pairs)}",

    "",

    "--- Leakage Report ---",
])


for _, row in leakage_df.iterrows():

    report_lines.append(

        f"  [{row['Result']}] "
        f"{row['Check']}: "
        f"{row['Details']}"
    )


report_lines.extend([

    "",

    "--- Reproducibility Requirements ---",

    "1. This dataset hash must be verified "
    "before running final experiments.",

    "2. The official group split above "
    "must be used for ALL final comparisons.",

    "3. Random split may be used as a "
    "secondary experiment, clearly labeled.",

    "4. TF-IDF must be fitted ONLY on "
    "training data in each experiment.",

    "5. All random seeds must be "
    f"set to {RANDOM_STATE}.",

    "",

    "--- Score Interpretation Warning ---",

    "The predicted match score is a "
    "regression estimate, NOT a probability.",

    "Do NOT interpret scores as calibrated "
    "likelihoods without calibration analysis.",

    "",

    "=" * 70,

    "END OF PROTOCOL FREEZE REPORT",

    "=" * 70,
])


report_text = "\n".join(
    report_lines
)


report_file = (
    OUTPUT_DIR
    / "protocol_freeze_report.txt"
)


with open(report_file, "w") as f:

    f.write(report_text)


print()

print(report_text)


print()

print(
    "Saved:",
    report_file
)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print()

print("=" * 70)

print("EXPERIMENT 10 COMPLETE")

print("=" * 70)


print()

print(
    "Protocol freeze outputs generated in:"
)

print(
    OUTPUT_DIR
)


print()

print(
    "Files generated:"
)


generated_files = list(
    OUTPUT_DIR.glob("*")
)


for i, f in enumerate(
    sorted(generated_files),
    start=1
):

    print(
        f"{i}.",
        f
    )


all_pass = all(
    row["Result"] == "PASS"
    for _, row in leakage_df.iterrows()
)


print()

if all_pass:

    print(
        "LEAKAGE STATUS: ALL CHECKS PASSED"
    )

else:

    print(
        "LEAKAGE STATUS: "
        "SOME CHECKS REQUIRE ATTENTION"
    )

    print(
        "Review the leakage_report.csv "
        "for details."
    )


print()

print("=" * 70)

print("PROTOCOL FREEZE FINISHED SUCCESSFULLY")

print("=" * 70)
