"""
dataset_validation.py

Experiment 7: Dataset and Target Validation
=============================================

Purpose
-------
Validate the research dataset before finalizing model claims.

Checks
------
1. Dataset size and schema
2. Missing values
3. Duplicate rows
4. Duplicate resumes
5. Duplicate job descriptions
6. Duplicate resume-job pairs
7. Number of unique job groups
8. Candidates per job group
9. Target distribution
10. Target statistics
11. Target distribution by job group
12. Job-group target variance
13. Potential leakage indicators
14. Highly duplicated groups
15. Train/test leakage risk
16. Dataset imbalance
17. Research recommendations

Outputs
-------
dataset/processed/dataset_validation/
    dataset_overview.csv
    missing_values.csv
    duplicate_analysis.csv
    job_group_statistics.csv
    target_distribution.csv
    target_by_job_group.csv
    leakage_analysis.csv
    validation_summary.csv
    target_distribution.png
    candidates_per_job_group.png
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

from sklearn.model_selection import GroupShuffleSplit

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

OUTPUT_DIR = (
    PROCESSED_DIR
    / "dataset_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# CONFIGURATION
# ==========================================================

RANDOM_STATE = 42

TEST_SIZE = 0.20

LOW_TARGET_THRESHOLD = 0.20

HIGH_TARGET_THRESHOLD = 0.80

MIN_GROUP_SIZE = 5


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
# NORMALIZE COLUMN NAMES
# ==========================================================

df.columns = (

    df.columns

    .str.strip()

    .str.lower()

)


# ==========================================================
# BASIC INFORMATION
# ==========================================================

print()

print(
    "Number of rows:",
    len(df)
)

print(
    "Number of columns:",
    len(df.columns)
)


print()

print(
    "Columns:"
)

for i, column in enumerate(
    df.columns,
    start=1
):

    print(
        f"{i:2d}. {column}"
    )


# ==========================================================
# REQUIRED RESEARCH COLUMNS
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


missing_required = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_required:

    print()

    print(
        "WARNING: Missing research columns:"
    )

    for column in missing_required:

        print(
            " -",
            column
        )


# ==========================================================
# DATASET OVERVIEW
# ==========================================================

section(
    "DATASET OVERVIEW"
)


overview = pd.DataFrame(

    [

        {

            "Rows":
            len(df),

            "Columns":
            len(df.columns),

            "Unique_Rows":
            len(
                df.drop_duplicates()
            ),

            "Duplicate_Rows":
            int(
                df.duplicated()
                .sum()
            ),

            "Memory_MB":
            round(
                df.memory_usage(
                    deep=True
                ).sum()
                /
                1024
                /
                1024,
                2
            ),

        }

    ]

)


print()

print(
    overview.to_string(
        index=False
    )
)


# ==========================================================
# MISSING VALUES
# ==========================================================

section(
    "MISSING VALUE ANALYSIS"
)


missing_rows = []


for column in df.columns:

    missing_count = int(
        df[column]
        .isna()
        .sum()
    )

    missing_percentage = (

        missing_count
        /
        len(df)
        *
        100

    )


    missing_rows.append(

        {

            "Column":
            column,

            "Missing_Count":
            missing_count,

            "Missing_Percentage":
            missing_percentage,

            "Non_Missing_Count":
            len(df)
            -
            missing_count,

        }

    )


missing_df = pd.DataFrame(
    missing_rows
)


missing_df = (

    missing_df

    .sort_values(
        "Missing_Percentage",
        ascending=False
    )

    .reset_index(
        drop=True
    )

)


print()

print(
    missing_df.head(
        20
    ).to_string(
        index=False
    )
)


# ==========================================================
# DUPLICATE ROWS
# ==========================================================

section(
    "DUPLICATE ANALYSIS"
)


duplicate_count = int(
    df.duplicated()
    .sum()
)


duplicate_percentage = (

    duplicate_count
    /
    len(df)
    *
    100

)


print()

print(
    "Duplicate complete rows:",
    duplicate_count
)


print(
    "Duplicate percentage:",
    f"{duplicate_percentage:.2f}%"
)


# ==========================================================
# TEXT NORMALIZATION
# ==========================================================

def safe_text(value):

    if pd.isna(value):

        return ""

    return str(value).strip()


def normalize_text(value):

    text = safe_text(
        value
    )

    text = (
        text
        .lower()
    )

    text = (
        " ".join(
            text.split()
        )
    )

    return text


# ==========================================================
# BUILD RESUME REPRESENTATION
# ==========================================================

section(
    "BUILDING RESUME REPRESENTATIONS"
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


available_candidate_columns = [

    column

    for column in candidate_columns

    if column in df.columns

]


def combine_columns(
    row,
    columns
):

    values = []

    for column in columns:

        value = safe_text(
            row[column]
        )

        if value:

            values.append(
                value
            )

    return " ".join(
        values
    )


df["_candidate_text"] = df.apply(

    lambda row:

    combine_columns(
        row,
        available_candidate_columns
    ),

    axis=1

)


# ==========================================================
# BUILD JOB REPRESENTATION
# ==========================================================

job_columns = [

    "job_position_name",

    "skills_required",

    "educational_requirements",

    "experiencere_requirement",

]


available_job_columns = [

    column

    for column in job_columns

    if column in df.columns

]


df["_job_text"] = df.apply(

    lambda row:

    combine_columns(
        row,
        available_job_columns
    ),

    axis=1

)


df["_candidate_normalized"] = (

    df["_candidate_text"]

    .apply(
        normalize_text
    )

)


df["_job_normalized"] = (

    df["_job_text"]

    .apply(
        normalize_text
    )

)


df["_resume_job_pair"] = (

    df["_candidate_normalized"]

    + " ||| "

    + df["_job_normalized"]

)


# ==========================================================
# DUPLICATE RESUMES
# ==========================================================

section(
    "DUPLICATE RESUME ANALYSIS"
)


unique_resumes = (
    df["_candidate_normalized"]
    .nunique()
)


duplicate_resume_rows = (

    len(df)
    -
    unique_resumes

)


duplicate_resume_percentage = (

    duplicate_resume_rows
    /
    len(df)
    *
    100

)


print()

print(
    "Unique resume representations:",
    unique_resumes
)


print(
    "Repeated resume rows:",
    duplicate_resume_rows
)


print(
    "Repeated resume percentage:",
    f"{duplicate_resume_percentage:.2f}%"
)


# ==========================================================
# DUPLICATE JOBS
# ==========================================================

section(
    "DUPLICATE JOB ANALYSIS"
)


unique_jobs = (
    df["_job_normalized"]
    .nunique()
)


duplicate_job_rows = (

    len(df)
    -
    unique_jobs

)


duplicate_job_percentage = (

    duplicate_job_rows
    /
    len(df)
    *
    100

)


print()

print(
    "Unique job representations:",
    unique_jobs
)


print(
    "Repeated job rows:",
    duplicate_job_rows
)


print(
    "Repeated job percentage:",
    f"{duplicate_job_percentage:.2f}%"
)


# ==========================================================
# DUPLICATE RESUME-JOB PAIRS
# ==========================================================

section(
    "DUPLICATE RESUME-JOB PAIR ANALYSIS"
)


unique_pairs = (
    df["_resume_job_pair"]
    .nunique()
)


duplicate_pair_rows = (

    len(df)
    -
    unique_pairs

)


duplicate_pair_percentage = (

    duplicate_pair_rows
    /
    len(df)
    *
    100

)


print()

print(
    "Unique resume-job pairs:",
    unique_pairs
)


print(
    "Repeated resume-job pair rows:",
    duplicate_pair_rows
)


print(
    "Repeated pair percentage:",
    f"{duplicate_pair_percentage:.2f}%"
)


# ==========================================================
# JOB GROUPS
# ==========================================================

section(
    "JOB GROUP ANALYSIS"
)


df["job_group"] = (
    df["_job_normalized"]
)


job_group_counts = (

    df["job_group"]

    .value_counts()

)


print()

print(
    "Unique job groups:",
    len(job_group_counts)
)


print()

print(
    "Candidates per job group:"
)


print(

    job_group_counts

    .describe()

    .to_string()

)


# ==========================================================
# JOB GROUP STATISTICS
# ==========================================================

group_statistics = (

    df

    .groupby(
        "job_group"
    )

    .agg(

        Candidate_Count=(
            "matched_score",
            "count"
        ),

        Mean_Target=(
            "matched_score",
            "mean"
        ),

        Median_Target=(
            "matched_score",
            "median"
        ),

        Std_Target=(
            "matched_score",
            "std"
        ),

        Min_Target=(
            "matched_score",
            "min"
        ),

        Max_Target=(
            "matched_score",
            "max"
        ),

        Target_Range=(
            "matched_score",
            lambda x:
            x.max() - x.min()
        ),

    )

    .reset_index()

)


group_statistics = (

    group_statistics

    .sort_values(
        "Candidate_Count",
        ascending=False
    )

)


print()

print(
    "Largest job groups:"
)


print(

    group_statistics.head(
        15
    ).to_string(
        index=False
    )

)


# ==========================================================
# TARGET VALIDATION
# ==========================================================

section(
    "TARGET VALIDATION"
)


if "matched_score" not in df.columns:

    raise ValueError(
        "matched_score column is missing."
    )


target = pd.to_numeric(

    df["matched_score"],

    errors="coerce"

)


invalid_target_count = int(
    target.isna().sum()
)


out_of_range_count = int(

    (
        (target < 0)
        |
        (target > 1)
    )

    .sum()

)


target = target.clip(
    0,
    1
)


print()

print(
    "Invalid target values:",
    invalid_target_count
)


print(
    "Out-of-range target values:",
    out_of_range_count
)


print()

print(
    "Target statistics:"
)


print(
    target.describe().to_string()
)


# ==========================================================
# TARGET DISTRIBUTION
# ==========================================================

target_distribution = pd.DataFrame(

    {

        "Statistic": [

            "Count",

            "Mean",

            "Std",

            "Min",

            "25th Percentile",

            "Median",

            "75th Percentile",

            "Max",

        ],

        "Value": [

            target.count(),

            target.mean(),

            target.std(),

            target.min(),

            target.quantile(
                0.25
            ),

            target.median(),

            target.quantile(
                0.75
            ),

            target.max(),

        ]

    }

)


# ==========================================================
# TARGET BINS
# ==========================================================

target_bins = [

    0.0,

    0.1,

    0.2,

    0.3,

    0.4,

    0.5,

    0.6,

    0.7,

    0.8,

    0.9,

    1.0,

]


target_bin_labels = [

    "0.00-0.09",

    "0.10-0.19",

    "0.20-0.29",

    "0.30-0.39",

    "0.40-0.49",

    "0.50-0.59",

    "0.60-0.69",

    "0.70-0.79",

    "0.80-0.89",

    "0.90-1.00",

]


target_categories = pd.cut(

    target,

    bins=target_bins,

    labels=target_bin_labels,

    include_lowest=True,

)


target_distribution_bins = (

    target_categories

    .value_counts(
        sort=False
    )

    .reset_index()

)


target_distribution_bins.columns = [

    "Target_Range",

    "Sample_Count",

]


target_distribution_bins[

    "Percentage"

] = (

    target_distribution_bins[
        "Sample_Count"
    ]

    /
    len(df)

    *
    100

)


print()

print(
    "Target distribution:"
)


print(

    target_distribution_bins

    .to_string(
        index=False
    )

)


# ==========================================================
# TARGET IMBALANCE
# ==========================================================

section(
    "TARGET IMBALANCE ANALYSIS"
)


low_count = int(
    (target < LOW_TARGET_THRESHOLD)
    .sum()
)


medium_count = int(

    (
        (target >= LOW_TARGET_THRESHOLD)
        &
        (target < HIGH_TARGET_THRESHOLD)
    )

    .sum()

)


high_count = int(
    (
        target >= HIGH_TARGET_THRESHOLD
    )
    .sum()
)


print()

print(
    "Low-score samples (<0.20):",
    low_count,
    f"({low_count / len(df):.2%})"
)


print(
    "Middle-score samples (0.20-0.79):",
    medium_count,
    f"({medium_count / len(df):.2%})"
)


print(
    "High-score samples (>=0.80):",
    high_count,
    f"({high_count / len(df):.2%})"
)


# ==========================================================
# WITHIN-JOB TARGET VARIANCE
# ==========================================================

section(
    "WITHIN-JOB TARGET VARIANCE"
)


valid_group_statistics = group_statistics[
    group_statistics[
        "Candidate_Count"
    ]
    >= MIN_GROUP_SIZE
].copy()


print()

print(
    "Groups with at least",
    MIN_GROUP_SIZE,
    "candidates:",
    len(valid_group_statistics)
)


print()

print(
    "Mean within-job target standard deviation:",
    valid_group_statistics[
        "Std_Target"
    ].mean()
)


print(
    "Median within-job target standard deviation:",
    valid_group_statistics[
        "Std_Target"
    ].median()
)


print(
    "Mean within-job target range:",
    valid_group_statistics[
        "Target_Range"
    ].mean()
)


# ==========================================================
# JOB GROUP TARGET ANALYSIS
# ==========================================================

target_by_job_group = (

    group_statistics

    .sort_values(
        "Mean_Target"
    )

)


print()

print(
    "Lowest mean-target job groups:"
)


print(

    target_by_job_group.head(
        10
    ).to_string(
        index=False
    )

)


print()

print(
    "Highest mean-target job groups:"
)


print(

    target_by_job_group.tail(
        10
    ).to_string(
        index=False
    )

)


# ==========================================================
# POTENTIAL LEAKAGE ANALYSIS
# ==========================================================

section(
    "POTENTIAL LEAKAGE ANALYSIS"
)


# ----------------------------------------------------------
# Exact duplicate rows
# ----------------------------------------------------------

exact_duplicate_count = int(

    df.duplicated(
        keep=False
    ).sum()

)


# ----------------------------------------------------------
# Same resume with different target
# ----------------------------------------------------------

resume_target_variation = (

    df

    .groupby(
        "_candidate_normalized"
    )

    ["matched_score"]

    .nunique()

)


same_resume_different_target = int(

    (
        resume_target_variation > 1
    )

    .sum()

)


# ----------------------------------------------------------
# Same job with different target
# ----------------------------------------------------------

job_target_variation = (

    df

    .groupby(
        "_job_normalized"
    )

    ["matched_score"]

    .nunique()

)


same_job_different_target = int(

    (
        job_target_variation > 1
    )

    .sum()

)


# ----------------------------------------------------------
# Same resume-job pair with different target
# ----------------------------------------------------------

pair_target_variation = (

    df

    .groupby(
        "_resume_job_pair"
    )

    ["matched_score"]

    .nunique()

)


same_pair_different_target = int(

    (
        pair_target_variation > 1
    )

    .sum()

)


print()

print(
    "Rows belonging to exact duplicate rows:",
    exact_duplicate_count
)


print(
    "Resume identities with multiple target values:",
    same_resume_different_target
)


print(
    "Job identities with multiple target values:",
    same_job_different_target
)


print(
    "Resume-job pairs with multiple target values:",
    same_pair_different_target
)


# ==========================================================
# GROUP SPLIT LEAKAGE CHECK
# ==========================================================

section(
    "GROUP SPLIT LEAKAGE CHECK"
)


splitter = GroupShuffleSplit(

    n_splits=1,

    test_size=TEST_SIZE,

    random_state=RANDOM_STATE,

)


train_idx, test_idx = next(

    splitter.split(

        df,

        target,

        groups=df["job_group"]

    )

)


train_groups = set(

    df.iloc[
        train_idx
    ][
        "job_group"
    ]

)


test_groups = set(

    df.iloc[
        test_idx
    ][
        "job_group"
    ]

)


overlap_groups = (

    train_groups
    &
    test_groups

)


print()

print(
    "Training samples:",
    len(train_idx)
)


print(
    "Testing samples:",
    len(test_idx)
)


print(
    "Training groups:",
    len(train_groups)
)


print(
    "Testing groups:",
    len(test_groups)
)


print(
    "Overlapping job groups:",
    len(overlap_groups)
)


if overlap_groups:

    print()

    print(
        "WARNING: Group leakage detected!"
    )

else:

    print()

    print(
        "PASS: No job-group overlap."
    )


# ==========================================================
# RESUME LEAKAGE CHECK
# ==========================================================

train_resumes = set(

    df.iloc[
        train_idx
    ][
        "_candidate_normalized"
    ]

)


test_resumes = set(

    df.iloc[
        test_idx
    ][
        "_candidate_normalized"
    ]

)


overlap_resumes = (

    train_resumes
    &
    test_resumes

)


print()

print(
    "Resume representations shared between "
    "train and test:",
    len(overlap_resumes)
)


if overlap_resumes:

    print(
        "WARNING: Resume-level overlap exists."
    )

else:

    print(
        "PASS: No resume-level overlap."
    )


# ==========================================================
# JOB LEAKAGE CHECK
# ==========================================================

train_jobs = set(

    df.iloc[
        train_idx
    ][
        "_job_normalized"
    ]

)


test_jobs = set(

    df.iloc[
        test_idx
    ][
        "_job_normalized"
    ]

)


overlap_jobs = (

    train_jobs
    &
    test_jobs

)


print()

print(
    "Job descriptions shared between "
    "train and test:",
    len(overlap_jobs)
)


if overlap_jobs:

    print(
        "WARNING: Job-level overlap exists."
    )

else:

    print(
        "PASS: No job-level overlap."
    )


# ==========================================================
# VALIDATION SUMMARY
# ==========================================================

section(
    "RESEARCH VALIDATION SUMMARY"
)


validation_rows = [

    {

        "Check":
        "Dataset rows",

        "Value":
        len(df),

        "Status":
        "INFO",

    },

    {

        "Check":
        "Dataset columns",

        "Value":
        len(df.columns),

        "Status":
        "INFO",

    },

    {

        "Check":
        "Duplicate complete rows",

        "Value":
        duplicate_count,

        "Status":
        (
            "WARNING"
            if duplicate_count > 0
            else "PASS"
        ),

    },

    {

        "Check":
        "Unique resumes",

        "Value":
        unique_resumes,

        "Status":
        "INFO",

    },

    {

        "Check":
        "Unique job groups",

        "Value":
        len(job_group_counts),

        "Status":
        "INFO",

    },

    {

        "Check":
        "Unique resume-job pairs",

        "Value":
        unique_pairs,

        "Status":
        "INFO",

    },

    {

        "Check":
        "Invalid target values",

        "Value":
        invalid_target_count,

        "Status":
        (
            "WARNING"
            if invalid_target_count > 0
            else "PASS"
        ),

    },

    {

        "Check":
        "Out-of-range targets",

        "Value":
        out_of_range_count,

        "Status":
        (
            "WARNING"
            if out_of_range_count > 0
            else "PASS"
        ),

    },

    {

        "Check":
        "Train/test job-group overlap",

        "Value":
        len(overlap_groups),

        "Status":
        (
            "FAIL"
            if overlap_groups
            else "PASS"
        ),

    },

    {

        "Check":
        "Train/test resume overlap",

        "Value":
        len(overlap_resumes),

        "Status":
        (
            "WARNING"
            if overlap_resumes
            else "PASS"
        ),

    },

    {

        "Check":
        "Train/test job overlap",

        "Value":
        len(overlap_jobs),

        "Status":
        (
            "WARNING"
            if overlap_jobs
            else "PASS"
        ),

    },

    {

        "Check":
        "Low-score target percentage",

        "Value":
        low_count / len(df),

        "Status":
        "INFO",

    },

    {

        "Check":
        "High-score target percentage",

        "Value":
        high_count / len(df),

        "Status":
        "INFO",

    },

    {

        "Check":
        "Mean within-job target std",

        "Value":
        valid_group_statistics[
            "Std_Target"
        ].mean(),

        "Status":
        "INFO",

    },

]


validation_summary = pd.DataFrame(
    validation_rows
)


print()

print(

    validation_summary.to_string(
        index=False
    )

)


# ==========================================================
# RESEARCH RECOMMENDATIONS
# ==========================================================

section(
    "RESEARCH RECOMMENDATIONS"
)


recommendations = []


if duplicate_count > 0:

    recommendations.append(

        "Investigate duplicate rows before final modeling."

    )


if same_pair_different_target > 0:

    recommendations.append(

        "Investigate identical resume-job pairs with "
        "different target values."

    )


if len(overlap_groups) == 0:

    recommendations.append(

        "Group-based evaluation successfully prevents "
        "job-group leakage."

    )


if len(overlap_resumes) > 0:

    recommendations.append(

        "Consider resume-level grouping if the same "
        "candidate appears across multiple jobs."

    )


if len(overlap_jobs) > 0:

    recommendations.append(

        "Use job-group splitting for generalization "
        "to unseen job descriptions."

    )


if low_count < 100:

    recommendations.append(

        "Very low match-score samples are relatively "
        "rare; report this limitation."

    )


if high_count / len(df) > 0.30:

    recommendations.append(

        "High-match candidates represent a substantial "
        "portion of the dataset; inspect target imbalance."

    )


recommendations.append(

    "Use group-based evaluation as the primary "
    "generalization experiment."

)


recommendations.append(

    "Do not claim that random row-level splits represent "
    "generalization to unseen job descriptions."

)


for i, recommendation in enumerate(
    recommendations,
    start=1
):

    print(
        f"{i}. {recommendation}"
    )


# ==========================================================
# SAVE OUTPUTS
# ==========================================================

section(
    "SAVING VALIDATION RESULTS"
)


overview_file = (
    OUTPUT_DIR
    / "dataset_overview.csv"
)


overview.to_csv(
    overview_file,
    index=False
)


missing_file = (
    OUTPUT_DIR
    / "missing_values.csv"
)


missing_df.to_csv(
    missing_file,
    index=False
)


duplicate_analysis = pd.DataFrame(

    [

        {

            "Metric":
            "Duplicate complete rows",

            "Count":
            duplicate_count,

            "Percentage":
            duplicate_percentage,

        },

        {

            "Metric":
            "Repeated resume rows",

            "Count":
            duplicate_resume_rows,

            "Percentage":
            duplicate_resume_percentage,

        },

        {

            "Metric":
            "Repeated job rows",

            "Count":
            duplicate_job_rows,

            "Percentage":
            duplicate_job_percentage,

        },

        {

            "Metric":
            "Repeated resume-job pairs",

            "Count":
            duplicate_pair_rows,

            "Percentage":
            duplicate_pair_percentage,

        },

    ]

)


duplicate_file = (
    OUTPUT_DIR
    / "duplicate_analysis.csv"
)


duplicate_analysis.to_csv(
    duplicate_file,
    index=False
)


group_file = (
    OUTPUT_DIR
    / "job_group_statistics.csv"
)


group_statistics.to_csv(
    group_file,
    index=False
)


target_file = (
    OUTPUT_DIR
    / "target_distribution.csv"
)


target_distribution_bins.to_csv(
    target_file,
    index=False
)


target_group_file = (
    OUTPUT_DIR
    / "target_by_job_group.csv"
)


target_by_job_group.to_csv(
    target_group_file,
    index=False
)


leakage_analysis = pd.DataFrame(

    [

        {

            "Metric":
            "Train job groups",

            "Value":
            len(train_groups),

        },

        {

            "Metric":
            "Test job groups",

            "Value":
            len(test_groups),

        },

        {

            "Metric":
            "Overlapping job groups",

            "Value":
            len(overlap_groups),

        },

        {

            "Metric":
            "Overlapping resumes",

            "Value":
            len(overlap_resumes),

        },

        {

            "Metric":
            "Overlapping jobs",

            "Value":
            len(overlap_jobs),

        },

        {

            "Metric":
            "Resume-job pairs with multiple targets",

            "Value":
            same_pair_different_target,

        },

    ]

)


leakage_file = (
    OUTPUT_DIR
    / "leakage_analysis.csv"
)


leakage_analysis.to_csv(
    leakage_file,
    index=False
)


validation_file = (
    OUTPUT_DIR
    / "validation_summary.csv"
)


validation_summary.to_csv(
    validation_file,
    index=False
)


# ==========================================================
# PLOT 1 — TARGET DISTRIBUTION
# ==========================================================

section(
    "GENERATING TARGET DISTRIBUTION PLOT"
)


plt.figure(
    figsize=(10, 6)
)


plt.hist(

    target,

    bins=20,

)


plt.xlabel(
    "Matched Score"
)


plt.ylabel(
    "Number of Samples"
)


plt.title(
    "Distribution of Resume-Job Matching Scores"
)


plt.tight_layout()


target_plot_file = (
    OUTPUT_DIR
    / "target_distribution.png"
)


plt.savefig(

    target_plot_file,

    dpi=300

)


plt.close()


# ==========================================================
# PLOT 2 — CANDIDATES PER JOB GROUP
# ==========================================================

plt.figure(
    figsize=(10, 6)
)


plt.hist(

    job_group_counts,

    bins=20,

)


plt.xlabel(
    "Candidates per Job Group"
)


plt.ylabel(
    "Number of Job Groups"
)


plt.title(
    "Distribution of Candidates Across Job Groups"
)


plt.tight_layout()


group_plot_file = (
    OUTPUT_DIR
    / "candidates_per_job_group.png"
)


plt.savefig(

    group_plot_file,

    dpi=300

)


plt.close()


# ==========================================================
# CLEAN TEMPORARY COLUMNS
# ==========================================================

temporary_columns = [

    "_candidate_text",

    "_job_text",

    "_candidate_normalized",

    "_job_normalized",

    "_resume_job_pair",

]


df.drop(

    columns=[
        column
        for column in temporary_columns
        if column in df.columns
    ],

    inplace=True,

)


# ==========================================================
# FINAL OUTPUT
# ==========================================================

section(
    "EXPERIMENT 7 COMPLETE"
)


print()

print(
    "Validation files generated:"
)


print()

print(
    overview_file
)


print(
    missing_file
)


print(
    duplicate_file
)


print(
    group_file
)


print(
    target_file
)


print(
    target_group_file
)


print(
    leakage_file
)


print(
    validation_file
)


print(
    target_plot_file
)


print(
    group_plot_file
)


print()

print("=" * 70)

print(
    "DATASET VALIDATION FINISHED SUCCESSFULLY"
)

print("=" * 70)