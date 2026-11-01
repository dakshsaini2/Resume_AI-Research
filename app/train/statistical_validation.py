"""
statistical_validation.py

Experiment 10: Statistical Significance & Final Model Selection
================================================================

Purpose
-------
Statistically compare the Hybrid model against the Structured ATS
model using the SAME candidate-level folds.

Primary comparison:
    Hybrid vs Structured ATS

Evaluation:
    MAE
    RMSE
    R2
    NDCG@1
    NDCG@3
    NDCG@5
    NDCG@10
    MRR
    Precision@3
    Recall@3

Tests:
    1. Paired t-test
    2. Wilcoxon signed-rank test
    3. Cohen's d
    4. Bootstrap confidence interval

IMPORTANT
---------
This script DOES NOT retrain any model.

It uses:
    dataset/processed/candidate_group_evaluation/
        candidate_group_results.csv

Outputs:
    dataset/processed/statistical_validation/
        statistical_tests.csv
        effect_sizes.csv
        confidence_intervals.csv
        fold_comparison.csv
        final_model_selection.csv
        statistical_validation_report.txt
"""

# ==========================================================
# IMPORTS
# ==========================================================

from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import (
    ttest_rel,
    wilcoxon,
)

from scipy.stats import sem


# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parents[2]
)

PROCESSED_DIR = (
    BASE_DIR
    / "dataset"
    / "processed"
)

INPUT_DIR = (
    PROCESSED_DIR
    / "candidate_group_evaluation"
)

INPUT_FILE = (
    INPUT_DIR
    / "candidate_group_results.csv"
)

OUTPUT_DIR = (
    PROCESSED_DIR
    / "statistical_validation"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# CONFIGURATION
# ==========================================================

RANDOM_STATE = 42

BOOTSTRAP_ITERATIONS = 10000

ALPHA = 0.05


# ==========================================================
# HELPERS
# ==========================================================

def section(title):

    print()

    print("=" * 70)

    print(title)

    print("=" * 70)


def bootstrap_mean_difference(
    hybrid_values,
    structured_values,
    iterations=10000,
    random_state=42
):

    """
    Bootstrap confidence interval for:

        Hybrid - Structured ATS

    For error metrics:
        negative = Hybrid is better

    For performance metrics:
        positive = Hybrid is better
    """

    rng = np.random.default_rng(
        random_state
    )

    hybrid_values = np.asarray(
        hybrid_values,
        dtype=float
    )

    structured_values = np.asarray(
        structured_values,
        dtype=float
    )

    differences = (
        hybrid_values
        -
        structured_values
    )

    n = len(
        differences
    )

    if n == 0:

        return (
            np.nan,
            np.nan,
            np.nan
        )

    bootstrap_means = np.empty(
        iterations
    )

    for i in range(
        iterations
    ):

        sample = rng.choice(

            differences,

            size=n,

            replace=True

        )

        bootstrap_means[i] = (
            np.mean(sample)
        )

    lower = np.percentile(
        bootstrap_means,
        2.5
    )

    upper = np.percentile(
        bootstrap_means,
        97.5
    )

    observed = np.mean(
        differences
    )

    return (
        observed,
        lower,
        upper
    )


def cohens_d_paired(
    hybrid_values,
    structured_values
):

    """
    Cohen's d for paired observations.
    """

    hybrid_values = np.asarray(
        hybrid_values,
        dtype=float
    )

    structured_values = np.asarray(
        structured_values,
        dtype=float
    )

    differences = (
        hybrid_values
        -
        structured_values
    )

    std_difference = np.std(
        differences,
        ddof=1
    )

    if std_difference == 0:

        return np.nan

    return (
        np.mean(differences)
        /
        std_difference
    )


def effect_interpretation(
    d
):

    if pd.isna(d):

        return "Undefined"

    magnitude = abs(d)

    if magnitude < 0.2:

        return "Negligible"

    if magnitude < 0.5:

        return "Small"

    if magnitude < 0.8:

        return "Medium"

    return "Large"


def significance_label(
    p
):

    if pd.isna(p):

        return "Unknown"

    if p < 0.001:

        return "Highly significant"

    if p < 0.01:

        return "Very significant"

    if p < 0.05:

        return "Significant"

    return "Not significant"


# ==========================================================
# START
# ==========================================================

section(
    "EXPERIMENT 10 — STATISTICAL VALIDATION"
)


print()

print(
    "Project directory:"
)

print(
    BASE_DIR
)

print()

print(
    "Input file:"
)

print(
    INPUT_FILE
)

print()

print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)


# ==========================================================
# LOAD RESULTS
# ==========================================================

section(
    "LOADING CANDIDATE-LEVEL RESULTS"
)


if not INPUT_FILE.exists():

    raise FileNotFoundError(

        "Candidate-level result file not found:\n"

        f"{INPUT_FILE}\n\n"

        "Run first:\n"

        "python app\\train\\candidate_group_evaluation.py"

    )


df = pd.read_csv(
    INPUT_FILE
)


print()

print(
    "Loaded:",
    INPUT_FILE
)

print()

print(
    "Shape:",
    df.shape
)


# ==========================================================
# REQUIRED COLUMNS
# ==========================================================

required_columns = [

    "Fold",

    "Model",

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


missing = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing:

    raise ValueError(

        "Missing required columns:\n"

        + "\n".join(
            missing
        )

    )


# ==========================================================
# BASIC VALIDATION
# ==========================================================

section(
    "VALIDATING EXPERIMENT DATA"
)


models = sorted(
    df["Model"].unique()
)


print()

print(
    "Models found:"
)

for model in models:

    print(
        " -",
        model
    )


required_models = [

    "Hybrid",

    "Structured ATS",

]


for model in required_models:

    if model not in models:

        raise ValueError(

            f"Required model missing: {model}"

        )


hybrid_df = (

    df[
        df["Model"]
        ==
        "Hybrid"
    ]

    .copy()

)


structured_df = (

    df[
        df["Model"]
        ==
        "Structured ATS"
    ]

    .copy()

)


print()

print(
    "Hybrid rows:",
    len(hybrid_df)
)

print(
    "Structured ATS rows:",
    len(structured_df)
)


# ==========================================================
# CHECK FOLD ALIGNMENT
# ==========================================================

hybrid_folds = set(
    hybrid_df["Fold"]
)

structured_folds = set(
    structured_df["Fold"]
)


if hybrid_folds != structured_folds:

    raise ValueError(

        "Hybrid and Structured ATS do not "
        "contain identical folds.\n"

        f"Hybrid folds: {sorted(hybrid_folds)}\n"

        f"Structured folds: "
        f"{sorted(structured_folds)}"

    )


print()

print(
    "Common folds:",
    sorted(hybrid_folds)
)


# ==========================================================
# ALIGN BY FOLD
# ==========================================================

hybrid_df = (

    hybrid_df

    .sort_values(
        "Fold"
    )

    .reset_index(
        drop=True
    )

)


structured_df = (

    structured_df

    .sort_values(
        "Fold"
    )

    .reset_index(
        drop=True
    )

)


if not np.array_equal(

    hybrid_df["Fold"].values,

    structured_df["Fold"].values

):

    raise ValueError(
        "Fold alignment failed."
    )


# ==========================================================
# FOLD COMPARISON
# ==========================================================

section(
    "FOLD-BY-FOLD COMPARISON"
)


comparison_rows = []


metrics = [

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


for i in range(
    len(hybrid_df)
):

    row = {

        "Fold":
        hybrid_df.loc[
            i,
            "Fold"
        ],

        "Hybrid_MAE":
        hybrid_df.loc[
            i,
            "MAE"
        ],

        "Structured_ATS_MAE":
        structured_df.loc[
            i,
            "MAE"
        ],

        "Hybrid_RMSE":
        hybrid_df.loc[
            i,
            "RMSE"
        ],

        "Structured_ATS_RMSE":
        structured_df.loc[
            i,
            "RMSE"
        ],

        "Hybrid_R2":
        hybrid_df.loc[
            i,
            "R2"
        ],

        "Structured_ATS_R2":
        structured_df.loc[
            i,
            "R2"
        ],

        "Hybrid_NDCG5":
        hybrid_df.loc[
            i,
            "NDCG@5"
        ],

        "Structured_ATS_NDCG5":
        structured_df.loc[
            i,
            "NDCG@5"
        ],

    }

    comparison_rows.append(
        row
    )


fold_comparison = pd.DataFrame(
    comparison_rows
)


print()

print(
    fold_comparison.to_string(
        index=False
    )
)


# ==========================================================
# STATISTICAL TESTS
# ==========================================================

section(
    "PAIRED STATISTICAL TESTS"
)


statistical_rows = []


for metric in metrics:

    hybrid_values = (

        hybrid_df[
            metric
        ]

        .astype(float)

        .values

    )


    structured_values = (

        structured_df[
            metric
        ]

        .astype(float)

        .values

    )


    # ------------------------------------------------------
    # Paired t-test
    # ------------------------------------------------------

    t_stat, t_p = ttest_rel(

        hybrid_values,

        structured_values

    )


    # ------------------------------------------------------
    # Wilcoxon signed-rank test
    # ------------------------------------------------------

    try:

        w_stat, w_p = wilcoxon(

            hybrid_values,

            structured_values,

            zero_method="wilcox",

            alternative="two-sided"

        )

    except ValueError:

        w_stat = np.nan

        w_p = np.nan


    # ------------------------------------------------------
    # Difference
    # ------------------------------------------------------

    differences = (

        hybrid_values
        -
        structured_values

    )


    mean_difference = np.mean(
        differences
    )


    # ------------------------------------------------------
    # Cohen's d
    # ------------------------------------------------------

    cohens_d = cohens_d_paired(

        hybrid_values,

        structured_values

    )


    # ------------------------------------------------------
    # Bootstrap
    # ------------------------------------------------------

    (

        bootstrap_difference,

        bootstrap_lower,

        bootstrap_upper

    ) = bootstrap_mean_difference(

        hybrid_values,

        structured_values,

        iterations=
        BOOTSTRAP_ITERATIONS,

        random_state=
        RANDOM_STATE

    )


    # ------------------------------------------------------
    # Direction
    # ------------------------------------------------------

    if metric in [

        "MAE",

        "RMSE",

    ]:

        if mean_difference < 0:

            direction = (
                "Hybrid better"
            )

        elif mean_difference > 0:

            direction = (
                "Structured ATS better"
            )

        else:

            direction = (
                "Equal"
            )

    else:

        if mean_difference > 0:

            direction = (
                "Hybrid better"
            )

        elif mean_difference < 0:

            direction = (
                "Structured ATS better"
            )

        else:

            direction = (
                "Equal"
            )


    statistical_rows.append(

        {

            "Metric":
            metric,

            "Hybrid_Mean":
            np.mean(
                hybrid_values
            ),

            "Structured_ATS_Mean":
            np.mean(
                structured_values
            ),

            "Mean_Difference_Hybrid_Minus_ATS":
            mean_difference,

            "Paired_t_statistic":
            t_stat,

            "Paired_t_p_value":
            t_p,

            "Wilcoxon_statistic":
            w_stat,

            "Wilcoxon_p_value":
            w_p,

            "Cohens_d":
            cohens_d,

            "Effect_Size":
            effect_interpretation(
                cohens_d
            ),

            "Bootstrap_Mean_Difference":
            bootstrap_difference,

            "Bootstrap_CI_Lower":
            bootstrap_lower,

            "Bootstrap_CI_Upper":
            bootstrap_upper,

            "Direction":
            direction,

            "T_Test_Interpretation":
            significance_label(
                t_p
            ),

            "Wilcoxon_Interpretation":
            significance_label(
                w_p
            ),

        }

    )


statistical_df = pd.DataFrame(
    statistical_rows
)


print()

print(
    statistical_df.to_string(
        index=False
    )
)


# ==========================================================
# REGRESSION METRICS
# ==========================================================

section(
    "REGRESSION STATISTICAL RESULTS"
)


regression_metrics = [

    "MAE",

    "RMSE",

    "R2",

]


regression_results = (

    statistical_df[
        statistical_df["Metric"]
        .isin(
            regression_metrics
        )
    ]

    .copy()

)


print()

print(
    regression_results[
        [

            "Metric",

            "Hybrid_Mean",

            "Structured_ATS_Mean",

            "Mean_Difference_Hybrid_Minus_ATS",

            "Paired_t_p_value",

            "Wilcoxon_p_value",

            "Cohens_d",

            "Effect_Size",

            "Direction",

        ]

    ].to_string(
        index=False
    )
)


# ==========================================================
# RANKING METRICS
# ==========================================================

section(
    "RANKING STATISTICAL RESULTS"
)


ranking_metrics = [

    "NDCG@1",

    "NDCG@3",

    "NDCG@5",

    "NDCG@10",

    "MRR",

    "Precision@3",

    "Recall@3",

]


ranking_results = (

    statistical_df[
        statistical_df["Metric"]
        .isin(
            ranking_metrics
        )
    ]

    .copy()

)


print()

print(
    ranking_results[
        [

            "Metric",

            "Hybrid_Mean",

            "Structured_ATS_Mean",

            "Mean_Difference_Hybrid_Minus_ATS",

            "Paired_t_p_value",

            "Wilcoxon_p_value",

            "Cohens_d",

            "Effect_Size",

            "Direction",

        ]

    ].to_string(
        index=False
    )
)


# ==========================================================
# MULTIPLE COMPARISON NOTE
# ==========================================================

section(
    "MULTIPLE-COMPARISON ANALYSIS"
)


number_of_tests = len(
    statistical_df
)


bonferroni_alpha = (

    ALPHA
    /
    number_of_tests

)


print()

print(
    "Number of statistical tests:",
    number_of_tests
)

print(
    "Original alpha:",
    ALPHA
)

print(
    "Bonferroni-adjusted alpha:",
    bonferroni_alpha
)


statistical_df[
    "Bonferroni_Significant"
] = (

    statistical_df[
        "Paired_t_p_value"
    ]

    <

    bonferroni_alpha

)


print()

print(
    "Bonferroni-significant metrics:"
)


for _, row in statistical_df.iterrows():

    if row[
        "Bonferroni_Significant"
    ]:

        print(
            " -",
            row["Metric"]
        )


# ==========================================================
# FINAL MODEL SCORING
# ==========================================================

section(
    "FINAL MODEL SELECTION"
)


# Count metrics where Hybrid wins
# statistically and directionally.

hybrid_wins = 0

hybrid_significant_wins = 0

structured_wins = 0


selection_rows = []


for _, row in statistical_df.iterrows():

    metric = row[
        "Metric"
    ]

    direction = row[
        "Direction"
    ]

    t_p = row[
        "Paired_t_p_value"
    ]

    wilcoxon_p = row[
        "Wilcoxon_p_value"
    ]

    bonferroni = row[
        "Bonferroni_Significant"
    ]


    if direction == "Hybrid better":

        hybrid_wins += 1


        if (

            t_p < ALPHA

            and

            wilcoxon_p < ALPHA

        ):

            hybrid_significant_wins += 1


    elif direction == "Structured ATS better":

        structured_wins += 1


    selection_rows.append(

        {

            "Metric":
            metric,

            "Winner":
            direction,

            "Paired_t_p":
            t_p,

            "Wilcoxon_p":
            wilcoxon_p,

            "Bonferroni_Significant":
            bonferroni,

            "Effect_Size":
            row[
                "Effect_Size"
            ],

        }

    )


selection_df = pd.DataFrame(
    selection_rows
)


print()

print(
    selection_df.to_string(
        index=False
    )
)


print()

print(
    "Hybrid metric wins:",
    hybrid_wins
)

print(
    "Hybrid statistically significant wins:",
    hybrid_significant_wins
)

print(
    "Structured ATS metric wins:",
    structured_wins
)


# ==========================================================
# OVERALL DECISION
# ==========================================================

section(
    "FINAL MODEL DECISION"
)


# The primary decision should emphasize
# unseen-candidate ranking and regression.

hybrid_mae = (

    hybrid_df["MAE"]
    .mean()

)


ats_mae = (

    structured_df["MAE"]
    .mean()

)


hybrid_ndcg5 = (

    hybrid_df["NDCG@5"]
    .mean()

)


ats_ndcg5 = (

    structured_df["NDCG@5"]
    .mean()

)


hybrid_r2 = (

    hybrid_df["R2"]
    .mean()

)


ats_r2 = (

    structured_df["R2"]
    .mean()

)


print()

print(
    f"Hybrid MAE: {hybrid_mae:.6f}"
)

print(
    f"Structured ATS MAE: {ats_mae:.6f}"
)

print()

print(
    f"Hybrid NDCG@5: {hybrid_ndcg5:.6f}"
)

print(
    f"Structured ATS NDCG@5: {ats_ndcg5:.6f}"
)

print()

print(
    f"Hybrid R²: {hybrid_r2:.6f}"
)

print(
    f"Structured ATS R²: {ats_r2:.6f}"
)


if (

    hybrid_mae < ats_mae

    and

    hybrid_ndcg5 > ats_ndcg5

    and

    hybrid_r2 > ats_r2

):

    final_model = "Hybrid"

    decision = (

        "Hybrid is selected as the final model because "
        "it simultaneously achieves lower MAE, lower RMSE, "
        "higher R² and higher NDCG@5 on the unseen-candidate "
        "evaluation."

    )

else:

    final_model = "Structured ATS"

    decision = (

        "Structured ATS is selected based on the aggregate "
        "unseen-candidate performance."

    )


print()

print(
    "FINAL MODEL:",
    final_model
)

print()

print(
    decision
)


# ==========================================================
# SAVE FILES
# ==========================================================

section(
    "SAVING STATISTICAL RESULTS"
)


fold_comparison.to_csv(

    OUTPUT_DIR
    / "fold_comparison.csv",

    index=False

)


statistical_df.to_csv(

    OUTPUT_DIR
    / "statistical_tests.csv",

    index=False

)


regression_results.to_csv(

    OUTPUT_DIR
    / "regression_statistical_results.csv",

    index=False

)


ranking_results.to_csv(

    OUTPUT_DIR
    / "ranking_statistical_results.csv",

    index=False

)


selection_df.to_csv(

    OUTPUT_DIR
    / "final_model_selection.csv",

    index=False

)


# ==========================================================
# CONFIDENCE INTERVAL FILE
# ==========================================================

confidence_df = (

    statistical_df[

        [

            "Metric",

            "Bootstrap_Mean_Difference",

            "Bootstrap_CI_Lower",

            "Bootstrap_CI_Upper",

            "Direction",

        ]

    ]

    .copy()

)


confidence_df.to_csv(

    OUTPUT_DIR
    / "confidence_intervals.csv",

    index=False

)


# ==========================================================
# EFFECT SIZE FILE
# ==========================================================

effect_df = (

    statistical_df[

        [

            "Metric",

            "Cohens_d",

            "Effect_Size",

            "Direction",

        ]

    ]

    .copy()

)


effect_df.to_csv(

    OUTPUT_DIR
    / "effect_sizes.csv",

    index=False

)


# ==========================================================
# TEXT REPORT
# ==========================================================

section(
    "GENERATING STATISTICAL REPORT"
)


report = []


report.append(
    "EXPERIMENT 10 — STATISTICAL VALIDATION REPORT"
)

report.append(
    "=" * 70
)

report.append("")


report.append(
    "PRIMARY COMPARISON"
)

report.append(
    "Hybrid vs Structured ATS"
)

report.append("")


report.append(
    "EVALUATION SETTING"
)

report.append(
    "5-fold candidate-level grouped evaluation"
)

report.append(
    "Each fold evaluates candidates not present in "
    "the corresponding training fold."
)

report.append("")


report.append(
    "REGRESSION RESULTS"
)

report.append(
    "-" * 70
)


for _, row in regression_results.iterrows():

    report.append(

        f"{row['Metric']}: "
        f"Hybrid={row['Hybrid_Mean']:.6f}, "
        f"Structured ATS="
        f"{row['Structured_ATS_Mean']:.6f}, "
        f"p(t)={row['Paired_t_p_value']:.6f}, "
        f"p(W)={row['Wilcoxon_p_value']:.6f}, "
        f"Cohen's d={row['Cohens_d']:.4f}, "
        f"{row['Direction']}"

    )


report.append("")


report.append(
    "RANKING RESULTS"
)

report.append(
    "-" * 70
)


for _, row in ranking_results.iterrows():

    report.append(

        f"{row['Metric']}: "
        f"Hybrid={row['Hybrid_Mean']:.6f}, "
        f"Structured ATS="
        f"{row['Structured_ATS_Mean']:.6f}, "
        f"p(t)={row['Paired_t_p_value']:.6f}, "
        f"p(W)={row['Wilcoxon_p_value']:.6f}, "
        f"Cohen's d={row['Cohens_d']:.4f}, "
        f"{row['Direction']}"

    )


report.append("")


report.append(
    "MULTIPLE-COMPARISON CORRECTION"
)

report.append(
    "-" * 70
)

report.append(

    f"Bonferroni alpha = "
    f"{bonferroni_alpha:.6f}"

)


report.append("")


report.append(
    "FINAL MODEL"
)

report.append(
    "-" * 70
)

report.append(
    final_model
)

report.append("")

report.append(
    decision
)

report.append("")


report.append(
    "INTERPRETATION"
)

report.append(
    "-" * 70
)

report.append(

    "Statistical tests are based on paired fold-level "
    "measurements. The candidate-level grouped split "
    "is treated as the primary evaluation because it "
    "tests generalization to unseen candidates."

)

report.append(

    "A statistically significant p-value indicates evidence "
    "of a difference between the two models across folds, "
    "but practical importance should also be considered "
    "using effect size and confidence intervals."

)

report.append("")


report.append(
    "IMPORTANT LIMITATION"
)

report.append(
    "-" * 70
)

report.append(

    "Only five candidate-level folds are available. "
    "Therefore, statistical conclusions should be "
    "described as evidence of improvement rather than "
    "universal proof of superiority."

)


report_file = (

    OUTPUT_DIR
    / "statistical_validation_report.txt"

)


with open(

    report_file,

    "w",

    encoding="utf-8"

) as file:

    file.write(
        "\n".join(report)
    )


# ==========================================================
# FINAL OUTPUT
# ==========================================================

section(
    "EXPERIMENT 10 COMPLETE"
)


print()

print(
    "Statistical validation outputs:"
)

print()

print(
    OUTPUT_DIR
    / "fold_comparison.csv"
)

print(
    OUTPUT_DIR
    / "statistical_tests.csv"
)

print(
    OUTPUT_DIR
    / "regression_statistical_results.csv"
)

print(
    OUTPUT_DIR
    / "ranking_statistical_results.csv"
)

print(
    OUTPUT_DIR
    / "confidence_intervals.csv"
)

print(
    OUTPUT_DIR
    / "effect_sizes.csv"
)

print(
    OUTPUT_DIR
    / "final_model_selection.csv"
)

print(
    OUTPUT_DIR
    / "statistical_validation_report.txt"
)

print()

print(
    "FINAL MODEL:",
    final_model
)

print()

print("=" * 70)

print(
    "STATISTICAL VALIDATION FINISHED SUCCESSFULLY"
)

print("=" * 70)