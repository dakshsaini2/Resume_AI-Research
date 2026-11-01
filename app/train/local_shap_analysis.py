"""
local_shap_analysis.py

Experiment 9: Local SHAP Explanations
======================================

Purpose
-------
Provide per-sample SHAP explanations that reveal WHY
the model assigned a particular score to a specific
resume-job pair.

This extends Experiment 4 (global SHAP) by adding:
1. Local waterfall plots for individual predictions
2. Force plots for representative cases
3. Positive/negative feature contribution tables
4. Human-readable "Why this score?" output
5. Analysis of best, worst, and median cases

The roadmap requires local explainability for:
- Representative high-match cases
- Representative low-match cases
- Difficult/surprising cases (high error)
- Median cases

Outputs
-------
dataset/processed/local_shap/
    local_shap_explanations.csv
    local_shap_high_match.png
    local_shap_low_match.png
    local_shap_median_match.png
    local_shap_worst_error_over.png
    local_shap_worst_error_under.png
    explanation_examples.csv
    explanation_examples.txt

Important
---------
This script DOES NOT retrain the model.

It loads:
    - trained Random Forest
    - trained scaler
    - existing feature configuration

and generates local explanations for individual samples.
"""

# ==========================================================
# IMPORTS
# ==========================================================

from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import shap


warnings.filterwarnings("ignore")

# Use non-interactive backend for server/script execution.
matplotlib.use("Agg")


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
    / "local_shap"
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
# IMPORT PROJECT CONFIGURATION
# ==========================================================

from config import (
    MODEL_FILE,
    SCALER_FILE,
    FEATURE_COLUMNS,
)


# ==========================================================
# IMPORT FEATURE ENGINEERING
# ==========================================================

from feature_engineering import generate_features


# ==========================================================
# CONFIGURATION
# ==========================================================

RANDOM_STATE = 42

TARGET_COLUMN = "matched_score"

# Number of representative cases to explain.

NUM_CASES_PER_CATEGORY = 3

# Sample size for SHAP analysis.

MAX_SHAP_SAMPLES = 500


# ==========================================================
# HUMAN-READABLE FEATURE NAMES
# ==========================================================

FEATURE_DISPLAY_NAMES = {

    "tfidf_similarity":
        "Keyword Overlap (TF-IDF)",

    "semantic_similarity":
        "Meaning Similarity (SBERT)",

    "skill_match_score":
        "Skill Match Ratio",

    "education_match_score":
        "Education Match",

    "experience_match_score":
        "Experience Match",

    "certification_match_score":
        "Certification Match",

    "resume_length":
        "Resume Text Length",

    "job_length":
        "Job Description Length",

    "resume_word_count":
        "Resume Word Count",

    "job_word_count":
        "Job Word Count",

    "candidate_skill_count":
        "Candidate Skills Count",

    "required_skill_count":
        "Required Skills Count",

    "skill_overlap_count":
        "Overlapping Skills Count",

    "education_exact_match":
        "Exact Degree Match",
}


# ==========================================================
# PUBLICATION PLOT SETTINGS
# ==========================================================

plt.rcParams.update({

    "figure.dpi": 150,

    "savefig.dpi": 300,

    "font.size": 11,

    "axes.titlesize": 13,

    "axes.labelsize": 12,

    "xtick.labelsize": 10,

    "ytick.labelsize": 10,

    "legend.fontsize": 10,

    "figure.figsize": (10, 6),
})


# ==========================================================
# DATASET
# ==========================================================

DATASET_FILE = (
    BASE_DIR
    / "dataset"
    / "Resume_Data_For_Ranking.csv"
)


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

print()

print("=" * 70)

print("EXPERIMENT 9: LOCAL SHAP EXPLANATIONS")

print("=" * 70)


print()

print("=" * 70)

print("LOADING TRAINED MODEL")

print("=" * 70)


print()

print(
    "Model file:",
    MODEL_FILE
)


if not Path(MODEL_FILE).exists():

    raise FileNotFoundError(

        "\nTrained model not found:\n"

        f"{MODEL_FILE}\n\n"

        "Run the research training script first."
    )


model = joblib.load(
    MODEL_FILE
)


print()

print(
    "Model loaded successfully."
)

print(
    "Model type:",
    type(model).__name__
)


# ==========================================================
# LOAD SCALER
# ==========================================================

print()

print("=" * 70)

print("LOADING FEATURE SCALER")

print("=" * 70)


print()

print(
    "Scaler file:",
    SCALER_FILE
)


if not Path(SCALER_FILE).exists():

    raise FileNotFoundError(

        "\nScaler not found:\n"

        f"{SCALER_FILE}\n\n"

        "Run the research training script first."
    )


scaler = joblib.load(
    SCALER_FILE
)


print()

print(
    "Scaler loaded successfully."
)


# ==========================================================
# LOAD DATASET
# ==========================================================

print()

print("=" * 70)

print("LOADING DATASET")

print("=" * 70)


if not DATASET_FILE.exists():

    raise FileNotFoundError(

        "\nDataset not found:\n"

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


# ==========================================================
# BUILD TEXT
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
# SAMPLE FOR SHAP
# ==========================================================

print()

print("=" * 70)

print("SELECTING SAMPLE FOR LOCAL SHAP")

print("=" * 70)


if len(df) > MAX_SHAP_SAMPLES:

    shap_df = df.sample(

        n=MAX_SHAP_SAMPLES,

        random_state=RANDOM_STATE

    ).reset_index(
        drop=True
    )

else:

    shap_df = df.copy().reset_index(
        drop=True
    )


print()

print(
    "Total dataset samples:",
    len(df)
)

print(
    "Samples selected for SHAP:",
    len(shap_df)
)


# ==========================================================
# GENERATE FEATURES
# ==========================================================

print()

print("=" * 70)

print("GENERATING FEATURES FOR SHAP SAMPLE")

print("=" * 70)


feature_rows = []

raw_feature_rows = []


for index, row in shap_df.iterrows():

    if index % 50 == 0:

        print(
            f"Processing sample "
            f"{index + 1}/"
            f"{len(shap_df)}..."
        )

    features = generate_features(
        row["candidate_text"],
        row["job_text"]
    )

    model_features = {

        feature:
        features[feature]

        for feature in FEATURE_COLUMNS
    }

    feature_rows.append(
        model_features
    )

    raw_feature_rows.append(
        model_features.copy()
    )


# ==========================================================
# CREATE FEATURE MATRIX
# ==========================================================

X = pd.DataFrame(

    feature_rows,

    columns=FEATURE_COLUMNS
)


X = (

    X

    .replace(
        [np.inf, -np.inf],
        np.nan
    )

    .fillna(0)
)


X_raw = pd.DataFrame(

    raw_feature_rows,

    columns=FEATURE_COLUMNS
)


# ==========================================================
# SCALE FEATURES
# ==========================================================

print()

print("=" * 70)

print("SCALING FEATURES")

print("=" * 70)


X_scaled = scaler.transform(X)


X_scaled = pd.DataFrame(
    X_scaled,
    columns=FEATURE_COLUMNS,
    index=X.index
)


# ==========================================================
# GENERATE PREDICTIONS
# ==========================================================

print()

print("=" * 70)

print("GENERATING PREDICTIONS")

print("=" * 70)


predictions = model.predict(
    X_scaled
)

predictions = np.clip(
    predictions,
    0,
    1
)


actual = shap_df[
    TARGET_COLUMN
].values


residuals = actual - predictions

absolute_errors = np.abs(residuals)


print()

print(
    "Predictions generated:",
    len(predictions)
)


# ==========================================================
# SHAP TREE EXPLAINER
# ==========================================================

print()

print("=" * 70)

print("BUILDING SHAP TREE EXPLAINER")

print("=" * 70)


explainer = shap.TreeExplainer(
    model
)


# ==========================================================
# COMPUTE SHAP VALUES
# ==========================================================

print()

print("=" * 70)

print("COMPUTING LOCAL SHAP VALUES")

print("=" * 70)


shap_values = explainer.shap_values(
    X_scaled
)


if isinstance(
    shap_values,
    list
):
    shap_values = shap_values[0]


shap_values = np.asarray(
    shap_values
)


if shap_values.ndim == 3:
    shap_values = shap_values[:, :, 0]


expected_value = explainer.expected_value

if isinstance(expected_value, (list, np.ndarray)):
    expected_value = expected_value[0]


print()

print(
    "SHAP matrix shape:",
    shap_values.shape
)

print(
    "Expected value (base score):",
    f"{expected_value:.6f}"
)


# ==========================================================
# SELECT REPRESENTATIVE CASES
# ==========================================================

print()

print("=" * 70)

print("SELECTING REPRESENTATIVE CASES")

print("=" * 70)


# Categories of interest.

case_selections = {}


# 1. High-match cases (actual >= 0.8).

high_mask = actual >= 0.8

if np.sum(high_mask) > 0:

    high_indices = np.where(high_mask)[0]

    selected = np.random.RandomState(
        RANDOM_STATE
    ).choice(

        high_indices,

        min(
            NUM_CASES_PER_CATEGORY,
            len(high_indices)
        ),

        replace=False
    )

    case_selections["high_match"] = selected


# 2. Low-match cases (actual <= 0.3).

low_mask = actual <= 0.3

if np.sum(low_mask) > 0:

    low_indices = np.where(low_mask)[0]

    selected = np.random.RandomState(
        RANDOM_STATE + 1
    ).choice(

        low_indices,

        min(
            NUM_CASES_PER_CATEGORY,
            len(low_indices)
        ),

        replace=False
    )

    case_selections["low_match"] = selected


# 3. Median cases (closest to median actual).

median_score = np.median(actual)

distances_to_median = np.abs(
    actual - median_score
)

median_indices = np.argsort(
    distances_to_median
)[:NUM_CASES_PER_CATEGORY]

case_selections["median_match"] = (
    median_indices
)


# 4. Worst over-prediction (largest positive error).

over_pred_errors = predictions - actual

worst_over_indices = np.argsort(
    over_pred_errors
)[-NUM_CASES_PER_CATEGORY:][::-1]

case_selections["worst_over_prediction"] = (
    worst_over_indices
)


# 5. Worst under-prediction (largest negative error).

worst_under_indices = np.argsort(
    over_pred_errors
)[:NUM_CASES_PER_CATEGORY]

case_selections["worst_under_prediction"] = (
    worst_under_indices
)


for category, indices in case_selections.items():

    print(
        f"\n{category}: {len(indices)} cases"
    )

    for idx in indices:

        print(
            f"  Sample {idx}: "
            f"actual={actual[idx]:.4f}, "
            f"predicted={predictions[idx]:.4f}, "
            f"error={residuals[idx]:.4f}"
        )


# ==========================================================
# GENERATE LOCAL EXPLANATION FOR A SINGLE SAMPLE
# ==========================================================

def generate_explanation(
    sample_index,
    shap_values_array,
    feature_values,
    raw_feature_values,
    predicted_score,
    actual_score,
    base_value,
):
    """
    Generate a human-readable explanation for why
    the model assigned a particular score.
    """

    contributions = []

    for i, feature in enumerate(
        FEATURE_COLUMNS
    ):

        display_name = (
            FEATURE_DISPLAY_NAMES.get(
                feature,
                feature
            )
        )

        shap_val = shap_values_array[i]

        raw_val = raw_feature_values[feature]

        direction = (
            "increased"
            if shap_val > 0
            else "decreased"
        )

        contributions.append({

            "Feature":
                feature,

            "Display_Name":
                display_name,

            "Feature_Value":
                round(
                    float(raw_val),
                    4
                ),

            "SHAP_Value":
                round(
                    float(shap_val),
                    6
                ),

            "Absolute_SHAP":
                round(
                    abs(float(shap_val)),
                    6
                ),

            "Direction":
                direction,
        })

    contrib_df = pd.DataFrame(
        contributions
    )

    contrib_df = contrib_df.sort_values(

        by="Absolute_SHAP",

        ascending=False

    ).reset_index(
        drop=True
    )

    # Build human-readable explanation.

    lines = []

    lines.append(
        f"Predicted Score: "
        f"{predicted_score:.4f} "
        f"({predicted_score * 100:.1f}%)"
    )

    lines.append(
        f"Actual Score:    "
        f"{actual_score:.4f} "
        f"({actual_score * 100:.1f}%)"
    )

    lines.append(
        f"Base Score:      "
        f"{base_value:.4f}"
    )

    lines.append("")

    lines.append("Top factors that INCREASED the score:")

    positive = contrib_df[
        contrib_df["SHAP_Value"] > 0
    ].head(5)

    for _, row in positive.iterrows():

        lines.append(

            f"  + {row['Display_Name']}: "
            f"value={row['Feature_Value']}, "
            f"impact=+{row['SHAP_Value']:.4f}"
        )

    if len(positive) == 0:
        lines.append("  (none)")

    lines.append("")

    lines.append("Top factors that DECREASED the score:")

    negative = contrib_df[
        contrib_df["SHAP_Value"] < 0
    ].head(5)

    for _, row in negative.iterrows():

        lines.append(

            f"  - {row['Display_Name']}: "
            f"value={row['Feature_Value']}, "
            f"impact={row['SHAP_Value']:.4f}"
        )

    if len(negative) == 0:
        lines.append("  (none)")

    return contrib_df, "\n".join(lines)


# ==========================================================
# GENERATE WATERFALL PLOTS FOR REPRESENTATIVE CASES
# ==========================================================

print()

print("=" * 70)

print("GENERATING LOCAL SHAP WATERFALL PLOTS")

print("=" * 70)


plot_categories = {

    "high_match":
        "High Match Case",

    "low_match":
        "Low Match Case",

    "median_match":
        "Median Match Case",

    "worst_over_prediction":
        "Worst Over-Prediction",

    "worst_under_prediction":
        "Worst Under-Prediction",
}


all_explanations = []

explanation_texts = []


for category, display_title in (
    plot_categories.items()
):

    if category not in case_selections:
        continue

    indices = case_selections[category]

    # Use the first case for the waterfall plot.

    primary_index = indices[0]


    print()

    print(
        f"Generating waterfall plot for "
        f"{display_title}..."
    )

    print(
        f"  Sample index: {primary_index}"
    )

    print(
        f"  Actual:    "
        f"{actual[primary_index]:.4f}"
    )

    print(
        f"  Predicted: "
        f"{predictions[primary_index]:.4f}"
    )


    # Generate explanation.

    contrib_df, explanation_text = (

        generate_explanation(

            primary_index,

            shap_values[primary_index],

            X_scaled.iloc[primary_index],

            X_raw.iloc[primary_index],

            predictions[primary_index],

            actual[primary_index],

            expected_value,
        )
    )


    contrib_df["Category"] = category

    contrib_df["Sample_Index"] = primary_index

    contrib_df["Actual_Score"] = (
        actual[primary_index]
    )

    contrib_df["Predicted_Score"] = (
        predictions[primary_index]
    )


    all_explanations.append(
        contrib_df
    )


    explanation_texts.append(
        f"\n{'=' * 60}\n"
        f"{display_title.upper()} "
        f"(Sample {primary_index})\n"
        f"{'=' * 60}\n"
        f"{explanation_text}"
    )


    # Generate waterfall plot.

    shap_explanation = shap.Explanation(

        values=shap_values[primary_index],

        base_values=expected_value,

        data=X_scaled.iloc[
            primary_index
        ].values,

        feature_names=FEATURE_COLUMNS,
    )


    fig, ax = plt.subplots(
        figsize=(10, 8)
    )


    shap.plots.waterfall(

        shap_explanation,

        show=False,

        max_display=14,
    )


    plt.title(

        f"Local SHAP Explanation — {display_title}\n"
        f"Actual: {actual[primary_index]:.3f} | "
        f"Predicted: {predictions[primary_index]:.3f}",

        fontsize=12,

        fontweight="bold",

        pad=15
    )


    plt.tight_layout()


    plot_filename = (
        f"local_shap_"
        f"{category}.png"
    )


    plot_file = (
        OUTPUT_DIR
        / plot_filename
    )


    plt.savefig(

        plot_file,

        dpi=300,

        bbox_inches="tight"
    )


    plt.close("all")


    print(
        f"  Saved: {plot_file}"
    )


    # Also generate explanations for
    # additional cases in this category.

    for extra_index in indices[1:]:

        extra_contrib, extra_text = (

            generate_explanation(

                extra_index,

                shap_values[extra_index],

                X_scaled.iloc[extra_index],

                X_raw.iloc[extra_index],

                predictions[extra_index],

                actual[extra_index],

                expected_value,
            )
        )

        extra_contrib["Category"] = category

        extra_contrib["Sample_Index"] = (
            extra_index
        )

        extra_contrib["Actual_Score"] = (
            actual[extra_index]
        )

        extra_contrib["Predicted_Score"] = (
            predictions[extra_index]
        )

        all_explanations.append(
            extra_contrib
        )

        explanation_texts.append(
            f"\n{'-' * 60}\n"
            f"{display_title} "
            f"(Sample {extra_index})\n"
            f"{'-' * 60}\n"
            f"{extra_text}"
        )


# ==========================================================
# SAVE ALL LOCAL EXPLANATIONS
# ==========================================================

print()

print("=" * 70)

print("SAVING LOCAL SHAP EXPLANATIONS")

print("=" * 70)


all_explanations_df = pd.concat(

    all_explanations,

    ignore_index=True
)


explanations_file = (
    OUTPUT_DIR
    / "local_shap_explanations.csv"
)


all_explanations_df.to_csv(
    explanations_file,
    index=False
)


print()

print(
    "Saved:",
    explanations_file
)

print(
    "Total explanation records:",
    len(all_explanations_df)
)


# ==========================================================
# SAVE EXPLANATION EXAMPLES
# ==========================================================

print()

print("=" * 70)

print("SAVING EXPLANATION EXAMPLES")

print("=" * 70)


# Create a summary table of example cases.

example_rows = []


for category, indices in (
    case_selections.items()
):

    display_title = (
        plot_categories.get(
            category,
            category
        )
    )

    for idx in indices:

        # Top 3 positive and negative contributions.

        sample_shap = shap_values[idx]

        sorted_indices = np.argsort(
            np.abs(sample_shap)
        )[::-1]

        top_positive = []

        top_negative = []


        for fi in sorted_indices:

            feature = FEATURE_COLUMNS[fi]

            display = (
                FEATURE_DISPLAY_NAMES.get(
                    feature,
                    feature
                )
            )

            val = sample_shap[fi]

            if val > 0 and len(top_positive) < 3:

                top_positive.append(
                    f"{display} (+{val:.4f})"
                )

            elif val < 0 and len(top_negative) < 3:

                top_negative.append(
                    f"{display} ({val:.4f})"
                )


        example_rows.append({

            "Category":
                display_title,

            "Sample_Index":
                idx,

            "Actual_Score":
                round(
                    float(actual[idx]),
                    4
                ),

            "Predicted_Score":
                round(
                    float(predictions[idx]),
                    4
                ),

            "Error":
                round(
                    float(residuals[idx]),
                    4
                ),

            "Top_Positive_1":
                (
                    top_positive[0]
                    if len(top_positive) > 0
                    else ""
                ),

            "Top_Positive_2":
                (
                    top_positive[1]
                    if len(top_positive) > 1
                    else ""
                ),

            "Top_Positive_3":
                (
                    top_positive[2]
                    if len(top_positive) > 2
                    else ""
                ),

            "Top_Negative_1":
                (
                    top_negative[0]
                    if len(top_negative) > 0
                    else ""
                ),

            "Top_Negative_2":
                (
                    top_negative[1]
                    if len(top_negative) > 1
                    else ""
                ),

            "Top_Negative_3":
                (
                    top_negative[2]
                    if len(top_negative) > 2
                    else ""
                ),
        })


examples_df = pd.DataFrame(
    example_rows
)


examples_csv_file = (
    OUTPUT_DIR
    / "explanation_examples.csv"
)


examples_df.to_csv(
    examples_csv_file,
    index=False
)


print()

print(
    "Saved:",
    examples_csv_file
)


# Save text explanations.

full_explanation_text = "\n".join(
    explanation_texts
)


examples_txt_file = (
    OUTPUT_DIR
    / "explanation_examples.txt"
)


with open(examples_txt_file, "w") as f:

    f.write(full_explanation_text)


print()

print(
    "Saved:",
    examples_txt_file
)


# ==========================================================
# DISPLAY EXPLANATIONS
# ==========================================================

print()

print("=" * 70)

print("LOCAL SHAP EXPLANATION EXAMPLES")

print("=" * 70)


print(full_explanation_text)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print()

print("=" * 70)

print("EXPERIMENT 9 COMPLETE")

print("=" * 70)


print()

print(
    "Local SHAP outputs generated in:"
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


print()

print("=" * 70)

print("LOCAL SHAP ANALYSIS FINISHED SUCCESSFULLY")

print("=" * 70)
