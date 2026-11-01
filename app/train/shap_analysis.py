"""
shap_analysis.py

Experiment 4: SHAP Explainability
=================================

Purpose
-------
Explain the predictions of the trained Hybrid Random Forest
used in the Resume Screening research framework.

The analysis provides:

1. Global SHAP feature importance
2. SHAP beeswarm summary plot
3. SHAP bar importance plot
4. Individual SHAP contribution values
5. Feature importance CSV

Outputs
-------
dataset/processed/shap/
    shap_summary.png
    shap_bar.png
    shap_values.csv
    shap_feature_importance.csv

Important
---------
This script DOES NOT retrain the model.

It loads:
    - trained Random Forest
    - trained scaler
    - existing feature configuration

and explains the already-trained research model.
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
import matplotlib.pyplot as plt
import shap

from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer


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


SHAP_DIR = (

    PROCESSED_DIR
    / "shap"

)


SHAP_DIR.mkdir(

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

# Number of samples used for SHAP analysis.
#
# We do not need to explain all 9,544 samples.
# A representative sample is sufficient for global
# explainability and dramatically reduces computation.

MAX_SHAP_SAMPLES = 500


# Fixed seed for reproducibility.

RANDOM_STATE = 42


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
# FEATURE INFORMATION
# ==========================================================

print()

print("=" * 70)

print("RESEARCH FEATURES")

print("=" * 70)


print()

print(

    "Number of model features:",

    len(FEATURE_COLUMNS)

)


for index, feature in enumerate(

    FEATURE_COLUMNS,

    start=1

):

    print(

        f"{index:2d}. {feature}"

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

]


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    raise ValueError(

        "\nMissing required dataset columns:\n"

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
# BUILD CANDIDATE TEXT
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
# SAMPLE DATA
# ==========================================================

print()

print("=" * 70)

print("SELECTING SHAP SAMPLE")

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

print("GENERATING RESEARCH FEATURES")

print("=" * 70)


feature_rows = []


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


# ==========================================================
# CREATE FEATURE MATRIX
# ==========================================================

X = pd.DataFrame(

    feature_rows,

    columns=FEATURE_COLUMNS

)


# ==========================================================
# NUMERICAL CLEANING
# ==========================================================

X = (

    X

    .replace(

        [np.inf, -np.inf],

        np.nan

    )

    .fillna(0)

)


# ==========================================================
# VALIDATE FEATURES
# ==========================================================

missing_features = [

    feature

    for feature in FEATURE_COLUMNS

    if feature not in X.columns

]


if missing_features:

    raise ValueError(

        "\nMissing model features:\n"

        +

        "\n".join(

            missing_features

        )

    )


print()

print(

    "Feature matrix shape:",

    X.shape

)


# ==========================================================
# APPLY SAME SCALER USED DURING TRAINING
# ==========================================================

print()

print("=" * 70)

print("SCALING FEATURES")

print("=" * 70)


X_scaled = scaler.transform(

    X

)


X_scaled = pd.DataFrame(

    X_scaled,

    columns=FEATURE_COLUMNS,

    index=X.index

)


print()

print(

    "Scaled feature matrix:",

    X_scaled.shape

)


# ==========================================================
# VERIFY MODEL INPUT
# ==========================================================

if hasattr(

    model,

    "n_features_in_"

):

    expected_features = (

        model.n_features_in_

    )


    actual_features = (

        X_scaled.shape[1]

    )


    print()

    print(

        "Model expects:",

        expected_features,

        "features"

    )


    print(

        "Provided:",

        actual_features,

        "features"

    )


    if expected_features != actual_features:

        raise ValueError(

            "\nFeature count mismatch!\n"

            f"Model expects {expected_features}, "

            f"but received {actual_features}."

        )


# ==========================================================
# GENERATE MODEL PREDICTIONS
# ==========================================================

print()

print("=" * 70)

print("GENERATING MODEL PREDICTIONS")

print("=" * 70)


predictions = model.predict(

    X_scaled

)


predictions = np.clip(

    predictions,

    0,

    1

)


print()

print(

    "Prediction minimum:",

    predictions.min()

)


print(

    "Prediction maximum:",

    predictions.max()

)


print(

    "Prediction mean:",

    predictions.mean()

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

print("COMPUTING SHAP VALUES")

print("=" * 70)


shap_values = explainer.shap_values(

    X_scaled

)


# ==========================================================
# HANDLE SHAP OUTPUT FORMAT
# ==========================================================

# For RandomForestRegressor, SHAP normally returns:

# samples × features

# But this check protects the experiment against
# different SHAP versions.

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


print()

print(

    "SHAP matrix shape:",

    shap_values.shape

)


expected_shape = (

    len(X_scaled),

    len(FEATURE_COLUMNS)

)


if shap_values.shape != expected_shape:

    raise ValueError(

        "\nUnexpected SHAP matrix shape.\n"

        f"Expected: {expected_shape}\n"

        f"Received: {shap_values.shape}"

    )


# ==========================================================
# SAVE SHAP VALUES
# ==========================================================

print()

print("=" * 70)

print("SAVING SHAP VALUES")

print("=" * 70)


shap_values_df = pd.DataFrame(

    shap_values,

    columns=FEATURE_COLUMNS

)


shap_values_df.insert(

    0,

    "sample_id",

    np.arange(

        len(shap_values_df)

    )

)


shap_values_df.insert(

    1,

    "predicted_score",

    predictions

)


shap_values_file = (

    SHAP_DIR

    / "shap_values.csv"

)


shap_values_df.to_csv(

    shap_values_file,

    index=False

)


print()

print(

    "Saved:",

    shap_values_file

)


# ==========================================================
# GLOBAL SHAP FEATURE IMPORTANCE
# ==========================================================

print()

print("=" * 70)

print("CALCULATING GLOBAL SHAP IMPORTANCE")

print("=" * 70)


mean_abs_shap = np.mean(

    np.abs(

        shap_values

    ),

    axis=0

)


feature_importance_df = pd.DataFrame({

    "Feature":

        FEATURE_COLUMNS,

    "Mean_Absolute_SHAP":

        mean_abs_shap,

})


feature_importance_df = (

    feature_importance_df

    .sort_values(

        by="Mean_Absolute_SHAP",

        ascending=False

    )

    .reset_index(

        drop=True

    )

)


# Add normalized importance.

total_importance = (

    feature_importance_df[

        "Mean_Absolute_SHAP"

    ].sum()

)


if total_importance > 0:

    feature_importance_df[

        "Relative_Importance"

    ] = (

        feature_importance_df[

            "Mean_Absolute_SHAP"

        ]

        /

        total_importance

    )


else:

    feature_importance_df[

        "Relative_Importance"

    ] = 0


# ==========================================================
# SAVE FEATURE IMPORTANCE
# ==========================================================

feature_importance_file = (

    SHAP_DIR

    / "shap_feature_importance.csv"

)


feature_importance_df.to_csv(

    feature_importance_file,

    index=False

)


print()

print(

    "Saved:",

    feature_importance_file

)


# ==========================================================
# DISPLAY FEATURE IMPORTANCE
# ==========================================================

print()

print("=" * 70)

print("GLOBAL SHAP FEATURE IMPORTANCE")

print("=" * 70)


print(

    feature_importance_df.to_string(

        index=False

    )

)


# ==========================================================
# SHAP SUMMARY / BEESWARM PLOT
# ==========================================================

print()

print("=" * 70)

print("GENERATING SHAP SUMMARY PLOT")

print("=" * 70)


plt.figure(

    figsize=(10, 7)

)


shap.summary_plot(

    shap_values,

    X_scaled,

    feature_names=FEATURE_COLUMNS,

    show=False,

)


plt.title(

    "SHAP Summary Plot - Hybrid Resume Screening Model"

)


plt.tight_layout()


summary_plot_file = (

    SHAP_DIR

    / "shap_summary.png"

)


plt.savefig(

    summary_plot_file,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print()

print(

    "Saved:",

    summary_plot_file

)


# ==========================================================
# SHAP BAR PLOT
# ==========================================================

print()

print("=" * 70)

print("GENERATING SHAP BAR PLOT")

print("=" * 70)


plt.figure(

    figsize=(10, 7)

)


shap.summary_plot(

    shap_values,

    X_scaled,

    feature_names=FEATURE_COLUMNS,

    plot_type="bar",

    show=False,

)


plt.title(

    "Global SHAP Feature Importance - Hybrid Model"

)


plt.tight_layout()


bar_plot_file = (

    SHAP_DIR

    / "shap_bar.png"

)


plt.savefig(

    bar_plot_file,

    dpi=300,

    bbox_inches="tight"

)


plt.close()


print()

print(

    "Saved:",

    bar_plot_file

)


# ==========================================================
# INDIVIDUAL FEATURE EFFECT STATISTICS
# ==========================================================

print()

print("=" * 70)

print("CALCULATING FEATURE EFFECT STATISTICS")

print("=" * 70)


effect_rows = []


for feature_index, feature in enumerate(

    FEATURE_COLUMNS

):

    values = shap_values[

        :,

        feature_index

    ]


    positive_count = np.sum(

        values > 0

    )


    negative_count = np.sum(

        values < 0

    )


    zero_count = np.sum(

        values == 0

    )


    effect_rows.append({

        "Feature":

            feature,

        "Mean_SHAP":

            np.mean(values),

        "Mean_Absolute_SHAP":

            np.mean(

                np.abs(values)

            ),

        "Positive_Contributions":

            positive_count,

        "Negative_Contributions":

            negative_count,

        "Zero_Contributions":

            zero_count,

    })


effect_df = pd.DataFrame(

    effect_rows

)


effect_df = (

    effect_df

    .sort_values(

        by="Mean_Absolute_SHAP",

        ascending=False

    )

    .reset_index(

        drop=True

    )

)


# ==========================================================
# DISPLAY EFFECT ANALYSIS
# ==========================================================

print()

print(

    effect_df.to_string(

        index=False

    )

)


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print()

print("=" * 70)

print("EXPERIMENT 4 COMPLETE")

print("=" * 70)


print()

print(

    "SHAP outputs generated in:"

)

print(

    SHAP_DIR

)


print()

print(

    "Files generated:"

)


print(

    "1.",

    summary_plot_file

)


print(

    "2.",

    bar_plot_file

)


print(

    "3.",

    shap_values_file

)


print(

    "4.",

    feature_importance_file

)


print()

print("=" * 70)

print("SHAP ANALYSIS FINISHED SUCCESSFULLY")

print("=" * 70)