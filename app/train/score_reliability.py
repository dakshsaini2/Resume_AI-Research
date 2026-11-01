"""
score_reliability.py

Experiment 8: Score Reliability and Calibration Analysis
========================================================

Purpose
-------
Evaluate whether the predicted match scores from the trained
Hybrid Random Forest are reliable across the full target range.

This is a critical requirement for research publishability.
Without this analysis, match scores cannot be interpreted
as meaningful indicators of resume-job compatibility.

Analyses
--------
1. Predicted versus actual score scatter plot
2. Residual distribution (histogram + density)
3. MAE / RMSE / R² by target score range
4. Predicted score distribution versus actual distribution
5. Residual versus predicted (heteroscedasticity check)
6. Score bin calibration table
7. Quantile-quantile residual plot

Outputs
-------
dataset/processed/score_reliability/
    predicted_vs_actual.csv
    predicted_vs_actual.png
    residual_distribution.png
    residual_vs_predicted.png
    score_distribution_comparison.png
    error_by_score_bin.csv
    error_by_score_bin.png
    calibration_summary.csv
    qq_residuals.png
    reliability_report.txt

Important
---------
This script DOES NOT retrain the model.

It loads:
    - trained Random Forest
    - trained scaler
    - existing feature configuration

and analyzes the prediction reliability of the
already-trained research model.
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

from scipy import stats

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
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
    / "score_reliability"
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

# Score bin edges for calibration analysis.
# These define the ranges over which we measure
# reliability of the model predictions.

SCORE_BINS = [
    0.0,
    0.2,
    0.4,
    0.6,
    0.8,
    1.0,
]

SCORE_BIN_LABELS = [
    "0.0 - 0.2",
    "0.2 - 0.4",
    "0.4 - 0.6",
    "0.6 - 0.8",
    "0.8 - 1.0",
]


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

    "figure.figsize": (8, 6),
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

print("EXPERIMENT 8: SCORE RELIABILITY AND CALIBRATION")

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

print(
    "Target range:",
    df[TARGET_COLUMN].min(),
    "to",
    df[TARGET_COLUMN].max()
)


# ==========================================================
# BUILD CANDIDATE AND JOB TEXT
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
# GENERATE FEATURES FOR ALL SAMPLES
# ==========================================================

print()

print("=" * 70)

print("GENERATING RESEARCH FEATURES")

print("=" * 70)


feature_rows = []


for index, row in df.iterrows():

    if index % 500 == 0:

        print(
            f"Processing sample "
            f"{index + 1}/"
            f"{len(df)}..."
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


X = (

    X

    .replace(
        [np.inf, -np.inf],
        np.nan
    )

    .fillna(0)
)


print()

print(
    "Feature matrix shape:",
    X.shape
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


print()

print(
    "Scaled feature matrix:",
    X_scaled.shape
)


# ==========================================================
# GENERATE PREDICTIONS
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


actual = df[TARGET_COLUMN].values


residuals = actual - predictions


print()

print(
    "Predictions generated:",
    len(predictions)
)

print(
    "Prediction range:",
    f"{predictions.min():.4f}",
    "to",
    f"{predictions.max():.4f}"
)

print(
    "Actual range:",
    f"{actual.min():.4f}",
    "to",
    f"{actual.max():.4f}"
)


# ==========================================================
# OVERALL METRICS
# ==========================================================

print()

print("=" * 70)

print("OVERALL REGRESSION METRICS")

print("=" * 70)


overall_mae = mean_absolute_error(
    actual,
    predictions
)


overall_rmse = np.sqrt(
    mean_squared_error(
        actual,
        predictions
    )
)


overall_r2 = r2_score(
    actual,
    predictions
)


print()

print(f"MAE:  {overall_mae:.6f}")

print(f"RMSE: {overall_rmse:.6f}")

print(f"R²:   {overall_r2:.6f}")


# ==========================================================
# SAVE PREDICTED VS ACTUAL DATA
# ==========================================================

print()

print("=" * 70)

print("SAVING PREDICTED VS ACTUAL DATA")

print("=" * 70)


results_df = pd.DataFrame({

    "sample_index":
        np.arange(len(actual)),

    "actual_score":
        actual,

    "predicted_score":
        predictions,

    "residual":
        residuals,

    "absolute_error":
        np.abs(residuals),
})


results_file = (
    OUTPUT_DIR
    / "predicted_vs_actual.csv"
)


results_df.to_csv(
    results_file,
    index=False
)


print()

print(
    "Saved:",
    results_file
)


# ==========================================================
# PLOT 1: PREDICTED VS ACTUAL SCATTER
# ==========================================================

print()

print("=" * 70)

print("GENERATING PREDICTED VS ACTUAL PLOT")

print("=" * 70)


fig, ax = plt.subplots(
    figsize=(8, 8)
)


ax.scatter(

    actual,

    predictions,

    alpha=0.15,

    s=8,

    color="#2563EB",

    edgecolors="none",

    label="Samples"
)


# Perfect prediction line.

ax.plot(

    [0, 1],

    [0, 1],

    color="#DC2626",

    linewidth=2,

    linestyle="--",

    label="Perfect Prediction",

    zorder=5
)


# Linear regression fit.

slope, intercept, r_value, p_value, std_err = (
    stats.linregress(
        actual,
        predictions
    )
)


fit_x = np.linspace(0, 1, 100)

fit_y = slope * fit_x + intercept


ax.plot(

    fit_x,

    fit_y,

    color="#059669",

    linewidth=2,

    linestyle="-",

    label=(
        f"Linear Fit "
        f"(r={r_value:.3f})"
    ),

    zorder=4
)


ax.set_xlabel(
    "Actual Matched Score"
)

ax.set_ylabel(
    "Predicted Matched Score"
)

ax.set_title(
    "Predicted vs Actual Score — "
    "Hybrid Random Forest Model"
)

ax.set_xlim(-0.05, 1.05)

ax.set_ylim(-0.05, 1.05)

ax.set_aspect("equal")

ax.legend(
    loc="upper left"
)

ax.grid(
    True,
    alpha=0.3,
    linestyle=":"
)


# Add metrics text box.

metrics_text = (
    f"MAE = {overall_mae:.4f}\n"
    f"RMSE = {overall_rmse:.4f}\n"
    f"R² = {overall_r2:.4f}\n"
    f"N = {len(actual)}"
)


ax.text(

    0.97,

    0.03,

    metrics_text,

    transform=ax.transAxes,

    fontsize=10,

    verticalalignment="bottom",

    horizontalalignment="right",

    bbox=dict(
        boxstyle="round,pad=0.4",
        facecolor="white",
        edgecolor="gray",
        alpha=0.9
    )
)


plt.tight_layout()


scatter_file = (
    OUTPUT_DIR
    / "predicted_vs_actual.png"
)


plt.savefig(

    scatter_file,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print()

print(
    "Saved:",
    scatter_file
)


# ==========================================================
# PLOT 2: RESIDUAL DISTRIBUTION
# ==========================================================

print()

print("=" * 70)

print("GENERATING RESIDUAL DISTRIBUTION PLOT")

print("=" * 70)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)


# Histogram.

axes[0].hist(

    residuals,

    bins=60,

    color="#2563EB",

    alpha=0.7,

    edgecolor="white",

    linewidth=0.5,

    density=True,

    label="Residuals"
)


# Overlay normal distribution.

mu = np.mean(residuals)

sigma = np.std(residuals)

x_normal = np.linspace(
    residuals.min(),
    residuals.max(),
    200
)

y_normal = stats.norm.pdf(
    x_normal,
    mu,
    sigma
)


axes[0].plot(
    x_normal,
    y_normal,
    color="#DC2626",
    linewidth=2,
    label=(
        f"Normal "
        f"(μ={mu:.4f}, σ={sigma:.4f})"
    )
)


axes[0].axvline(
    x=0,
    color="#059669",
    linewidth=1.5,
    linestyle="--",
    alpha=0.8
)


axes[0].set_xlabel(
    "Residual (Actual − Predicted)"
)

axes[0].set_ylabel(
    "Density"
)

axes[0].set_title(
    "Residual Distribution"
)

axes[0].legend(
    fontsize=9
)

axes[0].grid(
    True,
    alpha=0.3,
    linestyle=":"
)


# Summary statistics.

residual_stats = (
    f"Mean: {mu:.5f}\n"
    f"Std:  {sigma:.5f}\n"
    f"Skew: {stats.skew(residuals):.4f}\n"
    f"Kurt: {stats.kurtosis(residuals):.4f}"
)


axes[0].text(

    0.97,

    0.97,

    residual_stats,

    transform=axes[0].transAxes,

    fontsize=9,

    verticalalignment="top",

    horizontalalignment="right",

    bbox=dict(
        boxstyle="round,pad=0.3",
        facecolor="white",
        edgecolor="gray",
        alpha=0.9
    )
)


# Box plot.

bp = axes[1].boxplot(

    residuals,

    vert=True,

    patch_artist=True,

    boxprops=dict(
        facecolor="#DBEAFE",
        color="#2563EB"
    ),

    medianprops=dict(
        color="#DC2626",
        linewidth=2
    ),

    whiskerprops=dict(
        color="#2563EB"
    ),

    capprops=dict(
        color="#2563EB"
    ),

    flierprops=dict(
        marker="o",
        markerfacecolor="#FCA5A5",
        markersize=3,
        alpha=0.5
    )
)


axes[1].set_ylabel(
    "Residual (Actual − Predicted)"
)

axes[1].set_title(
    "Residual Box Plot"
)

axes[1].set_xticklabels(
    ["Hybrid Model"]
)

axes[1].grid(
    True,
    alpha=0.3,
    linestyle=":",
    axis="y"
)


plt.suptitle(

    "Prediction Residual Analysis — "
    "Hybrid Random Forest",

    fontsize=14,

    fontweight="bold",

    y=1.02
)


plt.tight_layout()


residual_file = (
    OUTPUT_DIR
    / "residual_distribution.png"
)


plt.savefig(

    residual_file,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print()

print(
    "Saved:",
    residual_file
)


# ==========================================================
# PLOT 3: RESIDUAL VS PREDICTED
# ==========================================================

print()

print("=" * 70)

print("GENERATING RESIDUAL VS PREDICTED PLOT")

print("=" * 70)


fig, ax = plt.subplots(
    figsize=(8, 6)
)


ax.scatter(

    predictions,

    residuals,

    alpha=0.15,

    s=8,

    color="#7C3AED",

    edgecolors="none"
)


ax.axhline(
    y=0,
    color="#DC2626",
    linewidth=2,
    linestyle="--"
)


ax.set_xlabel(
    "Predicted Score"
)

ax.set_ylabel(
    "Residual (Actual − Predicted)"
)

ax.set_title(
    "Residual vs Predicted — "
    "Heteroscedasticity Check"
)

ax.grid(
    True,
    alpha=0.3,
    linestyle=":"
)


plt.tight_layout()


residual_vs_pred_file = (
    OUTPUT_DIR
    / "residual_vs_predicted.png"
)


plt.savefig(

    residual_vs_pred_file,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print()

print(
    "Saved:",
    residual_vs_pred_file
)


# ==========================================================
# PLOT 4: SCORE DISTRIBUTION COMPARISON
# ==========================================================

print()

print("=" * 70)

print("GENERATING SCORE DISTRIBUTION COMPARISON")

print("=" * 70)


fig, ax = plt.subplots(
    figsize=(10, 6)
)


ax.hist(

    actual,

    bins=50,

    alpha=0.6,

    color="#2563EB",

    edgecolor="white",

    linewidth=0.5,

    label="Actual Scores",

    density=True
)


ax.hist(

    predictions,

    bins=50,

    alpha=0.6,

    color="#DC2626",

    edgecolor="white",

    linewidth=0.5,

    label="Predicted Scores",

    density=True
)


ax.set_xlabel(
    "Match Score"
)

ax.set_ylabel(
    "Density"
)

ax.set_title(
    "Distribution Comparison: "
    "Actual vs Predicted Scores"
)

ax.legend()

ax.grid(
    True,
    alpha=0.3,
    linestyle=":"
)


plt.tight_layout()


distribution_file = (
    OUTPUT_DIR
    / "score_distribution_comparison.png"
)


plt.savefig(

    distribution_file,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print()

print(
    "Saved:",
    distribution_file
)


# ==========================================================
# ANALYSIS: ERROR BY SCORE BIN
# ==========================================================

print()

print("=" * 70)

print("CALCULATING ERROR BY SCORE BIN")

print("=" * 70)


results_df["score_bin"] = pd.cut(

    results_df["actual_score"],

    bins=SCORE_BINS,

    labels=SCORE_BIN_LABELS,

    include_lowest=True
)


bin_analysis = []


for label in SCORE_BIN_LABELS:

    mask = (
        results_df["score_bin"]
        == label
    )

    bin_data = results_df[mask]

    if len(bin_data) == 0:
        continue

    bin_actual = bin_data[
        "actual_score"
    ].values

    bin_predicted = bin_data[
        "predicted_score"
    ].values

    bin_mae = mean_absolute_error(
        bin_actual,
        bin_predicted
    )

    bin_rmse = np.sqrt(
        mean_squared_error(
            bin_actual,
            bin_predicted
        )
    )

    bin_r2 = r2_score(
        bin_actual,
        bin_predicted
    ) if len(bin_actual) > 1 else float("nan")

    mean_actual = np.mean(
        bin_actual
    )

    mean_predicted = np.mean(
        bin_predicted
    )

    mean_residual = np.mean(
        bin_data["residual"].values
    )

    bin_analysis.append({

        "Score_Range":
            label,

        "Sample_Count":
            len(bin_data),

        "Mean_Actual":
            round(mean_actual, 6),

        "Mean_Predicted":
            round(mean_predicted, 6),

        "Mean_Residual":
            round(mean_residual, 6),

        "MAE":
            round(bin_mae, 6),

        "RMSE":
            round(bin_rmse, 6),

        "R2":
            round(bin_r2, 6)
            if not np.isnan(bin_r2)
            else "N/A",

        "Bias_Direction":
            "Over-predicting"
            if mean_residual < 0
            else "Under-predicting",
    })


bin_df = pd.DataFrame(
    bin_analysis
)


bin_file = (
    OUTPUT_DIR
    / "error_by_score_bin.csv"
)


bin_df.to_csv(
    bin_file,
    index=False
)


print()

print(
    bin_df.to_string(
        index=False
    )
)

print()

print(
    "Saved:",
    bin_file
)


# ==========================================================
# PLOT 5: ERROR BY SCORE BIN
# ==========================================================

print()

print("=" * 70)

print("GENERATING ERROR BY SCORE BIN PLOT")

print("=" * 70)


fig, axes = plt.subplots(
    1,
    2,
    figsize=(14, 6)
)


numeric_bin_df = bin_df[
    bin_df["R2"] != "N/A"
].copy()


x_positions = range(
    len(numeric_bin_df)
)


# MAE and RMSE bar chart.

bar_width = 0.35


axes[0].bar(

    [
        p - bar_width / 2
        for p in x_positions
    ],

    numeric_bin_df["MAE"].astype(float),

    bar_width,

    label="MAE",

    color="#2563EB",

    alpha=0.8,

    edgecolor="white"
)


axes[0].bar(

    [
        p + bar_width / 2
        for p in x_positions
    ],

    numeric_bin_df["RMSE"].astype(float),

    bar_width,

    label="RMSE",

    color="#DC2626",

    alpha=0.8,

    edgecolor="white"
)


axes[0].set_xlabel(
    "Actual Score Range"
)

axes[0].set_ylabel(
    "Error"
)

axes[0].set_title(
    "MAE and RMSE by Score Range"
)

axes[0].set_xticks(
    list(x_positions)
)

axes[0].set_xticklabels(

    numeric_bin_df[
        "Score_Range"
    ].values,

    rotation=30,

    ha="right"
)

axes[0].legend()

axes[0].grid(
    True,
    alpha=0.3,
    linestyle=":",
    axis="y"
)


# Mean residual (bias) bar chart.

colors = [

    "#DC2626"
    if float(r) < 0
    else "#059669"

    for r in numeric_bin_df[
        "Mean_Residual"
    ].values
]


axes[1].bar(

    list(x_positions),

    numeric_bin_df[
        "Mean_Residual"
    ].astype(float),

    color=colors,

    alpha=0.8,

    edgecolor="white"
)


axes[1].axhline(
    y=0,
    color="black",
    linewidth=1,
    linestyle="-"
)


axes[1].set_xlabel(
    "Actual Score Range"
)

axes[1].set_ylabel(
    "Mean Residual (Bias)"
)

axes[1].set_title(
    "Prediction Bias by Score Range\n"
    "(Positive = Under-predicting, "
    "Negative = Over-predicting)"
)

axes[1].set_xticks(
    list(x_positions)
)

axes[1].set_xticklabels(

    numeric_bin_df[
        "Score_Range"
    ].values,

    rotation=30,

    ha="right"
)

axes[1].grid(
    True,
    alpha=0.3,
    linestyle=":",
    axis="y"
)


plt.suptitle(

    "Score Reliability Analysis "
    "by Target Range",

    fontsize=14,

    fontweight="bold",

    y=1.02
)


plt.tight_layout()


bin_plot_file = (
    OUTPUT_DIR
    / "error_by_score_bin.png"
)


plt.savefig(

    bin_plot_file,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print()

print(
    "Saved:",
    bin_plot_file
)


# ==========================================================
# PLOT 6: QUANTILE-QUANTILE PLOT OF RESIDUALS
# ==========================================================

print()

print("=" * 70)

print("GENERATING QQ PLOT OF RESIDUALS")

print("=" * 70)


fig, ax = plt.subplots(
    figsize=(7, 7)
)


stats.probplot(

    residuals,

    dist="norm",

    plot=ax
)


ax.set_title(
    "Q-Q Plot of Prediction Residuals"
)

ax.grid(
    True,
    alpha=0.3,
    linestyle=":"
)


plt.tight_layout()


qq_file = (
    OUTPUT_DIR
    / "qq_residuals.png"
)


plt.savefig(

    qq_file,

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print()

print(
    "Saved:",
    qq_file
)


# ==========================================================
# CALIBRATION SUMMARY
# ==========================================================

print()

print("=" * 70)

print("GENERATING CALIBRATION SUMMARY")

print("=" * 70)


# Normality test on residuals.

shapiro_stat, shapiro_p = stats.shapiro(

    np.random.choice(
        residuals,
        min(5000, len(residuals)),
        replace=False
    )
)


# Correlation between predicted and actual.

pearson_r, pearson_p = stats.pearsonr(
    actual,
    predictions
)

spearman_r, spearman_p = stats.spearmanr(
    actual,
    predictions
)


# Over-prediction / under-prediction counts.

over_count = int(
    np.sum(predictions > actual)
)

under_count = int(
    np.sum(predictions < actual)
)

exact_count = int(
    np.sum(predictions == actual)
)


calibration_summary = pd.DataFrame([

    {
        "Metric": "Overall MAE",
        "Value": f"{overall_mae:.6f}"
    },

    {
        "Metric": "Overall RMSE",
        "Value": f"{overall_rmse:.6f}"
    },

    {
        "Metric": "Overall R²",
        "Value": f"{overall_r2:.6f}"
    },

    {
        "Metric": "Pearson r",
        "Value": f"{pearson_r:.6f}"
    },

    {
        "Metric": "Pearson p-value",
        "Value": f"{pearson_p:.2e}"
    },

    {
        "Metric": "Spearman ρ",
        "Value": f"{spearman_r:.6f}"
    },

    {
        "Metric": "Spearman p-value",
        "Value": f"{spearman_p:.2e}"
    },

    {
        "Metric": "Mean Residual",
        "Value": f"{np.mean(residuals):.6f}"
    },

    {
        "Metric": "Std Residual",
        "Value": f"{np.std(residuals):.6f}"
    },

    {
        "Metric": "Residual Skewness",
        "Value": f"{stats.skew(residuals):.6f}"
    },

    {
        "Metric": "Residual Kurtosis",
        "Value": f"{stats.kurtosis(residuals):.6f}"
    },

    {
        "Metric": "Shapiro-Wilk Statistic",
        "Value": f"{shapiro_stat:.6f}"
    },

    {
        "Metric": "Shapiro-Wilk p-value",
        "Value": f"{shapiro_p:.2e}"
    },

    {
        "Metric": "Over-predictions",
        "Value": str(over_count)
    },

    {
        "Metric": "Under-predictions",
        "Value": str(under_count)
    },

    {
        "Metric": "Exact predictions",
        "Value": str(exact_count)
    },

    {
        "Metric": "Total samples",
        "Value": str(len(actual))
    },

    {
        "Metric": "Linear fit slope",
        "Value": f"{slope:.6f}"
    },

    {
        "Metric": "Linear fit intercept",
        "Value": f"{intercept:.6f}"
    },

    {
        "Metric": "Linear fit r-value",
        "Value": f"{r_value:.6f}"
    },
])


calibration_file = (
    OUTPUT_DIR
    / "calibration_summary.csv"
)


calibration_summary.to_csv(
    calibration_file,
    index=False
)


print()

print(
    calibration_summary.to_string(
        index=False
    )
)


print()

print(
    "Saved:",
    calibration_file
)


# ==========================================================
# RELIABILITY REPORT
# ==========================================================

print()

print("=" * 70)

print("GENERATING RELIABILITY REPORT")

print("=" * 70)


report_lines = [

    "=" * 70,

    "EXPERIMENT 8: SCORE RELIABILITY REPORT",

    "=" * 70,

    "",

    "Model: Hybrid Random Forest (14 features)",

    f"Dataset: {len(actual)} samples",

    "",

    "--- Overall Metrics ---",

    f"MAE:  {overall_mae:.6f}",

    f"RMSE: {overall_rmse:.6f}",

    f"R²:   {overall_r2:.6f}",

    "",

    "--- Correlation ---",

    f"Pearson r:  {pearson_r:.6f}"
    f" (p = {pearson_p:.2e})",

    f"Spearman ρ: {spearman_r:.6f}"
    f" (p = {spearman_p:.2e})",

    "",

    "--- Residual Statistics ---",

    f"Mean residual: {np.mean(residuals):.6f}",

    f"Std residual:  {np.std(residuals):.6f}",

    f"Skewness:      {stats.skew(residuals):.6f}",

    f"Kurtosis:      {stats.kurtosis(residuals):.6f}",

    "",

    "--- Prediction Direction ---",

    f"Over-predictions:  {over_count}",

    f"Under-predictions: {under_count}",

    f"Exact predictions: {exact_count}",

    "",

    "--- Calibration Assessment ---",

    (
        "The predicted match score is a "
        "regression estimate, NOT a probability."
    ),

    (
        "It should be interpreted as a relative "
        "compatibility indicator, not a calibrated "
        "likelihood."
    ),

    "",

    "--- Score Range Reliability ---",
]


for _, row in bin_df.iterrows():

    report_lines.append(

        f"  {row['Score_Range']}: "
        f"MAE={row['MAE']}, "
        f"Bias={row['Bias_Direction']}"
    )


report_lines.extend([

    "",

    "=" * 70,

    "END OF RELIABILITY REPORT",

    "=" * 70,
])


report_text = "\n".join(
    report_lines
)


report_file = (
    OUTPUT_DIR
    / "reliability_report.txt"
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

print("EXPERIMENT 8 COMPLETE")

print("=" * 70)


print()

print(
    "Score reliability outputs generated in:"
)

print(
    OUTPUT_DIR
)


print()

print(
    "Files generated:"
)

output_files = [
    results_file,
    scatter_file,
    residual_file,
    residual_vs_pred_file,
    distribution_file,
    bin_file,
    bin_plot_file,
    qq_file,
    calibration_file,
    report_file,
]


for i, f in enumerate(
    output_files,
    start=1
):

    print(
        f"{i}.",
        f
    )


print()

print("=" * 70)

print("SCORE RELIABILITY ANALYSIS FINISHED SUCCESSFULLY")

print("=" * 70)
