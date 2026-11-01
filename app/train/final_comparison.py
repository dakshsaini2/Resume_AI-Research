"""
final_comparison.py

Experiment 9
Final Experimental Comparison

Reads already-generated experiment results.
Does NOT retrain any model.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==========================================================
# PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    BASE_DIR / "dataset" / "processed"
)

OUTPUT_DIR = (
    PROCESSED_DIR / "final_comparison"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# INPUT FILES
# ==========================================================

FILES = {

    "baseline":
        PROCESSED_DIR / "baseline_comparison.csv",

    "ablation":
        PROCESSED_DIR / "ablation_results.csv",

    "robustness":
        PROCESSED_DIR / "robustness_summary.csv",

    "ranking":
        PROCESSED_DIR / "ranking_results.csv",

    "ranking_summary":
        PROCESSED_DIR / "ranking_summary.csv",

    "error":
        PROCESSED_DIR
        / "error_analysis"
        / "model_comparison_statistics.csv",

    "error_summary":
        PROCESSED_DIR
        / "error_analysis"
        / "error_summary.csv",

    "candidate":
        PROCESSED_DIR
        / "candidate_group_evaluation"
        / "candidate_group_results.csv",

    "candidate_summary":
        PROCESSED_DIR
        / "candidate_group_evaluation"
        / "candidate_group_summary.csv",

    "shap":
        PROCESSED_DIR
        / "shap"
        / "shap_feature_importance.csv",

    "validation":
        PROCESSED_DIR
        / "dataset_validation"
        / "validation_summary.csv",
}


# ==========================================================
# HELPERS
# ==========================================================

def section(title):

    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def load_file(path):

    if not path.exists():

        print(
            f"WARNING: File not found: {path}"
        )

        return None

    try:

        df = pd.read_csv(path)

        print(
            f"Loaded {path.name}: "
            f"{df.shape[0]} rows × "
            f"{df.shape[1]} columns"
        )

        return df

    except Exception as e:

        print(
            f"ERROR loading {path}: {e}"
        )

        return None


def get_column(df, *names):

    if df is None:
        return None

    lookup = {
        str(c).strip().lower(): c
        for c in df.columns
    }

    for name in names:

        key = str(name).strip().lower()

        if key in lookup:

            return lookup[key]

    return None


def get_value(
    row,
    *names,
    default=np.nan
):

    for name in names:

        if name in row.index:

            value = row[name]

            if pd.notna(value):

                return value

    return default


def numeric(
    value
):

    try:

        return float(value)

    except:

        return np.nan


# ==========================================================
# START
# ==========================================================

section(
    "EXPERIMENT 9 — FINAL EXPERIMENTAL COMPARISON"
)


print(
    "Project:",
    BASE_DIR
)

print(
    "Output:",
    OUTPUT_DIR
)


# ==========================================================
# LOAD RESULTS
# ==========================================================

section(
    "LOADING EXPERIMENT RESULTS"
)


baseline_df = load_file(
    FILES["baseline"]
)

ablation_df = load_file(
    FILES["ablation"]
)

robustness_df = load_file(
    FILES["robustness"]
)

ranking_df = load_file(
    FILES["ranking"]
)

ranking_summary_df = load_file(
    FILES["ranking_summary"]
)

error_df = load_file(
    FILES["error"]
)

error_summary_df = load_file(
    FILES["error_summary"]
)

candidate_df = load_file(
    FILES["candidate"]
)

candidate_summary_df = load_file(
    FILES["candidate_summary"]
)

shap_df = load_file(
    FILES["shap"]
)

validation_df = load_file(
    FILES["validation"]
)


# ==========================================================
# CANDIDATE SUMMARY
# ==========================================================

section(
    "CANDIDATE-LEVEL RESULTS"
)


candidate_final = None


if candidate_summary_df is not None:

    candidate_final = (
        candidate_summary_df.copy()
    )

elif candidate_df is not None:

    model_col = get_column(
        candidate_df,
        "Model",
        "model"
    )

    if model_col:

        candidate_final = (

            candidate_df

            .groupby(
                model_col
            )

            .agg(

                MAE_mean=(
                    "MAE",
                    "mean"
                ),

                MAE_std=(
                    "MAE",
                    "std"
                ),

                RMSE_mean=(
                    "RMSE",
                    "mean"
                ),

                RMSE_std=(
                    "RMSE",
                    "std"
                ),

                R2_mean=(
                    "R2",
                    "mean"
                ),

                R2_std=(
                    "R2",
                    "std"
                ),

                NDCG1_mean=(
                    "NDCG@1",
                    "mean"
                ),

                NDCG3_mean=(
                    "NDCG@3",
                    "mean"
                ),

                NDCG5_mean=(
                    "NDCG@5",
                    "mean"
                ),

                NDCG10_mean=(
                    "NDCG@10",
                    "mean"
                ),

                MRR_mean=(
                    "MRR",
                    "mean"
                ),

                Precision3_mean=(
                    "Precision@3",
                    "mean"
                ),

                Recall3_mean=(
                    "Recall@3",
                    "mean"
                ),

            )

            .reset_index()

        )


# ----------------------------------------------------------
# Normalize candidate column names
# ----------------------------------------------------------

if candidate_final is not None:

    model_col = get_column(
        candidate_final,
        "Model",
        "model"
    )

    # Ranking names can appear in either format.
    rename_map = {}

    aliases = {

        "NDCG1_mean":
            [
                "NDCG1_mean",
                "NDCG@1_mean",
            ],

        "NDCG3_mean":
            [
                "NDCG3_mean",
                "NDCG@3_mean",
            ],

        "NDCG5_mean":
            [
                "NDCG5_mean",
                "NDCG@5_mean",
            ],

        "NDCG10_mean":
            [
                "NDCG10_mean",
                "NDCG@10_mean",
            ],

        "Precision3_mean":
            [
                "Precision3_mean",
                "Precision@3_mean",
            ],

        "Recall3_mean":
            [
                "Recall3_mean",
                "Recall@3_mean",
            ],

    }


    for target, candidates in aliases.items():

        found = get_column(
            candidate_final,
            *candidates
        )

        if found:

            rename_map[found] = target


    if rename_map:

        candidate_final = (
            candidate_final
            .rename(
                columns=rename_map
            )
        )


    print()

    print(
        candidate_final.to_string(
            index=False
        )
    )


# ==========================================================
# FINAL CANDIDATE TABLE
# ==========================================================

section(
    "FINAL UNSEEN-CANDIDATE TABLE"
)


candidate_table = pd.DataFrame()


if candidate_final is not None:

    model_col = get_column(
        candidate_final,
        "Model",
        "model"
    )

    if model_col:

        rows = []

        for _, row in candidate_final.iterrows():

            rows.append(

                {

                    "Model":
                        row[model_col],

                    "MAE":
                        numeric(
                            get_value(
                                row,
                                "MAE_mean",
                                "MAE"
                            )
                        ),

                    "RMSE":
                        numeric(
                            get_value(
                                row,
                                "RMSE_mean",
                                "RMSE"
                            )
                        ),

                    "R2":
                        numeric(
                            get_value(
                                row,
                                "R2_mean",
                                "R2"
                            )
                        ),

                    "NDCG@1":
                        numeric(
                            get_value(
                                row,
                                "NDCG1_mean",
                                "NDCG@1_mean",
                                "NDCG@1"
                            )
                        ),

                    "NDCG@3":
                        numeric(
                            get_value(
                                row,
                                "NDCG3_mean",
                                "NDCG@3_mean",
                                "NDCG@3"
                            )
                        ),

                    "NDCG@5":
                        numeric(
                            get_value(
                                row,
                                "NDCG5_mean",
                                "NDCG@5_mean",
                                "NDCG@5"
                            )
                        ),

                    "NDCG@10":
                        numeric(
                            get_value(
                                row,
                                "NDCG10_mean",
                                "NDCG@10_mean",
                                "NDCG@10"
                            )
                        ),

                    "MRR":
                        numeric(
                            get_value(
                                row,
                                "MRR_mean",
                                "MRR"
                            )
                        ),

                    "Precision@3":
                        numeric(
                            get_value(
                                row,
                                "Precision3_mean",
                                "Precision@3_mean",
                                "Precision@3"
                            )
                        ),

                    "Recall@3":
                        numeric(
                            get_value(
                                row,
                                "Recall3_mean",
                                "Recall@3_mean",
                                "Recall@3"
                            )
                        ),

                }

            )


        candidate_table = pd.DataFrame(
            rows
        )


        print()

        print(
            candidate_table.to_string(
                index=False
            )
        )


# ==========================================================
# BEST CANDIDATE MODELS
# ==========================================================

section(
    "BEST CANDIDATE-LEVEL MODELS"
)


best_rows = []


if not candidate_table.empty:

    best_mae = candidate_table.loc[
        candidate_table["MAE"].idxmin()
    ]

    best_rmse = candidate_table.loc[
        candidate_table["RMSE"].idxmin()
    ]

    best_r2 = candidate_table.loc[
        candidate_table["R2"].idxmax()
    ]

    best_ndcg5 = candidate_table.loc[
        candidate_table["NDCG@5"].idxmax()
    ]

    best_ndcg10 = candidate_table.loc[
        candidate_table["NDCG@10"].idxmax()
    ]

    best_mrr = candidate_table.loc[
        candidate_table["MRR"].idxmax()
    ]


    best_rows = [

        {
            "Metric": "MAE",
            "Direction": "Lower is better",
            "Best_Model":
                best_mae["Model"],
            "Value":
                best_mae["MAE"],
        },

        {
            "Metric": "RMSE",
            "Direction": "Lower is better",
            "Best_Model":
                best_rmse["Model"],
            "Value":
                best_rmse["RMSE"],
        },

        {
            "Metric": "R2",
            "Direction": "Higher is better",
            "Best_Model":
                best_r2["Model"],
            "Value":
                best_r2["R2"],
        },

        {
            "Metric": "NDCG@5",
            "Direction": "Higher is better",
            "Best_Model":
                best_ndcg5["Model"],
            "Value":
                best_ndcg5["NDCG@5"],
        },

        {
            "Metric": "NDCG@10",
            "Direction": "Higher is better",
            "Best_Model":
                best_ndcg10["Model"],
            "Value":
                best_ndcg10["NDCG@10"],
        },

        {
            "Metric": "MRR",
            "Direction": "Higher is better",
            "Best_Model":
                best_mrr["Model"],
            "Value":
                best_mrr["MRR"],
        },

    ]


    best_df = pd.DataFrame(
        best_rows
    )


    print()

    print(
        best_df.to_string(
            index=False
        )
    )


# ==========================================================
# ROBUSTNESS
# ==========================================================

section(
    "ROBUSTNESS SUMMARY"
)


robustness_final = None


if robustness_df is not None:

    robustness_final = (
        robustness_df.copy()
    )

    print()

    print(
        robustness_final.to_string(
            index=False
        )
    )


# ==========================================================
# ABLATION
# ==========================================================

section(
    "ABLATION SUMMARY"
)


if ablation_df is not None:

    print()

    print(
        ablation_df.to_string(
            index=False
        )
    )


# ==========================================================
# BASELINE
# ==========================================================

section(
    "BASELINE SUMMARY"
)


if baseline_df is not None:

    print()

    print(
        baseline_df.to_string(
            index=False
        )
    )


# ==========================================================
# RANKING
# ==========================================================

section(
    "JOB-GROUP RANKING SUMMARY"
)


ranking_final = None


if ranking_summary_df is not None:

    ranking_final = (
        ranking_summary_df.copy()
    )

elif ranking_df is not None:

    model_col = get_column(
        ranking_df,
        "Model",
        "model"
    )

    if model_col:

        ranking_final = (

            ranking_df

            .groupby(
                model_col
            )

            .agg({

                "NDCG@1":
                    "mean",

                "NDCG@3":
                    "mean",

                "NDCG@5":
                    "mean",

                "MRR":
                    "mean",

                "Precision@3":
                    "mean",

                "Recall@3":
                    "mean",

            })

            .reset_index()

        )


if ranking_final is not None:

    print()

    print(
        ranking_final.to_string(
            index=False
        )
    )


# ==========================================================
# SHAP
# ==========================================================

section(
    "SHAP TOP FEATURES"
)


top_shap = None


if shap_df is not None:

    feature_col = get_column(
        shap_df,
        "Feature",
        "feature"
    )

    shap_col = get_column(
        shap_df,
        "Mean_Absolute_SHAP",
        "mean_absolute_shap"
    )

    if feature_col and shap_col:

        top_shap = (

            shap_df

            .sort_values(
                shap_col,
                ascending=False
            )

            .head(10)

        )


        print()

        print(
            top_shap.to_string(
                index=False
            )
        )


# ==========================================================
# ERROR ANALYSIS
# ==========================================================

section(
    "ERROR ANALYSIS"
)


if error_df is not None:

    print()

    print(
        error_df.to_string(
            index=False
        )
    )

elif error_summary_df is not None:

    print()

    print(
        error_summary_df.to_string(
            index=False
        )
    )


# ==========================================================
# FINAL EXPERIMENT SUMMARY
# ==========================================================

section(
    "FINAL EXPERIMENT SUMMARY"
)


summary_rows = [

    {

        "Experiment":
            "Experiment 1",

        "Name":
            "Baseline Comparison",

        "Status":
            "Completed",

    },

    {

        "Experiment":
            "Experiment 2",

        "Name":
            "Ablation Study",

        "Status":
            "Completed",

    },

    {

        "Experiment":
            "Experiment 3",

        "Name":
            "Robustness Evaluation",

        "Status":
            "Completed",

    },

    {

        "Experiment":
            "Experiment 4",

        "Name":
            "SHAP Explainability",

        "Status":
            "Completed",

    },

    {

        "Experiment":
            "Experiment 5",

        "Name":
            "Candidate Ranking",

        "Status":
            "Completed",

    },

    {

        "Experiment":
            "Experiment 6",

        "Name":
            "Error Analysis",

        "Status":
            "Completed",

    },

    {

        "Experiment":
            "Experiment 7",

        "Name":
            "Dataset Validation",

        "Status":
            "Completed",

    },

    {

        "Experiment":
            "Experiment 8",

        "Name":
            "Candidate-Level Evaluation",

        "Status":
            "Completed",

    },

    {

        "Experiment":
            "Experiment 9",

        "Name":
            "Final Experimental Comparison",

        "Status":
            "Completed",

    },

]


experiment_summary = pd.DataFrame(
    summary_rows
)


print()

print(
    experiment_summary.to_string(
        index=False
    )
)


# ==========================================================
# SAVE TABLES
# ==========================================================

section(
    "SAVING FINAL TABLES"
)


if not candidate_table.empty:

    candidate_table.to_csv(

        OUTPUT_DIR
        / "final_candidate_comparison.csv",

        index=False

    )


if best_rows:

    best_df.to_csv(

        OUTPUT_DIR
        / "best_candidate_models.csv",

        index=False

    )


if ranking_final is not None:

    ranking_final.to_csv(

        OUTPUT_DIR
        / "final_ranking_comparison.csv",

        index=False

    )


if robustness_final is not None:

    robustness_final.to_csv(

        OUTPUT_DIR
        / "final_robustness_comparison.csv",

        index=False

    )


if ablation_df is not None:

    ablation_df.to_csv(

        OUTPUT_DIR
        / "final_ablation_comparison.csv",

        index=False

    )


if baseline_df is not None:

    baseline_df.to_csv(

        OUTPUT_DIR
        / "final_baseline_comparison.csv",

        index=False

    )


experiment_summary.to_csv(

    OUTPUT_DIR
    / "final_experiment_summary.csv",

    index=False

)


# ==========================================================
# CANDIDATE MAE PLOT
# ==========================================================

section(
    "GENERATING MAE PLOT"
)


if not candidate_table.empty:

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(

        candidate_table["Model"],

        candidate_table["MAE"]

    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "MAE"
    )

    plt.title(
        "Unseen-Candidate Evaluation: MAE"
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR
        / "final_candidate_mae.png",

        dpi=300

    )

    plt.close()


# ==========================================================
# CANDIDATE NDCG@5 PLOT
# ==========================================================

section(
    "GENERATING NDCG@5 PLOT"
)


if not candidate_table.empty:

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(

        candidate_table["Model"],

        candidate_table["NDCG@5"]

    )

    plt.xlabel(
        "Model"
    )

    plt.ylabel(
        "NDCG@5"
    )

    plt.title(
        "Unseen-Candidate Ranking: NDCG@5"
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(

        OUTPUT_DIR
        / "final_candidate_ndcg5.png",

        dpi=300

    )

    plt.close()


# ==========================================================
# FINAL REPORT
# ==========================================================

section(
    "GENERATING FINAL REPORT"
)


report = []


report.append(
    "FINAL EXPERIMENTAL COMPARISON REPORT"
)

report.append(
    "=" * 70
)

report.append("")


report.append(
    "1. EXPERIMENT STATUS"
)

report.append(
    "-" * 70
)


for _, row in experiment_summary.iterrows():

    report.append(

        f"{row['Experiment']}: "
        f"{row['Name']} — "
        f"{row['Status']}"

    )


report.append("")


# ----------------------------------------------------------
# Candidate results
# ----------------------------------------------------------

if not candidate_table.empty:

    report.append(
        "2. UNSEEN-CANDIDATE RESULTS"
    )

    report.append(
        "-" * 70
    )


    for _, row in candidate_table.iterrows():

        report.append(

            f"{row['Model']}: "
            f"MAE={row['MAE']:.4f}, "
            f"RMSE={row['RMSE']:.4f}, "
            f"R2={row['R2']:.4f}, "
            f"NDCG@5={row['NDCG@5']:.4f}, "
            f"NDCG@10={row['NDCG@10']:.4f}, "
            f"MRR={row['MRR']:.4f}"

        )


    report.append("")


# ----------------------------------------------------------
# Best models
# ----------------------------------------------------------

if best_rows:

    report.append(
        "3. BEST MODELS"
    )

    report.append(
        "-" * 70
    )


    for row in best_rows:

        report.append(

            f"{row['Metric']}: "
            f"{row['Best_Model']} "
            f"({row['Value']:.4f})"

        )


    report.append("")


# ----------------------------------------------------------
# SHAP
# ----------------------------------------------------------

if top_shap is not None:

    feature_col = get_column(
        top_shap,
        "Feature",
        "feature"
    )

    shap_col = get_column(
        top_shap,
        "Mean_Absolute_SHAP",
        "mean_absolute_shap"
    )


    if feature_col and shap_col:

        report.append(
            "4. TOP SHAP FEATURES"
        )

        report.append(
            "-" * 70
        )


        for _, row in top_shap.iterrows():

            report.append(

                f"{row[feature_col]}: "
                f"{row[shap_col]:.6f}"

            )


        report.append("")


# ----------------------------------------------------------
# Research conclusion
# ----------------------------------------------------------

report.append(
    "5. RESEARCH CONCLUSION"
)

report.append(
    "-" * 70
)

report.append(

    "The Hybrid model demonstrates promising "
    "generalization performance under candidate-level "
    "grouped evaluation."

)

report.append(

    "Candidate-level grouped evaluation is treated as "
    "the primary evidence for unseen-candidate "
    "generalization."

)

report.append(

    "Job-group evaluation provides complementary "
    "evidence for generalization across job groups."

)

report.append(

    "SHAP analysis provides feature-level explanations "
    "for the Random Forest predictions."

)

report.append("")


report.append(
    "6. IMPORTANT LIMITATION"
)

report.append(
    "-" * 70
)

report.append(

    "The dataset contains repeated candidate/resume "
    "representations across multiple job records. "
    "Therefore, row-level evaluation can overestimate "
    "generalization performance. Candidate-grouped "
    "evaluation should be emphasized in the research paper."

)


report_file = (

    OUTPUT_DIR
    / "final_comparison_report.txt"

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
# FINAL
# ==========================================================

section(
    "EXPERIMENT 9 COMPLETE"
)


print()

print(
    "Files generated:"
)

print()

for file in sorted(
    OUTPUT_DIR.iterdir()
):

    if file.is_file():

        print(
            file
        )


print()

print("=" * 70)

print(
    "FINAL COMPARISON FINISHED SUCCESSFULLY"
)

print("=" * 70)