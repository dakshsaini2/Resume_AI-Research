"""
generate_paper_figures.py

Publication Figure and Table Generator
=======================================

Purpose
-------
Consolidate all experiment outputs into publication-ready
figures and tables in a single directory, suitable for
direct inclusion in the research paper.

This script reads existing experiment results (it does NOT
rerun experiments) and generates consistently formatted,
high-resolution outputs.

Required Inputs (from prior experiments)
----------------------------------------
dataset/processed/ablation_results.csv
dataset/processed/ranking_v2_summary.csv
dataset/processed/robustness_results.csv
dataset/processed/error_analysis/error_by_score_range.csv
dataset/processed/error_analysis/error_by_job_group.csv
dataset/processed/shap/shap_feature_importance.csv
dataset/processed/evaluation_metrics.csv

Outputs
-------
dataset/processed/paper_figures/
    fig_ablation_comparison.png
    fig_ranking_comparison.png
    fig_robustness_stability.png
    fig_error_by_score_range.png
    fig_model_comparison.png
    table_ablation_results.csv
    table_ranking_results.csv
    table_feature_definitions.csv
    table_dataset_statistics.csv
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
import matplotlib.ticker as mtick


warnings.filterwarnings("ignore")


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

OUTPUT_DIR = (
    PROCESSED_DIR
    / "paper_figures"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ==========================================================
# PUBLICATION PLOT STYLE
# ==========================================================

plt.rcParams.update({

    "figure.dpi": 150,

    "savefig.dpi": 300,

    "font.size": 11,

    "font.family": "serif",

    "axes.titlesize": 14,

    "axes.labelsize": 12,

    "xtick.labelsize": 10,

    "ytick.labelsize": 10,

    "legend.fontsize": 9,

    "figure.figsize": (8, 5),

    "axes.grid": True,

    "grid.alpha": 0.3,

    "grid.linestyle": ":",
})


# ==========================================================
# COLOR PALETTE
# ==========================================================

COLORS = {
    "hybrid":     "#2563EB",
    "tfidf":      "#059669",
    "sbert":      "#7C3AED",
    "ats":        "#DC2626",
    "accent1":    "#F59E0B",
    "accent2":    "#EC4899",
    "neutral":    "#6B7280",
}


# ==========================================================
# START
# ==========================================================

print()

print("=" * 70)

print("PUBLICATION FIGURE AND TABLE GENERATOR")

print("=" * 70)


# ==========================================================
# FIGURE 1: ABLATION COMPARISON
# ==========================================================

print()

print("=" * 70)

print("FIGURE 1: ABLATION COMPARISON")

print("=" * 70)


ablation_file = (
    PROCESSED_DIR
    / "ablation_results.csv"
)

if ablation_file.exists():

    ablation = pd.read_csv(
        ablation_file
    )

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 5)
    )

    experiments = ablation[
        "Experiment"
    ].values

    short_names = [
        name.replace("Hybrid - ", "−")
        .replace("Full Hybrid", "Full")
        for name in experiments
    ]

    # MAE.

    bars = axes[0].barh(

        range(len(experiments)),

        ablation["MAE"].values,

        color=[
            COLORS["hybrid"]
            if "Full" in n
            else COLORS["neutral"]
            for n in experiments
        ],

        alpha=0.85,

        edgecolor="white"
    )

    axes[0].set_yticks(
        range(len(experiments))
    )

    axes[0].set_yticklabels(
        short_names
    )

    axes[0].set_xlabel("MAE")

    axes[0].set_title("MAE by Configuration")

    axes[0].invert_yaxis()


    # RMSE.

    axes[1].barh(

        range(len(experiments)),

        ablation["RMSE"].values,

        color=[
            COLORS["hybrid"]
            if "Full" in n
            else COLORS["neutral"]
            for n in experiments
        ],

        alpha=0.85,

        edgecolor="white"
    )

    axes[1].set_yticks(
        range(len(experiments))
    )

    axes[1].set_yticklabels(
        short_names
    )

    axes[1].set_xlabel("RMSE")

    axes[1].set_title("RMSE by Configuration")

    axes[1].invert_yaxis()


    # R².

    axes[2].barh(

        range(len(experiments)),

        ablation["R2"].values,

        color=[
            COLORS["hybrid"]
            if "Full" in n
            else COLORS["neutral"]
            for n in experiments
        ],

        alpha=0.85,

        edgecolor="white"
    )

    axes[2].set_yticks(
        range(len(experiments))
    )

    axes[2].set_yticklabels(
        short_names
    )

    axes[2].set_xlabel("R²")

    axes[2].set_title("R² by Configuration")

    axes[2].invert_yaxis()


    plt.suptitle(

        "Seven-Way Ablation Study Results",

        fontsize=15,

        fontweight="bold",

        y=1.02
    )


    plt.tight_layout()

    fig_file = (
        OUTPUT_DIR
        / "fig_ablation_comparison.png"
    )

    plt.savefig(
        fig_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"  Saved: {fig_file}")


    # Save formatted table.

    table_file = (
        OUTPUT_DIR
        / "table_ablation_results.csv"
    )

    ablation_table = ablation[[
        "Experiment",
        "MAE",
        "RMSE",
        "R2",
        "Number_of_features"
    ]].copy()

    ablation_table["MAE"] = (
        ablation_table["MAE"]
        .round(4)
    )

    ablation_table["RMSE"] = (
        ablation_table["RMSE"]
        .round(4)
    )

    ablation_table["R2"] = (
        ablation_table["R2"]
        .round(4)
    )

    ablation_table.to_csv(
        table_file,
        index=False
    )

    print(f"  Saved: {table_file}")


else:

    print("  SKIPPED: ablation_results.csv not found")


# ==========================================================
# FIGURE 2: RANKING COMPARISON
# ==========================================================

print()

print("=" * 70)

print("FIGURE 2: RANKING COMPARISON")

print("=" * 70)


ranking_file = (
    PROCESSED_DIR
    / "ranking_v2_summary.csv"
)


if ranking_file.exists():

    ranking = pd.read_csv(
        ranking_file
    )

    models = ranking["Model"].values

    model_colors = [
        COLORS.get(
            m.lower().replace(" ", "_")
            .replace("-", "_"),
            COLORS["neutral"]
        )
        for m in models
    ]

    # Assign specific colors.

    color_map = {
        "Hybrid": COLORS["hybrid"],
        "Structured ATS": COLORS["ats"],
        "SBERT Only": COLORS["sbert"],
        "TF-IDF Only": COLORS["tfidf"],
    }

    model_colors = [
        color_map.get(m, COLORS["neutral"])
        for m in models
    ]


    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 5)
    )


    # NDCG@5.

    ndcg5 = ranking["NDCG@5_mean"].values

    ndcg5_std = ranking["NDCG@5_std"].values

    axes[0].bar(

        range(len(models)),

        ndcg5,

        yerr=ndcg5_std,

        color=model_colors,

        alpha=0.85,

        edgecolor="white",

        capsize=3
    )

    axes[0].set_xticks(
        range(len(models))
    )

    axes[0].set_xticklabels(
        models,
        rotation=25,
        ha="right"
    )

    axes[0].set_ylabel("NDCG@5")

    axes[0].set_title("NDCG@5")

    axes[0].set_ylim(0.8, 0.92)


    # MRR.

    mrr = ranking["MRR_mean"].values

    mrr_std = ranking["MRR_std"].values

    axes[1].bar(

        range(len(models)),

        mrr,

        yerr=mrr_std,

        color=model_colors,

        alpha=0.85,

        edgecolor="white",

        capsize=3
    )

    axes[1].set_xticks(
        range(len(models))
    )

    axes[1].set_xticklabels(
        models,
        rotation=25,
        ha="right"
    )

    axes[1].set_ylabel("MRR")

    axes[1].set_title("Mean Reciprocal Rank")

    axes[1].set_ylim(0.75, 0.98)


    # Precision@3.

    p3 = ranking["Precision@3_mean"].values

    p3_std = ranking["Precision@3_std"].values

    axes[2].bar(

        range(len(models)),

        p3,

        yerr=p3_std,

        color=model_colors,

        alpha=0.85,

        edgecolor="white",

        capsize=3
    )

    axes[2].set_xticks(
        range(len(models))
    )

    axes[2].set_xticklabels(
        models,
        rotation=25,
        ha="right"
    )

    axes[2].set_ylabel("Precision@3")

    axes[2].set_title("Precision@3")

    axes[2].set_ylim(0.7, 0.9)


    plt.suptitle(

        "Candidate Ranking Performance Comparison",

        fontsize=15,

        fontweight="bold",

        y=1.02
    )


    plt.tight_layout()


    fig_file = (
        OUTPUT_DIR
        / "fig_ranking_comparison.png"
    )

    plt.savefig(
        fig_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"  Saved: {fig_file}")


    # Save formatted ranking table.

    table_cols = [
        "Model",
        "NDCG@1_mean", "NDCG@3_mean",
        "NDCG@5_mean", "NDCG@10_mean",
        "MRR_mean",
        "Precision@3_mean",
        "Recall@3_mean",
    ]

    ranking_table = ranking[
        table_cols
    ].copy()

    for col in table_cols[1:]:

        ranking_table[col] = (
            ranking_table[col].round(4)
        )

    table_file = (
        OUTPUT_DIR
        / "table_ranking_results.csv"
    )

    ranking_table.to_csv(
        table_file,
        index=False
    )

    print(f"  Saved: {table_file}")


else:

    print(
        "  SKIPPED: "
        "ranking_v2_summary.csv not found"
    )


# ==========================================================
# FIGURE 3: ROBUSTNESS STABILITY
# ==========================================================

print()

print("=" * 70)

print("FIGURE 3: ROBUSTNESS STABILITY")

print("=" * 70)


robustness_file = (
    PROCESSED_DIR
    / "robustness_results.csv"
)


if robustness_file.exists():

    robustness = pd.read_csv(
        robustness_file
    )

    models_r = robustness[
        "Model"
    ].unique()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(16, 5)
    )


    for model_name in models_r:

        mask = (
            robustness["Model"] == model_name
        )

        model_data = robustness[mask]

        color = color_map.get(
            model_name,
            COLORS["neutral"]
        )

        if "Hybrid -" in model_name:
            color = COLORS["accent1"]

        seeds = model_data["Seed"].values

        axes[0].plot(
            seeds,
            model_data["MAE"].values,
            marker="o",
            label=model_name,
            linewidth=1.5,
            markersize=5,
        )

        axes[1].plot(
            seeds,
            model_data["RMSE"].values,
            marker="s",
            label=model_name,
            linewidth=1.5,
            markersize=5,
        )

        axes[2].plot(
            seeds,
            model_data["R2"].values,
            marker="^",
            label=model_name,
            linewidth=1.5,
            markersize=5,
        )


    axes[0].set_xlabel("Random Seed")

    axes[0].set_ylabel("MAE")

    axes[0].set_title(
        "MAE Stability Across Seeds"
    )


    axes[1].set_xlabel("Random Seed")

    axes[1].set_ylabel("RMSE")

    axes[1].set_title(
        "RMSE Stability Across Seeds"
    )


    axes[2].set_xlabel("Random Seed")

    axes[2].set_ylabel("R²")

    axes[2].set_title(
        "R² Stability Across Seeds"
    )

    axes[2].legend(

        bbox_to_anchor=(1.05, 1),

        loc="upper left",

        fontsize=8
    )


    plt.suptitle(

        "Multi-Seed Robustness Evaluation",

        fontsize=15,

        fontweight="bold",

        y=1.02
    )


    plt.tight_layout()


    fig_file = (
        OUTPUT_DIR
        / "fig_robustness_stability.png"
    )

    plt.savefig(
        fig_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"  Saved: {fig_file}")

else:

    print(
        "  SKIPPED: "
        "robustness_results.csv not found"
    )


# ==========================================================
# FIGURE 4: MODEL COMPARISON RADAR-STYLE BAR
# ==========================================================

print()

print("=" * 70)

print("FIGURE 4: MODEL COMPARISON SUMMARY")

print("=" * 70)


eval_file = (
    PROCESSED_DIR
    / "evaluation_metrics.csv"
)

if eval_file.exists() and ranking_file.exists():

    eval_metrics = pd.read_csv(
        eval_file
    )

    ranking_data = pd.read_csv(
        ranking_file
    )

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5)
    )

    # Regression metrics.

    metrics = eval_metrics.set_index(
        "Metric"
    )["Value"].to_dict()

    metric_names = list(metrics.keys())

    metric_values = [
        float(v) for v in metrics.values()
    ]

    bar_colors = [
        COLORS["hybrid"],
        COLORS["ats"],
        COLORS["sbert"],
    ][:len(metric_names)]

    axes[0].bar(

        range(len(metric_names)),

        metric_values,

        color=bar_colors,

        alpha=0.85,

        edgecolor="white"
    )

    axes[0].set_xticks(
        range(len(metric_names))
    )

    axes[0].set_xticklabels(
        metric_names
    )

    axes[0].set_title(
        "Hybrid Model Regression Performance"
    )

    axes[0].set_ylabel("Value")

    # Add value labels.

    for i, v in enumerate(metric_values):

        axes[0].text(
            i, v + 0.01,
            f"{v:.4f}",
            ha="center",
            fontsize=9,
            fontweight="bold"
        )


    # Ranking comparison across models.

    if "Model" in ranking_data.columns:

        ndcg_cols = [
            c for c in ranking_data.columns
            if "NDCG" in c and "_mean" in c
        ]

        for _, row in (
            ranking_data.iterrows()
        ):

            model_name = row["Model"]

            values = [
                row[c] for c in ndcg_cols
            ]

            k_values = [
                c.replace(
                    "NDCG@", ""
                ).replace(
                    "_mean", ""
                )
                for c in ndcg_cols
            ]

            color = color_map.get(
                model_name,
                COLORS["neutral"]
            )

            axes[1].plot(

                k_values,

                values,

                marker="o",

                label=model_name,

                linewidth=2,

                markersize=6,

                color=color
            )

        axes[1].set_xlabel("K")

        axes[1].set_ylabel("NDCG@K")

        axes[1].set_title(
            "Ranking Quality (NDCG@K)"
        )

        axes[1].legend()


    plt.suptitle(

        "Overall Model Performance Summary",

        fontsize=15,

        fontweight="bold",

        y=1.02
    )


    plt.tight_layout()


    fig_file = (
        OUTPUT_DIR
        / "fig_model_comparison.png"
    )

    plt.savefig(
        fig_file,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(f"  Saved: {fig_file}")


else:

    print(
        "  SKIPPED: "
        "required input files not found"
    )


# ==========================================================
# TABLE: FEATURE DEFINITIONS
# ==========================================================

print()

print("=" * 70)

print("TABLE: FEATURE DEFINITIONS")

print("=" * 70)


feature_definitions = pd.DataFrame([

    {
        "No": 1,
        "Feature": "tfidf_similarity",
        "Group": "Lexical",
        "Description": (
            "Cosine similarity between "
            "TF-IDF vectors of resume "
            "and job texts"
        ),
    },

    {
        "No": 2,
        "Feature": "semantic_similarity",
        "Group": "Semantic",
        "Description": (
            "Cosine similarity between "
            "SBERT embeddings of resume "
            "and job texts"
        ),
    },

    {
        "No": 3,
        "Feature": "skill_match_score",
        "Group": "Structured ATS",
        "Description": (
            "Ratio of matched skills "
            "to required skills"
        ),
    },

    {
        "No": 4,
        "Feature": "education_match_score",
        "Group": "Structured ATS",
        "Description": (
            "Degree level match ratio "
            "(candidate vs required)"
        ),
    },

    {
        "No": 5,
        "Feature": "experience_match_score",
        "Group": "Structured ATS",
        "Description": (
            "Experience years match ratio "
            "(capped at 1.0)"
        ),
    },

    {
        "No": 6,
        "Feature": "certification_match_score",
        "Group": "Structured ATS",
        "Description": (
            "Ratio of matched certifications "
            "to required certifications"
        ),
    },

    {
        "No": 7,
        "Feature": "resume_length",
        "Group": "Text Statistics",
        "Description": (
            "Character length of "
            "preprocessed resume text"
        ),
    },

    {
        "No": 8,
        "Feature": "job_length",
        "Group": "Text Statistics",
        "Description": (
            "Character length of "
            "preprocessed job text"
        ),
    },

    {
        "No": 9,
        "Feature": "resume_word_count",
        "Group": "Text Statistics",
        "Description": (
            "Word count of "
            "preprocessed resume text"
        ),
    },

    {
        "No": 10,
        "Feature": "job_word_count",
        "Group": "Text Statistics",
        "Description": (
            "Word count of "
            "preprocessed job text"
        ),
    },

    {
        "No": 11,
        "Feature": "candidate_skill_count",
        "Group": "Structured ATS",
        "Description": (
            "Number of skills "
            "extracted from resume"
        ),
    },

    {
        "No": 12,
        "Feature": "required_skill_count",
        "Group": "Structured ATS",
        "Description": (
            "Number of skills "
            "required by job"
        ),
    },

    {
        "No": 13,
        "Feature": "skill_overlap_count",
        "Group": "Structured ATS",
        "Description": (
            "Count of skills found "
            "in both resume and job"
        ),
    },

    {
        "No": 14,
        "Feature": "education_exact_match",
        "Group": "Structured ATS",
        "Description": (
            "Binary indicator: 1 if degree "
            "names match exactly"
        ),
    },
])


feature_file = (
    OUTPUT_DIR
    / "table_feature_definitions.csv"
)


feature_definitions.to_csv(
    feature_file,
    index=False
)


print(f"  Saved: {feature_file}")


# ==========================================================
# TABLE: DATASET STATISTICS
# ==========================================================

print()

print("=" * 70)

print("TABLE: DATASET STATISTICS")

print("=" * 70)


DATASET_FILE = (
    BASE_DIR
    / "dataset"
    / "Resume_Data_For_Ranking.csv"
)


if DATASET_FILE.exists():

    full_df = pd.read_csv(
        DATASET_FILE
    )

    full_df.columns = (
        full_df.columns
        .str.strip()
        .str.lower()
    )

    full_df["matched_score"] = pd.to_numeric(
        full_df["matched_score"],
        errors="coerce"
    )

    dataset_stats = pd.DataFrame([

        {
            "Statistic": "Total samples",
            "Value": len(full_df)
        },

        {
            "Statistic": "Total columns",
            "Value": full_df.shape[1]
        },

        {
            "Statistic": "Unique job groups",
            "Value": (
                full_df["job_position_name"]
                .nunique()
            )
        },

        {
            "Statistic": "Target variable",
            "Value": "matched_score"
        },

        {
            "Statistic": "Target min",
            "Value": round(
                full_df["matched_score"]
                .min(), 4
            )
        },

        {
            "Statistic": "Target max",
            "Value": round(
                full_df["matched_score"]
                .max(), 4
            )
        },

        {
            "Statistic": "Target mean",
            "Value": round(
                full_df["matched_score"]
                .mean(), 4
            )
        },

        {
            "Statistic": "Target std",
            "Value": round(
                full_df["matched_score"]
                .std(), 4
            )
        },

        {
            "Statistic": "Target median",
            "Value": round(
                full_df["matched_score"]
                .median(), 4
            )
        },

        {
            "Statistic": "Model",
            "Value": "RandomForestRegressor"
        },

        {
            "Statistic": "Number of features",
            "Value": 14
        },

        {
            "Statistic": "Embedding model",
            "Value": "all-MiniLM-L6-v2"
        },
    ])

    stats_file = (
        OUTPUT_DIR
        / "table_dataset_statistics.csv"
    )

    dataset_stats.to_csv(
        stats_file,
        index=False
    )

    print(f"  Saved: {stats_file}")


else:

    print("  SKIPPED: dataset not found")


# ==========================================================
# FINAL SUMMARY
# ==========================================================

print()

print("=" * 70)

print("PUBLICATION FIGURES COMPLETE")

print("=" * 70)


print()

print(
    "All publication figures and tables "
    "generated in:"
)

print(OUTPUT_DIR)


print()

generated = list(
    OUTPUT_DIR.glob("*")
)

for i, f in enumerate(
    sorted(generated),
    start=1
):

    print(f"{i}. {f.name}")


print()

print("=" * 70)

print("FIGURE GENERATION FINISHED SUCCESSFULLY")

print("=" * 70)
