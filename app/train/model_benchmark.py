"""
model_benchmark.py

Model Benchmark for Resume AI Optimization Evaluator
"""
import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, ndcg_score
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    VotingRegressor,
    StackingRegressor
)
from sklearn.linear_model import RidgeCV

from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parents[2]
DATASET_FILE = BASE_DIR / "dataset" / "processed" / "final_dataset_with_features.csv"
OUTPUT_DIR = BASE_DIR / "dataset" / "processed" / "model_benchmark"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "tfidf_similarity",
    "semantic_similarity",
    "skill_match_score",
    "education_match_score",
    "experience_match_score",
    "certification_match_score",
    "candidate_skill_count",
    "required_skill_count",
    "skill_overlap_count",
    "education_exact_match",
    "resume_length",
    "job_length",
    "resume_word_count",
    "job_word_count",
]

TARGET_COLUMN = "matched_score"

def compute_ranking_metrics(df_group, actual_col, predicted_col):
    actual = df_group[actual_col].values
    predicted = df_group[predicted_col].values
    n = len(actual)
    if n < 2: return None
    
    ndcg_5 = ndcg_score([actual], [predicted], k=min(5, n))
    
    threshold = np.percentile(actual, 75)
    relevant = actual >= threshold
    ranked_indices = np.argsort(-predicted)
    ranked_relevant = relevant[ranked_indices]
    
    mrr = 0.0
    for i, is_relevant in enumerate(ranked_relevant):
        if is_relevant:
            mrr = 1.0 / (i + 1)
            break
            
    return {"NDCG@5": ndcg_5, "MRR": mrr}

def main():
    print("Loading precomputed dataset...")
    df = pd.read_csv(DATASET_FILE)
    
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN] + FEATURE_COLUMNS).copy()
    
    df["job_group"] = df["job_position_name"].str.strip().str.lower()
    
    group_split = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=RANDOM_STATE)
    train_idx, test_idx = next(group_split.split(df, groups=df["job_group"]))
    
    X_train = df.iloc[train_idx][FEATURE_COLUMNS]
    y_train = df.iloc[train_idx][TARGET_COLUMN]
    X_test = df.iloc[test_idx][FEATURE_COLUMNS]
    y_test = df.iloc[test_idx][TARGET_COLUMN]
    test_df = df.iloc[test_idx].copy()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples:  {len(X_test)}")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    rf = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    xgb = XGBRegressor(n_estimators=300, learning_rate=0.05, random_state=RANDOM_STATE, n_jobs=-1)
    et = ExtraTreesRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    hgb = HistGradientBoostingRegressor(max_iter=300, random_state=RANDOM_STATE)
    
    voting = VotingRegressor([
        ('rf', rf),
        ('xgb', xgb),
        ('et', et),
        ('hgb', hgb)
    ])
    
    stacking = StackingRegressor(
        estimators=[
            ('rf', rf),
            ('xgb', xgb),
            ('et', et),
            ('hgb', hgb)
        ],
        final_estimator=RidgeCV(),
        n_jobs=-1
    )
    
    models = {
        "Random Forest": rf,
        "XGBoost": xgb,
        "Extra Trees": et,
        "HistGradientBoosting": hgb,
        "Voting Regressor": voting,
        "Stacking Regressor": stacking
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        model.fit(X_train_scaled, y_train)
        
        preds = model.predict(X_test_scaled)
        
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        test_df["pred"] = preds
        
        group_metrics = []
        for _, group in test_df.groupby("job_group"):
            metrics = compute_ranking_metrics(group, TARGET_COLUMN, "pred")
            if metrics:
                group_metrics.append(metrics)
                
        metrics_df = pd.DataFrame(group_metrics)
        mean_ndcg = metrics_df["NDCG@5"].mean()
        mean_mrr = metrics_df["MRR"].mean()
        
        results.append({
            "Model": name,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "NDCG@5": mean_ndcg,
            "MRR": mean_mrr
        })
        print(f"MAE: {mae:.4f} | RMSE: {rmse:.4f} | R2: {r2:.4f} | NDCG@5: {mean_ndcg:.4f} | MRR: {mean_mrr:.4f}")
        
    results_df = pd.DataFrame(results)
    print("\n\n======================================================================")
    print("FINAL BENCHMARK RESULTS")
    print("======================================================================")
    print(results_df.to_string(index=False))
    
    results_df.to_csv(OUTPUT_DIR / "model_benchmark_results.csv", index=False)
    print(f"\nSaved results to {OUTPUT_DIR / 'model_benchmark_results.csv'}")
    
if __name__ == "__main__":
    main()
