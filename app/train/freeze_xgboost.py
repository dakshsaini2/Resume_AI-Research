"""
freeze_xgboost.py

Retrains XGBoost on the entire dataset using the 14 features
and saves the model and scaler.
"""
import sys
from pathlib import Path
import warnings
import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

BASE_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BASE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from config import FEATURE_COLUMNS

DATASET_FILE = BASE_DIR / "dataset" / "processed" / "final_dataset_with_features.csv"
MODEL_DIR = BASE_DIR / "models"

TARGET_COLUMN = "matched_score"

def main():
    print("Loading dataset...")
    df = pd.read_csv(DATASET_FILE)
    df[TARGET_COLUMN] = pd.to_numeric(df[TARGET_COLUMN], errors="coerce")
    df = df.dropna(subset=[TARGET_COLUMN] + FEATURE_COLUMNS).copy()
    
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    
    print(f"Training on {len(X)} samples with {len(FEATURE_COLUMNS)} features...")
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_scaled, y)
    
    print("Saving XGBoost model...")
    joblib.dump(model, MODEL_DIR / "best_resume_model.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    
    print(f"Saved to {MODEL_DIR / 'best_resume_model.pkl'}!")

if __name__ == "__main__":
    main()
