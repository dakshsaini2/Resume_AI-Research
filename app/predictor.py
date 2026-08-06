"""
predictor.py

Load trained model and predict
resume-job matching score.
"""

import joblib
import pandas as pd

from config import (
    MODEL_FILE,
    SCALER_FILE,
    FEATURE_COLUMNS,
    STRONG_MATCH,
    MODERATE_MATCH,
)

from feature_engineering import generate_features

# ==========================================================
# LOAD MODEL
# ==========================================================

model = joblib.load(MODEL_FILE)

scaler = joblib.load(SCALER_FILE)


# ==========================================================
# PREDICTION
# ==========================================================

def predict_resume_score(
    resume_text,
    job_text,
):

    # ---------------------------------------
    # Generate Features
    # ---------------------------------------

    features = generate_features(
        resume_text,
        job_text
    )

    # ---------------------------------------
    # Extract UI Data
    # ---------------------------------------

    candidate_skills = features["candidate_skills"]

    required_skills = features["required_skills"]

    matched_skills = features["matched_skills"]

    missing_skills = features["missing_skills"]

    candidate_degree = features["candidate_degree"]

    required_degree = features["required_degree"]

    candidate_experience = features["candidate_experience"]

    required_experience = features["required_experience"]

    candidate_certifications = features["candidate_certifications"]

    required_certifications = features["required_certifications"]

    # ---------------------------------------
    # Keep only Model Features
    # ---------------------------------------

    model_features = {

        key: features[key]

        for key in FEATURE_COLUMNS

    }

    X = pd.DataFrame(

        [model_features],

        columns=FEATURE_COLUMNS

    )

    X_scaled = scaler.transform(X)

    score = float(model.predict(X_scaled)[0])

    score = max(0.0, min(score, 1.0))

    percentage = round(score * 100, 2)

    # ---------------------------------------
    # Recommendation
    # ---------------------------------------

    if percentage >= STRONG_MATCH:

        recommendation = "🟢 Strong Match"

    elif percentage >= MODERATE_MATCH:

        recommendation = "🟡 Moderate Match"

    else:

        recommendation = "🔴 Weak Match"

    # ---------------------------------------
    # Return
    # ---------------------------------------

    return {

        "score": percentage,

        "recommendation": recommendation,

        "features": model_features,

        "candidate_skills": candidate_skills,

        "required_skills": required_skills,

        "matched_skills": matched_skills,

        "missing_skills": missing_skills,

        "candidate_degree": candidate_degree,

        "required_degree": required_degree,

        "candidate_experience": candidate_experience,

        "required_experience": required_experience,

        "candidate_certifications": candidate_certifications,

        "required_certifications": required_certifications,

    }


# ==========================================================
# DEBUG
# ==========================================================

if __name__ == "__main__":

    resume = """
    Python Developer

    Skills:
    Python
    SQL
    GitHub
    TensorFlow

    Experience:
    2 years

    Education:
    Bachelor of Technology

    Certification:
    AWS
    """

    job = """
    Python Developer

    Skills Required:
    Python
    SQL
    Git
    Docker

    Experience:
    2 years

    Education:
    B.Tech

    Preferred Certification:
    AWS
    """

    result = predict_resume_score(
        resume,
        job
    )

    from pprint import pprint

    pprint(result)