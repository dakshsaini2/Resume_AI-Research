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
    EXCELLENT_MATCH,
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

    # ======================================================
    # GENERATE FEATURES
    # ======================================================

    features = generate_features(
        resume_text,
        job_text
    )


    # ======================================================
    # EXTRACT UI DATA
    # ======================================================

    candidate_skills = features["candidate_skills"]

    required_skills = features["required_skills"]

    matched_skills = features["matched_skills"]

    missing_skills = features["missing_skills"]


    candidate_degree = features["candidate_degree"]

    required_degree = features["required_degree"]


    candidate_experience = features["candidate_experience"]

    required_experience = features["required_experience"]


    candidate_certifications = (
        features["candidate_certifications"]
    )

    required_certifications = (
        features["required_certifications"]
    )


    # ======================================================
    # EXTRACT MATCH SCORES
    # ======================================================

    skill_match_score = float(
        features["skill_match_score"]
    )

    education_match_score = float(
        features["education_match_score"]
    )

    experience_match_score = float(
        features["experience_match_score"]
    )

    certification_match_score = float(
        features["certification_match_score"]
    )

    tfidf_similarity = float(
        features["tfidf_similarity"]
    )

    semantic_similarity = float(
        features["semantic_similarity"]
    )


    # ======================================================
    # KEEP ONLY MODEL FEATURES
    # ======================================================

    model_features = {
        key: features[key]
        for key in FEATURE_COLUMNS
    }


    # ======================================================
    # CREATE MODEL DATAFRAME
    # ======================================================

    X = pd.DataFrame(
        [model_features],
        columns=FEATURE_COLUMNS
    )


    # ======================================================
    # SCALE FEATURES
    # ======================================================

    X_scaled = scaler.transform(X)


    # ======================================================
    # RANDOM FOREST PREDICTION
    # ======================================================

    model_score = float(
        model.predict(X_scaled)[0]
    )


    # ======================================================
    # CLAMP MODEL SCORE
    # ======================================================

    model_score = max(
        0.0,
        min(model_score, 1.0)
    )


    model_percentage = round(
        model_score * 100,
        2
    )


    # ======================================================
    # RULE-BASED ATS ANALYSIS
    # ======================================================
    #
    # This does NOT replace the trained model.
    #
    # It provides transparent recruiter-facing signals.
    #
    # ======================================================

    if required_experience <= 0:

        experience_status = "Not Required"

    elif candidate_experience >= required_experience:

        experience_status = "Meets Requirement"

    else:

        experience_status = "Below Requirement"


    cand_deg_lower = candidate_degree.strip().lower()
    req_deg_lower = required_degree.strip().lower()

    if req_deg_lower == "not specified" or not req_deg_lower:
        education_status = "Not Specified"
    elif cand_deg_lower == "not specified" or not cand_deg_lower:
        education_status = "Not Provided"
    elif education_match_score >= 1.0:
        education_status = "Meets Requirement"
    else:
        education_status = "Below Requirement"


    if required_certifications:

        certification_status = (
            "Meets Requirement"
            if certification_match_score >= 1.0
            else "Partial Match"
        )

    else:

        certification_status = "Not Required"


    if required_skills:

        skill_status = (
            "Strong"
            if skill_match_score >= 0.75
            else
            "Moderate"
            if skill_match_score >= 0.50
            else
            "Weak"
        )

    else:

        skill_status = "Not Specified"


    # ======================================================
    # FINAL RECOMMENDATION
    # ======================================================
    #
    # IMPORTANT:
    #
    # The final ATS percentage remains the trained
    # Random Forest prediction.
    #
    # The additional rule-based signals are returned
    # separately so we don't silently change the
    # behavior of your trained model.
    #
    # ======================================================

    percentage = model_percentage


    if percentage >= EXCELLENT_MATCH:

        recommendation = "🌟 Excellent Match"

    elif percentage >= STRONG_MATCH:

        recommendation = "🟢 Strong Match"

    elif percentage >= MODERATE_MATCH:

        recommendation = "🟡 Moderate Match"

    else:

        recommendation = "🔴 Weak Match"


    # ======================================================
    # RETURN RESULT
    # ======================================================

    return {

        # --------------------------------------------------
        # FINAL MODEL SCORE
        # --------------------------------------------------

        "score": percentage,

        "model_score": model_percentage,

        "recommendation": recommendation,


        # --------------------------------------------------
        # MODEL FEATURES
        # --------------------------------------------------

        "features": model_features,


        # --------------------------------------------------
        # SKILLS
        # --------------------------------------------------

        "candidate_skills": candidate_skills,

        "required_skills": required_skills,

        "matched_skills": matched_skills,

        "missing_skills": missing_skills,


        # --------------------------------------------------
        # EDUCATION
        # --------------------------------------------------

        "candidate_degree": candidate_degree,

        "required_degree": required_degree,

        "education_match_score": (
            education_match_score
        ),

        "education_status": education_status,


        # --------------------------------------------------
        # EXPERIENCE
        # --------------------------------------------------

        "candidate_experience": candidate_experience,

        "required_experience": required_experience,

        "experience_match_score": (
            experience_match_score
        ),

        "experience_status": experience_status,


        # --------------------------------------------------
        # CERTIFICATIONS
        # --------------------------------------------------

        "candidate_certifications": (
            candidate_certifications
        ),

        "required_certifications": (
            required_certifications
        ),

        "certification_match_score": (
            certification_match_score
        ),

        "certification_status": (
            certification_status
        ),


        # --------------------------------------------------
        # SKILL SCORE
        # --------------------------------------------------

        "skill_match_score": skill_match_score,

        "skill_status": skill_status,


        # --------------------------------------------------
        # NLP SCORES
        # --------------------------------------------------

        "tfidf_similarity": tfidf_similarity,

        "semantic_similarity": (
            semantic_similarity
        ),
    }


# ==========================================================
# DEBUG / TEST
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