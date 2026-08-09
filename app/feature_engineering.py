"""
feature_engineering.py

Generate features for the Resume Screening System.
"""

from pathlib import Path
import joblib
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import (
    preprocess_resume,
    preprocess_job
)

from utils import (
    extract_skills,
    extract_degree,
    extract_experience,
    extract_certifications,
    calculate_skill_overlap,
    calculate_missing_skills,
    skill_match_score,
)

from config import (
    MODEL_DIR,
    SBERT_MODEL,
)

# ==========================================================
# LOAD TF-IDF
# ==========================================================

tfidf = joblib.load(
    MODEL_DIR / "tfidf_vectorizer.pkl"
)

# ==========================================================
# LOAD SBERT
# ==========================================================

sentence_model = SentenceTransformer(
    SBERT_MODEL
)

# ==========================================================
# TF-IDF SIMILARITY
# ==========================================================

def compute_tfidf_similarity(
    resume_text,
    job_text,
):

    vectors = tfidf.transform(
        [
            resume_text,
            job_text,
        ]
    )

    similarity = cosine_similarity(
        vectors[0],
        vectors[1]
    )[0][0]

    return float(similarity)


# ==========================================================
# SEMANTIC SIMILARITY
# ==========================================================

def compute_semantic_similarity(
    resume_text,
    job_text,
):

    embeddings = sentence_model.encode(

        [
            resume_text,
            job_text,
        ],

        convert_to_numpy=True

    )

    similarity = cosine_similarity(

        embeddings[0].reshape(1, -1),

        embeddings[1].reshape(1, -1),

    )[0][0]

    return float(similarity)
# ==========================================================
# FEATURE GENERATION
# ==========================================================

def generate_features(
    resume_text,
    job_text
):

    # ---------------------------------------
    # PREPROCESS
    # ---------------------------------------

    resume = preprocess_resume(resume_text)

    job = preprocess_job(job_text)

    resume_clean = resume["processed_text"]

    job_clean = job["processed_text"]

    # ---------------------------------------
    # SKILLS
    # ---------------------------------------

    candidate_skills = extract_skills(
        resume_text
    )

    required_skills = extract_skills(
        job_text
    )

    matched_skills = calculate_skill_overlap(

        candidate_skills,

        required_skills

    )

    missing_skills = calculate_missing_skills(

        candidate_skills,

        required_skills

    )

    skill_score = skill_match_score(

        candidate_skills,

        required_skills

    )

    # ---------------------------------------
    # EDUCATION
    # ---------------------------------------

    candidate_degree = extract_degree(
        resume_text
    )

    required_degree = extract_degree(
        job_text
    )

    education_exact_match = int(
        candidate_degree.strip().lower()
        ==
        required_degree.strip().lower()
    )

    education_match_score = float(
        education_exact_match
    )


    # ---------------------------------------
    # EXPERIENCE
    # ---------------------------------------

    candidate_exp = extract_experience(
        resume_text
    )

    required_exp = extract_experience(
        job_text
    )

    if required_exp <= 0:

        experience_match_score = 1.0

    elif candidate_exp <= 0:

        experience_match_score = 0.0

    else:

        experience_match_score = min(
            candidate_exp / required_exp,
            1.0
        )

            # ---------------------------------------
    # CERTIFICATIONS
    # ---------------------------------------

    candidate_certifications = extract_certifications(
        resume_text
    )

    required_certifications = extract_certifications(
        job_text
    )

    if len(required_certifications) == 0:

        certification_match_score = 1.0

    else:

        certification_match_score = (
            len(
                set(candidate_certifications)
                &
                set(required_certifications)
            )
            /
            len(required_certifications)
        )

    # ---------------------------------------
    # TF-IDF SIMILARITY
    # ---------------------------------------

    tfidf_similarity = compute_tfidf_similarity(

        resume_clean,

        job_clean

    )

    # ---------------------------------------
    # SEMANTIC SIMILARITY
    # ---------------------------------------

    semantic_similarity = compute_semantic_similarity(

        resume_clean,

        job_clean

    )

    # ---------------------------------------
    # TEXT STATISTICS
    # ---------------------------------------

    resume_length = len(resume_clean)

    job_length = len(job_clean)

    resume_word_count = len(
        resume_clean.split()
    )

    job_word_count = len(
        job_clean.split()
    )

    candidate_skill_count = len(
        candidate_skills
    )

    required_skill_count = len(
        required_skills
    )

    skill_overlap_count = len(
        matched_skills
    )
        # ---------------------------------------
    # FINAL FEATURE DICTIONARY
    # ---------------------------------------

    features = {

        # Model Features
        "tfidf_similarity": tfidf_similarity,

        "semantic_similarity": semantic_similarity,

        "skill_match_score": skill_score,

        "education_match_score": education_match_score,

        "experience_match_score": experience_match_score,

        "certification_match_score": certification_match_score,

        "resume_length": resume_length,

        "job_length": job_length,

        "resume_word_count": resume_word_count,

        "job_word_count": job_word_count,

        "candidate_skill_count": candidate_skill_count,

        "required_skill_count": required_skill_count,

        "skill_overlap_count": skill_overlap_count,

        "education_exact_match": education_exact_match,

        # -----------------------------
        # UI Features (Not used by model)
        # -----------------------------

        "candidate_skills": candidate_skills,

        "required_skills": required_skills,

        "matched_skills": matched_skills,

        "missing_skills": missing_skills,

        "candidate_degree": candidate_degree,

        "required_degree": required_degree,

        "candidate_experience": candidate_exp,

        "required_experience": required_exp,

        "candidate_certifications": candidate_certifications,

        "required_certifications": required_certifications,
    }

    return features


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
    Docker

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
    Docker
    Git

    Experience:
    2 years

    Education:
    B.Tech

    Preferred Certification:
    AWS
    """

    features = generate_features(
        resume,
        job
    )

    from pprint import pprint

    pprint(features)