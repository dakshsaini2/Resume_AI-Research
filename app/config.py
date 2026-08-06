"""
config.py

Configuration file for Resume Screening System
"""

from pathlib import Path

# ==========================================================
# PROJECT PATHS
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"

DATASET_DIR = BASE_DIR / "dataset"

# ==========================================================
# MODEL FILES
# ==========================================================

MODEL_FILE = MODEL_DIR / "best_resume_model.pkl"

SCALER_FILE = MODEL_DIR / "scaler.pkl"

TFIDF_FILE = MODEL_DIR / "tfidf_vectorizer.pkl"

# ==========================================================
# SENTENCE TRANSFORMER
# ==========================================================

SBERT_MODEL = "all-MiniLM-L6-v2"

# ==========================================================
# FEATURE ORDER
# MUST MATCH TRAINING NOTEBOOK
# ==========================================================

FEATURE_COLUMNS = [
    "tfidf_similarity",
    "semantic_similarity",
    "skill_match_score",
    "education_match_score",
    "experience_match_score",
    "certification_match_score",
    "resume_length",
    "job_length",
    "resume_word_count",
    "job_word_count",
    "candidate_skill_count",
    "required_skill_count",
    "skill_overlap_count",
    "education_exact_match",
]

# ==========================================================
# SCORE THRESHOLDS
# ==========================================================

STRONG_MATCH = 80

MODERATE_MATCH = 60

# ==========================================================
# DEGREE NORMALIZATION
# ==========================================================

DEGREE_MAP = {

    "b.tech": "btech",
    "b tech": "btech",
    "btech": "btech",
    "b.e": "btech",
    "be": "btech",
    "bachelor of technology": "btech",

    "m.tech": "mtech",
    "m tech": "mtech",
    "mtech": "mtech",
    "master of technology": "mtech",

    "bca": "bca",
    "mca": "mca",

    "b.sc": "bsc",
    "bsc": "bsc",
    "bachelor of science": "bsc",

    "m.sc": "msc",
    "msc": "msc",

    "mba": "mba",

    "phd": "phd"
}

# ==========================================================
# SKILL SYNONYMS
# ==========================================================

SKILL_SYNONYMS = {

    "github": "git",
    "gitlab": "git",

    "mysql": "sql",
    "postgresql": "sql",
    "sqlite": "sql",

    "tensorflow": "machine learning",
    "keras": "machine learning",
    "pytorch": "machine learning",
    "scikit-learn": "machine learning",

    "react.js": "react",
    "node.js": "node",

    "artificial intelligence": "machine learning"
}