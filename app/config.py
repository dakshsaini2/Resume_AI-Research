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
# Recommendation Thresholds
# ==========================================================
# Recommendation Thresholds
# ==========================================================

EXCELLENT_MATCH = 90
STRONG_MATCH = 75
MODERATE_MATCH = 60
# ==========================================================
# DEGREE NORMALIZATION & HIERARCHY
# ==========================================================

DEGREE_MAP = {
    # Bachelor's Engineering / Technology
    "b.tech": "btech",
    "b tech": "btech",
    "btech": "btech",
    "b.e": "btech",
    "be": "btech",
    "bachelor of technology": "btech",
    "bachelor of engineering": "btech",

    # Master's Engineering / Technology
    "m.tech": "mtech",
    "m tech": "mtech",
    "mtech": "mtech",
    "m.e": "mtech",
    "me": "mtech",
    "master of technology": "mtech",
    "master of engineering": "mtech",

    # Computer Applications
    "bca": "bca",
    "b.c.a": "bca",
    "bachelor of computer applications": "bca",
    "mca": "mca",
    "m.c.a": "mca",
    "master of computer applications": "mca",

    # Science
    "b.sc": "bsc",
    "bsc": "bsc",
    "b.s": "bsc",
    "bs": "bsc",
    "bachelor of science": "bsc",

    "m.sc": "msc",
    "msc": "msc",
    "m.s": "msc",
    "ms": "msc",
    "master of science": "msc",

    # Arts & Business
    "b.a": "ba",
    "ba": "ba",
    "bachelor of arts": "ba",
    "m.a": "ma",
    "ma": "ma",
    "master of arts": "ma",
    "mba": "mba",
    "master of business administration": "mba",

    # General / Doctorate
    "bachelor": "btech",
    "bachelors": "btech",
    "master": "mtech",
    "masters": "mtech",
    "degree": "btech",
    "diploma": "diploma",
    "phd": "phd",
    "ph.d": "phd",
    "doctorate": "phd"
}

DEGREE_LEVELS = {
    "not specified": 0,
    "diploma": 1,
    "ba": 1.5,
    "ma": 2.5,
    "bsc": 2,
    "bca": 2,
    "btech": 2,
    "msc": 3,
    "mca": 3,
    "mtech": 3,
    "mba": 3,
    "phd": 4
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