"""
==========================================================
Resume AI Research - FastAPI Backend
==========================================================

API:

POST /api/v1/analyze

Input:
    resume PDF/DOCX
    job description

Output:
    resume-job match score
    category
    matched skills
    missing skills
    education analysis
    experience analysis
    certification analysis
    TF-IDF similarity
    SBERT similarity

==========================================================
"""

from pathlib import Path
import sys
import tempfile
import shutil
import os

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
)

from fastapi.middleware.cors import CORSMiddleware


# ==========================================================
# PATH SETUP
# ==========================================================

CURRENT_DIR = Path(__file__).resolve().parent

APP_DIR = CURRENT_DIR.parent

BASE_DIR = APP_DIR.parent


if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))


# ==========================================================
# IMPORT PREDICTOR
# ==========================================================

from predictor import predict_resume_score

from utils import load_resume


# ==========================================================
# FASTAPI APPLICATION
# ==========================================================

app = FastAPI(
    title="Resume AI Research API",
    description=(
        "Hybrid Resume-Job Matching API "
        "using TF-IDF, SBERT and structured ATS features."
    ),
    version="1.0.0",
)


# ==========================================================
# CORS
# ==========================================================


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================================
# HEALTH CHECK
# ==========================================================

@app.get("/")
def root():

    return {
        "application": "Resume AI Research",
        "status": "running",
        "version": "1.0.0",
    }


@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model": "RandomForestRegressor",
        "features": 14,
        "embedding_model": "all-MiniLM-L6-v2",
    }


# ==========================================================
# ANALYZE RESUME
# ==========================================================

@app.post("/api/v1/analyze")
async def analyze_resume(

    resume: UploadFile = File(...),

    job_description: str = Form(...),

):
    """
    Analyze an uploaded resume against a job description.
    """

    # ------------------------------------------------------
    # Validate filename
    # ------------------------------------------------------

    if not resume.filename:

        raise HTTPException(
            status_code=400,
            detail="Resume filename is missing.",
        )


    extension = Path(
        resume.filename
    ).suffix.lower()


    allowed_extensions = {
        ".pdf",
        ".docx",
        ".doc",
    }


    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported resume format. "
                "Use PDF, DOCX or DOC."
            ),
        )


    # ------------------------------------------------------
    # Validate job description
    # ------------------------------------------------------

    job_description = (
        job_description.strip()
    )


    if len(job_description) < 20:

        raise HTTPException(
            status_code=400,
            detail=(
                "Job description must contain "
                "at least 20 characters."
            ),
        )


    temporary_path = None


    try:

        # --------------------------------------------------
        # Save uploaded file temporarily
        # --------------------------------------------------

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:

            temporary_path = Path(
                temp_file.name
            )

            shutil.copyfileobj(
                resume.file,
                temp_file,
            )


        # --------------------------------------------------
        # Extract resume text
        # --------------------------------------------------

        resume_text = load_resume(
        temporary_path
    )


        if not resume_text:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not extract text "
                    "from the uploaded resume."
                ),
            )


        resume_text = str(
            resume_text
        ).strip()


        if len(resume_text) < 20:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Very little text was extracted "
                    "from the resume. "
                    "The file may be scanned/image-based."
                ),
            )


        # --------------------------------------------------
        # MODEL INFERENCE
        # --------------------------------------------------

        result = predict_resume_score(

            resume_text,

            job_description,

        )


        # --------------------------------------------------
        # Remove internal feature vector
        # --------------------------------------------------

        result.pop(
            "features",
            None,
        )


        # --------------------------------------------------
        # Add metadata
        # --------------------------------------------------

        result["resume_filename"] = (
            resume.filename
        )

        result["resume_characters"] = (
            len(resume_text)
        )

        result["resume_words"] = (
            len(resume_text.split())
        )


        return {
            "success": True,
            "data": result,
        }


    except HTTPException:

        raise


    except Exception as error:

        print(
            "Inference error:",
            repr(error),
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Resume analysis failed: "
                f"{str(error)}"
            ),

        )


    finally:

        # --------------------------------------------------
        # Cleanup temporary file
        # --------------------------------------------------

        if temporary_path is not None:

            try:

                if temporary_path.exists():

                    temporary_path.unlink()

            except Exception:

                pass


# ==========================================================
# EXPLAIN ENDPOINT — "Why this score?"
# ==========================================================

# Lazy-loaded explainer to avoid startup cost
# unless the endpoint is actually called.

_shap_explainer = None


def _get_shap_explainer():
    """Load SHAP TreeExplainer on first call."""

    global _shap_explainer

    if _shap_explainer is None:

        import shap

        from predictor import model as rf_model

        _shap_explainer = shap.TreeExplainer(
            rf_model
        )

    return _shap_explainer


# Human-readable feature names for explanations.

FEATURE_DISPLAY_NAMES = {
    "tfidf_similarity":
        "Keyword Overlap (TF-IDF)",
    "semantic_similarity":
        "Meaning Similarity (SBERT)",
    "skill_match_score":
        "Skill Match Ratio",
    "education_match_score":
        "Education Match",
    "experience_match_score":
        "Experience Match",
    "certification_match_score":
        "Certification Match",
    "resume_length":
        "Resume Text Length",
    "job_length":
        "Job Description Length",
    "resume_word_count":
        "Resume Word Count",
    "job_word_count":
        "Job Word Count",
    "candidate_skill_count":
        "Candidate Skills Count",
    "required_skill_count":
        "Required Skills Count",
    "skill_overlap_count":
        "Overlapping Skills Count",
    "education_exact_match":
        "Exact Degree Match",
}


@app.post("/api/v1/explain")
async def explain_resume(

    resume: UploadFile = File(...),

    job_description: str = Form(...),

):
    """
    Analyze a resume and return a 'Why this score?'
    explanation with per-feature SHAP contributions.
    """

    import numpy as np
    import pandas as pd

    from predictor import (
        model as rf_model,
        scaler as rf_scaler,
    )

    from config import FEATURE_COLUMNS

    from feature_engineering import generate_features

    # --------------------------------------------------
    # Validate inputs (same as /analyze)
    # --------------------------------------------------

    if not resume.filename:

        raise HTTPException(
            status_code=400,
            detail="Resume filename is missing.",
        )

    extension = Path(
        resume.filename
    ).suffix.lower()

    allowed_extensions = {
        ".pdf",
        ".docx",
        ".doc",
    }

    if extension not in allowed_extensions:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported resume format. "
                "Use PDF, DOCX or DOC."
            ),
        )

    job_description = (
        job_description.strip()
    )

    if len(job_description) < 20:

        raise HTTPException(
            status_code=400,
            detail=(
                "Job description must contain "
                "at least 20 characters."
            ),
        )

    temporary_path = None

    try:

        # Save uploaded file temporarily.

        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=extension,
        ) as temp_file:

            temporary_path = Path(
                temp_file.name
            )

            shutil.copyfileobj(
                resume.file,
                temp_file,
            )

        # Extract resume text.

        resume_text = load_resume(
            temporary_path
        )

        if not resume_text:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Could not extract text "
                    "from the uploaded resume."
                ),
            )

        resume_text = str(
            resume_text
        ).strip()

        if len(resume_text) < 20:

            raise HTTPException(
                status_code=400,
                detail=(
                    "Very little text was extracted "
                    "from the resume."
                ),
            )

        # --------------------------------------------------
        # Generate features and prediction.
        # --------------------------------------------------

        features = generate_features(
            resume_text,
            job_description,
        )

        model_features = {
            key: features[key]
            for key in FEATURE_COLUMNS
        }

        X = pd.DataFrame(
            [model_features],
            columns=FEATURE_COLUMNS
        )

        X_scaled = rf_scaler.transform(X)

        prediction = float(
            rf_model.predict(X_scaled)[0]
        )

        prediction = max(
            0.0,
            min(prediction, 1.0)
        )

        # --------------------------------------------------
        # Compute SHAP explanation.
        # --------------------------------------------------

        explainer = _get_shap_explainer()

        shap_values = explainer.shap_values(
            X_scaled
        )

        if isinstance(shap_values, list):
            shap_values = shap_values[0]

        shap_values = np.asarray(
            shap_values
        ).flatten()

        expected_value = explainer.expected_value

        if isinstance(
            expected_value,
            (list, np.ndarray)
        ):
            expected_value = float(
                expected_value[0]
            )

        # Build explanation response.

        contributions = []

        for i, feature in enumerate(
            FEATURE_COLUMNS
        ):

            display_name = (
                FEATURE_DISPLAY_NAMES.get(
                    feature,
                    feature
                )
            )

            shap_val = float(shap_values[i])

            contributions.append({

                "feature": feature,

                "display_name": display_name,

                "feature_value": round(
                    float(
                        model_features[feature]
                    ),
                    4
                ),

                "shap_value": round(
                    shap_val,
                    6
                ),

                "direction": (
                    "positive"
                    if shap_val > 0
                    else "negative"
                    if shap_val < 0
                    else "neutral"
                ),
            })

        # Sort by absolute impact.

        contributions.sort(

            key=lambda x: abs(
                x["shap_value"]
            ),

            reverse=True
        )

        # Top positive and negative factors.

        positive_factors = [
            c for c in contributions
            if c["direction"] == "positive"
        ][:5]

        negative_factors = [
            c for c in contributions
            if c["direction"] == "negative"
        ][:5]

        return {

            "success": True,

            "data": {

                "predicted_score": round(
                    prediction * 100,
                    2
                ),

                "base_score": round(
                    expected_value * 100,
                    2
                ),

                "all_contributions":
                    contributions,

                "top_positive_factors":
                    positive_factors,

                "top_negative_factors":
                    negative_factors,

                "explanation_summary": (
                    f"The model predicted a "
                    f"{prediction * 100:.1f}% "
                    f"match score. The base "
                    f"expectation was "
                    f"{expected_value * 100:.1f}%. "
                    f"The strongest positive "
                    f"factor was "
                    f"'{positive_factors[0]['display_name']}' "
                    f"and the strongest negative "
                    f"factor was "
                    f"'{negative_factors[0]['display_name']}'."
                    if positive_factors
                    and negative_factors
                    else
                    f"The model predicted a "
                    f"{prediction * 100:.1f}% "
                    f"match score."
                ),
            },
        }


    except HTTPException:

        raise


    except Exception as error:

        print(
            "Explanation error:",
            repr(error),
        )

        raise HTTPException(

            status_code=500,

            detail=(
                "Score explanation failed: "
                f"{str(error)}"
            ),
        )


    finally:

        if temporary_path is not None:

            try:

                if temporary_path.exists():

                    temporary_path.unlink()

            except Exception:

                pass


# ==========================================================
# SERVER ENTRY POINT
# ==========================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(

        "app.api.main:app",

        host="127.0.0.1",

        port=8000,

        reload=True,

    )