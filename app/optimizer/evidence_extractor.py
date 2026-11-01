"""
evidence_extractor.py

Extracts a strictly factual, structured inventory of a candidate's 
skills, education, experience, certifications, projects, employers, 
dates, and achievements from raw resume text using schema-constrained LLM inference.
"""
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
import json
import logging

try:
    from app.optimizer.gemini_client import call_gemini_with_retry, GeminiAPIException
except ImportError:
    from gemini_client import call_gemini_with_retry, GeminiAPIException

logger = logging.getLogger("evidence_extractor")


# ==========================================================
# PYDANTIC SCHEMAS FOR STRUCTURED EXTRACTION
# ==========================================================

class ResumeEvidence(BaseModel):
    skills: List[str] = Field(
        default_factory=list,
        description="Exhaustive list of all technical and soft skills explicitly mentioned in the resume"
    )
    education: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of all educational degrees and institutions. Keys: 'institution', 'degree', 'major', 'dates'"
    )
    experience: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of all work experiences. Keys: 'employer', 'position', 'dates', 'achievements'"
    )
    projects: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of all specific projects. Keys: 'name', 'description', 'skills_used'"
    )
    certifications: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of all certifications. Keys: 'name', 'provider', 'date'"
    )


# ==========================================================
# EXTRACTOR FUNCTION
# ==========================================================

def extract_evidence(resume_text: str, api_key: str, max_retries: int = 5) -> ResumeEvidence:
    """
    Extracts a factual evidence inventory from resume text using schema-constrained Gemini inference
    with bounded retries and exponential backoff for transient errors.
    """
    if not api_key:
        raise ValueError("A Google API key is required for evidence extraction.")

    prompt = (
        "You are a strict factual extraction engine. Your job is to extract an evidence inventory from the provided resume.\n"
        "You MUST NOT invent, infer, or hallucinate any skills, experiences, dates, or achievements that are not explicitly "
        "written in the text. If information is missing, leave the field as an empty list. "
        "Precision and factuality are your highest priorities.\n\n"
        "Return a valid JSON object matching the following schema:\n"
        "{\n"
        '  "skills": ["skill1", "skill2"],\n'
        '  "education": [{"institution": "...", "degree": "...", "major": "...", "dates": "..."}],\n'
        '  "experience": [{"employer": "...", "position": "...", "dates": "...", "achievements": ["..."]}],\n'
        '  "projects": [{"name": "...", "description": "...", "skills_used": ["..."]}],\n'
        '  "certifications": [{"name": "...", "provider": "...", "date": "..."}]\n'
        "}\n\n"
        f"--- RESUME TEXT ---\n{resume_text}"
    )

    parsed_dict, telemetry = call_gemini_with_retry(
        prompt=prompt,
        api_key=api_key,
        schema_model=ResumeEvidence,
        temperature=0.0,
        max_retries=max_retries,
        max_output_tokens=4096,
    )

    return ResumeEvidence(**parsed_dict)


if __name__ == "__main__":
    import os
    import sys
    from dotenv import load_dotenv
    load_dotenv()

    key = os.environ.get("GOOGLE_API_KEY")
    if not key:
        print("GOOGLE_API_KEY not set.")
        sys.exit(0)

    sample_resume = '''
    Daksh Saini
    Software Engineer
    
    Skills: Python, React, Machine Learning, SQL
    
    Experience:
    Google - Software Engineer (Jan 2021 - Present)
    - Developed a resume parsing system using Python and LLMs.
    - Improved processing speed by 40%.
    
    Education:
    B.Tech in Computer Science, MIT, 2020
    
    Certifications:
    AWS Certified Solutions Architect
    '''
    print("Testing Evidence Extraction...")
    evidence = extract_evidence(sample_resume, key)
    print(json.dumps(evidence.dict(), indent=2))
