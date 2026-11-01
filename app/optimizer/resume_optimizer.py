"""
resume_optimizer.py

JD-aware rewriting and reorganization of a resume, supporting:
1. Evidence-Preserving Optimization: Heavily constrained by the original evidence inventory
   to prevent hallucinations while tailoring structure and relevance.
2. Unconstrained Score-Oriented Baseline: Optimizes purely for matching score without
   strict evidence constraints, serving as the empirical baseline for hallucination testing.
"""

from typing import Optional
from pydantic import BaseModel, Field
import json
import logging

try:
    from app.optimizer.gemini_client import call_gemini_with_retry, GeminiAPIException
except ImportError:
    from gemini_client import call_gemini_with_retry, GeminiAPIException

logger = logging.getLogger("resume_optimizer")


# ==========================================================
# PYDANTIC SCHEMA FOR OPTIMIZED RESUME
# ==========================================================

class OptimizedResume(BaseModel):
    summary: str = Field(
        default="",
        description="A professional summary tailored to the job description."
    )
    skills: str = Field(
        default="",
        description="A comma-separated list of skills reordered to prioritize those relevant to the JD."
    )
    experience: str = Field(
        default="",
        description="Work experience section rewritten to emphasize achievements relevant to the JD."
    )
    education: str = Field(
        default="",
        description="Education section detailing institution, degree, and coursework."
    )
    certifications_and_projects: str = Field(
        default="",
        description="Certifications and projects relevant to the JD."
    )


# ==========================================================
# OPTIMIZER FUNCTIONS
# ==========================================================

def optimize_resume(
    evidence_json: str,
    job_description: str,
    api_key: str,
    max_retries: int = 5
) -> OptimizedResume:
    """
    Optimizes a resume for a specific JD, constrained by the provided factual evidence.
    Uses schema-constrained decoding and bounded exponential backoff.
    """
    if not api_key:
        raise ValueError("A Google API key is required for optimization.")

    prompt = (
        "You are an expert resume optimizer. Your task is to rewrite the candidate's resume to maximize relevance to the provided Job Description.\n\n"
        "CRITICAL CONSTRAINT: You are provided with an 'Evidence Inventory' containing the absolute ground truth facts about the candidate. "
        "You MUST NOT invent, hallucinate, or infer any skill, tool, degree, experience, or achievement that is not explicitly present in the Evidence Inventory. "
        "If the Job Description asks for 'Python' and the Evidence Inventory only has 'Java', you CANNOT add Python. "
        "You may reorder, rephrase, and emphasize existing facts to highlight relevance, but the factual integrity must remain 100% intact.\n\n"
        "Return a valid JSON object matching the following schema:\n"
        "{\n"
        '  "summary": "...",\n'
        '  "skills": "...",\n'
        '  "experience": "...",\n'
        '  "education": "...",\n'
        '  "certifications_and_projects": "..."\n'
        "}\n\n"
        f"--- EVIDENCE INVENTORY ---\n{evidence_json}\n\n"
        f"--- JOB DESCRIPTION ---\n{job_description}"
    )

    parsed_dict, telemetry = call_gemini_with_retry(
        prompt=prompt,
        api_key=api_key,
        schema_model=OptimizedResume,
        temperature=0.1,
        max_retries=max_retries,
        max_output_tokens=4096,
    )

    return OptimizedResume(**parsed_dict)


def unconstrained_optimize(
    resume_text: str,
    job_description: str,
    api_key: str,
    max_retries: int = 5
) -> OptimizedResume:
    """
    Optimizes a resume for a specific JD without factual constraints (Score-Oriented Optimization).
    Tests whether unconstrained score optimization introduces hallucinations and superficial claims.
    """
    if not api_key:
        raise ValueError("A Google API key is required for optimization.")

    prompt = (
        "You are an expert resume optimizer. Your task is to rewrite the candidate's resume to absolutely maximize "
        "the compatibility score for the provided Job Description.\n\n"
        "You should aggressively tailor, reorganize, and adapt the candidate's background to match the exact requirements "
        "and keywords of the employer. Present the candidate as an ideal fit.\n\n"
        "Return a valid JSON object matching the following schema:\n"
        "{\n"
        '  "summary": "...",\n'
        '  "skills": "...",\n'
        '  "experience": "...",\n'
        '  "education": "...",\n'
        '  "certifications_and_projects": "..."\n'
        "}\n\n"
        f"--- ORIGINAL RESUME ---\n{resume_text}\n\n"
        f"--- JOB DESCRIPTION ---\n{job_description}"
    )

    parsed_dict, telemetry = call_gemini_with_retry(
        prompt=prompt,
        api_key=api_key,
        schema_model=OptimizedResume,
        temperature=0.5,
        max_retries=max_retries,
        max_output_tokens=4096,
    )

    return OptimizedResume(**parsed_dict)


if __name__ == "__main__":
    pass
