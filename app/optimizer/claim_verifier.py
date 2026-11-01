"""
claim_verifier.py

Adversarial claim verification engine:
Detects unsupported generated claims by cross-referencing an
optimized resume against the original evidence inventory.

Saves structured audit logs capturing:
- Extracted evidence inventory
- Candidate claims evaluated
- Supporting evidence matches
- Specific unsupported claims flagged
- Verification decision and reasoning
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel, Field
import json
import logging
from datetime import datetime

try:
    from app.optimizer.gemini_client import call_gemini_with_retry, GeminiAPIException
except ImportError:
    from gemini_client import call_gemini_with_retry, GeminiAPIException

logger = logging.getLogger("claim_verifier")


# ==========================================================
# PYDANTIC SCHEMA FOR VERIFICATION
# ==========================================================

class VerificationResult(BaseModel):
    is_faithful: bool = Field(
        description="True if the optimized resume is 100% faithful to the evidence inventory with zero hallucinations. False otherwise."
    )
    supported_claims: List[str] = Field(
        default_factory=list,
        description="List of verified claims present in the optimized resume that have direct backing in the evidence inventory."
    )
    unsupported_claims: List[str] = Field(
        default_factory=list,
        description="List of specific skills, experiences, dates, employers, or achievements in the resume NOT supported by the evidence inventory."
    )
    total_claims_evaluated: int = Field(
        default=0,
        description="Total number of discrete qualification and experience claims identified and audited."
    )
    reasoning: str = Field(
        default="",
        description="Detailed step-by-step audit reasoning explaining why the resume was approved or flagged."
    )


# ==========================================================
# VERIFIER FUNCTION
# ==========================================================

def verify_claims(
    evidence_json: str,
    optimized_resume_text: str,
    api_key: str,
    max_retries: int = 5
) -> VerificationResult:
    """
    Verifies that all claims in the optimized resume are supported by the evidence inventory.
    Employs an adversarial fact-checking prompt with schema-constrained JSON output.
    """
    if not api_key:
        raise ValueError("A Google API key is required for verification.")

    prompt = (
        "You are a strict, adversarial fact-checker for candidate resumes. Your job is to audit a rewritten resume "
        "against the candidate's absolute ground-truth Evidence Inventory to detect any hallucinations or unsupported claims.\n\n"
        "You will be given:\n"
        "1. EVIDENCE INVENTORY: The absolute ground-truth facts about the candidate.\n"
        "2. OPTIMIZED RESUME: The resume text being audited.\n\n"
        "AUDIT RULES:\n"
        "- LEGITIMATE PARAPHRASING IS ALLOWED: If the candidate claims 'Built an image classification pipeline' and the evidence states 'Trained CNN for image tagging', this is SUPPORTED.\n"
        "- HALLUCINATIONS ARE FORBIDDEN: If the resume introduces ANY programming language, library, credential, employer, job title, degree, metric, or accomplishment NOT explicitly supported by the Evidence Inventory, flag it as unsupported.\n"
        "- INVENTED CREDENTIALS: If the candidate lists a certification, degree, or skill absent from the Evidence Inventory, add it to unsupported_claims.\n"
        "- IF ANY unsupported claim is detected, set is_faithful to false. Only set is_faithful to true if 100% of claims are substantiated.\n\n"
        "Return a valid JSON object matching the following schema:\n"
        "{\n"
        '  "is_faithful": true,\n'
        '  "supported_claims": ["claim 1", "claim 2"],\n'
        '  "unsupported_claims": [],\n'
        '  "total_claims_evaluated": 5,\n'
        '  "reasoning": "Step-by-step breakdown of evidence cross-checking..."\n'
        "}\n\n"
        f"--- EVIDENCE INVENTORY ---\n{evidence_json}\n\n"
        f"--- OPTIMIZED RESUME ---\n{optimized_resume_text}"
    )

    parsed_dict, telemetry = call_gemini_with_retry(
        prompt=prompt,
        api_key=api_key,
        schema_model=VerificationResult,
        temperature=0.0,
        max_retries=max_retries,
        max_output_tokens=4096,
    )

    # Ensure total_claims_evaluated is consistent
    res = VerificationResult(**parsed_dict)
    if res.total_claims_evaluated == 0:
        res.total_claims_evaluated = len(res.supported_claims) + len(res.unsupported_claims)
    
    # Enforce logical consistency: if unsupported_claims exist, is_faithful must be False
    if len(res.unsupported_claims) > 0:
        res.is_faithful = False

    return res


# ==========================================================
# AUDIT LOGGING FUNCTION
# ==========================================================

def log_verification_audit(
    audit_log_path: Path,
    sample_id: Any,
    job_title: str,
    condition: str,
    evidence_inventory: Any,
    resume_text: str,
    verification: VerificationResult,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Appends a complete audit record to a JSON-Lines file.
    Guarantees full reproducibility and auditable records of every fact, claim, decision, and reasoning.
    """
    audit_log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sample_id": str(sample_id),
        "job_title": job_title,
        "condition": condition,
        "evidence_inventory": evidence_inventory if isinstance(evidence_inventory, dict) else json.loads(evidence_inventory) if isinstance(evidence_inventory, str) else str(evidence_inventory),
        "resume_text": resume_text,
        "verification_result": {
            "is_faithful": verification.is_faithful,
            "supported_claims": verification.supported_claims,
            "unsupported_claims": verification.unsupported_claims,
            "total_claims_evaluated": verification.total_claims_evaluated,
            "reasoning": verification.reasoning,
        },
        "metadata": metadata or {},
    }

    with open(audit_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
