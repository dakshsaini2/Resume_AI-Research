"""
batch_optimization.py

Evidence-Preserving vs. Unconstrained Resume Optimization Experiment
-------------------------------------------------------------------
Implements:
1. Resilient API calls with exponential backoff on transient errors (503, 429, 500).
2. Schema-constrained output and robust JSON parsing/repair to prevent malformed text errors.
3. Complete audit logging of extracted evidence, rewritten claims, and verifier decisions.
4. Rigorous experiment accounting:
   - Tracks attempted, completed, and failed samples explicitly (no silent skipping).
   - Matched pairs analysis where BOTH conditions succeed.
   - Explicit definitions for sample-level and claim-level unsupported claim denominators.
"""

import sys
import os
import json
import time
import argparse
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
import numpy as np
import warnings
from dotenv import load_dotenv

# Ensure app is importable
BASE_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BASE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from app.optimizer.evidence_extractor import extract_evidence
    from app.optimizer.resume_optimizer import optimize_resume, unconstrained_optimize, OptimizedResume
    from app.optimizer.claim_verifier import verify_claims, log_verification_audit, VerificationResult
    from app.optimizer.gemini_client import (
        GeminiAPIException,
        GeminiJSONParseException,
        GeminiQuotaExhaustedException
    )
    from app.predictor import predict_resume_score
except ImportError:
    from evidence_extractor import extract_evidence
    from resume_optimizer import optimize_resume, unconstrained_optimize, OptimizedResume
    from claim_verifier import verify_claims, log_verification_audit, VerificationResult
    from gemini_client import (
        GeminiAPIException,
        GeminiJSONParseException,
        GeminiQuotaExhaustedException
    )
    from predictor import predict_resume_score

# Silence third-party and Streamlit warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
import logging
for log_name in ["streamlit", "streamlit.runtime", "transformers", "sentence_transformers"]:
    logging.getLogger(log_name).setLevel(logging.ERROR)

DATASET_FILE = BASE_DIR / "dataset" / "Resume_Data_For_Ranking.csv"
OUTPUT_DIR = BASE_DIR / "dataset" / "processed" / "optimization_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT_LOG_FILE = OUTPUT_DIR / "verification_audit_log.jsonl"
ALL_RESULTS_FILE = OUTPUT_DIR / "optimization_results_all.csv"
MATCHED_RESULTS_FILE = OUTPUT_DIR / "optimization_results_matched.csv"
FAILURES_FILE = OUTPUT_DIR / "optimization_failures.csv"
ACCOUNTING_SUMMARY_FILE = OUTPUT_DIR / "experiment_accounting_summary.json"


def combine_columns(row: pd.Series, columns: List[str]) -> str:
    """Concatenates non-empty column values into a clean text block."""
    values = []
    for col in columns:
        if col in row:
            val = str(row[col]).strip()
            if val and val.lower() != "nan":
                values.append(val)
    return " ".join(values)


def format_resume(optimized: OptimizedResume) -> str:
    """Formats the structured resume into a standardized textual representation."""
    return f"""Professional Summary:
{optimized.summary}

Skills:
{optimized.skills}

Experience:
{optimized.experience}

Education:
{optimized.education}

Certifications & Projects:
{optimized.certifications_and_projects}
"""


def run_batch_optimization(num_samples: int = 5, random_seed: int = 42, sleep_between_calls: float = 8.0):
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        print("\n[FATAL ERROR] GOOGLE_API_KEY not found in environment or .env file.")
        sys.exit(1)

    print("======================================================================")
    print("RESUME OPTIMIZATION EXPERIMENT: EVIDENCE-PRESERVING VS. UNCONSTRAINED")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print(f"Target Samples: {num_samples} | Seed: {random_seed}")
    print("======================================================================\n")

    if not DATASET_FILE.exists():
        print(f"[FATAL ERROR] Dataset file not found at: {DATASET_FILE}")
        sys.exit(1)

    df = pd.read_csv(DATASET_FILE)
    df.columns = df.columns.str.strip().str.lower()

    candidate_columns = [
        "career_objective", "skills", "degree_names", "major_field_of_studies",
        "professional_company_names", "positions", "certification_skills"
    ]
    job_columns = [
        "job_position_name", "skills_required", "educational_requirements", "experiencere_requirement"
    ]

    sample_df = df.sample(n=num_samples, random_state=random_seed).copy().reset_index()

    # Accounting state
    accounting = {
        "attempted": 0,
        "completed_original": 0,
        "completed_unconstrained": 0,
        "completed_preserving": 0,
        "completed_both_matched": 0,
        "failed_samples": 0,
        "failures_by_stage": {
            "evidence_extraction": 0,
            "unconstrained_optimization": 0,
            "unconstrained_verification": 0,
            "preserving_optimization": 0,
            "preserving_verification": 0,
        }
    }

    all_records = []
    failures = []

    # Check for existing completed records to resume
    existing_completed = {}
    if ALL_RESULTS_FILE.exists():
        try:
            prev_df = pd.read_csv(ALL_RESULTS_FILE)
            for _, r in prev_df.iterrows():
                if r.get("status") == "COMPLETED_MATCHED":
                    existing_completed[str(r.get("sample_id"))] = r.to_dict()
        except Exception:
            pass

    quota_exhausted = False

    for idx, row in sample_df.iterrows():
        sample_num = idx + 1
        sample_id = str(row.get("index", idx))
        job_title = str(row.get("job_position_name", "Unknown Job"))
        accounting["attempted"] += 1

        print(f"\n>>> [Sample {sample_num}/{num_samples}] ID: {sample_id} | Position: {job_title}")

        # Check if already completed in previous run
        if sample_id in existing_completed:
            print("  [RESUMED] Sample already completed in previous run. Reusing matched result.")
            saved_rec = existing_completed[sample_id]
            accounting["completed_original"] += 1
            accounting["completed_unconstrained"] += 1
            accounting["completed_preserving"] += 1
            accounting["completed_both_matched"] += 1
            all_records.append(saved_rec)
            continue

        orig_resume_text = combine_columns(row, candidate_columns)
        jd_text = combine_columns(row, job_columns)

        sample_record: Dict[str, Any] = {
            "sample_index": sample_num,
            "sample_id": sample_id,
            "job_title": job_title,
            "status": "IN_PROGRESS",
            "failed_stage": None,
            "error_message": None,
            "original_score": None,
            "unconstrained_score": None,
            "unconstrained_gain": None,
            "unconstrained_faithful": None,
            "unconstrained_hallucinations_count": None,
            "unconstrained_total_claims": None,
            "preserving_score": None,
            "preserving_gain": None,
            "preserving_faithful": None,
            "preserving_hallucinations_count": None,
            "preserving_total_claims": None,
            "timestamp": datetime.now().isoformat() + "Z",
        }

        # Step 1: Compute Original Score
        try:
            print("  [1/5] Calculating Original Matching Score...")
            orig_pred = predict_resume_score(orig_resume_text, jd_text)
            orig_score = float(orig_pred["score"])
            sample_record["original_score"] = orig_score
            accounting["completed_original"] += 1
            print(f"        Original Score: {orig_score:.2f}%")
        except Exception as e:
            err_msg = f"Failed to compute original score: {str(e)}"
            print(f"  [ERROR] {err_msg}")
            sample_record["status"] = "FAILED"
            sample_record["failed_stage"] = "original_score"
            sample_record["error_message"] = err_msg
            failures.append({"sample_id": sample_id, "stage": "original_score", "error": err_msg})
            all_records.append(sample_record)
            continue

        # Step 2: Extract Ground Truth Evidence
        evidence_obj = None
        evidence_json = None
        try:
            print("  [2/5] Extracting Evidence Inventory (Gemini)...")
            evidence_obj = extract_evidence(orig_resume_text, api_key)
            evidence_json = json.dumps(evidence_obj.dict())
            time.sleep(sleep_between_calls)
        except GeminiQuotaExhaustedException as qe:
            err_msg = f"Daily API quota exhausted: {str(qe)}"
            print(f"\n[FATAL QUOTA LIMIT] {err_msg}")
            accounting["failed_samples"] += 1
            accounting["failures_by_stage"]["evidence_extraction"] += 1
            sample_record["status"] = "FAILED"
            sample_record["failed_stage"] = "evidence_extraction"
            sample_record["error_message"] = err_msg
            failures.append({"sample_id": sample_id, "stage": "evidence_extraction", "error": err_msg})
            all_records.append(sample_record)
            quota_exhausted = True
            break
        except Exception as e:
            err_msg = f"Evidence extraction failed: {str(e)}"
            print(f"  [ERROR] {err_msg}")
            accounting["failed_samples"] += 1
            accounting["failures_by_stage"]["evidence_extraction"] += 1
            sample_record["status"] = "FAILED"
            sample_record["failed_stage"] = "evidence_extraction"
            sample_record["error_message"] = err_msg
            failures.append({"sample_id": sample_id, "stage": "evidence_extraction", "error": err_msg})
            all_records.append(sample_record)
            pd.DataFrame(failures).to_csv(FAILURES_FILE, index=False)
            continue

        # Step 3: Condition B — Unconstrained Optimization & Audit
        unconstrained_succeeded = False
        try:
            print("  [3/5] Generating Condition B (Unconstrained Optimization)...")
            unconstrained_res = unconstrained_optimize(orig_resume_text, jd_text, api_key)
            unconstrained_text = format_resume(unconstrained_res)
            time.sleep(sleep_between_calls)

            unconstrained_score = float(predict_resume_score(unconstrained_text, jd_text)["score"])
            sample_record["unconstrained_score"] = unconstrained_score
            sample_record["unconstrained_gain"] = unconstrained_score - orig_score

            # Verify unconstrained claims against evidence
            uncon_verify: VerificationResult = verify_claims(evidence_json, unconstrained_text, api_key)
            time.sleep(sleep_between_calls)

            sample_record["unconstrained_faithful"] = uncon_verify.is_faithful
            sample_record["unconstrained_hallucinations_count"] = len(uncon_verify.unsupported_claims)
            sample_record["unconstrained_total_claims"] = uncon_verify.total_claims_evaluated
            unconstrained_succeeded = True
            accounting["completed_unconstrained"] += 1

            # Log audit trail
            log_verification_audit(
                audit_log_path=AUDIT_LOG_FILE,
                sample_id=sample_id,
                job_title=job_title,
                condition="unconstrained",
                evidence_inventory=evidence_obj.dict(),
                resume_text=unconstrained_text,
                verification=uncon_verify,
                metadata={"original_score": orig_score, "score_after": unconstrained_score}
            )

            print(f"        Score: {unconstrained_score:.2f}% (Gain: {unconstrained_score - orig_score:+.2f}%) | "
                  f"Faithful: {uncon_verify.is_faithful} | Hallucinations: {len(uncon_verify.unsupported_claims)}")
        except GeminiQuotaExhaustedException as qe:
            err_msg = f"Daily API quota exhausted: {str(qe)}"
            print(f"\n[FATAL QUOTA LIMIT] {err_msg}")
            accounting["failed_samples"] += 1
            accounting["failures_by_stage"]["unconstrained_optimization"] += 1
            sample_record["status"] = "FAILED"
            sample_record["failed_stage"] = "unconstrained_pipeline"
            sample_record["error_message"] = err_msg
            failures.append({"sample_id": sample_id, "stage": "unconstrained_pipeline", "error": err_msg})
            all_records.append(sample_record)
            quota_exhausted = True
            break
        except Exception as e:
            err_msg = f"Unconstrained pipeline failed: {str(e)}"
            print(f"  [ERROR] {err_msg}")
            accounting["failures_by_stage"]["unconstrained_optimization"] += 1
            sample_record["failed_stage"] = "unconstrained_pipeline"
            sample_record["error_message"] = err_msg
            failures.append({"sample_id": sample_id, "stage": "unconstrained_pipeline", "error": err_msg})

        # Step 4: Condition C — Evidence-Preserving Optimization & Audit
        preserving_succeeded = False
        try:
            print("  [4/5] Generating Condition C (Evidence-Preserving Optimization)...")
            preserving_res = optimize_resume(evidence_json, jd_text, api_key)
            preserving_text = format_resume(preserving_res)
            time.sleep(sleep_between_calls)

            preserving_score = float(predict_resume_score(preserving_text, jd_text)["score"])
            sample_record["preserving_score"] = preserving_score
            sample_record["preserving_gain"] = preserving_score - orig_score

            # Verify preserving claims against evidence
            pres_verify: VerificationResult = verify_claims(evidence_json, preserving_text, api_key)
            time.sleep(sleep_between_calls)

            sample_record["preserving_faithful"] = pres_verify.is_faithful
            sample_record["preserving_hallucinations_count"] = len(pres_verify.unsupported_claims)
            sample_record["preserving_total_claims"] = pres_verify.total_claims_evaluated
            preserving_succeeded = True
            accounting["completed_preserving"] += 1

            # Log audit trail
            log_verification_audit(
                audit_log_path=AUDIT_LOG_FILE,
                sample_id=sample_id,
                job_title=job_title,
                condition="evidence_preserving",
                evidence_inventory=evidence_obj.dict(),
                resume_text=preserving_text,
                verification=pres_verify,
                metadata={"original_score": orig_score, "score_after": preserving_score}
            )

            print(f"        Score: {preserving_score:.2f}% (Gain: {preserving_score - orig_score:+.2f}%) | "
                  f"Faithful: {pres_verify.is_faithful} | Hallucinations: {len(pres_verify.unsupported_claims)}")
        except GeminiQuotaExhaustedException as qe:
            err_msg = f"Daily API quota exhausted: {str(qe)}"
            print(f"\n[FATAL QUOTA LIMIT] {err_msg}")
            accounting["failed_samples"] += 1
            accounting["failures_by_stage"]["preserving_optimization"] += 1
            sample_record["status"] = "FAILED"
            sample_record["failed_stage"] = "preserving_pipeline"
            sample_record["error_message"] = err_msg
            failures.append({"sample_id": sample_id, "stage": "preserving_pipeline", "error": err_msg})
            all_records.append(sample_record)
            quota_exhausted = True
            break
        except Exception as e:
            err_msg = f"Preserving pipeline failed: {str(e)}"
            print(f"  [ERROR] {err_msg}")
            accounting["failures_by_stage"]["preserving_optimization"] += 1
            sample_record["failed_stage"] = "preserving_pipeline"
            sample_record["error_message"] = err_msg
            failures.append({"sample_id": sample_id, "stage": "preserving_pipeline", "error": err_msg})

        # Step 5: Mark Sample Status
        if unconstrained_succeeded and preserving_succeeded:
            sample_record["status"] = "COMPLETED_MATCHED"
            accounting["completed_both_matched"] += 1
        elif unconstrained_succeeded or preserving_succeeded:
            sample_record["status"] = "COMPLETED_PARTIAL"
        else:
            sample_record["status"] = "FAILED"
            accounting["failed_samples"] += 1

        all_records.append(sample_record)

        # Save all results incrementally
        pd.DataFrame(all_records).to_csv(ALL_RESULTS_FILE, index=False)
        if failures:
            pd.DataFrame(failures).to_csv(FAILURES_FILE, index=False)

    # ==========================================================
    # FINAL METRICS & ACCOUNTING ANALYSIS
    # ==========================================================
    df_all = pd.DataFrame(all_records)
    df_matched = df_all[df_all["status"] == "COMPLETED_MATCHED"].copy()

    # Save matched records separately
    df_matched.to_csv(MATCHED_RESULTS_FILE, index=False)

    print("\n" + "=" * 70)
    print("FINAL EXPERIMENT ACCOUNTING & RESULTS")
    print("=" * 70)

    print("\n--- 1. SAMPLE ACCOUNTING AUDIT ---")
    print(f"Total Attempted Samples              : {accounting['attempted']}")
    print(f"Fully Completed Matched Pairs (Both) : {accounting['completed_both_matched']}")
    print(f"Condition B Completed (Unconstrained): {accounting['completed_unconstrained']}")
    print(f"Condition C Completed (Preserving)   : {accounting['completed_preserving']}")
    print(f"Total Failed Samples                 : {accounting['failed_samples']}")
    print(f"Failures by Stage: {accounting['failures_by_stage']}")

    accounting_summary: Dict[str, Any] = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "sample_accounting": accounting,
        "matched_pairs_metrics": {},
        "hallucination_and_factuality_rates": {},
    }

    if not df_matched.empty:
        # Matched pairs statistical calculations
        avg_orig = df_matched["original_score"].mean()
        avg_uncon_gain = df_matched["unconstrained_gain"].mean()
        std_uncon_gain = df_matched["unconstrained_gain"].std() if len(df_matched) > 1 else 0.0

        avg_pres_gain = df_matched["preserving_gain"].mean()
        std_pres_gain = df_matched["preserving_gain"].std() if len(df_matched) > 1 else 0.0

        paired_diff = df_matched["unconstrained_gain"] - df_matched["preserving_gain"]
        avg_paired_diff = paired_diff.mean()

        print("\n--- 2. MATCHED PAIRS OPTIMIZATION GAIN (N = {}) ---".format(len(df_matched)))
        print(f"Mean Original Compatibility Score    : {avg_orig:.2f}%")
        print(f"Condition B (Unconstrained) Gain    : +{avg_uncon_gain:.2f}% (std: {std_uncon_gain:.2f}%)")
        print(f"Condition C (Preserving) Gain       : +{avg_pres_gain:.2f}% (std: {std_pres_gain:.2f}%)")
        print(f"Superficial Advantage (B - C)       : +{avg_paired_diff:.2f}%")

        accounting_summary["matched_pairs_metrics"] = {
            "n_matched_pairs": len(df_matched),
            "mean_original_score": float(avg_orig),
            "unconstrained": {
                "mean_gain": float(avg_uncon_gain),
                "std_gain": float(std_uncon_gain),
                "min_gain": float(df_matched["unconstrained_gain"].min()),
                "max_gain": float(df_matched["unconstrained_gain"].max()),
            },
            "preserving": {
                "mean_gain": float(avg_pres_gain),
                "std_gain": float(std_pres_gain),
                "min_gain": float(df_matched["preserving_gain"].min()),
                "max_gain": float(df_matched["preserving_gain"].max()),
            },
            "paired_difference_mean": float(avg_paired_diff),
        }

    # Denominator definitions & rate reporting
    print("\n--- 3. FACTUALITY & HALLUCINATION RATES (EXPLICIT DENOMINATORS) ---")
    
    # Condition B (Unconstrained)
    df_uncon = df_all.dropna(subset=["unconstrained_faithful"]).copy()
    n_uncon = len(df_uncon)
    if n_uncon > 0:
        uncon_hallucinating_resumes = (df_uncon["unconstrained_faithful"] == False).sum()
        uncon_hallucination_rate = (uncon_hallucinating_resumes / n_uncon) * 100
        uncon_total_hallucinations = df_uncon["unconstrained_hallucinations_count"].sum()
        uncon_total_claims = df_uncon["unconstrained_total_claims"].sum()
        uncon_density = uncon_total_hallucinations / n_uncon

        print(f"Condition B (Unconstrained) [Denominator N_completed = {n_uncon} resumes]:")
        print(f"  • Sample-Level Hallucination Rate: {uncon_hallucinating_resumes}/{n_uncon} ({uncon_hallucination_rate:.1f}%)")
        print(f"  • Claim-Level Hallucination Density: {uncon_total_hallucinations} hallucinations across {n_uncon} resumes ({uncon_density:.2f} per resume)")
        if uncon_total_claims > 0:
            print(f"  • Factuality of Total Claims: {(uncon_total_claims - uncon_total_hallucinations)/uncon_total_claims * 100:.1f}%")

        accounting_summary["hallucination_and_factuality_rates"]["unconstrained"] = {
            "denominator_completed_resumes": int(n_uncon),
            "hallucinating_resumes_count": int(uncon_hallucinating_resumes),
            "sample_level_hallucination_rate_pct": float(uncon_hallucination_rate),
            "total_unsupported_claims": int(uncon_total_hallucinations),
            "total_evaluated_claims": int(uncon_total_claims),
            "unsupported_claim_density_per_resume": float(uncon_density),
        }

    # Condition C (Preserving)
    df_pres = df_all.dropna(subset=["preserving_faithful"]).copy()
    n_pres = len(df_pres)
    if n_pres > 0:
        pres_hallucinating_resumes = (df_pres["preserving_faithful"] == False).sum()
        pres_hallucination_rate = (pres_hallucinating_resumes / n_pres) * 100
        pres_total_hallucinations = df_pres["preserving_hallucinations_count"].sum()
        pres_total_claims = df_pres["preserving_total_claims"].sum()
        pres_density = pres_total_hallucinations / n_pres

        print(f"\nCondition C (Evidence-Preserving) [Denominator N_completed = {n_pres} resumes]:")
        print(f"  • Sample-Level Hallucination Rate: {pres_hallucinating_resumes}/{n_pres} ({pres_hallucination_rate:.1f}%)")
        print(f"  • Claim-Level Hallucination Density: {pres_total_hallucinations} hallucinations across {n_pres} resumes ({pres_density:.2f} per resume)")
        if pres_total_claims > 0:
            print(f"  • Factuality of Total Claims: {(pres_total_claims - pres_total_hallucinations)/pres_total_claims * 100:.1f}%")

        accounting_summary["hallucination_and_factuality_rates"]["preserving"] = {
            "denominator_completed_resumes": int(n_pres),
            "hallucinating_resumes_count": int(pres_hallucinating_resumes),
            "sample_level_hallucination_rate_pct": float(pres_hallucination_rate),
            "total_unsupported_claims": int(pres_total_hallucinations),
            "total_evaluated_claims": int(pres_total_claims),
            "unsupported_claim_density_per_resume": float(pres_density),
        }

    # Write accounting summary JSON
    with open(ACCOUNTING_SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(accounting_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("SAVED OUTPUT ARTIFACTS:")
    print(f"  • Master Results (All)       : {ALL_RESULTS_FILE}")
    print(f"  • Matched Pairs Results      : {MATCHED_RESULTS_FILE}")
    print(f"  • Audit Log (JSON-Lines)     : {AUDIT_LOG_FILE}")
    print(f"  • Failure Log                : {FAILURES_FILE}")
    print(f"  • Accounting Summary         : {ACCOUNTING_SUMMARY_FILE}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Resume Optimization Experiment")
    parser.add_argument("--samples", type=int, default=5, help="Number of samples to process (default: 5)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--sleep", type=float, default=8.0, help="Sleep duration in seconds between Gemini calls (default: 8.0)")
    args = parser.parse_args()

    run_batch_optimization(num_samples=args.samples, random_seed=args.seed, sleep_between_calls=args.sleep)
