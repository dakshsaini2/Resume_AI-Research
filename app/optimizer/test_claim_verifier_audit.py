"""
test_claim_verifier_audit.py

Comprehensive Audit Test Suite for the Claim Verifier:
Tests the verifier against:
1. Known-supported claims (exact ground truth matching)
2. Legitimate paraphrases (stylistic reorganization and terminology alignment)
3. Deliberately invented qualifications (hallucinated skills, certifications, degrees)
4. Subtle qualification inflation (inflated titles, unearned senior tenure)

Saves detailed audit records and benchmark metrics to:
- dataset/processed/optimization_experiment/verifier_test_audit_log.jsonl
- dataset/processed/optimization_experiment/verifier_audit_benchmark.json
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

# Ensure app is importable
BASE_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BASE_DIR / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from app.optimizer.claim_verifier import verify_claims, log_verification_audit, VerificationResult
except ImportError:
    from claim_verifier import verify_claims, log_verification_audit, VerificationResult

OUTPUT_DIR = BASE_DIR / "dataset" / "processed" / "optimization_experiment"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_LOG_FILE = OUTPUT_DIR / "verifier_test_audit_log.jsonl"
BENCHMARK_FILE = OUTPUT_DIR / "verifier_audit_benchmark.json"

# ==========================================================
# TEST CASES
# ==========================================================

AUDIT_TEST_CASES = [
    {
        "id": "TC-01-KNOWN-SUPPORTED",
        "category": "known_supported",
        "description": "Candidate claims exact skills, degree, and role from evidence inventory.",
        "evidence": {
            "skills": ["Python", "PostgreSQL", "Docker", "Git"],
            "education": [{"institution": "State University", "degree": "B.S.", "major": "Computer Science", "dates": "2018 - 2022"}],
            "experience": [{"employer": "Acme Corp", "position": "Software Engineer", "dates": "2022 - 2024", "achievements": ["Maintained REST APIs with PostgreSQL", "Automated deployments using Docker"]}],
            "projects": [{"name": "Task Tracker", "description": "CLI tool written in Python", "skills_used": ["Python", "Git"]}],
            "certifications": []
        },
        "resume": """
Professional Summary:
Software Engineer with 2 years of experience building Python backend services and Docker containers.

Skills:
Python, PostgreSQL, Docker, Git

Experience:
Acme Corp — Software Engineer (2022 - 2024)
- Maintained backend REST APIs leveraging PostgreSQL databases.
- Containerized deployment workflows using Docker.

Education:
B.S. in Computer Science, State University (2018 - 2022)

Projects:
Task Tracker: Python CLI utility version-controlled with Git.
""",
        "expected_faithful": True,
        "expected_flagged_keywords": []
    },
    {
        "id": "TC-02-LEGITIMATE-PARAPHRASE",
        "category": "legitimate_paraphrase",
        "description": "Reworded achievements and terminology alignment without introducing new factual claims.",
        "evidence": {
            "skills": ["Java", "Spring Boot", "MySQL", "JUnit"],
            "education": [{"institution": "Tech Institute", "degree": "Bachelor of Technology", "major": "Information Technology", "dates": "2017 - 2021"}],
            "experience": [{"employer": "FinTech Labs", "position": "Backend Developer", "dates": "2021 - Present", "achievements": ["Engineered transaction service with 99.9% uptime", "Wrote unit tests improving code coverage"]}],
            "projects": [],
            "certifications": [{"name": "Oracle Certified Java Associate", "provider": "Oracle", "date": "2021"}]
        },
        "resume": """
Professional Summary:
Results-oriented Backend Developer specialized in the Java ecosystem, delivering high-availability financial systems.

Skills:
Java, Spring Boot, MySQL, JUnit

Experience:
FinTech Labs — Backend Developer (2021 - Present)
- Architected payment processing services maintaining three-nines (99.9%) production availability.
- Authored comprehensive test suites with JUnit to enhance codebase test coverage.

Education:
B.Tech in Information Technology, Tech Institute (2017 - 2021)

Certifications:
Oracle Certified Java Associate (Oracle, 2021)
""",
        "expected_faithful": True,
        "expected_flagged_keywords": []
    },
    {
        "id": "TC-03-INVENTED-QUALIFICATIONS",
        "category": "deliberately_invented",
        "description": "Resume injects unearned cloud certifications, Kubernetes, and Golang not present in evidence.",
        "evidence": {
            "skills": ["HTML", "CSS", "JavaScript"],
            "education": [{"institution": "City College", "degree": "Associate Degree", "major": "Web Design", "dates": "2020 - 2022"}],
            "experience": [{"employer": "Digital Agency", "position": "Junior Front-End Developer", "dates": "2022 - 2023", "achievements": ["Created responsive landing pages in HTML/CSS"]}],
            "projects": [],
            "certifications": []
        },
        "resume": """
Professional Summary:
Senior Full Stack Engineer with deep expertise in cloud-native microservices, Golang, and Kubernetes orchestration.

Skills:
HTML, CSS, JavaScript, Golang, Kubernetes, AWS, Terraform

Experience:
Digital Agency — Lead Cloud Engineer (2022 - 2023)
- Designed Kubernetes clusters and microservices in Golang on AWS.
- Created responsive landing pages in HTML/CSS.

Education:
Associate Degree in Web Design, City College (2020 - 2022)

Certifications:
AWS Certified Solutions Architect — Professional
Certified Kubernetes Administrator (CKA)
""",
        "expected_faithful": False,
        "expected_flagged_keywords": ["Golang", "Kubernetes", "AWS", "CKA", "Lead Cloud Engineer"]
    },
    {
        "id": "TC-04-QUALIFICATION-INFLATION",
        "category": "subtle_inflation",
        "description": "Candidate was an intern for 3 months, resume claims 5 years senior managerial leadership.",
        "evidence": {
            "skills": ["Data Analysis", "Excel", "SQL"],
            "education": [{"institution": "Metro University", "degree": "B.A.", "major": "Economics", "dates": "2020 - 2024"}],
            "experience": [{"employer": "Retail Co", "position": "Summer Data Intern", "dates": "Jun 2023 - Aug 2023", "achievements": ["Analyzed weekly sales spreadsheets in Excel"]}],
            "projects": [],
            "certifications": []
        },
        "resume": """
Professional Summary:
Senior Strategic Analytics Manager with 5+ years of cross-functional executive leadership directing enterprise analytics teams.

Skills:
Data Analysis, Excel, SQL, Executive Stakeholder Management, Team Leadership

Experience:
Retail Co — Senior Analytics Manager (2019 - 2024)
- Spearheaded company-wide data initiatives, managing a team of 12 data analysts.
- Optimized multi-million dollar sales pipelines using SQL and Excel.

Education:
B.A. in Economics, Metro University (2020 - 2024)
""",
        "expected_faithful": False,
        "expected_flagged_keywords": ["Senior", "5+ years", "Executive", "team of 12"]
    }
]


def run_claim_verifier_audit():
    load_dotenv()
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        print("[ERROR] GOOGLE_API_KEY not configured in environment or .env file.")
        sys.exit(1)

    print("======================================================================")
    print("CLAIM VERIFIER AUDIT BENCHMARK")
    print(f"Timestamp: {datetime.utcnow().isoformat()}Z")
    print(f"Total Test Cases: {len(AUDIT_TEST_CASES)}")
    print("======================================================================\n")

    results = []
    tp = fp = tn = fn = 0

    for i, tc in enumerate(AUDIT_TEST_CASES, 1):
        print(f"--- Running Test Case {i}/{len(AUDIT_TEST_CASES)}: [{tc['id']}] ---")
        print(f"Category: {tc['category']} | Description: {tc['description']}")
        
        evidence_json = json.dumps(tc["evidence"])
        
        try:
            verdict: VerificationResult = verify_claims(
                evidence_json=evidence_json,
                optimized_resume_text=tc["resume"],
                api_key=api_key,
                max_retries=5
            )
            
            # Log complete audit record
            log_verification_audit(
                audit_log_path=AUDIT_LOG_FILE,
                sample_id=tc["id"],
                job_title=f"Audit Test: {tc['category']}",
                condition=tc["category"],
                evidence_inventory=tc["evidence"],
                resume_text=tc["resume"],
                verification=verdict,
                metadata={"test_id": tc["id"], "expected_faithful": tc["expected_faithful"]}
            )

            is_faithful_actual = verdict.is_faithful
            is_faithful_expected = tc["expected_faithful"]
            passed = (is_faithful_actual == is_faithful_expected)

            # Confusion matrix math (Positive = Hallucination detected / Not Faithful)
            # Actual Positive: resume has hallucinations (expected_faithful == False)
            # Actual Negative: resume is clean (expected_faithful == True)
            actual_has_hallucination = not is_faithful_expected
            predicted_has_hallucination = not is_faithful_actual

            if actual_has_hallucination and predicted_has_hallucination:
                tp += 1
            elif not actual_has_hallucination and not predicted_has_hallucination:
                tn += 1
            elif not actual_has_hallucination and predicted_has_hallucination:
                fp += 1
            elif actual_has_hallucination and not predicted_has_hallucination:
                fn += 1

            status_str = "PASS" if passed else "FAIL"
            print(f"  Result: [{status_str}]")
            print(f"  Expected Faithful: {is_faithful_expected} | Actual: {is_faithful_actual}")
            print(f"  Supported Claims Count: {len(verdict.supported_claims)}")
            print(f"  Unsupported Claims Flagged ({len(verdict.unsupported_claims)}): {verdict.unsupported_claims}")
            print(f"  Reasoning Snippet: {verdict.reasoning[:140]}...")
            print("-" * 70)

            results.append({
                "test_id": tc["id"],
                "category": tc["category"],
                "passed": passed,
                "expected_faithful": is_faithful_expected,
                "actual_faithful": is_faithful_actual,
                "unsupported_count": len(verdict.unsupported_claims),
                "unsupported_claims": verdict.unsupported_claims,
                "supported_count": len(verdict.supported_claims),
                "reasoning": verdict.reasoning
            })

        except Exception as e:
            print(f"  [ERROR] Execution failed for test case {tc['id']}: {e}")
            results.append({
                "test_id": tc["id"],
                "category": tc["category"],
                "passed": False,
                "error": str(e)
            })

    total = len(AUDIT_TEST_CASES)
    passed_count = sum(1 for r in results if r.get("passed", False))
    accuracy = (passed_count / total) * 100 if total > 0 else 0
    precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 100.0
    recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 100.0
    fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0.0

    benchmark_summary = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "total_test_cases": total,
        "passed_tests": passed_count,
        "accuracy_percent": accuracy,
        "confusion_matrix": {
            "true_positives_hallucinations_detected": tp,
            "true_negatives_clean_verified": tn,
            "false_positives_clean_flagged": fp,
            "false_negatives_hallucinations_missed": fn,
        },
        "metrics": {
            "hallucination_detection_precision": precision,
            "hallucination_detection_recall": recall,
            "false_positive_rate": fpr,
        },
        "detailed_results": results
    }

    with open(BENCHMARK_FILE, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    print("\n======================================================================")
    print("CLAIM VERIFIER AUDIT RESULTS SUMMARY")
    print("======================================================================")
    print(f"Passed: {passed_count}/{total} ({accuracy:.1f}%)")
    print(f"Hallucination Detection Precision : {precision:.1f}%")
    print(f"Hallucination Detection Recall    : {recall:.1f}%")
    print(f"False Positive Rate (Clean Flagged): {fpr:.1f}%")
    print(f"Audit Log Saved to : {AUDIT_LOG_FILE}")
    print(f"Benchmark Saved to : {BENCHMARK_FILE}")
    print("======================================================================\n")


if __name__ == "__main__":
    run_claim_verifier_audit()
