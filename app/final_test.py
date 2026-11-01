# -*- coding: utf-8 -*-
"""
final_test.py

FINAL COMPREHENSIVE TEST — validates every feature of the Resume AI system:

  1. Skill Extraction & Synonym Resolution
  2. Education Degree Extraction & Normalization
  3. Education Hierarchy Matching
  4. Experience Extraction (years, months, weeks, date ranges)
  5. Certification Extraction
  6. Skill Overlap & Missing Skill Calculation
  7. Skill Match Scoring
  8. TF-IDF Similarity (directional checks)
  9. SBERT Semantic Similarity (directional checks)
  10. Preprocessing Pipeline
  11. Hybrid ATS Scoring (end-to-end, 10 scenarios)
  12. Edge Cases (empty inputs, missing sections)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from utils import (
    extract_skills,
    extract_degree,
    extract_experience,
    extract_certifications,
    calculate_skill_overlap,
    calculate_missing_skills,
    skill_match_score,
)
from preprocessing import preprocess_resume, preprocess_job
from feature_engineering import (
    compute_tfidf_similarity,
    compute_semantic_similarity,
    generate_features,
)
from predictor import predict_resume_score


# ==============================================================
# HELPERS
# ==============================================================

passed = 0
failed = 0
total = 0


def check(test_name, condition, details=""):
    global passed, failed, total
    total += 1
    if condition:
        passed += 1
        print(f"    PASS  {test_name}")
    else:
        failed += 1
        print(f"    FAIL  {test_name}  -->  {details}")


def section(title):
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


# ==============================================================
# 1. SKILL EXTRACTION & SYNONYM RESOLUTION
# ==============================================================

section("1. SKILL EXTRACTION & SYNONYM RESOLUTION")

# Basic extraction
skills_1 = extract_skills("I know Python, Java, and Docker")
check("Detects Python", "python" in skills_1, f"got {skills_1}")
check("Detects Java", "java" in skills_1, f"got {skills_1}")
check("Detects Docker", "docker" in skills_1, f"got {skills_1}")

# Synonym: GitHub -> Git
skills_2 = extract_skills("Experience with GitHub and MySQL")
check("GitHub resolves to Git (synonym)", "git" in skills_2, f"got {skills_2}")
check("MySQL resolves to SQL (synonym)", "sql" in skills_2, f"got {skills_2}")
check("GitHub itself is also retained", "github" in skills_2, f"got {skills_2}")

# ML inference: TensorFlow -> Machine Learning
skills_3 = extract_skills("Built models using TensorFlow and Keras")
check("TensorFlow triggers ML inference", "machine learning" in skills_3, f"got {skills_3}")

# Multi-word skills
skills_4 = extract_skills("Worked on deep learning and computer vision projects using power bi")
check("Detects 'deep learning'", "deep learning" in skills_4, f"got {skills_4}")
check("Detects 'computer vision'", "computer vision" in skills_4, f"got {skills_4}")
check("Detects 'power bi'", "power bi" in skills_4, f"got {skills_4}")

# No false positives on unrelated text
skills_5 = extract_skills("I enjoy cooking pasta and gardening on weekends")
check("No false positives on unrelated text", len(skills_5) == 0, f"got {skills_5}")

# C/C++/C# detection
skills_6 = extract_skills("Proficient in C, C++, and C#")
check("Detects C", "c" in skills_6, f"got {skills_6}")
check("Detects C++", "c++" in skills_6, f"got {skills_6}")
check("Detects C#", "c#" in skills_6, f"got {skills_6}")


# ==============================================================
# 2. EDUCATION DEGREE EXTRACTION & NORMALIZATION
# ==============================================================

section("2. EDUCATION DEGREE EXTRACTION & NORMALIZATION")

check("'Bachelor of Technology' -> btech", extract_degree("Bachelor of Technology from IIT") == "btech")
check("'B.Tech' -> btech", extract_degree("B.Tech in Computer Science") == "btech")
check("'M.Tech' -> mtech", extract_degree("M.Tech in AI from IIT Delhi") == "mtech")
check("'BCA' -> bca", extract_degree("Bachelor of Computer Applications (BCA)") == "bca")
check("'MCA' -> mca", extract_degree("MCA from Amity University") == "mca")
check("'MBA' -> mba", extract_degree("MBA in Finance from ISB") == "mba")
check("'PhD' -> phd", extract_degree("PhD in Computer Science from Stanford") == "phd")
check("'B.Sc' -> bsc", extract_degree("B.Sc in Mathematics") == "bsc")
check("'M.Sc' -> msc", extract_degree("M.Sc in Data Science") == "msc")
check("'Diploma' -> diploma", extract_degree("Diploma in Culinary Arts") == "diploma")
check("No degree -> 'Not SPECIFIED'", "not" in extract_degree("I like painting").lower())


# ==============================================================
# 3. EDUCATION HIERARCHY MATCHING
# ==============================================================

section("3. EDUCATION HIERARCHY MATCHING")

f1 = generate_features("Education: M.Tech", "Education: B.Tech")
check("M.Tech >= B.Tech -> edu_match=1.0", f1["education_match_score"] == 1.0)

f2 = generate_features("Education: BCA", "Education: M.Tech")
check("BCA < M.Tech -> edu_match < 1.0", f2["education_match_score"] < 1.0, f"got {f2['education_match_score']}")

f3 = generate_features("Education: PhD", "Education: B.Tech")
check("PhD >= B.Tech -> edu_match=1.0", f3["education_match_score"] == 1.0)

f4 = generate_features("Education: Diploma", "Education: M.Tech")
check("Diploma < M.Tech -> edu_match < 0.5", f4["education_match_score"] < 0.5, f"got {f4['education_match_score']}")

f5 = generate_features("Education: B.Tech", "Education: B.Tech")
check("B.Tech == B.Tech -> exact_match=1", f5["education_exact_match"] == 1)

f6 = generate_features("Education: M.Tech", "Education: B.Tech")
check("M.Tech vs B.Tech -> exact_match=0 (higher, not exact)", f6["education_exact_match"] == 0)


# ==============================================================
# 4. EXPERIENCE EXTRACTION
# ==============================================================

section("4. EXPERIENCE EXTRACTION")

check("'2 years' -> 2.0", extract_experience("2 years of experience") == 2.0)
check("'5+ years' -> 5.0", extract_experience("5+ years of Python") == 5.0)
check("'3.5 years' -> 3.5", extract_experience("3.5 years in ML") == 3.5)
check("'6 months' -> 0.5", extract_experience("6 months internship") == 0.5)
check("'8 weeks' -> ~0.15", 0.1 <= extract_experience("8 weeks training") <= 0.2)
check("Date range: May 2022 to June 2022 -> ~0.17",
      0.1 <= extract_experience("May 2022 to June 2022") <= 0.25)
check("Date range: Jan 2020 to Dec 2022 -> ~3.0",
      2.5 <= extract_experience("Jan 2020 to Dec 2022") <= 3.1)
check("No experience text -> 0.0", extract_experience("I study mathematics") == 0.0)
check("Empty string -> 0.0", extract_experience("") == 0.0)

# Experience matching
f_exp1 = generate_features("Experience: 5 years", "Experience: 3 years")
check("5yr vs 3yr required -> exp_match=1.0", f_exp1["experience_match_score"] == 1.0)

f_exp2 = generate_features("Experience: 1 year", "Experience: 4 years")
check("1yr vs 4yr required -> exp_match=0.25", f_exp2["experience_match_score"] == 0.25)

f_exp3 = generate_features("No experience info", "Experience: 2 years")
check("No exp vs 2yr required -> exp_match=0.0", f_exp3["experience_match_score"] == 0.0)


# ==============================================================
# 5. CERTIFICATION EXTRACTION
# ==============================================================

section("5. CERTIFICATION EXTRACTION")

certs_1 = extract_certifications("AWS Solutions Architect and Docker Certified")
check("Detects AWS", "aws" in certs_1, f"got {certs_1}")
check("Detects Docker", "docker" in certs_1, f"got {certs_1}")

certs_2 = extract_certifications("Completed Coursera Machine Learning course with certificate of completion")
check("Detects Coursera", "coursera" in certs_2, f"got {certs_2}")

certs_3 = extract_certifications("Google Cloud Professional certification")
check("Detects Google Cloud", "google cloud" in certs_3, f"got {certs_3}")

certs_4 = extract_certifications("I enjoy hiking and cooking")
check("No false positives on unrelated text", len(certs_4) == 0, f"got {certs_4}")


# ==============================================================
# 6. SKILL OVERLAP & MISSING SKILLS
# ==============================================================

section("6. SKILL OVERLAP & MISSING SKILL CALCULATION")

cand = ["python", "django", "docker", "git"]
reqd = ["python", "django", "aws", "kubernetes"]

overlap = calculate_skill_overlap(cand, reqd)
missing = calculate_missing_skills(cand, reqd)

check("Overlap includes python", "python" in overlap)
check("Overlap includes django", "django" in overlap)
check("Overlap does NOT include aws", "aws" not in overlap)
check("Missing includes aws", "aws" in missing)
check("Missing includes kubernetes", "kubernetes" in missing)
check("Missing does NOT include python", "python" not in missing)
check("Overlap count is 2", len(overlap) == 2, f"got {len(overlap)}")
check("Missing count is 2", len(missing) == 2, f"got {len(missing)}")


# ==============================================================
# 7. SKILL MATCH SCORING
# ==============================================================

section("7. SKILL MATCH SCORING")

check("Perfect match -> 1.0",
      skill_match_score(["python", "sql"], ["python", "sql"]) == 1.0)
check("Half match -> 0.5",
      skill_match_score(["python"], ["python", "sql"]) == 0.5)
check("No match -> 0.0",
      skill_match_score(["java"], ["python", "sql"]) == 0.0)
check("Empty job skills -> 0.0",
      skill_match_score(["python"], []) == 0.0)
check("Superset (3/2) -> 1.0",
      skill_match_score(["python", "sql", "docker"], ["python", "sql"]) == 1.0)


# ==============================================================
# 8. TF-IDF SIMILARITY (DIRECTIONAL)
# ==============================================================

section("8. TF-IDF SIMILARITY (DIRECTIONAL)")

tfidf_high = compute_tfidf_similarity(
    "python developer django flask postgresql",
    "python developer django flask postgresql"
)
tfidf_low = compute_tfidf_similarity(
    "french cuisine pastry chef kitchen",
    "python developer django flask postgresql"
)
check(f"Identical text -> high TF-IDF ({tfidf_high:.4f})", tfidf_high > 0.7, f"got {tfidf_high}")
check(f"Unrelated text -> low TF-IDF ({tfidf_low:.4f})", tfidf_low < 0.3, f"got {tfidf_low}")
check("Identical > Unrelated", tfidf_high > tfidf_low)


# ==============================================================
# 9. SBERT SEMANTIC SIMILARITY (DIRECTIONAL)
# ==============================================================

section("9. SBERT SEMANTIC SIMILARITY (DIRECTIONAL)")

sbert_high = compute_semantic_similarity(
    "senior python backend developer with django and aws experience",
    "looking for experienced python developer with django and cloud skills"
)
sbert_medium = compute_semantic_similarity(
    "java spring boot microservices developer",
    "python django rest api backend developer"
)
sbert_low = compute_semantic_similarity(
    "professional chef specializing in french cuisine",
    "machine learning engineer with deep learning expertise"
)
check(f"Similar meaning -> high SBERT ({sbert_high:.4f})", sbert_high > 0.6, f"got {sbert_high}")
check(f"Same domain diff stack -> lower SBERT ({sbert_medium:.4f})", sbert_medium < sbert_high, f"got {sbert_medium}")
check(f"Unrelated -> low SBERT ({sbert_low:.4f})", sbert_low < 0.5, f"got {sbert_low}")
check("Similar meaning > Unrelated", sbert_high > sbert_low)


# ==============================================================
# 10. PREPROCESSING PIPELINE
# ==============================================================

section("10. PREPROCESSING PIPELINE")

pp = preprocess_resume("Python Developer! I have 3+ years of experience @ TechCorp.")
check("Preprocessing returns clean_text", "clean_text" in pp)
check("Preprocessing returns processed_text", "processed_text" in pp)
check("Special chars removed in clean_text", "@" not in pp["clean_text"])
check("Text is lowercased", pp["clean_text"] == pp["clean_text"].lower())
check("Word count > 0", pp["word_count"] > 0)


# ==============================================================
# 11. HYBRID ATS SCORING — 10 END-TO-END SCENARIOS
# ==============================================================

section("11. HYBRID ATS SCORING — 10 END-TO-END SCENARIOS")

SCENARIOS = [
    {
        "name": "Perfect Senior Python Dev",
        "expected": (80, 100),
        "resume": """
            Senior Python Developer. Skills: Python, Django, Flask, REST APIs, PostgreSQL,
            Docker, Kubernetes, AWS, Git, CI/CD. Experience: 5 years. Education: M.Tech
            in CS from IIT. Certifications: AWS Solutions Architect, Docker Certified.
        """,
        "job": """
            Senior Python Developer. Required: Python, Django, REST APIs, PostgreSQL,
            Docker, AWS, Git. Experience: 4+ years. Education: B.Tech. Cert: AWS.
        """
    },
    {
        "name": "Partial Match Junior",
        "expected": (30, 70),
        "resume": """
            Junior Web Developer. Skills: HTML, CSS, JavaScript, React. Experience: 1 year
            as frontend intern. Education: BCA. Certifications: Google Analytics.
        """,
        "job": """
            Full Stack Developer. Required: Python, Django, React, Node.js, PostgreSQL,
            Docker, AWS, Git, CI/CD. Experience: 3+ years. Education: B.Tech. Cert: AWS.
        """
    },
    {
        "name": "Chef vs ML Engineer (Unrelated)",
        "expected": (0, 40),
        "resume": """
            Professional Chef. Skills: French Cuisine, Italian Cooking, Pastry Making,
            Menu Planning, Food Safety. Experience: 8 years as Head Chef. Education:
            Diploma in Culinary Arts. Certifications: ServSafe Food Handler.
        """,
        "job": """
            ML Engineer. Required: Python, TensorFlow, PyTorch, Scikit-learn, SQL,
            Docker, Kubernetes, MLOps, Git. Experience: 3+ years. Education: M.Tech/PhD.
            Cert: Google ML Engineer, AWS ML Specialty.
        """
    },
    {
        "name": "Overqualified PhD for Junior role",
        "expected": (60, 100),
        "resume": """
            AI Research Scientist. Skills: Python, TensorFlow, PyTorch, Scikit-learn,
            Pandas, NumPy, SQL, Docker, AWS, Spark, Deep Learning, NLP, Git.
            Experience: 7 years at Google DeepMind. Education: PhD from Stanford.
            Certifications: Google ML Engineer, AWS ML Specialty, TF Developer.
        """,
        "job": """
            Junior Data Analyst. Required: Python, SQL, Excel, Pandas. Experience: 1+ year.
            Education: B.Tech/B.Sc. Cert: Google Data Analytics.
        """
    },
    {
        "name": "Java Dev for Python role (Wrong Stack)",
        "expected": (25, 60),
        "resume": """
            Java Backend Developer. Skills: Java, Spring Boot, Hibernate, Oracle,
            Maven, Jenkins, Kafka, JUnit. Experience: 4 years at Infosys.
            Education: B.Tech in CS. Certifications: Oracle Java, AWS Cloud Practitioner.
        """,
        "job": """
            Python Backend Developer. Required: Python, Django, Flask, PostgreSQL, Redis,
            Docker, Kubernetes, Git, REST APIs. Experience: 3+ years. Education: B.Tech.
            Cert: AWS Solutions Architect.
        """
    },
    {
        "name": "Fresher with all keywords",
        "expected": (45, 80),
        "resume": """
            Software Developer. Skills: Python, Django, Flask, PostgreSQL, Docker, AWS, Git.
            Experience: Fresher. Education: B.Tech. Certifications: None.
        """,
        "job": """
            Python Backend Developer. Required: Python, Django, Flask, PostgreSQL, Docker,
            AWS, Git. Experience: 2+ years. Education: B.Tech. Cert: AWS Solutions Architect.
        """
    },
    {
        "name": "Data Scientist perfect fit",
        "expected": (75, 100),
        "resume": """
            Data Scientist. Skills: Python, TensorFlow, PyTorch, Scikit-learn, SQL, Pandas,
            NumPy, Matplotlib, Docker, AWS, Git, NLP, Deep Learning.
            Experience: 4 years at Microsoft. Education: M.Sc in Data Science.
            Certifications: AWS ML Specialty, Google ML Engineer.
        """,
        "job": """
            Senior Data Scientist. Required: Python, TensorFlow, Scikit-learn, SQL, Pandas,
            Docker, AWS, Git. Experience: 3+ years. Education: M.Sc/M.Tech.
            Cert: AWS ML Specialty.
        """
    },
    {
        "name": "Marketing Manager for DevOps role",
        "expected": (0, 35),
        "resume": """
            Marketing Manager. Skills: SEO, Content Marketing, Social Media, Google Ads,
            Adobe Photoshop, Canva, Copywriting. Experience: 6 years at AdAgency.
            Education: MBA in Marketing. Certifications: Google Ads, HubSpot.
        """,
        "job": """
            DevOps Engineer. Required: Python, Docker, Kubernetes, AWS, Terraform,
            Ansible, Git, CI/CD, Linux. Experience: 3+ years. Education: B.Tech.
            Cert: AWS Solutions Architect, CKA.
        """
    },
    {
        "name": "React Dev for React role (Exact fit)",
        "expected": (70, 100),
        "resume": """
            Frontend Developer. Skills: JavaScript, TypeScript, React, React.js, Node.js,
            HTML, CSS, Git, Docker. Experience: 3 years at Startup.
            Education: B.Tech in IT. Certifications: AWS Cloud Practitioner.
        """,
        "job": """
            React Frontend Developer. Required: JavaScript, TypeScript, React, HTML, CSS,
            Git. Experience: 2+ years. Education: B.Tech. Cert: AWS.
        """
    },
    {
        "name": "Intern with minimal experience",
        "expected": (20, 55),
        "resume": """
            Computer Science Student. Skills: Python, C++, HTML. Experience: 6 months
            internship. Education: B.Tech (ongoing). Certifications: Coursera Python.
        """,
        "job": """
            Senior ML Engineer. Required: Python, TensorFlow, PyTorch, Scikit-learn,
            Docker, Kubernetes, AWS, MLOps, Spark, SQL. Experience: 5+ years.
            Education: M.Tech or PhD. Cert: Google ML Engineer, AWS ML.
        """
    },
]

scenario_results = []

for i, s in enumerate(SCENARIOS, 1):
    r = predict_resume_score(s["resume"], s["job"])
    score = r["score"]
    lo, hi = s["expected"]
    ok = lo <= score <= hi

    check(
        f"Scenario {i:>2}: {s['name']:<45} Score={score:>6.1f}%  (expected {lo}-{hi}%)",
        ok,
        f"score {score}% outside [{lo}, {hi}]"
    )

    scenario_results.append({
        "name": s["name"],
        "score": score,
        "expected": (lo, hi),
        "passed": ok,
        "skills_matched": r["matched_skills"],
        "skills_missing": r["missing_skills"],
        "skill_pct": r["skill_match_score"],
        "edu": r["education_match_score"],
        "exp": r["experience_match_score"],
        "cert": r["certification_match_score"],
        "recommendation": r["recommendation"],
    })


# ==============================================================
# 12. EDGE CASES
# ==============================================================

section("12. EDGE CASES")

# Empty or minimal inputs
r_empty = predict_resume_score("", "Python Developer needed")
check("Empty resume -> score < 50%", r_empty["score"] < 50, f"got {r_empty['score']}")

r_no_job = predict_resume_score("Senior Python Developer with 5 years", "")
check("Empty job desc -> still produces result", r_no_job["score"] is not None)

r_no_skills = predict_resume_score(
    "I am a hardworking and dedicated team player",
    "Python Developer. Required: Python, SQL, Docker"
)
check("No technical skills -> low score (<50%)", r_no_skills["score"] < 50, f"got {r_no_skills['score']}")
check("No skills matched", len(r_no_skills["matched_skills"]) == 0, f"got {r_no_skills['matched_skills']}")

r_same = predict_resume_score(
    "Python Django Flask SQL Docker Git AWS 3 years B.Tech",
    "Python Django Flask SQL Docker Git AWS 3 years B.Tech"
)
check("Identical resume/job -> high score (>75%)", r_same["score"] > 75, f"got {r_same['score']}")


# ==============================================================
# FINAL SUMMARY
# ==============================================================

section("FINAL RESULTS")

print(f"\n    Total Tests:   {total}")
print(f"    Passed:        {passed}")
print(f"    Failed:        {failed}")
print(f"    Pass Rate:     {passed/total*100:.1f}%")

if failed == 0:
    print(f"\n    ALL TESTS PASSED!")
else:
    print(f"\n    {failed} TEST(S) FAILED — review output above.")

# Scoring summary table
print(f"\n{'─' * 75}")
print(f"  {'#':>2}  {'Scenario':<45} {'Score':>6}  {'Expected':>10}  {'Result':>6}")
print(f"{'─' * 75}")
for i, r in enumerate(scenario_results, 1):
    lo, hi = r["expected"]
    tag = "PASS" if r["passed"] else "FAIL"
    print(f"  {i:>2}  {r['name']:<45} {r['score']:>5.1f}%  {lo:>3}-{hi:<3}%    {tag}")
print(f"{'─' * 75}")
