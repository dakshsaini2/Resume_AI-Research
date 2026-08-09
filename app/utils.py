"""
utils.py

Utility functions for Resume Screening System
"""

import re
import pdfplumber
from docx import Document

import spacy
from spacy.matcher import PhraseMatcher

from config import DEGREE_MAP, SKILL_SYNONYMS

# ==========================================================
# LOAD SPACY
# ==========================================================

nlp = spacy.load("en_core_web_sm")

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

# ==========================================================
# SKILL DATABASE
# ==========================================================

SKILLS = [

    "python",
    "java",
    "c",
    "c++",
    "c#",
    "javascript",
    "typescript",

    "html",
    "css",
    "react",
    "react.js",
    "angular",
    "vue",

    "node",
    "node.js",
    "express",

    "sql",
    "mysql",
    "postgresql",
    "sqlite",
    "mongodb",

    "machine learning",
    "deep learning",
    "artificial intelligence",

    "nlp",
    "computer vision",

    "tensorflow",
    "keras",
    "pytorch",
    "scikit-learn",

    "numpy",
    "pandas",
    "matplotlib",

    "aws",
    "azure",
    "gcp",

    "docker",
    "kubernetes",

    "git",
    "github",

    "linux",
    "power bi",
    "excel",

    "flask",
    "django",
    "fastapi",

    "spark",
    "hadoop",
]

# ==========================================================
# ADD PATTERNS TO MATCHER
# ==========================================================

patterns = [nlp.make_doc(skill) for skill in SKILLS]

matcher.add("SKILLS", patterns)

# ==========================================================
# EDUCATION
# ==========================================================

DEGREES = [

    "b.tech",
    "b tech",
    "btech",

    "b.e",
    "be",

    "bachelor of technology",

    "m.tech",
    "m tech",
    "mtech",

    "master of technology",

    "bca",
    "mca",

    "b.sc",
    "bsc",

    "m.sc",
    "msc",

    "mba",

    "phd",
]

# ==========================================================
# CERTIFICATIONS
# ==========================================================

CERTIFICATIONS = [

    "aws",
    "azure",
    "google cloud",

    "oracle",

    "ccna",

    "redhat",

    "salesforce",

    "tensorflow",

    "pmp",
]

# ==========================================================
# PDF READER
# ==========================================================

def extract_text_from_pdf(pdf_path):

    text = ""

    with pdfplumber.open(pdf_path) as pdf:

        for page in pdf.pages:

            page_text = page.extract_text()

            if page_text:

                text += page_text + "\n"

    return text

# ==========================================================
# DOCX READER
# ==========================================================

def extract_text_from_docx(docx_path):

    document = Document(docx_path)

    text = "\n".join(

        para.text

        for para in document.paragraphs

    )

    return text
# ==========================================================
# LOAD RESUME
# ==========================================================

def load_resume(file):

    filename = file.name.lower()

    if filename.endswith(".pdf"):

        return extract_text_from_pdf(file)

    elif filename.endswith(".docx"):

        return extract_text_from_docx(file)

    else:

        raise ValueError(
            "Only PDF and DOCX files are supported."
        )


# ==========================================================
# EXTRACT SKILLS
# ==========================================================

def extract_skills(text):

    text = text.lower()

    doc = nlp(text)

    matches = matcher(doc)

    found = set()

    # PhraseMatcher matches
    for match_id, start, end in matches:

        skill = doc[start:end].text.lower()

        found.add(skill)

    # Regex matching for short skills like C, C++, C#
    regex_skills = [
        "c",
        "c++",
        "c#",
        "sql",
        "aws",
        "gcp"
    ]

    for skill in regex_skills:

        pattern = r"\b" + re.escape(skill) + r"\b"

        if re.search(pattern, text):

            found.add(skill)

    # ------------------------------
    # Skill Synonyms
    # ------------------------------

    extra = set()

    for skill in found:

        if skill in SKILL_SYNONYMS:

            extra.add(SKILL_SYNONYMS[skill])

    found.update(extra)

    # ------------------------------
    # Infer Machine Learning
    # ------------------------------

    ml_related = {

        "tensorflow",
        "keras",
        "pytorch",
        "scikit-learn",
        "numpy",
        "pandas",
        "matplotlib"

    }

    if any(skill in found for skill in ml_related):

        found.add("machine learning")

    return sorted(found)


# ==========================================================
# EXPERIENCE
# ==========================================================

def extract_experience(text):
    """
    Extract professional experience from resume text.

    Supports:
        2 years
        3+ years
        6 months
        8 weeks
        8-week
        May 2022 to June 2022
        Sep 2024 - Dec 2024

    Returns experience in years as a float.
    """

    if not text:
        return 0.0

    text = text.lower()

    # ------------------------------------------------------
    # 1. Date-range experience
    # ------------------------------------------------------

    month_numbers = {
        "january": 1,
        "jan": 1,
        "february": 2,
        "feb": 2,
        "march": 3,
        "mar": 3,
        "april": 4,
        "apr": 4,
        "may": 5,
        "june": 6,
        "jun": 6,
        "july": 7,
        "jul": 7,
        "august": 8,
        "aug": 8,
        "september": 9,
        "sep": 9,
        "sept": 9,
        "october": 10,
        "oct": 10,
        "november": 11,
        "nov": 11,
        "december": 12,
        "dec": 12,
    }

    month_names = "|".join(
        sorted(month_numbers.keys(), key=len, reverse=True)
    )

    date_pattern = (
        rf"\b({month_names})\s+(\d{{4}})"
        rf"\s*(?:to|-|–|—)\s*"
        rf"({month_names})\s+(\d{{4}})\b"
    )

    date_matches = re.findall(date_pattern, text)

    if date_matches:

        total_months = 0

        for start_month_name, start_year, end_month_name, end_year in date_matches:

            start_month = month_numbers[start_month_name]
            end_month = month_numbers[end_month_name]

            start_year = int(start_year)
            end_year = int(end_year)

            start_value = (
                start_year * 12
                + start_month
            )

            end_value = (
                end_year * 12
                + end_month
            )

            months = end_value - start_value + 1

            if months > 0:
                total_months += months

        return round(total_months / 12.0, 2)

    # ------------------------------------------------------
    # 2. Explicit years
    # ------------------------------------------------------

    year_matches = re.findall(
        r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:year|years)\b",
        text
    )

    if year_matches:

        years = [
            float(value)
            for value in year_matches
        ]

        return round(max(years), 2)

    # ------------------------------------------------------
    # 3. Explicit months
    # ------------------------------------------------------

    month_matches = re.findall(
        r"\b(\d+(?:\.\d+)?)\s*\+?\s*(?:month|months)\b",
        text
    )

    if month_matches:

        months = [
            float(value)
            for value in month_matches
        ]

        return round(max(months) / 12.0, 2)

    # ------------------------------------------------------
    # 4. Explicit weeks
    # ------------------------------------------------------

    # Handles:
    #
    # 8 weeks
    # 8 week
    # 8-week
    # 8-weeks
    # 8 - week
    # 8 - weeks

    week_matches = re.findall(
        r"\b(\d+(?:\.\d+)?)\s*-?\s*(?:week|weeks)\b",
        text
    )

    if week_matches:

        weeks = [
            float(value)
            for value in week_matches
        ]

        return round(max(weeks) / 52.1429, 2)

    # ------------------------------------------------------
    # 5. Nothing detected
    # ------------------------------------------------------

    return 0.0
# ==========================================================
# DEGREE EXTRACTION
# ==========================================================

def extract_degree(text):
    """
    Extract and normalize the candidate/job degree.
    """

    if not text:
        return "Not SPECIFIED"

    text = text.lower()

    # Check longer degree names first
    # so "bachelor of technology" is detected correctly.
    for degree in sorted(
        DEGREE_MAP.keys(),
        key=len,
        reverse=True
    ):

        # Word-safe matching
        pattern = r"(?<!\w)" + re.escape(degree.lower()) + r"(?!\w)"

        if re.search(pattern, text):

            return DEGREE_MAP[degree]

    return "Not SPECIFIED"
# ==========================================================
# CERTIFICATION EXTRACTION
# ==========================================================

def extract_certifications(text):

    text = text.lower()

    found = set()

    for cert in CERTIFICATIONS:

        pattern = r"\b" + re.escape(cert.lower()) + r"\b"

        if re.search(pattern, text):

            found.add(cert)

    return sorted(found)


# ==========================================================
# SKILL OVERLAP
# ==========================================================

def calculate_skill_overlap(candidate_skills, job_skills):

    candidate = set(skill.lower() for skill in candidate_skills)

    job = set(skill.lower() for skill in job_skills)

    overlap = candidate.intersection(job)

    return sorted(list(overlap))


# ==========================================================
# MISSING SKILLS
# ==========================================================

def calculate_missing_skills(candidate_skills, job_skills):

    candidate = set(skill.lower() for skill in candidate_skills)

    job = set(skill.lower() for skill in job_skills)

    missing = job - candidate

    return sorted(list(missing))


# ==========================================================
# SKILL MATCH SCORE
# ==========================================================

def skill_match_score(candidate_skills, job_skills):

    if len(job_skills) == 0:
        return 0.0

    overlap = calculate_skill_overlap(
        candidate_skills,
        job_skills
    )

    return round(len(overlap) / len(job_skills), 4)


# ==========================================================
# CANDIDATE SUMMARY
# ==========================================================

def generate_candidate_summary(text):

    skills = extract_skills(text)

    degree = extract_degree(text)

    experience = extract_experience(text)

    certifications = extract_certifications(text)

    return {
        "degree": degree,
        "experience": experience,
        "skills": skills,
        "skill_count": len(skills),
        "certifications": certifications,
        "certification_count": len(certifications)
    }


# ==========================================================
# DEBUG
# ==========================================================
if __name__ == "__main__":

    test_1 = """
    Experience:
    2 years
    """

    test_2 = """
    Sidalceas EduTech – May 2022 to June 2022
    Finance Intern
    """

    test_3 = """
    THB Private Limited – Sep 2024 to Dec 2024
    Data Analyst Intern
    Worked as Data Analyst Intern for 2 months
    """

    test_4 = """
    Sidalceas EduTech – May 2022 to June 2022

    THB Private Limited – Sep 2024 to Dec 2024
    """

    print("Test 1:", extract_experience(test_1))
    print("Test 2:", extract_experience(test_2))
    print("Test 3:", extract_experience(test_3))
    print("Test 4:", extract_experience(test_4))