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

    text = text.lower()

    pattern = r"(\d+)\+?\s*(?:year|years)"

    matches = re.findall(pattern, text)

    if len(matches) == 0:

        return 0

    years = [int(x) for x in matches]

    return max(years)


# ==========================================================
# DEGREE EXTRACTION
# ==========================================================

def extract_degree(text):

    text = text.lower()

    for degree in DEGREE_MAP:

        if degree in text:

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

    sample = """
    Python Developer

    Skills:
    Python
    SQL
    Git
    Docker
    TensorFlow
    MySQL
    GitHub

    Experience:
    2 years

    Education:
    Bachelor of Technology

    Certification:
    AWS
    """

    print("Skills:", extract_skills(sample))
    print("Degree:", extract_degree(sample))
    print("Experience:", extract_experience(sample))
    print("Certifications:", extract_certifications(sample))