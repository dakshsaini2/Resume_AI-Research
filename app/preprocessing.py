"""
preprocessing.py

Text preprocessing for Resume Screening System
"""

import re
import string
import pandas as pd
import spacy

# ==========================================================
# LOAD SPACY
# ==========================================================

nlp = spacy.load("en_core_web_sm")

# ==========================================================
# CLEAN TEXT
# ==========================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove Email
    text = re.sub(r"\S+@\S+", " ", text)

    # Remove Phone Numbers
    text = re.sub(
        r"\+?\d[\d\s\-\(\)]{8,}",
        " ",
        text
    )

    # Remove punctuation
    text = text.translate(
        str.maketrans(
            "",
            "",
            string.punctuation
        )
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text

# ==========================================================
# LEMMATIZATION
# ==========================================================

def preprocess_text(text):

    text = clean_text(text)

    doc = nlp(text)

    tokens = []

    for token in doc:

        if token.is_stop:
            continue

        if token.is_punct:
            continue

        if token.is_space:
            continue

        lemma = token.lemma_.strip()

        if lemma == "":
            continue

        tokens.append(lemma.lower())

    return " ".join(tokens)
# ==========================================================
# WORD COUNT
# ==========================================================

def word_count(text):

    processed = preprocess_text(text)

    return len(processed.split())


# ==========================================================
# CHARACTER COUNT
# ==========================================================

def character_count(text):

    cleaned = clean_text(text)

    return len(cleaned)


# ==========================================================
# SENTENCE COUNT
# ==========================================================

def sentence_count(text):

    if pd.isna(text):
        return 0

    doc = nlp(str(text))

    return len(list(doc.sents))


# ==========================================================
# PREPROCESS RESUME
# ==========================================================

def preprocess_resume(resume_text):

    cleaned = clean_text(resume_text)

    processed = preprocess_text(resume_text)

    return {

        "clean_text": cleaned,

        "processed_text": processed,

        "char_count": character_count(resume_text),

        "word_count": word_count(resume_text),

        "sentence_count": sentence_count(resume_text),
    }


# ==========================================================
# PREPROCESS JOB DESCRIPTION
# ==========================================================

def preprocess_job(job_text):

    cleaned = clean_text(job_text)

    processed = preprocess_text(job_text)

    return {

        "clean_text": cleaned,

        "processed_text": processed,

        "char_count": character_count(job_text),

        "word_count": word_count(job_text),

        "sentence_count": sentence_count(job_text),
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
    Machine Learning
    Docker

    Experience:
    2 years

    Education:
    Bachelor of Technology
    """

    result = preprocess_resume(sample)

    print("Clean Text:")
    print(result["clean_text"])

    print("\nProcessed Text:")
    print(result["processed_text"])

    print("\nStatistics:")
    print("Characters:", result["char_count"])
    print("Words:", result["word_count"])
    print("Sentences:", result["sentence_count"])