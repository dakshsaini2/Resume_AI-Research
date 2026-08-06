"""
app.py

AI Resume Screening System
Professional ATS Dashboard
"""
import streamlit as st

from predictor import predict_resume_score

from utils import (
    load_resume,
    generate_candidate_summary,
)

from styles import load_css

from dashboard import show_dashboard

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide",
)
# ==========================================================
# LOAD CSS
# ==========================================================

load_css()

# ==========================================================
# HEADER
# ==========================================================
st.markdown("""
<div class='main-title'>
🍎 AI Resume Screening
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class='sub-title'>
Explainable Hybrid AI Framework using NLP, Sentence-BERT & Machine Learning
</div>
""", unsafe_allow_html=True)

# ==========================================================
# SIDEBAR
# ==========================================================
with st.sidebar:

    st.image(
        "https://img.icons8.com/fluency/96/artificial-intelligence.png",
        width=70
    )

    st.title("AI Resume Screening")

    st.caption("Version 2.0")

    st.divider()

    st.success("✔ Random Forest")
    st.success("✔ TF-IDF")
    st.success("✔ Sentence-BERT")
    st.success("✔ Explainable AI")

    st.divider()

    st.info("Developed by")

    st.write("**Daksh Saini**")
# ==========================================================
# INPUT SECTION
# ==========================================================

st.divider()

col1, col2 = st.columns(2)

with col1:

    st.subheader("📄 Upload Resume")

    resume_file = st.file_uploader(
        "Upload Resume (PDF or DOCX)",
        type=["pdf", "docx"]
    )

with col2:

    st.subheader("📝 Job Description")

    job_description = st.text_area(
        "Paste Job Description",
        height=250,
        placeholder="""
Example:

Python Developer

Required Skills:
Python
SQL
Machine Learning
TensorFlow
Docker
AWS

Experience:
2 Years

Education:
B.Tech
"""
    )

st.divider()

analyze = st.button(
    "🚀 Analyze Resume",
    use_container_width=True,
    type="primary"
)

# ==========================================================
# ANALYSIS
# ==========================================================
if analyze:

    if resume_file is None:
        st.error("Please upload a resume.")
        st.stop()

    if job_description.strip() == "":
        st.error("Please enter the job description.")
        st.stop()

    with st.spinner("Analyzing Resume..."):

        try:

            resume_text = load_resume(resume_file)

            result = predict_resume_score(
                resume_text,
                job_description
            )

            show_dashboard(result)

        except Exception as e:

            st.error(
                "An unexpected error occurred while analyzing the resume."
            )

            st.exception(e)