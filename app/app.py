import streamlit as st
from predictor import predict_resume_score
from utils import load_resume

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)

st.title("📄 AI Resume Screening System")
st.markdown("### Intelligent Resume Screening using NLP + Machine Learning")

uploaded_resume = st.file_uploader(
    "Upload Resume (PDF/DOCX)",
    type=["pdf", "docx"]
)

job_description = st.text_area(
    "Paste Job Description",
    height=250
)

if st.button("Analyze Resume"):

    if uploaded_resume is None:
        st.error("Please upload a resume.")
        st.stop()

    if job_description.strip() == "":
        st.error("Please enter a job description.")
        st.stop()

    with st.spinner("Analyzing Resume..."):

        resume_text = load_resume(uploaded_resume)

        result = predict_resume_score(
            resume_text,
            job_description
        )

    st.success("Analysis Complete!")

    score = result["score"]

    st.metric(
        "Resume Match Score",
        f"{score:.2f}%"
    )

    st.subheader("Recommendation")
    st.write(result["recommendation"])

    st.subheader("Feature Values")

    st.json(result["features"])