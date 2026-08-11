"""
app.py

AI Resume Screening System – Premium Edition
"""
import streamlit as st

# ==========================================================
# PAGE CONFIG  — must be the very first Streamlit command
# ==========================================================

st.set_page_config(
    page_title="ResumeAI – Intelligent Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ==========================================================
# STYLES
# ==========================================================

from styles import apply_styles, load_css

apply_styles()
load_css()

# ==========================================================
# APP IMPORTS
# ==========================================================

from predictor import predict_resume_score

from utils import (
    load_resume,
    generate_candidate_summary,
)

from dashboard import show_dashboard

# ==========================================================
# HEADER
# ==========================================================

st.html("""
<div style="
    background: linear-gradient(135deg, #0A1628 0%, #0F2A4A 55%, #1A3F6F 100%);
    border-radius: 24px;
    padding: 36px 44px;
    margin-bottom: 28px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 24px 64px rgba(0,77,130,0.35);
    position: relative;
    overflow: hidden;
">
    <div style="position:absolute;top:-60px;right:120px;width:260px;height:260px;
        background:radial-gradient(circle,rgba(0,180,216,0.3) 0%,transparent 70%);
        border-radius:50%;pointer-events:none;"></div>
    <div style="position:absolute;bottom:-80px;right:-40px;width:300px;height:300px;
        background:radial-gradient(circle,rgba(0,119,182,0.2) 0%,transparent 70%);
        border-radius:50%;pointer-events:none;"></div>
    <div style="position:absolute;top:50%;left:42%;transform:translate(-50%,-50%);
        width:400px;height:2px;
        background:linear-gradient(90deg,transparent,rgba(255,255,255,0.04),transparent);
        pointer-events:none;"></div>

    <div style="position:relative;z-index:1;">
        <div style="
            font-size: 38px;
            font-weight: 900;
            color: #FFFFFF;
            letter-spacing: -2px;
            line-height: 1;
            margin-bottom: 10px;
        ">🧠 ResumeAI</div>
        <div style="
            font-size: 15px;
            color: rgba(255,255,255,0.6);
            font-weight: 500;
            letter-spacing: 0.3px;
        ">Intelligent Resume Screening &amp; Job Matching &nbsp;·&nbsp; Powered by AI</div>
    </div>

    <div style="display:flex;align-items:center;gap:24px;position:relative;z-index:1;">
        <div style="text-align:right;">
            <div style="color:rgba(255,255,255,0.4);font-size:10px;font-weight:700;
                text-transform:uppercase;letter-spacing:1.2px;margin-bottom:5px;">AI Stack</div>
            <div style="color:#7DD3F7;font-size:13px;font-weight:600;line-height:1.5;">
                Random Forest &nbsp;·&nbsp; TF-IDF<br>Sentence-BERT &nbsp;·&nbsp; XAI
            </div>
        </div>
        <div style="
            display: flex;
            align-items: center;
            gap: 9px;
            padding: 11px 20px;
            background: rgba(16, 185, 129, 0.14);
            border: 1px solid rgba(16, 185, 129, 0.35);
            border-radius: 999px;
            color: #6EE7B7;
            font-size: 13px;
            font-weight: 700;
            backdrop-filter: blur(10px);
            white-space: nowrap;
        ">
            <span class="pulse-dot" style="
                width: 9px; height: 9px;
                background: #10B981;
                border-radius: 50%;
                display: inline-block;
            "></span>
            AI Analysis Ready
        </div>
    </div>
</div>
""")

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.html("""
    <div style="text-align:center;padding:20px 0 12px;">
        <div style="font-size:52px;margin-bottom:10px;line-height:1;">🧠</div>
        <div style="font-size:20px;font-weight:800;color:#E2E8F0;
            letter-spacing:-0.5px;margin-bottom:4px;">ResumeAI</div>
        <div style="font-size:11px;color:rgba(255,255,255,0.4);font-weight:500;
            text-transform:uppercase;letter-spacing:1px;">v2.0 · Professional</div>
    </div>
    """)

    st.divider()

    st.html("""
    <div style="color:rgba(255,255,255,0.4);font-size:10px;font-weight:700;
        text-transform:uppercase;letter-spacing:1.2px;margin-bottom:10px;padding:0 4px;">
        AI Models Active
    </div>
    """)

    st.success("✔ Random Forest Classifier")
    st.success("✔ TF-IDF Vectorizer")
    st.success("✔ Sentence-BERT (SBERT)")
    st.success("✔ Explainable AI (XAI)")

    st.divider()

    st.html("""
    <div style="
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 16px;
        padding: 18px;
        margin-top: 6px;
    ">
        <div style="color:rgba(255,255,255,0.4);font-size:10px;font-weight:700;
            text-transform:uppercase;letter-spacing:1px;margin-bottom:10px;">Developed by</div>
        <div style="color:#FFFFFF;font-size:17px;font-weight:800;
            letter-spacing:-0.3px;margin-bottom:3px;">Daksh Saini</div>
        <div style="color:rgba(255,255,255,0.45);font-size:12px;font-weight:500;">
            AI / ML Research
        </div>
    </div>
    """)

# ==========================================================
# INPUT SECTION
# ==========================================================

st.divider()

col1, col2 = st.columns(2, gap="large")

with col1:

    st.html("""
    <div style="margin-bottom:14px;">
        <div style="font-size:18px;font-weight:700;color:#0F172A;
            margin-bottom:4px;">📄 Upload Resume</div>
        <div style="font-size:13px;color:#6B7280;font-weight:500;">
            PDF or DOCX format supported
        </div>
    </div>
    """)

    resume_file = st.file_uploader(
        "Upload Resume",
        type=["pdf", "docx"],
        label_visibility="collapsed",
    )

with col2:

    st.html("""
    <div style="margin-bottom:14px;">
        <div style="font-size:18px;font-weight:700;color:#0F172A;
            margin-bottom:4px;">📝 Job Description</div>
        <div style="font-size:13px;color:#6B7280;font-weight:500;">
            Paste the complete job description
        </div>
    </div>
    """)

    job_description = st.text_area(
        "Job Description",
        height=240,
        label_visibility="collapsed",
        placeholder=(
            "Example:\n\nPython Developer\n\n"
            "Required Skills:\nPython, SQL, Machine Learning\n"
            "TensorFlow, Docker, AWS\n\n"
            "Experience: 2 Years\nEducation: B.Tech"
        ),
    )

st.divider()

analyze = st.button(
    "🚀  Analyze Resume",
    use_container_width=True,
    type="primary",
)

# ==========================================================
# ANALYSIS
# ==========================================================

if analyze:

    if resume_file is None:
        st.error("⚠️ Please upload a resume to continue.")
        st.stop()

    if job_description.strip() == "":
        st.error("⚠️ Please enter the job description to continue.")
        st.stop()

    with st.spinner("Analyzing resume with AI models…"):

        try:

            resume_text = load_resume(resume_file)

            result = predict_resume_score(
                resume_text,
                job_description,
            )

            show_dashboard(result)

        except Exception as e:

            st.error(
                "An unexpected error occurred while analyzing the resume."
            )

            st.exception(e)