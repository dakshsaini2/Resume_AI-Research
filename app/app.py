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
    initial_sidebar_state="collapsed",
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
# HEADER — Premium Glassmorphism Banner
# ==========================================================

st.html("""
<div style="
    background: linear-gradient(135deg, #06111F 0%, #0A2240 30%, #0D3B66 60%, #145DA0 100%);
    background-size: 200% 200%;
    animation: gradientShift 8s ease infinite;
    border-radius: 28px;
    padding: 44px 52px;
    margin-bottom: 36px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow:
        0 32px 80px rgba(6, 17, 31, 0.5),
        0 0 0 1px rgba(255,255,255,0.05) inset;
    position: relative;
    overflow: hidden;
">
    <!-- Ambient Glow Orbs -->
    <div style="position:absolute;top:-80px;right:100px;width:320px;height:320px;
        background:radial-gradient(circle,rgba(0,180,216,0.2) 0%,transparent 65%);
        border-radius:50%;pointer-events:none;animation:float 6s ease-in-out infinite;"></div>
    <div style="position:absolute;bottom:-100px;right:-60px;width:360px;height:360px;
        background:radial-gradient(circle,rgba(20,93,160,0.25) 0%,transparent 65%);
        border-radius:50%;pointer-events:none;"></div>
    <div style="position:absolute;top:-40px;left:30%;width:200px;height:200px;
        background:radial-gradient(circle,rgba(0,119,182,0.12) 0%,transparent 70%);
        border-radius:50%;pointer-events:none;"></div>

    <!-- Subtle Grid Lines -->
    <div style="position:absolute;inset:0;
        background-image:
            linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
            linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px);
        background-size: 40px 40px;
        pointer-events:none;"></div>

    <!-- Horizontal Accent Line -->
    <div style="position:absolute;top:50%;left:35%;transform:translate(-50%,-50%);
        width:500px;height:1px;
        background:linear-gradient(90deg,transparent,rgba(0,180,216,0.08),rgba(255,255,255,0.04),transparent);
        pointer-events:none;"></div>

    <!-- Brand Block -->
    <div style="position:relative;z-index:1;">
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:14px;">
            <div style="
                width: 52px; height: 52px;
                background: linear-gradient(135deg, #0077B6, #00B4D8);
                border-radius: 16px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 26px;
                box-shadow: 0 8px 24px rgba(0, 119, 182, 0.4);
            ">🧠</div>
            <div>
                <div style="
                    font-size: 36px;
                    font-weight: 900;
                    color: #FFFFFF;
                    letter-spacing: -1.5px;
                    line-height: 1;
                    font-family: 'Inter', sans-serif;
                ">ResumeAI</div>
            </div>
        </div>
        <div style="
            font-size: 14px;
            color: rgba(255,255,255,0.5);
            font-weight: 500;
            letter-spacing: 0.5px;
            padding-left: 66px;
            margin-top: -6px;
        ">Intelligent Resume Screening &amp; Job Matching &nbsp;·&nbsp; Powered by AI</div>
    </div>

    <!-- Right Side — AI Stack Info -->
    <div style="display:flex;align-items:center;gap:28px;position:relative;z-index:1;">
        <!-- Tech Stack Pills -->
        <div style="display:flex;flex-direction:column;align-items:flex-end;gap:8px;">
            <div style="color:rgba(255,255,255,0.35);font-size:10px;font-weight:700;
                text-transform:uppercase;letter-spacing:1.5px;">AI Stack</div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end;">
                <span style="
                    padding: 5px 12px;
                    background: rgba(255,255,255,0.06);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 8px;
                    color: rgba(255,255,255,0.7);
                    font-size: 11px;
                    font-weight: 600;
                    backdrop-filter: blur(4px);
                ">Random Forest</span>
                <span style="
                    padding: 5px 12px;
                    background: rgba(255,255,255,0.06);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 8px;
                    color: rgba(255,255,255,0.7);
                    font-size: 11px;
                    font-weight: 600;
                    backdrop-filter: blur(4px);
                ">TF-IDF</span>
            </div>
            <div style="display:flex;gap:6px;flex-wrap:wrap;justify-content:flex-end;">
                <span style="
                    padding: 5px 12px;
                    background: rgba(255,255,255,0.06);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 8px;
                    color: rgba(255,255,255,0.7);
                    font-size: 11px;
                    font-weight: 600;
                    backdrop-filter: blur(4px);
                ">Sentence-BERT</span>
                <span style="
                    padding: 5px 12px;
                    background: rgba(255,255,255,0.06);
                    border: 1px solid rgba(255,255,255,0.1);
                    border-radius: 8px;
                    color: rgba(255,255,255,0.7);
                    font-size: 11px;
                    font-weight: 600;
                    backdrop-filter: blur(4px);
                ">XAI</span>
            </div>
        </div>

        <!-- Status Badge -->
        <div style="
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 12px 22px;
            background: rgba(16, 185, 129, 0.1);
            border: 1px solid rgba(16, 185, 129, 0.25);
            border-radius: 14px;
            color: #6EE7B7;
            font-size: 13px;
            font-weight: 700;
            backdrop-filter: blur(12px);
            white-space: nowrap;
            box-shadow: 0 4px 16px rgba(16, 185, 129, 0.1);
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

<style>
    @keyframes gradientShift {
        0%   { background-position: 0% 50%; }
        50%  { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    @keyframes float {
        0%, 100% { transform: translateY(0); }
        50%      { transform: translateY(-12px); }
    }
</style>
""")

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.html("""
    <div style="text-align:center;padding:24px 0 16px;">
        <div style="
            width: 64px; height: 64px;
            background: linear-gradient(135deg, #0077B6, #00B4D8);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 32px;
            margin: 0 auto 14px;
            box-shadow: 0 8px 24px rgba(0, 119, 182, 0.35);
        ">🧠</div>
        <div style="font-size:22px;font-weight:800;color:#E2E8F0;
            letter-spacing:-0.5px;margin-bottom:5px;">ResumeAI</div>
        <div style="font-size:11px;color:rgba(255,255,255,0.35);font-weight:600;
            text-transform:uppercase;letter-spacing:1.5px;">v2.0 · Professional</div>
    </div>
    """)

    st.divider()

    st.html("""
    <div style="color:rgba(255,255,255,0.35);font-size:10px;font-weight:700;
        text-transform:uppercase;letter-spacing:1.5px;margin-bottom:12px;padding:0 4px;">
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
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 18px;
        padding: 20px;
        margin-top: 8px;
    ">
        <div style="color:rgba(255,255,255,0.35);font-size:10px;font-weight:700;
            text-transform:uppercase;letter-spacing:1.5px;margin-bottom:10px;">Developed by</div>
        <div style="color:#FFFFFF;font-size:18px;font-weight:800;
            letter-spacing:-0.3px;margin-bottom:4px;">Daksh Saini</div>
        <div style="color:rgba(255,255,255,0.4);font-size:12px;font-weight:500;">
            AI / ML Research
        </div>
    </div>
    """)

# ==========================================================
# INPUT SECTION — Card-based Layout
# ==========================================================

st.html("""
<div style="
    display: flex;
    align-items: center;
    gap: 12px;
    margin-bottom: 8px;
    margin-top: 8px;
">
    <div style="
        height: 2px;
        flex: 1;
        background: linear-gradient(90deg, rgba(0,119,182,0.2), transparent);
    "></div>
    <div style="
        font-size: 11px;
        font-weight: 700;
        color: #0077B6;
        text-transform: uppercase;
        letter-spacing: 2px;
        white-space: nowrap;
    ">Input Data</div>
    <div style="
        height: 2px;
        flex: 1;
        background: linear-gradient(90deg, transparent, rgba(0,119,182,0.2));
    "></div>
</div>
""")

col1, col2 = st.columns(2, gap="large")

with col1:

    st.html("""
    <div style="
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FBFF 100%);
        border: 1px solid rgba(0, 119, 182, 0.1);
        border-radius: 24px;
        padding: 28px 32px 12px;
        margin-bottom: 4px;
        box-shadow:
            0 4px 24px rgba(0, 119, 182, 0.06),
            0 1px 3px rgba(0,0,0,0.03);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #0077B6, #00B4D8, #48CAE4);
        "></div>
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:6px;">
            <div style="
                width: 42px; height: 42px;
                background: linear-gradient(135deg, #EBF5FB, #DBEAFE);
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
            ">📄</div>
            <div>
                <div style="font-size:17px;font-weight:800;color:#0A1628;
                    letter-spacing:-0.3px;">Upload Resume</div>
                <div style="font-size:12px;color:#64748B;font-weight:500;margin-top:2px;">
                    PDF or DOCX format supported · Max 200MB
                </div>
            </div>
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
    <div style="
        background: linear-gradient(135deg, #FFFFFF 0%, #F8FBFF 100%);
        border: 1px solid rgba(0, 119, 182, 0.1);
        border-radius: 24px;
        padding: 28px 32px 12px;
        margin-bottom: 4px;
        box-shadow:
            0 4px 24px rgba(0, 119, 182, 0.06),
            0 1px 3px rgba(0,0,0,0.03);
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0; left: 0; right: 0;
            height: 3px;
            background: linear-gradient(90deg, #00B4D8, #0077B6, #145DA0);
        "></div>
        <div style="display:flex;align-items:center;gap:14px;margin-bottom:6px;">
            <div style="
                width: 42px; height: 42px;
                background: linear-gradient(135deg, #EBF5FB, #DBEAFE);
                border-radius: 12px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 20px;
            ">📝</div>
            <div>
                <div style="font-size:17px;font-weight:800;color:#0A1628;
                    letter-spacing:-0.3px;">Job Description</div>
                <div style="font-size:12px;color:#64748B;font-weight:500;margin-top:2px;">
                    Paste the complete job description below
                </div>
            </div>
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

# ==========================================================
# CTA BUTTON — Prominent Analyze Action
# ==========================================================

st.html("""
<div style="height: 12px;"></div>
""")

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