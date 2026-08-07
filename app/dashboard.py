"""
dashboard.py

Professional ATS Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px

from charts import score_gauge
from components import apple_card
from dashboard_skills import skills_section
from report import generate_pdf


def show_dashboard(result):

    # =====================================================
    # EXTRACT RESULT
    # =====================================================

    score = result["score"]
    recommendation = result["recommendation"]

    features = result["features"]

    candidate_skills = result["candidate_skills"]
    required_skills = result["required_skills"]

    matched_skills = result["matched_skills"]
    missing_skills = result["missing_skills"]

    candidate_degree = result["candidate_degree"]
    required_degree = result["required_degree"]

    candidate_experience = result["candidate_experience"]
    required_experience = result["required_experience"]

    candidate_certifications = result["candidate_certifications"]
    required_certifications = result["required_certifications"]

    # =====================================================
    # ATS SCORE
    # =====================================================

    st.divider()

    st.subheader("🎯 ATS Resume Match")

    left, right = st.columns([1.7, 1])

    with left:
        score_gauge(score)

    with right:

        with st.container(border=True):
            apple_card(
                "🏆 Recruiter Verdict",
                recommendation
            )

        st.write("")

        with st.container(border=True):
            apple_card(
                "✅ Matched Skills",
                len(matched_skills)
            )

        st.write("")

        with st.container(border=True):
            apple_card(
                "📋 Required Skills",
                len(required_skills)
            )

    # =====================================================
    # CANDIDATE OVERVIEW
    # =====================================================

    st.divider()

    st.subheader("👤 Candidate Overview")

    c1, c2, c3, c4 = st.columns(4)

    with c1:

        with st.container(border=True):

            st.metric(
                "🎓 Degree",
                candidate_degree.upper()
            )

    with c2:

        with st.container(border=True):

            st.metric(
                "💼 Experience",
                f"{candidate_experience} Years"
            )

    with c3:

        with st.container(border=True):

            st.metric(
                "🛠 Skills",
                len(candidate_skills)
            )

    with c4:

        with st.container(border=True):

            st.metric(
                "📜 Certificates",
                len(candidate_certifications)
            )

    # =====================================================
    # SKILLS
    # =====================================================

    skills_section(
        matched_skills,
        missing_skills
    )

    # =====================================================
    # EDUCATION
    # =====================================================

    st.divider()

    st.subheader("🎓 Qualification Analysis")

    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.markdown("### 🎓 Education")

            st.write(
                f"**Candidate:** {candidate_degree.upper()}"
            )

            st.write(
                f"**Required:** {required_degree.upper()}"
            )

    with col2:

        with st.container(border=True):

            st.markdown("### 💼 Experience")

            st.write(
                f"**Candidate:** {candidate_experience} Years"
            )

            st.write(
                f"**Required:** {required_experience} Years"
            )

    with col3:

        with st.container(border=True):

            st.markdown("### 📜 Certifications")

            st.write("**Candidate**")

            if candidate_certifications:

                st.write(
                    ", ".join(candidate_certifications)
                )

            else:

                st.write("None")

            st.write("")

            st.write("**Required**")

            if required_certifications:

                st.write(
                    ", ".join(required_certifications)
                )

            else:

                st.write("None")
    # =====================================================
    # FEATURE IMPORTANCE
    # =====================================================

    st.divider()

    st.subheader("📊 Resume Analytics")

    feature_df = pd.DataFrame({
        "Feature": list(features.keys()),
        "Value": list(features.values())
    })

    # Make feature names recruiter-friendly
    feature_names = {
        "resume_length": "📄 Resume Length",
        "job_length": "📑 Job Description Length",
        "semantic_similarity": "🧠 Semantic Similarity",
        "tfidf_similarity": "📝 TF-IDF Similarity",
        "skill_match": "🛠 Skill Match",
        "education_match": "🎓 Education Match",
        "experience_match": "💼 Experience Match",
        "certification_match": "📜 Certification Match",
        "matched_skills": "✅ Matched Skills",
        "missing_skills": "❌ Missing Skills",
    }

    feature_df["Feature"] = feature_df["Feature"].replace(feature_names)

    # Keep numeric values only
    feature_df = feature_df[
        pd.to_numeric(feature_df["Value"], errors="coerce").notnull()
    ]

    feature_df["Value"] = feature_df["Value"].astype(float)

    max_value = feature_df["Value"].max()

    if max_value > 0:
        feature_df["Importance"] = (
            feature_df["Value"] / max_value
        ) * 100
    else:
        feature_df["Importance"] = 0

    feature_df = feature_df.sort_values(
        by="Importance",
        ascending=True
    )

    fig = px.bar(
        feature_df,
        x="Importance",
        y="Feature",
        orientation="h",
        text="Importance",
        color="Importance",
        color_continuous_scale="Blues"
    )

    fig.update_traces(
        texttemplate="%{x:.0f}%",
        textposition="outside",
        cliponaxis=False
    )

    fig.update_layout(

        paper_bgcolor="#F5F5F7",
        plot_bgcolor="white",

        height=550,

        font=dict(
            color="#1D1D1F",
            size=14
        ),

        coloraxis_showscale=False,

        margin=dict(
            l=220,
            r=40,
            t=20,
            b=20
        ),

        xaxis=dict(
            title="Relative Importance",
            showgrid=True,
            gridcolor="#E5E5EA",
            zeroline=False
        ),

        yaxis=dict(
            title=""
        )

    )

    st.plotly_chart(
        fig,
        use_container_width=True,
        config={
            "displaylogo": False
        }
    )

    # =====================================================
    # AI SUGGESTIONS
    # =====================================================

    st.divider()

    st.subheader("💡 AI Resume Suggestions")

    with st.container(border=True):

        if missing_skills:

            st.success(
                "Here are some improvements to increase your ATS score:"
            )

            for skill in missing_skills[:6]:

                st.write(f"✅ Learn **{skill}**")

        else:

            st.success(
                "Excellent! No major skill gaps detected."
            )

        if candidate_experience < required_experience:

            st.write(
                "💼 Gain more relevant project or internship experience."
            )

        if candidate_certifications == []:

            st.write(
                "📜 Add industry-recognized certifications."
            )

        st.write(
            "📄 Keep your resume to 1–2 pages."
        )

        st.write(
            "🔗 Add GitHub and LinkedIn profile links."
        )

    # =====================================================
    # ADVANCED FEATURES
    # =====================================================

    st.divider()

    with st.expander("🔍 View Raw Feature Values"):

        st.dataframe(
            feature_df,
            use_container_width=True
        )

        st.json(features)

    # =====================================================
    # PDF REPORT
    # =====================================================

    st.divider()

    st.subheader("📄 ATS Report")

    pdf_file = generate_pdf(result)

    with open(pdf_file, "rb") as file:

        st.download_button(

            label="📄 Download Professional ATS Report",

            data=file,

            file_name="ATS_Report.pdf",

            mime="application/pdf",

            use_container_width=True

        )

    # =====================================================
    # FOOTER
    # =====================================================

    st.divider()

    st.caption(
        "🍎 AI Resume Screening System • Developed by Daksh Saini"
    )