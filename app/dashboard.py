"""
dashboard.py

Professional ATS Dashboard
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from charts import score_gauge
from components import apple_card
from dashboard_skills import skills_section
from report import generate_pdf

from config import MODEL_FILE, FEATURE_COLUMNS


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

model = joblib.load(MODEL_FILE)


# ==========================================================
# SHOW DASHBOARD
# ==========================================================

def show_dashboard(result):

    # ======================================================
    # EXTRACT RESULT
    # ======================================================

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

    candidate_certifications = (
        result["candidate_certifications"]
    )

    required_certifications = (
        result["required_certifications"]
    )

    # Optional values returned by the updated predictor
    skill_match_score = result.get(
        "skill_match_score",
        features.get("skill_match_score", 0)
    )

    education_match_score = result.get(
        "education_match_score",
        features.get("education_match_score", 0)
    )

    experience_match_score = result.get(
        "experience_match_score",
        features.get("experience_match_score", 0)
    )

    certification_match_score = result.get(
        "certification_match_score",
        features.get("certification_match_score", 0)
    )

    tfidf_similarity = result.get(
        "tfidf_similarity",
        features.get("tfidf_similarity", 0)
    )

    semantic_similarity = result.get(
        "semantic_similarity",
        features.get("semantic_similarity", 0)
    )

    experience_status = result.get(
        "experience_status",
        "Not Available"
    )

    education_status = result.get(
        "education_status",
        "Not Available"
    )

    certification_status = result.get(
        "certification_status",
        "Not Available"
    )

    skill_status = result.get(
        "skill_status",
        "Not Available"
    )

    # ======================================================
    # ATS SCORE
    # ======================================================

    st.divider()

    st.subheader("🎯 ATS Resume Match")

    left, right = st.columns([1.7, 1])

    # ------------------------------------------------------
    # SCORE GAUGE
    # ------------------------------------------------------

    with left:

        with st.container(border=True):

            score_gauge(score)

    # ------------------------------------------------------
    # VERDICT CARDS
    # ------------------------------------------------------

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

    # ======================================================
    # CANDIDATE OVERVIEW
    # ======================================================

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

    # ======================================================
    # SKILLS
    # ======================================================

    skills_section(
        matched_skills,
        missing_skills
    )

    # ======================================================
    # QUALIFICATION ANALYSIS
    # ======================================================

    st.divider()

    st.subheader("🎓 Qualification Analysis")

    col1, col2, col3 = st.columns(3)

    # ------------------------------------------------------
    # EDUCATION
    # ------------------------------------------------------

    with col1:

        with st.container(border=True):

            st.markdown("### 🎓 Education")

            st.write(
                f"**Candidate:** "
                f"{candidate_degree.upper()}"
            )

            st.write(
                f"**Required:** "
                f"{required_degree.upper()}"
            )

            st.divider()

            if education_match_score >= 1:

                st.success(
                    "✅ Education requirement satisfied"
                )

            else:

                st.error(
                    "❌ Education requirement not satisfied"
                )

    # ------------------------------------------------------
    # EXPERIENCE
    # ------------------------------------------------------

    with col2:

        with st.container(border=True):

            st.markdown("### 💼 Experience")

            st.write(
                f"**Candidate:** "
                f"{candidate_experience} Years"
            )

            st.write(
                f"**Required:** "
                f"{required_experience} Years"
            )

            st.divider()

            if required_experience <= 0:

                st.info(
                    "ℹ️ Experience not specified"
                )

            elif candidate_experience >= required_experience:

                st.success(
                    "✅ Experience requirement satisfied"
                )

            else:

                st.error(
                    "❌ Experience requirement not satisfied"
                )

    # ------------------------------------------------------
    # CERTIFICATIONS
    # ------------------------------------------------------

    with col3:

        with st.container(border=True):

            st.markdown("### 📜 Certifications")

            st.write("**Candidate:**")

            if candidate_certifications:

                st.write(
                    ", ".join(candidate_certifications)
                )

            else:

                st.write("None")

            st.write("**Required:**")

            if required_certifications:

                st.write(
                    ", ".join(required_certifications)
                )

            else:

                st.write("None")

            st.divider()

            if certification_match_score >= 1:

                st.success(
                    "✅ Certification requirement satisfied"
                )

            elif certification_match_score > 0:

                st.warning(
                    "⚠️ Partial certification match"
                )

            else:

                if required_certifications:

                    st.error(
                        "❌ Required certifications missing"
                    )

                else:

                    st.info(
                        "ℹ️ Certifications not required"
                    )

    # ======================================================
    # MATCH QUALITY ANALYSIS
    # ======================================================

    st.divider()

    st.subheader("🔍 Match Quality Analysis")

    q1, q2, q3, q4, q5 = st.columns(5)

    # ------------------------------------------------------
    # SKILLS
    # ------------------------------------------------------

    with q1:

        with st.container(border=True):

            st.metric(
                "🛠 Skill Match",
                f"{skill_match_score * 100:.0f}%"
            )

            if skill_status == "Strong":

                st.success("Strong")

            elif skill_status == "Moderate":

                st.warning("Moderate")

            else:

                st.error("Weak")

    # ------------------------------------------------------
    # EDUCATION
    # ------------------------------------------------------

    with q2:

        with st.container(border=True):

            st.metric(
                "🎓 Education",
                f"{education_match_score * 100:.0f}%"
            )

            st.caption(
                education_status
            )

    # ------------------------------------------------------
    # EXPERIENCE
    # ------------------------------------------------------

    with q3:

        with st.container(border=True):

            st.metric(
                "💼 Experience",
                f"{experience_match_score * 100:.0f}%"
            )

            st.caption(
                experience_status
            )

    # ------------------------------------------------------
    # CERTIFICATION
    # ------------------------------------------------------

    with q4:

        with st.container(border=True):

            st.metric(
                "📜 Certification",
                f"{certification_match_score * 100:.0f}%"
            )

            st.caption(
                certification_status
            )

    # ------------------------------------------------------
    # SEMANTIC
    # ------------------------------------------------------

    with q5:

        with st.container(border=True):

            st.metric(
                "🧠 Semantic Match",
                f"{semantic_similarity * 100:.0f}%"
            )

            st.caption(
                "Sentence-BERT"
            )

    # ======================================================
    # NLP ANALYSIS
    # ======================================================

    st.divider()

    st.subheader("🧠 NLP Similarity Analysis")

    n1, n2 = st.columns(2)

    with n1:

        with st.container(border=True):

            st.markdown("### 📝 TF-IDF Similarity")

            st.progress(
                min(
                    max(tfidf_similarity, 0.0),
                    1.0
                )
            )

            st.write(
                f"**{tfidf_similarity * 100:.2f}%**"
            )

            st.caption(
                "Lexical similarity between the resume "
                "and job description."
            )

    with n2:

        with st.container(border=True):

            st.markdown("### 🧠 Sentence-BERT Similarity")

            st.progress(
                min(
                    max(semantic_similarity, 0.0),
                    1.0
                )
            )

            st.write(
                f"**{semantic_similarity * 100:.2f}%**"
            )

            st.caption(
                "Semantic similarity between the resume "
                "and job description."
            )

    # ======================================================
    # TRUE RANDOM FOREST FEATURE IMPORTANCE
    # ======================================================

    st.divider()

    st.subheader("📊 Model Feature Importance")

    st.caption(
        "Importance calculated directly from the trained "
        "Random Forest model."
    )

    # ------------------------------------------------------
    # CHECK MODEL SUPPORT
    # ------------------------------------------------------

    if hasattr(model, "feature_importances_"):

        importance_values = model.feature_importances_

        # Safety check
        if len(importance_values) == len(FEATURE_COLUMNS):

            importance_df = pd.DataFrame({

                "Feature": FEATURE_COLUMNS,

                "Importance": importance_values

            })

            # ------------------------------------------------
            # Friendly names
            # ------------------------------------------------

            feature_names = {

                "tfidf_similarity":
                    "📝 TF-IDF Similarity",

                "semantic_similarity":
                    "🧠 Semantic Similarity",

                "skill_match_score":
                    "🛠 Skill Match",

                "education_match_score":
                    "🎓 Education Match",

                "experience_match_score":
                    "💼 Experience Match",

                "certification_match_score":
                    "📜 Certification Match",

                "resume_length":
                    "📄 Resume Length",

                "job_length":
                    "📑 Job Description Length",

                "resume_word_count":
                    "📄 Resume Word Count",

                "job_word_count":
                    "📑 Job Description Word Count",

                "candidate_skill_count":
                    "🛠 Candidate Skill Count",

                "required_skill_count":
                    "📋 Required Skill Count",

                "skill_overlap_count":
                    "✅ Skill Overlap Count",

                "education_exact_match":
                    "🎓 Exact Education Match",
            }

            importance_df["Feature"] = (
                importance_df["Feature"]
                .map(feature_names)
                .fillna(importance_df["Feature"])
            )

            # ------------------------------------------------
            # Convert to percentage
            # ------------------------------------------------

            importance_df["Importance"] = (
                importance_df["Importance"] * 100
            )

            importance_df = importance_df.sort_values(
                by="Importance",
                ascending=True
            )

            # ------------------------------------------------
            # Plot
            # ------------------------------------------------

            fig = px.bar(

                importance_df,

                x="Importance",

                y="Feature",

                orientation="h",

                text="Importance",

                color="Importance",

                color_continuous_scale="Blues"

            )

            fig.update_traces(

                texttemplate="%{x:.1f}%",

                textposition="outside",

                cliponaxis=False

            )

            fig.update_layout(

                height=600,

                paper_bgcolor="#F5F5F7",

                plot_bgcolor="white",

                font=dict(

                    color="#1D1D1F",

                    size=14

                ),

                coloraxis_showscale=False,

                margin=dict(

                    l=230,

                    r=70,

                    t=30,

                    b=50

                ),

                xaxis=dict(

                    title="Model Importance (%)",

                    range=[
                        0,
                        max(
                            10,
                            importance_df["Importance"].max()
                            * 1.15
                        )
                    ],

                    showgrid=True,

                    gridcolor="#E5E5EA",

                    zeroline=False,

                    tickfont=dict(
                        color="#1D1D1F"
                    ),

                    title_font=dict(
                        color="#1D1D1F"
                    )

                ),

                yaxis=dict(

                    title="",

                    tickfont=dict(
                        color="#1D1D1F"
                    ),

                    automargin=True

                )

            )

            st.plotly_chart(

                fig,

                use_container_width=True,

                config={
                    "displaylogo": False
                }

            )

            # ------------------------------------------------
            # Importance table
            # ------------------------------------------------

            with st.expander(
                "🔍 View Model Feature Importance Values"
            ):

                display_df = importance_df.copy()

                display_df["Importance"] = (
                    display_df["Importance"]
                    .round(2)
                    .astype(str)
                    + "%"
                )

                st.dataframe(
                    display_df,
                    use_container_width=True,
                    hide_index=True
                )

        else:

            st.warning(
                "The number of model features does not "
                "match FEATURE_COLUMNS."
            )

    else:

        st.warning(
            "The loaded model does not expose "
            "`feature_importances_`."
        )

        # ======================================================
    # AI SUGGESTIONS
    # ======================================================

    st.divider()

    st.subheader("💡 AI Resume Suggestions")

    with st.container(border=True):

        specific_suggestion = False

        # --------------------------------------------------
        # MISSING SKILLS
        # --------------------------------------------------

        if missing_skills:

            st.success(
                "Here are some improvements to increase your ATS score:"
            )

            for skill in missing_skills[:6]:

                st.write(
                    f"✅ Learn **{skill}**"
                )

            specific_suggestion = True

        # --------------------------------------------------
        # EXPERIENCE
        # --------------------------------------------------

        if candidate_experience < required_experience:

            st.write(
                "💼 Gain more relevant project, internship, "
                "or professional experience."
            )

            specific_suggestion = True

        # --------------------------------------------------
        # CERTIFICATIONS
        # --------------------------------------------------

        if (
            not candidate_certifications
            and required_certifications
        ):

            st.write(
                "📜 Consider adding relevant industry-recognized "
                "certifications."
            )

            specific_suggestion = True

        # --------------------------------------------------
        # GENERAL SUGGESTIONS
        # --------------------------------------------------

        st.write(
            "📄 Keep the resume concise and preferably within 1–2 pages."
        )

        st.write(
            "🔗 Include GitHub and LinkedIn profile links."
        )

        # --------------------------------------------------
        # NO SPECIFIC GAPS
        # --------------------------------------------------

        if not specific_suggestion:

            st.success(
                "Excellent! No major improvement gaps detected."
            )

        # ======================================================
    # ADVANCED FEATURES
    # ======================================================

    st.divider()

    with st.expander("🔍 View Raw Feature Values"):

        raw_feature_df = pd.DataFrame({
            "Feature": list(features.keys()),
            "Value": list(features.values())
        })

        st.dataframe(
            raw_feature_df,
            use_container_width=True,
            hide_index=True
        )
        # ======================================================
    # PDF REPORT
    # ======================================================

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