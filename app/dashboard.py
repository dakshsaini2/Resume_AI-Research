"""
dashboard.py

Professional ATS Dashboard
"""
from report import generate_pdf
import streamlit as st
import pandas as pd
import plotly.express as px

from charts import score_gauge
from components import apple_card, skill_chip

def show_dashboard(result):

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
    # SCORE
    # =====================================================

    st.divider()

    st.subheader("🎯 Resume Match Result")

    left, right = st.columns([1.2, 1])

    with left:
        score_gauge(score)

    with right:

        apple_card("🏆 Recommendation", recommendation)

        apple_card("✅ Matched Skills", len(matched_skills))

        apple_card("🛠 Required Skills", len(required_skills))

    # =====================================================
    # SUMMARY
    # =====================================================

    st.divider()

    st.subheader("👤 Candidate Summary")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("🎓 Degree", candidate_degree.upper())

    c2.metric("💼 Experience", f"{candidate_experience} Years")

    c3.metric("🛠 Skills", len(candidate_skills))

    c4.metric("📜 Certificates", len(candidate_certifications))

    # =====================================================
    # SKILLS
    # =====================================================

    st.divider()

    left, right = st.columns(2)

    with left:

        st.subheader("✅ Matched Skills")

        if matched_skills:

            for skill in matched_skills:
                skill_chip(skill, True)

        else:

            st.warning("No matching skills found.")

    with right:

        st.subheader("❌ Missing Skills")

        if missing_skills:

            for skill in missing_skills:
                skill_chip(skill, False)

        else:

            st.success("No missing skills 🎉")

    # =====================================================
    # EDUCATION
    # =====================================================

    st.divider()

    c1, c2, c3 = st.columns(3)

    with c1:

        st.subheader("🎓 Education")

        st.write(f"Candidate : **{candidate_degree.upper()}**")

        st.write(f"Required : **{required_degree.upper()}**")

    with c2:

        st.subheader("💼 Experience")

        st.write(f"Candidate : **{candidate_experience} Years**")

        st.write(f"Required : **{required_experience} Years**")

    with c3:

        st.subheader("📜 Certifications")

        st.write("Candidate:")

        if candidate_certifications:
            st.write(", ".join(candidate_certifications))
        else:
            st.write("None")

        st.write("Required:")

        if required_certifications:
            st.write(", ".join(required_certifications))
        else:
            st.write("None")

    # =====================================================
    # FEATURE CHART
    # =====================================================

    st.divider()

    st.subheader("📊 Feature Importance")

    feature_df = pd.DataFrame({

        "Feature": list(features.keys()),

        "Value": list(features.values())

    })

    fig = px.bar(

        feature_df,

        x="Value",

        y="Feature",

        orientation="h",

        color="Value",

        color_continuous_scale="Blues"

    )

    fig.update_layout(

        paper_bgcolor="#F5F5F7",

        plot_bgcolor="white",

        height=500

    )

    st.plotly_chart(

        fig,

        use_container_width=True

    )

      # =========================
    # ADVANCED FEATURES
    # =========================

    with st.expander("🔍 Advanced Feature Values"):
        st.json(features)

    # =========================
    # PDF REPORT
    # =========================

    st.markdown("---")
    st.markdown("## 📄 ATS Report")

    pdf_file = generate_pdf(result)

    with open(pdf_file, "rb") as f:
        st.download_button(
            label="📄 Download ATS PDF Report",
            data=f,
            file_name="ATS_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )