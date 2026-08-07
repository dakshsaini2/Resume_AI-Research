"""
dashboard_skills.py

Professional Skills Dashboard
"""

import streamlit as st
from components import skill_chip


def skills_section(matched_skills, missing_skills):

    st.divider()

    st.subheader("🛠 Skills Analysis")

    left, right = st.columns(2)

    # ===========================
    # Matched Skills
    # ===========================

    with left:

        with st.container(border=True):

            st.markdown("### ✅ Matched Skills")

            if matched_skills:

                for skill in matched_skills:
                    skill_chip(skill, True)

                st.success(
                    f"{len(matched_skills)} skills matched."
                )

            else:

                st.warning(
                    "No matching skills found."
                )

    # ===========================
    # Missing Skills
    # ===========================

    with right:

        with st.container(border=True):

            st.markdown("### ❌ Missing Skills")

            if missing_skills:

                for skill in missing_skills:
                    skill_chip(skill, False)

                st.error(
                    f"{len(missing_skills)} skills missing."
                )

            else:

                st.success(
                    "No missing skills 🎉"
                )

    # ===========================
    # Skill Match Percentage
    # ===========================

    st.write("")

    total = len(matched_skills) + len(missing_skills)

    if total == 0:
        percentage = 0
    else:
        percentage = round(
            len(matched_skills) / total * 100,
            1
        )

    st.markdown("### 🎯 Skill Match")

    st.progress(percentage / 100)

    st.caption(f"{percentage}% Skills Matched")