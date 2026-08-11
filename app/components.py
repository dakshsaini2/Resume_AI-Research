import streamlit as st


def apple_card(title, value):

    st.metric(
        label=title,
        value=value
    )


def skill_chip(skill, matched=True):

    css = (
        "skill-match"
        if matched
        else "skill-missing"
    )

    icon = "✓" if matched else "+"

    st.markdown(
        f"""
        <span class="skill-chip {css}">
            {icon} {skill.title()}
        </span>
        """,
        unsafe_allow_html=True
    )