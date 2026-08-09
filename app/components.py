import streamlit as st

def apple_card(title, value):
    st.metric(label=title, value=value)


def skill_chip(skill, matched=True):

    css = "skill-match" if matched else "skill-missing"

    st.markdown(f"""
    <span class="skill-chip {css}">
        {skill.title()}
    </span>
    """, unsafe_allow_html=True)