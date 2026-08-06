import streamlit as st

def apple_card(title, value):
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="text-align:center;padding:10px;">
                <div style="font-size:18px;color:#6E6E73;font-weight:600;">
                    {title}
                </div>
                <div style="font-size:36px;color:#007AFF;font-weight:700;margin-top:10px;">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def skill_chip(skill, matched=True):

    css = "skill-match" if matched else "skill-missing"

    st.markdown(f"""
    <span class="skill-chip {css}">
        {skill.title()}
    </span>
    """, unsafe_allow_html=True)