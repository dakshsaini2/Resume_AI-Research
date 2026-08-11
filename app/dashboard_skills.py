"""
dashboard_skills.py

Premium Skills Dashboard — renders all HTML as single st.html() blocks
to prevent raw HTML code from showing on screen.
"""

import streamlit as st


def skills_section(matched_skills, missing_skills):

    st.divider()

    # ======================================================
    # COUNTS
    # ======================================================

    matched_count = len(matched_skills)
    missing_count = len(missing_skills)
    total = matched_count + missing_count
    percentage = round(matched_count / total * 100, 1) if total > 0 else 0

    # ======================================================
    # BUILD SKILL CHIPS HTML
    # ======================================================

    matched_chips_html = ""
    for skill in matched_skills:
        matched_chips_html += (
            f'<span class="skill-chip skill-match">✓ {skill.title()}</span>'
        )

    missing_chips_html = ""
    for skill in missing_skills:
        missing_chips_html += (
            f'<span class="skill-chip skill-missing">+ {skill.title()}</span>'
        )

    if not matched_chips_html:
        matched_chips_html = (
            '<p style="color:#9CA3AF;font-size:14px;margin:10px 0;">'
            'No matching skills found.</p>'
        )

    if not missing_chips_html:
        missing_chips_html = (
            '<p style="color:#065F46;font-size:14px;font-weight:600;margin:10px 0;">'
            '🎉 No required skills are missing!</p>'
        )

    # ======================================================
    # HEADER + SUMMARY CARDS
    # ======================================================

    st.html(f"""
    <div style="margin-bottom:6px;">
        <div style="
            font-size: 26px;
            font-weight: 800;
            color: #0F172A;
            letter-spacing: -0.5px;
            margin-bottom: 4px;
        ">🛠 Skills Analysis</div>
        <div style="font-size:14px;color:#6B7280;font-weight:500;">
            Compare candidate skills with the requirements of the target job.
        </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin:20px 0 24px;">

        <div class="metric-card">
            <div class="metric-label">✅ Matched Skills</div>
            <div class="metric-value">{matched_count}</div>
            <div class="metric-description">Skills in both resume and job</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">❌ Missing Skills</div>
            <div class="metric-value">{missing_count}</div>
            <div class="metric-description">Required skills not found</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">🎯 Skill Match</div>
            <div class="metric-value">{percentage}%</div>
            <div class="metric-description">Overall required skill coverage</div>
        </div>

    </div>

    <!-- SKILL CARDS SIDE BY SIDE -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;">

        <div class="skills-container">
            <div style="
                display:inline-block;
                background:linear-gradient(135deg,#ECFDF5,#D1FAE5);
                color:#065F46;
                font-size:14px;
                font-weight:700;
                padding:8px 16px;
                border-radius:10px;
                border:1px solid #A7F3D0;
                margin-bottom:14px;
            ">✅ Matched Skills</div>
            <div style="font-size:13px;color:#6B7280;margin-bottom:14px;font-weight:500;">
                Skills already present in the resume
            </div>
            <div style="line-height:2.2;">
                {matched_chips_html}
            </div>
            <div style="
                margin-top:16px;
                padding-top:12px;
                border-top:1px solid #E5E7EB;
                color:#065F46;
                font-size:13px;
                font-weight:700;
            ">✓ {matched_count} required skills matched</div>
        </div>

        <div class="skills-container">
            <div style="
                display:inline-block;
                background:linear-gradient(135deg,#FEF2F2,#FEE2E2);
                color:#991B1B;
                font-size:14px;
                font-weight:700;
                padding:8px 16px;
                border-radius:10px;
                border:1px solid #FECACA;
                margin-bottom:14px;
            ">❌ Missing Skills</div>
            <div style="font-size:13px;color:#6B7280;margin-bottom:14px;font-weight:500;">
                Skills that could improve the match
            </div>
            <div style="line-height:2.2;">
                {missing_chips_html}
            </div>
            <div style="
                margin-top:16px;
                padding-top:12px;
                border-top:1px solid #E5E7EB;
                color:#991B1B;
                font-size:13px;
                font-weight:700;
            ">+ {missing_count} skills could improve the match</div>
        </div>

    </div>

    <!-- SKILL COVERAGE PROGRESS -->
    <div class="analytics-card">
        <div style="font-size:16px;font-weight:800;color:#0A1628;margin-bottom:5px;">
            🎯 Skill Coverage Progress
        </div>
        <div style="font-size:13px;color:#64748B;margin-bottom:16px;font-weight:500;">
            Percentage of required job skills found in the candidate's resume.
        </div>
        <div style="
            background:#DBEAFE;
            border-radius:999px;
            height:14px;
            overflow:hidden;
            margin-bottom:12px;
        ">
            <div style="
                background:linear-gradient(90deg, #0077B6, #00B4D8);
                height:100%;
                width:{percentage}%;
                border-radius:999px;
            "></div>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center;">
            <span style="color:#065F46;font-size:13px;font-weight:700;">✓ {matched_count} matched</span>
            <span style="
                color:#0077B6;
                font-size:18px;
                font-weight:900;
                letter-spacing:-0.5px;
            ">{percentage}%</span>
            <span style="color:#991B1B;font-size:13px;font-weight:700;">{missing_count} missing ✗</span>
        </div>
    </div>
    """)