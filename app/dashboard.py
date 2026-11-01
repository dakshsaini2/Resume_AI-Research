"""
dashboard.py

Premium ATS Dashboard — all custom HTML uses st.html()
so nothing renders as raw code.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib

from report import generate_pdf

from config import MODEL_FILE, FEATURE_COLUMNS


# ==========================================================
# LOAD TRAINED MODEL & SCALER
# ==========================================================

model = joblib.load(MODEL_FILE)

from predictor import scaler



# ==========================================================
# SCORE COLOR HELPER
# ==========================================================

def _score_color(pct):
    if pct >= 90:
        return "#10B981"   # green
    elif pct >= 75:
        return "#0077B6"   # ocean blue
    elif pct >= 60:
        return "#F59E0B"   # amber
    else:
        return "#EF4444"   # red


# ==========================================================
# EXPLAINABLE AI HELPER
# ==========================================================

@st.cache_resource
def get_shap_explainer():
    import shap
    return shap.TreeExplainer(model)


# ==========================================================
# SHOW DASHBOARD
# ==========================================================


# --- Merged from charts.py ---
def score_gauge(score):

    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score,

            number={
                "suffix": "%",
                "font": {
                    "size": 48,
                    "color": "#0A1628",
                    "family": "Inter, sans-serif",
                }
            },

            title={
                "text": "ATS Match Score",
                "font": {
                    "size": 22,
                    "color": "#475569",
                    "family": "Inter, sans-serif",
                }
            },

            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickcolor": "#94A3B8",
                    "tickwidth": 1,
                    "dtick": 20,
                    "tickfont": {"color": "#64748B", "size": 12},
                },

                "bar": {
                    "color": "#0077B6",
                    "thickness": 0.75,
                },

                "bgcolor": "#F1F5F9",
                "borderwidth": 0,

                "steps": [
                    {"range": [0, 40],  "color": "#FEE2E2"},
                    {"range": [40, 60], "color": "#FEF3C7"},
                    {"range": [60, 75], "color": "#FEF9C3"},
                    {"range": [75, 90], "color": "#D1FAE5"},
                    {"range": [90, 100],"color": "#DBEAFE"},
                ],

                "threshold": {
                    "line": {"color": "#0077B6", "width": 3},
                    "thickness": 0.8,
                    "value": score,
                },
            }
        )
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        font=dict(
            color="#0A1628",
            size=14,
            family="Inter, sans-serif",
        ),
        margin=dict(t=40, b=20, l=40, r=40),
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

# --- Merged from dashboard_skills.py ---
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
        ">Skills Analysis</div>
        <div style="font-size:14px;color:#6B7280;font-weight:500;">
            Compare candidate skills with the requirements of the target job.
        </div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;margin:20px 0 24px;">

        <div class="metric-card">
            <div class="metric-label">Matched Skills</div>
            <div class="metric-value">{matched_count}</div>
            <div class="metric-description">Skills in both resume and job</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Missing Skills</div>
            <div class="metric-value">{missing_count}</div>
            <div class="metric-description">Required skills not found</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Skill Match</div>
            <div class="metric-value">{percentage}%</div>
            <div class="metric-description">Overall required skill coverage</div>
        </div>

    </div>

    <!-- SKILL CARDS SIDE BY SIDE -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;">

        <div class="skills-container">
            <div style="
                display:inline-block;
                background-color:#ECFDF5;
                color:#065F46;
                font-size:14px;
                font-weight:700;
                padding:8px 16px;
                border-radius:10px;
                border:1px solid #A7F3D0;
                margin-bottom:14px;
            ">Matched Skills</div>
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
                background-color:#FEF2F2;
                color:#991B1B;
                font-size:14px;
                font-weight:700;
                padding:8px 16px;
                border-radius:10px;
                border:1px solid #FECACA;
                margin-bottom:14px;
            ">Missing Skills</div>
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
            Skill Coverage Progress
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
                background-color:#0077B6;
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

def show_dashboard(result):

    # ======================================================
    # EXTRACT RESULT
    # ======================================================

    score               = result["score"]
    recommendation      = result["recommendation"]
    features            = result["features"]

    candidate_skills    = result["candidate_skills"]
    required_skills     = result["required_skills"]
    matched_skills      = result["matched_skills"]
    missing_skills      = result["missing_skills"]

    candidate_degree    = result["candidate_degree"]
    required_degree     = result["required_degree"]

    candidate_experience = result["candidate_experience"]
    required_experience  = result["required_experience"]

    candidate_certifications = result.get("candidate_certifications", [])
    required_certifications  = result.get("required_certifications", [])

    skill_match_score   = result.get("skill_match_score", features.get("skill_match_score", 0))
    education_match_score = result.get("education_match_score", features.get("education_match_score", 0))
    experience_match_score = result.get("experience_match_score", features.get("experience_match_score", 0))
    certification_match_score = result.get("certification_match_score", features.get("certification_match_score", 0))
    tfidf_similarity    = result.get("tfidf_similarity", features.get("tfidf_similarity", 0))
    semantic_similarity = result.get("semantic_similarity", features.get("semantic_similarity", 0))

    experience_status   = result.get("experience_status", "Not Available")
    education_status    = result.get("education_status", "Not Available")
    certification_status = result.get("certification_status", "Not Available")
    skill_status        = result.get("skill_status", "Not Available")

    score_color         = _score_color(score)

    # ======================================================
    # ATS SCORE SECTION
    # ======================================================

    st.divider()

    left, right = st.columns([1.5, 1], gap="large")

    with left:
        st.html("""
        <div style="
            font-size:11px;font-weight:700;color:#6366F1;
            text-transform:uppercase;letter-spacing:1.5px;
            margin-bottom:4px;
        ">ATS Match Score</div>
        """)
        score_gauge(score)

    with right:
        st.html(f"""
        <div style="
            background-color: #1E293B !important;
            border-radius: 24px;
            padding: 32px;
            min-height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            box-shadow:
                0 16px 48px rgba(13, 92, 158, 0.35),
                0 0 0 1px rgba(255,255,255,0.08) inset;
            position: relative;
            overflow: hidden;
        ">
            <!-- Ambient glow -->
            <div style="position:absolute;top:-50%;right:-20%;width:220px;height:220px;
                background:radial-gradient(circle,rgba(0,180,216,0.15) 0%,transparent 70%);
                border-radius:50%;pointer-events:none;"></div>

            <div style="color:rgba(255,255,255,0.5) !important;font-size:11px;font-weight:700;
                text-transform:uppercase;letter-spacing:1.5px;margin-bottom:14px;
                position:relative;z-index:1;">RECRUITER VERDICT</div>

            <div style="display:flex;align-items:center;gap:12px;margin-bottom:6px;
                position:relative;z-index:1;">
                <div style="width:14px;height:14px;border-radius:50%;
                    background:{'#10B981' if score >= 75 else '#F59E0B' if score >= 60 else '#EF4444'};
                    box-shadow:0 0 12px {'rgba(16,185,129,0.5)' if score >= 75 else 'rgba(245,158,11,0.5)' if score >= 60 else 'rgba(239,68,68,0.5)'};
                    flex-shrink:0;"></div>
                <div style="font-size:28px;font-weight:800;color:#FFFFFF !important;line-height:1.2;
                    letter-spacing:-0.5px;">{recommendation}</div>
            </div>

            <div style="color:rgba(255,255,255,0.45) !important;font-size:13px;margin-top:4px;
                position:relative;z-index:1;font-weight:500;">
                Based on skills, education, experience &amp; semantic similarity.
            </div>

            <div style="
                margin-top: 24px;
                display: flex;
                align-items: center;
                gap: 14px;
                position: relative;
                z-index: 1;
            ">
                <div style="
                    background: rgba(255,255,255,0.06);
                    border: 1px solid rgba(255,255,255,0.15);
                    border-radius: 14px;
                    padding: 14px 18px;
                    text-align: center;
                    flex: 1;
                    backdrop-filter: blur(8px);
                ">
                    <div style="font-size:28px;font-weight:900;color:#FFFFFF !important;letter-spacing:-1px;">
                        {score:.1f}%
                    </div>
                    <div style="font-size:10px;font-weight:700;color:rgba(255,255,255,0.45) !important;
                        text-transform:uppercase;letter-spacing:1.2px;margin-top:4px;">
                        ATS Score
                    </div>
                </div>
                <div style="
                    background: rgba(255,255,255,0.06);
                    border: 1px solid rgba(255,255,255,0.15);
                    border-radius: 14px;
                    padding: 14px 18px;
                    text-align: center;
                    flex: 1;
                    backdrop-filter: blur(8px);
                ">
                    <div style="font-size:28px;font-weight:900;color:#FFFFFF !important;letter-spacing:-1px;">
                        {len(matched_skills)}/{len(required_skills)}
                    </div>
                    <div style="font-size:10px;font-weight:700;color:rgba(255,255,255,0.45) !important;
                        text-transform:uppercase;letter-spacing:1.2px;margin-top:4px;">
                        Skills
                    </div>
                </div>
            </div>
        </div>
        """)

    # ======================================================
    # CANDIDATE OVERVIEW
    # ======================================================

    st.divider()

    st.html(f"""
    <div style="font-size:22px;font-weight:800;color:#0F172A;
        letter-spacing:-0.5px;margin-bottom:20px;">Candidate Overview</div>

    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:8px;">

        <div class="metric-card">
            <div class="metric-label">Education</div>
            <div class="metric-value" style="font-size:22px;letter-spacing:-0.5px;">
                {str(candidate_degree).upper()}
            </div>
            <div class="metric-description">Required: {str(required_degree).upper()}</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Experience</div>
            <div class="metric-value">{candidate_experience}</div>
            <div class="metric-description">Years &nbsp;·&nbsp; Required: {required_experience} yrs</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Skills Matched</div>
            <div class="metric-value">{len(matched_skills)}/{len(required_skills)}</div>
            <div class="metric-description">Required skills matched</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">Certifications</div>
            <div class="metric-value">{len(candidate_certifications)}</div>
            <div class="metric-description">Required: {len(required_certifications)}</div>
        </div>

    </div>
    """)

    # ======================================================
    # SKILLS SECTION
    # ======================================================

    skills_section(matched_skills, missing_skills)

    # ======================================================
    # QUALIFICATION ANALYSIS
    # ======================================================

    st.divider()
    st.subheader("Qualification Analysis")

    col1, col2, col3 = st.columns(3, gap="medium")

    # --------------------------------------------------
    # EDUCATION
    # --------------------------------------------------

    with col1:
        with st.container(border=True):
            st.markdown("### Education")
            st.write(f"**Candidate:** {str(candidate_degree).upper()}")
            st.write(f"**Required:** {str(required_degree).upper()}")
            st.divider()
            if education_match_score >= 1:
                st.success("Education requirement satisfied")
            else:
                st.error("Education requirement not satisfied")

    # --------------------------------------------------
    # EXPERIENCE
    # --------------------------------------------------

    with col2:
        with st.container(border=True):
            st.markdown("### Experience")
            st.write(f"**Candidate:** {candidate_experience} Years")
            st.write(f"**Required:** {required_experience} Years")
            st.divider()
            if required_experience <= 0:
                st.info("Experience not specified")
            elif candidate_experience >= required_experience:
                st.success("Experience requirement satisfied")
            else:
                st.error("Experience requirement not satisfied")

    # --------------------------------------------------
    # CERTIFICATIONS
    # --------------------------------------------------

    with col3:
        with st.container(border=True):
            st.markdown("### Certifications")
            st.write("**Candidate:**")
            st.write(", ".join(candidate_certifications) if candidate_certifications else "None")
            st.write("**Required:**")
            st.write(", ".join(required_certifications) if required_certifications else "None")
            st.divider()
            if certification_match_score >= 1:
                st.success("Certification requirement satisfied")
            elif certification_match_score > 0:
                st.warning("Partial certification match")
            else:
                if required_certifications:
                    st.error("Required certifications missing")
                else:
                    st.info("Certifications not required")

    # ======================================================
    # MATCH QUALITY ANALYSIS — single HTML block
    # ======================================================

    st.divider()

    # Helper for color coding
    def _bar_color(val):
        if val >= 0.75: return "#10B981"
        if val >= 0.5:  return "#F59E0B"
        return "#EF4444"

    def _skill_badge(status):
        colors = {
            "Strong":   ("#ECFDF5","#065F46","#A7F3D0"),
            "Moderate": ("#FFFBEB","#92400E","#FDE68A"),
            "Weak":     ("#FEF2F2","#991B1B","#FECACA"),
        }
        bg, txt, border = colors.get(status, ("#F3F4F6","#374151","#D1D5DB"))
        return f"""<span style="background:{bg};color:{txt};border:1px solid {border};
            padding:4px 12px;border-radius:999px;font-size:12px;font-weight:700;">
            {status}</span>"""

    st.html(f"""
    <div style="font-size:22px;font-weight:800;color:#0F172A;
        letter-spacing:-0.5px;margin-bottom:20px;">Match Quality Analysis</div>

    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:14px;">

        <!-- SKILLS -->
        <div class="metric-card" style="text-align:center;">
            <div class="metric-label" style="margin-bottom:10px;">Skill Match</div>
            <div style="font-size:28px;font-weight:900;color:{_bar_color(skill_match_score)};
                letter-spacing:-1px;">{skill_match_score*100:.0f}%</div>
            <div style="margin-top:10px;">{_skill_badge(skill_status)}</div>
        </div>

        <!-- EDUCATION -->
        <div class="metric-card" style="text-align:center;">
            <div class="metric-label" style="margin-bottom:10px;">Education</div>
            <div style="font-size:28px;font-weight:900;color:{_bar_color(education_match_score)};
                letter-spacing:-1px;">{education_match_score*100:.0f}%</div>
            <div style="margin-top:10px;font-size:12px;color:#6B7280;font-weight:600;">
                {education_status}
            </div>
        </div>

        <!-- EXPERIENCE -->
        <div class="metric-card" style="text-align:center;">
            <div class="metric-label" style="margin-bottom:10px;">Experience</div>
            <div style="font-size:28px;font-weight:900;color:{_bar_color(experience_match_score)};
                letter-spacing:-1px;">{experience_match_score*100:.0f}%</div>
            <div style="margin-top:10px;font-size:12px;color:#6B7280;font-weight:600;">
                {experience_status}
            </div>
        </div>

        <!-- CERTIFICATION -->
        <div class="metric-card" style="text-align:center;">
            <div class="metric-label" style="margin-bottom:10px;">Certification</div>
            <div style="font-size:28px;font-weight:900;color:{_bar_color(certification_match_score)};
                letter-spacing:-1px;">{certification_match_score*100:.0f}%</div>
            <div style="margin-top:10px;font-size:12px;color:#6B7280;font-weight:600;">
                {certification_status}
            </div>
        </div>

        <!-- SEMANTIC -->
        <div class="metric-card" style="text-align:center;">
            <div class="metric-label" style="margin-bottom:10px;">Semantic</div>
            <div style="font-size:28px;font-weight:900;color:{_bar_color(semantic_similarity)};
                letter-spacing:-1px;">{semantic_similarity*100:.0f}%</div>
            <div style="margin-top:10px;font-size:12px;color:#6B7280;font-weight:600;">
                Sentence-BERT
            </div>
        </div>

    </div>
    """)

    # ======================================================
    # NLP ANALYSIS — single HTML block
    # ======================================================

    st.divider()

    tfidf_pct   = min(max(float(tfidf_similarity), 0.0), 1.0)
    sbert_pct   = min(max(float(semantic_similarity), 0.0), 1.0)

    st.html(f"""
    <div style="font-size:22px;font-weight:800;color:#0A1628;
        letter-spacing:-0.5px;margin-bottom:20px;">NLP Similarity Analysis</div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">

        <div class="analytics-card">
            <div style="font-size:16px;font-weight:700;color:#0A1628;margin-bottom:4px;">
                TF-IDF Similarity
            </div>
            <div style="font-size:13px;color:#64748B;margin-bottom:16px;font-weight:500;">
                Lexical overlap between resume and job description.
            </div>
            <div style="background:#DBEAFE;border-radius:999px;height:12px;overflow:hidden;margin-bottom:10px;">
                <div style="background-color:#0077B6;
                    height:100%;width:{tfidf_pct*100:.1f}%;border-radius:999px;"></div>
            </div>
            <div style="font-size:22px;font-weight:800;color:#0077B6;letter-spacing:-0.5px;">
                {tfidf_similarity*100:.2f}%
            </div>
        </div>

        <div class="analytics-card">
            <div style="font-size:16px;font-weight:700;color:#0A1628;margin-bottom:4px;">
                Sentence-BERT Similarity
            </div>
            <div style="font-size:13px;color:#64748B;margin-bottom:16px;font-weight:500;">
                Deep semantic similarity via transformer embeddings.
            </div>
            <div style="background:#DBEAFE;border-radius:999px;height:12px;overflow:hidden;margin-bottom:10px;">
                <div style="background-color:#0096C7;
                    height:100%;width:{sbert_pct*100:.1f}%;border-radius:999px;"></div>
            </div>
            <div style="font-size:22px;font-weight:800;color:#0096C7;letter-spacing:-0.5px;">
                {semantic_similarity*100:.2f}%
            </div>
        </div>

    </div>
    """)

    # ======================================================
    # FEATURE IMPORTANCE
    # ======================================================

    st.divider()
    st.subheader("Model Feature Importance")

    st.caption(
        "Importance calculated directly from the trained Random Forest model."
    )

    if hasattr(model, "feature_importances_"):

        importance_values = model.feature_importances_

        if len(importance_values) == len(FEATURE_COLUMNS):

            importance_df = pd.DataFrame({
                "Feature":    FEATURE_COLUMNS,
                "Importance": importance_values,
            })

            feature_names = {
                "tfidf_similarity":         "TF-IDF Similarity",
                "semantic_similarity":      "Semantic Similarity",
                "skill_match_score":        "Skill Match",
                "education_match_score":    "Education Match",
                "experience_match_score":   "Experience Match",
                "certification_match_score":"Certification Match",
                "resume_length":            "Resume Length",
                "job_length":               "Job Length",
                "resume_word_count":        "Resume Word Count",
                "job_word_count":           "Job Word Count",
                "candidate_skill_count":    "Candidate Skill Count",
                "required_skill_count":     "Required Skill Count",
                "skill_overlap_count":      "Skill Overlap Count",
                "education_exact_match":    "Exact Education Match",
            }

            importance_df["Feature"] = (
                importance_df["Feature"]
                .map(feature_names)
                .fillna(importance_df["Feature"])
            )

            importance_df["Importance"] = importance_df["Importance"] * 100

            importance_df = importance_df.sort_values(
                by="Importance", ascending=True
            )

            fig = px.bar(
                importance_df,
                x="Importance",
                y="Feature",
                orientation="h",
                text="Importance",
                color="Importance",
                color_continuous_scale=[
                    [0.0,  "#DBEAFE"],
                    [0.5,  "#38BDF8"],
                    [1.0,  "#0077B6"],
                ],
            )

            fig.update_traces(
                texttemplate="%{x:.1f}%",
                textposition="outside",
                cliponaxis=False,
            )

            fig.update_layout(
                height=560,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#374151", size=13),
                coloraxis_showscale=False,
                margin=dict(l=240, r=80, t=20, b=40),
                xaxis=dict(
                    title="Model Importance (%)",
                    range=[0, max(10, importance_df["Importance"].max() * 1.2)],
                    showgrid=True,
                    gridcolor="#F3F4F6",
                    zeroline=False,
                    tickfont=dict(color="#6B7280"),
                    title_font=dict(color="#6B7280"),
                ),
                yaxis=dict(
                    title="",
                    tickfont=dict(color="#374151", size=13),
                    automargin=True,
                ),
            )

            st.plotly_chart(
                fig,
                use_container_width=True,
                config={"displaylogo": False},
            )

            with st.expander("View Raw Feature Importance Values"):
                display_df = importance_df.copy()
                display_df["Importance"] = (
                    display_df["Importance"].round(2).astype(str) + "%"
                )
                st.dataframe(display_df, use_container_width=True, hide_index=True)

        else:
            st.warning("Feature column count mismatch.")

    else:
        st.warning("The loaded model does not expose `feature_importances_`.")

    # ======================================================
    # WHY THIS SCORE? (LOCAL SHAP)
    # ======================================================

    st.divider()
    st.html("""
    <div style="font-size:22px;font-weight:800;color:#0F172A;
        letter-spacing:-0.5px;margin-bottom:8px;">Why this score? (Explainable AI)</div>
    <div style="font-size:14px;color:#64748B;font-weight:500;margin-bottom:20px;">
        Understand exactly how each feature influenced the final ATS score for this specific candidate.
    </div>
    """)

    with st.spinner("Generating AI explanation..."):
        try:
            import numpy as np
            explainer = get_shap_explainer()
            X = pd.DataFrame([features], columns=FEATURE_COLUMNS)
            X_scaled = scaler.transform(X)
            
            shap_values = explainer.shap_values(X_scaled)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]
            shap_values = np.asarray(shap_values).flatten()
            
            expected_value = explainer.expected_value
            if isinstance(expected_value, (list, np.ndarray)):
                expected_value = float(expected_value[0])
                
            feature_names = {
                "tfidf_similarity":         "Keyword Overlap (TF-IDF)",
                "semantic_similarity":      "Meaning Similarity (SBERT)",
                "skill_match_score":        "Skill Match",
                "education_match_score":    "Education Match",
                "experience_match_score":   "Experience Match",
                "certification_match_score":"Certification Match",
                "resume_length":            "Resume Length",
                "job_length":               "Job Length",
                "resume_word_count":        "Resume Word Count",
                "job_word_count":           "Job Word Count",
                "candidate_skill_count":    "Candidate Skill Count",
                "required_skill_count":     "Required Skill Count",
                "skill_overlap_count":      "Skill Overlap Count",
                "education_exact_match":    "Exact Education Match",
            }

            contributions = []
            for i, feature in enumerate(FEATURE_COLUMNS):
                display_name = feature_names.get(feature, feature)
                shap_val = float(shap_values[i])
                contributions.append({
                    "Feature": display_name,
                    "Value": round(float(features[feature]), 4),
                    "Impact": shap_val,
                    "Direction": "Positive" if shap_val > 0 else "Negative" if shap_val < 0 else "Neutral"
                })
                
            contributions.sort(key=lambda x: abs(x["Impact"]), reverse=True)
            
            pos_html = ""
            neg_html = ""
            
            for c in contributions:
                if c["Impact"] == 0: continue
                impact_pct = c["Impact"] * 100
                color = "#10B981" if impact_pct > 0 else "#EF4444"
                sign = "+" if impact_pct > 0 else ""
                row = f'''
                <div style="display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid #F1F5F9;">
                    <div style="font-size:13px;color:#475569;font-weight:600;">{c["Feature"]} <span style="color:#94A3B8;font-weight:400;margin-left:6px;">(value: {c["Value"]})</span></div>
                    <div style="font-size:13px;color:{color};font-weight:800;">{sign}{impact_pct:.2f}%</div>
                </div>
                '''
                if impact_pct > 0:
                    pos_html += row
                else:
                    neg_html += row
                    
            st.html(f'''
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;">
                <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:16px;padding:20px;">
                    <div style="font-size:14px;font-weight:800;color:#0F172A;margin-bottom:12px;display:flex;align-items:center;gap:8px;">
                        <span style="color:#10B981;font-size:18px;">+</span> Factors Increasing Score
                    </div>
                    {pos_html or '<div style="font-size:13px;color:#94A3B8;">No significant positive factors.</div>'}
                </div>
                <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:16px;padding:20px;">
                    <div style="font-size:14px;font-weight:800;color:#0F172A;margin-bottom:12px;display:flex;align-items:center;gap:8px;">
                        <span style="color:#EF4444;font-size:18px;">−</span> Factors Decreasing Score
                    </div>
                    {neg_html or '<div style="font-size:13px;color:#94A3B8;">No significant negative factors.</div>'}
                </div>
            </div>
            <div style="margin-top:16px;font-size:13px;color:#64748B;font-weight:500;text-align:center;">
                Base model expectation: <span style="font-weight:700;color:#0F172A;">{expected_value*100:.1f}%</span>
            </div>
            ''')
            
            with st.expander("View SHAP Waterfall Chart"):
                import shap
                import matplotlib.pyplot as plt
                
                fig, ax = plt.subplots(figsize=(10, 6))
                explanation = shap.Explanation(
                    values=shap_values,
                    base_values=expected_value,
                    data=X_scaled[0],
                    feature_names=[feature_names.get(f, f) for f in FEATURE_COLUMNS]
                )
                shap.plots.waterfall(explanation, max_display=10, show=False)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)
                
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")

    # ======================================================
    # AI SUGGESTIONS
    # ======================================================

    st.divider()

    # Build suggestions as HTML
    suggestion_items = ""
    specific = False

    for skill in missing_skills[:6]:
        suggestion_items += f"""
        <div style="
            display:flex;align-items:flex-start;gap:12px;
            padding:12px 16px;
            background-color:#EFF6FF;
            border:1px solid #C7D2FE;
            border-radius:12px;
            margin-bottom:10px;
        ">
            <div>
                <div style="font-size:14px;font-weight:700;color:#1E40AF;">
                    Learn <strong>{skill.title()}</strong>
                </div>
                <div style="font-size:12px;color:#6B7280;margin-top:2px;font-weight:500;">
                    Required skill not found in resume
                </div>
            </div>
        </div>"""
        specific = True

    if candidate_experience < required_experience:
        suggestion_items += """
        <div style="
            display:flex;align-items:flex-start;gap:12px;
            padding:12px 16px;
            background-color:#FFFBEB;
            border:1px solid #FDE68A;
            border-radius:12px;
            margin-bottom:10px;
        ">
            <div>
                <div style="font-size:14px;font-weight:700;color:#92400E;">
                    Gain More Experience
                </div>
                <div style="font-size:12px;color:#6B7280;margin-top:2px;font-weight:500;">
                    Seek relevant project, internship, or professional roles
                </div>
            </div>
        </div>"""
        specific = True

    if not candidate_certifications and required_certifications:
        suggestion_items += """
        <div style="
            display:flex;align-items:flex-start;gap:12px;
            padding:12px 16px;
            background-color:#F0FDF4;
            border:1px solid #BBF7D0;
            border-radius:12px;
            margin-bottom:10px;
        ">
            <div>
                <div style="font-size:14px;font-weight:700;color:#065F46;">
                    Add Industry Certifications
                </div>
                <div style="font-size:12px;color:#6B7280;margin-top:2px;font-weight:500;">
                    Certifications can significantly boost your ATS score
                </div>
            </div>
        </div>"""
        specific = True

    # Always-present general tips
    suggestion_items += """
    <div style="
        display:flex;align-items:flex-start;gap:12px;
        padding:12px 16px;
        background:#F9FAFB;
        border:1px solid #E5E7EB;
        border-radius:12px;
        margin-bottom:10px;
    ">
        <div style="font-size:14px;font-weight:600;color:#374151;">
            Keep resume concise — ideally 1–2 pages
        </div>
    </div>
    <div style="
        display:flex;align-items:flex-start;gap:12px;
        padding:12px 16px;
        background:#F9FAFB;
        border:1px solid #E5E7EB;
        border-radius:12px;
        margin-bottom:10px;
    ">
        <div style="font-size:14px;font-weight:600;color:#374151;">
            Include GitHub and LinkedIn profile links
        </div>
    </div>"""

    if not specific:
        no_gaps_banner = """
        <div style="
            padding:16px 20px;
            background-color:#ECFDF5;
            border:1px solid #6EE7B7;
            border-radius:14px;
            margin-bottom:16px;
            font-size:15px;font-weight:700;color:#065F46;
        ">Excellent! No major improvement gaps detected.</div>"""
    else:
        no_gaps_banner = ""

    st.html(f"""
    <div style="font-size:22px;font-weight:800;color:#0F172A;
        letter-spacing:-0.5px;margin-bottom:20px;">AI Resume Suggestions</div>

    <div style="
        background:#FFFFFF;
        border:1px solid rgba(0,119,182,0.12);
        border-radius:20px;
        padding:24px;
        box-shadow:0 4px 20px rgba(0,119,182,0.06);
    ">
        {no_gaps_banner}
        {suggestion_items}
    </div>
    """)

    # ======================================================
    # RAW FEATURE VALUES
    # ======================================================

    st.divider()

    with st.expander("View Raw Feature Values"):

        raw_df = pd.DataFrame({
            "Feature": list(features.keys()),
            "Value":   list(features.values()),
        })

        st.dataframe(raw_df, use_container_width=True, hide_index=True)

    # ======================================================
    # PDF REPORT
    # ======================================================

    st.divider()
    st.subheader("ATS Report")

    pdf_file = generate_pdf(result)

    with open(pdf_file, "rb") as f:

        st.download_button(
            label="Download Professional ATS Report",
            data=f,
            file_name="ATS_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )