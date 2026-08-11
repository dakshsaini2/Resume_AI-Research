"""
dashboard.py

Premium ATS Dashboard — all custom HTML uses st.html()
so nothing renders as raw code.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from charts import score_gauge
from dashboard_skills import skills_section
from report import generate_pdf

from config import MODEL_FILE, FEATURE_COLUMNS


# ==========================================================
# LOAD TRAINED MODEL
# ==========================================================

model = joblib.load(MODEL_FILE)


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
# SHOW DASHBOARD
# ==========================================================

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
        ">🎯 ATS Match Score</div>
        """)
        score_gauge(score)

    with right:
        st.html(f"""
        <div class="verdict-card">
            <div class="verdict-title">🏆 Recruiter Verdict</div>
            <div class="verdict-value">{recommendation}</div>
            <div class="metric-description-light">
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
                    background: rgba(255,255,255,0.08);
                    border: 1px solid rgba(0,180,216,0.25);
                    border-radius: 12px;
                    padding: 10px 18px;
                    text-align: center;
                    flex: 1;
                ">
                    <div style="font-size:26px;font-weight:900;color:#FFFFFF;letter-spacing:-1px;">
                        {score:.1f}%
                    </div>
                    <div style="font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);
                        text-transform:uppercase;letter-spacing:1px;margin-top:3px;">
                        ATS Score
                    </div>
                </div>
                <div style="
                    background: rgba(255,255,255,0.08);
                    border: 1px solid rgba(0,180,216,0.25);
                    border-radius: 12px;
                    padding: 10px 18px;
                    text-align: center;
                    flex: 1;
                ">
                    <div style="font-size:26px;font-weight:900;color:#FFFFFF;letter-spacing:-1px;">
                        {len(matched_skills)}/{len(required_skills)}
                    </div>
                    <div style="font-size:10px;font-weight:700;color:rgba(255,255,255,0.4);
                        text-transform:uppercase;letter-spacing:1px;margin-top:3px;">
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
        letter-spacing:-0.5px;margin-bottom:20px;">👤 Candidate Overview</div>

    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:8px;">

        <div class="metric-card">
            <div class="metric-label">🎓 Education</div>
            <div class="metric-value" style="font-size:22px;letter-spacing:-0.5px;">
                {str(candidate_degree).upper()}
            </div>
            <div class="metric-description">Required: {str(required_degree).upper()}</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">💼 Experience</div>
            <div class="metric-value">{candidate_experience}</div>
            <div class="metric-description">Years &nbsp;·&nbsp; Required: {required_experience} yrs</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">🛠 Skills Matched</div>
            <div class="metric-value">{len(matched_skills)}/{len(required_skills)}</div>
            <div class="metric-description">Required skills matched</div>
        </div>

        <div class="metric-card">
            <div class="metric-label">📜 Certifications</div>
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
    st.subheader("🎓 Qualification Analysis")

    col1, col2, col3 = st.columns(3, gap="medium")

    # --------------------------------------------------
    # EDUCATION
    # --------------------------------------------------

    with col1:
        with st.container(border=True):
            st.markdown("### 🎓 Education")
            st.write(f"**Candidate:** {str(candidate_degree).upper()}")
            st.write(f"**Required:** {str(required_degree).upper()}")
            st.divider()
            if education_match_score >= 1:
                st.success("✅ Education requirement satisfied")
            else:
                st.error("❌ Education requirement not satisfied")

    # --------------------------------------------------
    # EXPERIENCE
    # --------------------------------------------------

    with col2:
        with st.container(border=True):
            st.markdown("### 💼 Experience")
            st.write(f"**Candidate:** {candidate_experience} Years")
            st.write(f"**Required:** {required_experience} Years")
            st.divider()
            if required_experience <= 0:
                st.info("ℹ️ Experience not specified")
            elif candidate_experience >= required_experience:
                st.success("✅ Experience requirement satisfied")
            else:
                st.error("❌ Experience requirement not satisfied")

    # --------------------------------------------------
    # CERTIFICATIONS
    # --------------------------------------------------

    with col3:
        with st.container(border=True):
            st.markdown("### 📜 Certifications")
            st.write("**Candidate:**")
            st.write(", ".join(candidate_certifications) if candidate_certifications else "None")
            st.write("**Required:**")
            st.write(", ".join(required_certifications) if required_certifications else "None")
            st.divider()
            if certification_match_score >= 1:
                st.success("✅ Certification requirement satisfied")
            elif certification_match_score > 0:
                st.warning("⚠️ Partial certification match")
            else:
                if required_certifications:
                    st.error("❌ Required certifications missing")
                else:
                    st.info("ℹ️ Certifications not required")

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
        letter-spacing:-0.5px;margin-bottom:20px;">🔍 Match Quality Analysis</div>

    <div style="display:grid;grid-template-columns:repeat(5,1fr);gap:14px;">

        <!-- SKILLS -->
        <div class="metric-card" style="text-align:center;">
            <div class="metric-label" style="margin-bottom:10px;">🛠 Skill Match</div>
            <div style="font-size:28px;font-weight:900;color:{_bar_color(skill_match_score)};
                letter-spacing:-1px;">{skill_match_score*100:.0f}%</div>
            <div style="margin-top:10px;">{_skill_badge(skill_status)}</div>
        </div>

        <!-- EDUCATION -->
        <div class="metric-card" style="text-align:center;">
            <div class="metric-label" style="margin-bottom:10px;">🎓 Education</div>
            <div style="font-size:28px;font-weight:900;color:{_bar_color(education_match_score)};
                letter-spacing:-1px;">{education_match_score*100:.0f}%</div>
            <div style="margin-top:10px;font-size:12px;color:#6B7280;font-weight:600;">
                {education_status}
            </div>
        </div>

        <!-- EXPERIENCE -->
        <div class="metric-card" style="text-align:center;">
            <div class="metric-label" style="margin-bottom:10px;">💼 Experience</div>
            <div style="font-size:28px;font-weight:900;color:{_bar_color(experience_match_score)};
                letter-spacing:-1px;">{experience_match_score*100:.0f}%</div>
            <div style="margin-top:10px;font-size:12px;color:#6B7280;font-weight:600;">
                {experience_status}
            </div>
        </div>

        <!-- CERTIFICATION -->
        <div class="metric-card" style="text-align:center;">
            <div class="metric-label" style="margin-bottom:10px;">📜 Certification</div>
            <div style="font-size:28px;font-weight:900;color:{_bar_color(certification_match_score)};
                letter-spacing:-1px;">{certification_match_score*100:.0f}%</div>
            <div style="margin-top:10px;font-size:12px;color:#6B7280;font-weight:600;">
                {certification_status}
            </div>
        </div>

        <!-- SEMANTIC -->
        <div class="metric-card" style="text-align:center;">
            <div class="metric-label" style="margin-bottom:10px;">🧠 Semantic</div>
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
        letter-spacing:-0.5px;margin-bottom:20px;">🧠 NLP Similarity Analysis</div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;">

        <div class="analytics-card">
            <div style="font-size:16px;font-weight:700;color:#0A1628;margin-bottom:4px;">
                📝 TF-IDF Similarity
            </div>
            <div style="font-size:13px;color:#64748B;margin-bottom:16px;font-weight:500;">
                Lexical overlap between resume and job description.
            </div>
            <div style="background:#DBEAFE;border-radius:999px;height:12px;overflow:hidden;margin-bottom:10px;">
                <div style="background:linear-gradient(90deg,#0077B6,#00B4D8);
                    height:100%;width:{tfidf_pct*100:.1f}%;border-radius:999px;"></div>
            </div>
            <div style="font-size:22px;font-weight:800;color:#0077B6;letter-spacing:-0.5px;">
                {tfidf_similarity*100:.2f}%
            </div>
        </div>

        <div class="analytics-card">
            <div style="font-size:16px;font-weight:700;color:#0A1628;margin-bottom:4px;">
                🧠 Sentence-BERT Similarity
            </div>
            <div style="font-size:13px;color:#64748B;margin-bottom:16px;font-weight:500;">
                Deep semantic similarity via transformer embeddings.
            </div>
            <div style="background:#DBEAFE;border-radius:999px;height:12px;overflow:hidden;margin-bottom:10px;">
                <div style="background:linear-gradient(90deg,#0096C7,#00B4D8);
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
    st.subheader("📊 Model Feature Importance")

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
                "tfidf_similarity":         "📝 TF-IDF Similarity",
                "semantic_similarity":      "🧠 Semantic Similarity",
                "skill_match_score":        "🛠 Skill Match",
                "education_match_score":    "🎓 Education Match",
                "experience_match_score":   "💼 Experience Match",
                "certification_match_score":"📜 Certification Match",
                "resume_length":            "📄 Resume Length",
                "job_length":               "📑 Job Length",
                "resume_word_count":        "📄 Resume Word Count",
                "job_word_count":           "📑 Job Word Count",
                "candidate_skill_count":    "🛠 Candidate Skill Count",
                "required_skill_count":     "📋 Required Skill Count",
                "skill_overlap_count":      "✅ Skill Overlap Count",
                "education_exact_match":    "🎓 Exact Education Match",
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

            with st.expander("🔍 View Raw Feature Importance Values"):
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
            background:linear-gradient(135deg,#EFF6FF,#EDE9FE);
            border:1px solid #C7D2FE;
            border-radius:12px;
            margin-bottom:10px;
        ">
            <span style="font-size:18px;margin-top:1px;">📚</span>
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
            background:linear-gradient(135deg,#FFFBEB,#FEF3C7);
            border:1px solid #FDE68A;
            border-radius:12px;
            margin-bottom:10px;
        ">
            <span style="font-size:18px;margin-top:1px;">💼</span>
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
            background:linear-gradient(135deg,#F0FDF4,#DCFCE7);
            border:1px solid #BBF7D0;
            border-radius:12px;
            margin-bottom:10px;
        ">
            <span style="font-size:18px;margin-top:1px;">📜</span>
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
        <span style="font-size:18px;margin-top:1px;">📄</span>
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
        <span style="font-size:18px;margin-top:1px;">🔗</span>
        <div style="font-size:14px;font-weight:600;color:#374151;">
            Include GitHub and LinkedIn profile links
        </div>
    </div>"""

    if not specific:
        no_gaps_banner = """
        <div style="
            padding:16px 20px;
            background:linear-gradient(135deg,#ECFDF5,#D1FAE5);
            border:1px solid #6EE7B7;
            border-radius:14px;
            margin-bottom:16px;
            font-size:15px;font-weight:700;color:#065F46;
        ">🎉 Excellent! No major improvement gaps detected.</div>"""
    else:
        no_gaps_banner = ""

    st.html(f"""
    <div style="font-size:22px;font-weight:800;color:#0F172A;
        letter-spacing:-0.5px;margin-bottom:20px;">💡 AI Resume Suggestions</div>

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

    with st.expander("🔍 View Raw Feature Values"):

        raw_df = pd.DataFrame({
            "Feature": list(features.keys()),
            "Value":   list(features.values()),
        })

        st.dataframe(raw_df, use_container_width=True, hide_index=True)

    # ======================================================
    # PDF REPORT
    # ======================================================

    st.divider()
    st.subheader("📄 ATS Report")

    pdf_file = generate_pdf(result)

    with open(pdf_file, "rb") as f:

        st.download_button(
            label="📄  Download Professional ATS Report",
            data=f,
            file_name="ATS_Report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )