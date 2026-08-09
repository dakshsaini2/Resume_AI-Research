"""
report.py

Professional PDF ATS Report
"""

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    KeepTogether,
)

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import (
    getSampleStyleSheet,
    ParagraphStyle,
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.units import mm


# ==========================================================
# GENERATE PDF
# ==========================================================

def generate_pdf(result, filename="ATS_Report.pdf"):

    # ======================================================
    # DOCUMENT
    # ======================================================

    doc = SimpleDocTemplate(
        filename,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
    )

    # ======================================================
    # COLORS
    # ======================================================

    BLUE = colors.HexColor("#007AFF")
    DARK = colors.HexColor("#1D1D1F")
    GREY = colors.HexColor("#6E6E73")
    LIGHT_GREY = colors.HexColor("#F5F5F7")
    BORDER = colors.HexColor("#D2D2D7")
    GREEN = colors.HexColor("#34C759")
    RED = colors.HexColor("#FF3B30")
    WHITE = colors.white

    # ======================================================
    # STYLES
    # ======================================================

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "ATS_Title",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=25,
        leading=30,
        textColor=DARK,
        alignment=TA_CENTER,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "ATS_Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=14,
        textColor=GREY,
        alignment=TA_CENTER,
        spaceAfter=20,
    )

    section_style = ParagraphStyle(
        "ATS_Section",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=20,
        textColor=DARK,
        spaceBefore=8,
        spaceAfter=10,
    )

    body_style = ParagraphStyle(
        "ATS_Body",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=DARK,
    )

    small_style = ParagraphStyle(
        "ATS_Small",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=GREY,
    )

    score_style = ParagraphStyle(
        "ATS_Score",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=30,
        leading=35,
        textColor=BLUE,
        alignment=TA_CENTER,
    )

    recommendation_style = ParagraphStyle(
        "ATS_Recommendation",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=13,
        leading=18,
        textColor=GREEN,
        alignment=TA_CENTER,
    )

    # ======================================================
    # STORY
    # ======================================================

    story = []

    # ======================================================
    # HEADER
    # ======================================================

    story.append(
        Paragraph(
            "AI Resume Screening Report",
            title_style,
        )
    )

    story.append(
        Paragraph(
            "Explainable AI powered Resume & Job Description Analysis",
            subtitle_style,
        )
    )

    # ======================================================
    # SCORE CARD
    # ======================================================

    score = float(result.get("score", 0))
    recommendation = result.get(
        "recommendation",
        "Not Available"
    )

    score_data = [
        [
            Paragraph(
                f"{score:.2f}%",
                score_style
            )
        ],
        [
            Paragraph(
                "ATS MATCH SCORE",
                small_style
            )
        ],
        [
            Paragraph(
                recommendation,
                recommendation_style
            )
        ],
    ]

    score_table = Table(
        score_data,
        colWidths=[170 * mm],
        rowHeights=[18 * mm, 8 * mm, 10 * mm],
    )

    score_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, -1),
                LIGHT_GREY
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                1,
                BORDER
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "ALIGN",
                (0, 0),
                (-1, -1),
                "CENTER"
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5
            ),
        ])
    )

    story.append(score_table)

    story.append(Spacer(1, 15))

    # ======================================================
    # CANDIDATE SUMMARY
    # ======================================================

    story.append(
        Paragraph(
            "👤 Candidate Summary",
            section_style
        )
    )

    candidate_degree = result.get(
        "candidate_degree",
        "Unknown"
    )

    required_degree = result.get(
        "required_degree",
        "Unknown"
    )

    candidate_experience = result.get(
        "candidate_experience",
        0
    )

    required_experience = result.get(
        "required_experience",
        0
    )

    candidate_skills = result.get(
        "candidate_skills",
        []
    )

    candidate_certifications = result.get(
        "candidate_certifications",
        []
    )

    summary_data = [
        [
            "Field",
            "Candidate",
            "Required"
        ],
        [
            "Education",
            str(candidate_degree).upper(),
            str(required_degree).upper()
        ],
        [
            "Experience",
            f"{candidate_experience} Years",
            f"{required_experience} Years"
        ],
        [
            "Skills",
            str(len(candidate_skills)),
            str(
                len(
                    result.get(
                        "required_skills",
                        []
                    )
                )
            )
        ],
        [
            "Certifications",
            str(len(candidate_certifications)),
            str(
                len(
                    result.get(
                        "required_certifications",
                        []
                    )
                )
            )
        ],
    ]

    summary_table = Table(
        summary_data,
        colWidths=[
            45 * mm,
            60 * mm,
            60 * mm
        ],
        repeatRows=1,
    )

    summary_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                BLUE
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                WHITE
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "BACKGROUND",
                (0, 1),
                (-1, -1),
                WHITE
            ),
            (
                "TEXTCOLOR",
                (0, 1),
                (-1, -1),
                DARK
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                BORDER
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
        ])
    )

    story.append(summary_table)

    story.append(Spacer(1, 15))

    # ======================================================
    # SKILLS ANALYSIS
    # ======================================================

    matched_skills = result.get(
        "matched_skills",
        []
    )

    missing_skills = result.get(
        "missing_skills",
        []
    )

    story.append(
        Paragraph(
            "🛠 Skills Analysis",
            section_style
        )
    )

    matched_text = (
        ", ".join(matched_skills)
        if matched_skills
        else "None"
    )

    missing_text = (
        ", ".join(missing_skills)
        if missing_skills
        else "None"
    )

    skills_data = [
        [
            Paragraph(
                "<b>Matched Skills</b>",
                body_style
            ),
            Paragraph(
                "<b>Missing Skills</b>",
                body_style
            ),
        ],
        [
            Paragraph(
                matched_text,
                body_style
            ),
            Paragraph(
                missing_text,
                body_style
            ),
        ],
    ]

    skills_table = Table(
        skills_data,
        colWidths=[
            82 * mm,
            82 * mm
        ],
    )

    skills_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (0, 0),
                colors.HexColor("#E8F5E9")
            ),
            (
                "BACKGROUND",
                (1, 0),
                (1, 0),
                colors.HexColor("#FFEBEE")
            ),
            (
                "BACKGROUND",
                (0, 1),
                (0, 1),
                colors.HexColor("#F6FFF7")
            ),
            (
                "BACKGROUND",
                (1, 1),
                (1, 1),
                colors.HexColor("#FFF8F8")
            ),
            (
                "BOX",
                (0, 0),
                (-1, -1),
                0.5,
                BORDER
            ),
            (
                "INNERGRID",
                (0, 0),
                (-1, -1),
                0.5,
                BORDER
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
        ])
    )

    story.append(skills_table)

    story.append(Spacer(1, 15))

    # ======================================================
    # CERTIFICATIONS
    # ======================================================

    required_certifications = result.get(
        "required_certifications",
        []
    )

    story.append(
        Paragraph(
            "📜 Certifications",
            section_style
        )
    )

    certification_data = [
        [
            "Candidate Certifications",
            "Required Certifications"
        ],
        [
            (
                ", ".join(candidate_certifications)
                if candidate_certifications
                else "None"
            ),
            (
                ", ".join(required_certifications)
                if required_certifications
                else "None"
            ),
        ],
    ]

    certification_table = Table(
        certification_data,
        colWidths=[
            82 * mm,
            82 * mm
        ]
    )

    certification_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                BLUE
            ),
            (
                "TEXTCOLOR",
                (0, 0),
                (-1, 0),
                WHITE
            ),
            (
                "FONTNAME",
                (0, 0),
                (-1, 0),
                "Helvetica-Bold"
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                BORDER
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "TOP"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                8
            ),
        ])
    )

    story.append(certification_table)

    story.append(Spacer(1, 15))

    # ======================================================
    # FEATURE VALUES
    # ======================================================

    features = result.get(
        "features",
        {}
    )

    if features:

        story.append(
            Paragraph(
                "📊 Resume Analysis Features",
                section_style
            )
        )

        feature_data = [
            ["Feature", "Value"]
        ]

        for key, value in features.items():

            feature_data.append([
                str(key).replace("_", " ").title(),
                f"{value:.4f}"
                if isinstance(value, float)
                else str(value)
            ])

        feature_table = Table(
            feature_data,
            colWidths=[
                105 * mm,
                55 * mm
            ],
            repeatRows=1
        )

        feature_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    BLUE
                ),
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 0),
                    WHITE
                ),
                (
                    "FONTNAME",
                    (0, 0),
                    (-1, 0),
                    "Helvetica-Bold"
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    BORDER
                ),
                (
                    "BACKGROUND",
                    (0, 1),
                    (-1, -1),
                    WHITE
                ),
                (
                    "TEXTCOLOR",
                    (0, 1),
                    (-1, -1),
                    DARK
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    8
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
            ])
        )

        story.append(feature_table)

        story.append(Spacer(1, 15))

    # ======================================================
    # AI SUGGESTIONS
    # ======================================================

    story.append(
        Paragraph(
            "💡 AI Suggestions",
            section_style
        )
    )

    suggestions = []

    for skill in missing_skills[:5]:

        suggestions.append(
            f"Improve knowledge of {skill}."
        )

    if candidate_experience < required_experience:

        suggestions.append(
            "Gain additional relevant project, internship, or professional experience."
        )

    if not candidate_certifications:

        suggestions.append(
            "Consider adding relevant industry certifications."
        )

    suggestions.append(
        "Keep the resume concise and focused on skills relevant to the target job."
    )

    suggestions.append(
        "Include relevant project, GitHub, LinkedIn, or portfolio links when applicable."
    )

    for suggestion in suggestions:

        story.append(
            Paragraph(
                f"• {suggestion}",
                body_style
            )
        )

        story.append(
            Spacer(1, 4)
        )

    # ======================================================
    # FOOTER
    # ======================================================

    story.append(Spacer(1, 20))

    story.append(
        Paragraph(
            "Generated by AI Resume Screening System",
            small_style
        )
    )

    story.append(
        Paragraph(
            "Developed by Daksh Saini",
            small_style
        )
    )

    # ======================================================
    # BUILD
    # ======================================================

    doc.build(story)

    return filename