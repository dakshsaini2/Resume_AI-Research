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
)

from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet


def generate_pdf(result, filename="ATS_Report.pdf"):

    doc = SimpleDocTemplate(filename)

    styles = getSampleStyleSheet()

    story = []

    # ============================
    # Title
    # ============================

    story.append(
        Paragraph(
            "<b>AI Resume Screening Report</b>",
            styles["Title"],
        )
    )

    story.append(Spacer(1, 20))

    # ============================
    # Score
    # ============================

    story.append(
        Paragraph(
            f"<b>ATS Match Score:</b> {result['score']}%",
            styles["Heading2"],
        )
    )

    story.append(
        Paragraph(
            f"<b>Recommendation:</b> {result['recommendation']}",
            styles["Heading2"],
        )
    )

    story.append(Spacer(1, 20))

    # ============================
    # Candidate Summary
    # ============================

    data = [
        ["Field", "Value"],
        ["Degree", result["candidate_degree"]],
        ["Required Degree", result["required_degree"]],
        ["Experience", str(result["candidate_experience"])],
        ["Required Experience", str(result["required_experience"])],
    ]

    table = Table(data)

    table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#007AFF")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 1, colors.grey),
            ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
            ("BOTTOMPADDING", (0, 0), (-1, 0), 10),
        ])
    )

    story.append(table)

    story.append(Spacer(1, 20))

    # ============================
    # Matched Skills
    # ============================

    story.append(
        Paragraph("<b>Matched Skills</b>", styles["Heading2"])
    )

    for skill in result["matched_skills"]:
        story.append(
            Paragraph(f"• {skill}", styles["BodyText"])
        )

    story.append(Spacer(1, 15))

    # ============================
    # Missing Skills
    # ============================

    story.append(
        Paragraph("<b>Missing Skills</b>", styles["Heading2"])
    )

    for skill in result["missing_skills"]:
        story.append(
            Paragraph(f"• {skill}", styles["BodyText"])
        )

    story.append(Spacer(1, 20))

    # ============================
    # AI Suggestions
    # ============================

    story.append(
        Paragraph("<b>AI Suggestions</b>", styles["Heading2"])
    )

    if result["missing_skills"]:
        for skill in result["missing_skills"][:5]:
            story.append(
                Paragraph(
                    f"• Improve knowledge of {skill}",
                    styles["BodyText"],
                )
            )
    else:
        story.append(
            Paragraph(
                "• Excellent profile. No major improvements suggested.",
                styles["BodyText"],
            )
        )

    doc.build(story)

    return filename