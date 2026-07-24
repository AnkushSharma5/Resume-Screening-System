from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet


def generate_report(result, output_path):
    """
    Generate ATS Resume Analysis Report.
    """

    doc = SimpleDocTemplate(output_path)

    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("<b>AI Resume Screening Report</b>", styles["Title"]))

    elements.append(Paragraph("<br/>", styles["Normal"]))

    elements.append(
        Paragraph(
            f"<b>Overall Score:</b> {result['overall_score']}%",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"<b>Similarity Score:</b> {result['similarity_score']}%",
            styles["Normal"]
        )
    )

    elements.append(
        Paragraph(
            f"<b>Skill Match Score:</b> {result['skill_match_score']}%",
            styles["Normal"]
        )
    )

    elements.append(Paragraph("<br/>", styles["Normal"]))

    elements.append(
        Paragraph(
            "<b>Matched Skills</b>",
            styles["Heading2"]
        )
    )

    for skill in result["matched_skills"]:
        elements.append(
            Paragraph(f"• {skill}", styles["Normal"])
        )

    elements.append(Paragraph("<br/>", styles["Normal"]))

    elements.append(
        Paragraph(
            "<b>Missing Skills</b>",
            styles["Heading2"]
        )
    )

    for skill in result["missing_skills"]:
        elements.append(
            Paragraph(f"• {skill}", styles["Normal"])
        )

    doc.build(elements)