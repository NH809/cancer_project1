from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet

def generate_pdf(name, result, confidence, stage, filename):

    doc = SimpleDocTemplate(filename)
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("AI MRI Cancer Report", styles['Title']))
    content.append(Spacer(1, 20))

    content.append(Paragraph(f"Patient Name: {name}", styles['Normal']))
    content.append(Paragraph(f"Result: {result}", styles['Normal']))
    content.append(Paragraph(f"Confidence: {confidence}%", styles['Normal']))
    content.append(Paragraph(f"Stage: {stage}", styles['Normal']))

    doc.build(content)