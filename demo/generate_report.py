import pandas as pd
from pathlib import Path
from PIL import Image as PILImage
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image,
    Table, TableStyle, PageBreak
)

REFERENCE = "a0081"

base_dir = Path(__file__).resolve().parent
project_dir = base_dir.parent

test_dir = project_dir / "dataset" / "Benchmark" / "Test"
reference_dir = project_dir / "dataset" / "Benchmark" / "Test_References"
results_dir = base_dir / "results"

output_path = base_dir / "INRetouch_Visual_Comparison.pdf"

df = pd.read_csv(results_dir / "metrics.csv")
df["preset"] = df["preset"].astype(str)
df["image"] = df["image"].astype(str)

# ---------------------------------------------------------------- styles
styles = getSampleStyleSheet()

styles.add(ParagraphStyle(
    name="ReportTitle", parent=styles["Title"],
    alignment=TA_CENTER, fontSize=18, leading=22, spaceAfter=8
))
styles.add(ParagraphStyle(
    name="SectionHeading", parent=styles["Heading2"],
    fontSize=11, leading=13, spaceBefore=5, spaceAfter=5
))
styles.add(ParagraphStyle(
    name="ImageLabel", parent=styles["Normal"],
    alignment=TA_CENTER, fontSize=8, leading=9
))

# ---------------------------------------------------------------- page geometry
LEFT, RIGHT, TOP, BOTTOM = 25, 25, 20, 20
FRAME_PAD = 6          # SimpleDocTemplate's default frame padding (each side)
SAFETY = 12            # spare points so rounding can never overflow the page

doc = SimpleDocTemplate(
    str(output_path), pagesize=A4,
    rightMargin=RIGHT, leftMargin=LEFT, topMargin=TOP, bottomMargin=BOTTOM
)

frame_w = A4[0] - LEFT - RIGHT - 2 * FRAME_PAD
frame_h = A4[1] - TOP - BOTTOM - 2 * FRAME_PAD

LABEL_H = 12
ROW_PAD = 2
ROW_GAP = 2

# ---------------------------------------------------------------- helpers
def find_image(directory, image_name):
    matches = sorted(
        p for p in directory.iterdir()
        if p.is_file()
        and p.stem.startswith(image_name)
        and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not matches:
        raise FileNotFoundError(
            f"No image starting with {image_name} found in {directory}"
        )
    if len(matches) > 1:
        print(f"Multiple matches for {image_name}. Using {matches[0].name}")
    return matches[0]


def make_image(path, max_width, max_height):
    with PILImage.open(path) as img:
        width, height = img.size
    scale = min(max_width / width, max_height / height)
    return Image(str(path), width=width * scale, height=height * scale)


def make_image_row(paths, labels, image_width, image_height, gap=2):
    """labels=None -> image-only row (no label strip)."""
    images = [make_image(p, image_width, image_height) for p in paths]
    col_widths = [img.drawWidth + gap for img in images]

    data = [images]
    row_heights = [image_height + ROW_PAD]
    if labels is not None:
        data.append([Paragraph(l, styles["ImageLabel"]) for l in labels])
        row_heights.append(LABEL_H)

    table = Table(data, colWidths=col_widths, rowHeights=row_heights,
                  hAlign="CENTER")
    table.setStyle(TableStyle([
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return table


def flowable_height(flowable):
    """Height a flowable will actually occupy, including space before/after."""
    _, h = flowable.wrap(frame_w, frame_h)
    return h + flowable.getSpaceBefore() + flowable.getSpaceAfter()


# ---------------------------------------------------------------- build
story = []
presets = sorted(df["preset"].unique(), key=int)

reference_width = 2.65 * inch
reference_height = 1.35 * inch
image_width = 2.05 * inch
max_image_height = 0.95 * inch

for preset in presets:
    preset_dir = f"Preset_{preset}"

    title = Paragraph(f"Preset {preset}", styles["ReportTitle"])
    heading_ref = Paragraph("Training Reference Transformation",
                            styles["SectionHeading"])
    heading_test = Paragraph("Test Image Comparisons",
                             styles["SectionHeading"])

    reference_original = find_image(reference_dir / "natural", REFERENCE)
    reference_retouched = find_image(reference_dir / "Presets" / preset_dir,
                                     REFERENCE)

    ref_row = make_image_row(
        [reference_original, reference_retouched],
        [f"Original ({REFERENCE})", "Retouched"],
        reference_width, reference_height
    )

    preset_data = df[df["preset"] == preset].sort_values("image")
    n = len(preset_data)

    # ---- work out how tall each test row may be so the page always fits
    fixed_h = (
        flowable_height(title)
        + flowable_height(heading_ref)
        + (reference_height + ROW_PAD + LABEL_H)   # reference row
        + 7                                         # spacer under it
        + flowable_height(heading_test)
        + (n - 1) * ROW_GAP                         # gaps between test rows
        + LABEL_H                                   # label strip, last row only
        + SAFETY
    )
    per_row = (frame_h - fixed_h) / n - ROW_PAD
    image_height = max(10, min(max_image_height, per_row))

    if per_row < max_image_height:
        print(f"Preset {preset}: {n} rows, image height reduced to "
              f"{image_height / inch:.2f} in to fit one page")

    # ---- assemble the page
    story.append(title)
    story.append(heading_ref)
    story.append(ref_row)
    story.append(Spacer(1, 7))
    story.append(heading_test)

    for i, row in enumerate(preset_data.itertuples()):
        name = row.image
        input_path = find_image(test_dir / "natural", name)
        output_image = find_image(results_dir / preset_dir, name)
        ground_truth = find_image(test_dir / "Presets" / preset_dir, name)

        is_last = i == n - 1
        story.append(make_image_row(
            [input_path, output_image, ground_truth],
            ["Input", "INRetouch", "Ground Truth"] if is_last else None,
            image_width, image_height
        ))
        if not is_last:                 # no trailing spacer after the last row
            story.append(Spacer(1, ROW_GAP))

    if preset != presets[-1]:
        story.append(PageBreak())

doc.build(story)
print(f"Report generated: {output_path}")