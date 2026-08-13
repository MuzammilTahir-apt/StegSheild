import os
import io
import uuid
from datetime import datetime

import torch
from torch import nn
from torchvision import transforms, models

from PIL import Image

from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    send_file
)

from werkzeug.utils import secure_filename

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.utils import ImageReader


# ============================================================
# STEGSHIELD - FLASK APPLICATION
# ============================================================

app = Flask(__name__)

app.secret_key = "stegshield-dev-secret-key"


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "resnet18_stegshield_new.pth"
)

UPLOAD_FOLDER = os.path.join(
    BASE_DIR,
    "static",
    "uploads"
)

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((224, 224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD RESNET18 MODEL
# ============================================================

print("Loading StegShield model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model = models.resnet18(weights=None)

num_features = model.fc.in_features

model.fc = nn.Linear(
    num_features,
    2
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()

class_names = checkpoint.get(
    "classes",
    ["cleanTrain", "stegoTrain"]
)

print("Classes:", class_names)

print("Model loaded successfully.")


# ============================================================
# PREDICTION
# ============================================================

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    image_width, image_height = image.size

    image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(device)

    with torch.no_grad():

        outputs = model(image_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            1
        )

    predicted_class = class_names[
        predicted.item()
    ]

    confidence = confidence.item() * 100

    probs_list = probabilities.squeeze(0).tolist()

    class_probabilities = {
        class_names[i]: probs_list[i] * 100
        for i in range(len(class_names))
    }

    return (
        predicted_class,
        confidence,
        class_probabilities,
        (image_width, image_height)
    )


# ============================================================
# HELPERS
# ============================================================

def format_file_size(num_bytes):

    if num_bytes < 1024 * 1024:

        return f"{num_bytes / 1024:.1f} KB"

    return f"{num_bytes / (1024 * 1024):.2f} MB"


def split_clean_stego(class_probabilities):

    clean_prob = 0.0
    stego_prob = 0.0

    for name, pct in class_probabilities.items():

        if "stego" in name.lower():

            stego_prob = pct

        else:

            clean_prob = pct

    return clean_prob, stego_prob


def compute_threat_level(stego_prob):

    if stego_prob >= 90:
        return "HIGH"

    if stego_prob >= 70:
        return "MEDIUM"

    if stego_prob >= 50:
        return "LOW"

    return "NONE"


# ============================================================
# PDF COLORS
# ============================================================

PDF_BG = colors.HexColor("#0a0f1c")
PDF_CARD_BG = colors.HexColor("#0f1729")
PDF_CARD_BORDER = colors.HexColor("#22314d")

PDF_CYAN = colors.HexColor("#22d3ee")
PDF_BLUE = colors.HexColor("#3b82f6")

PDF_WHITE = colors.HexColor("#f1f5f9")
PDF_GRAY = colors.HexColor("#94a3b8")
PDF_GRAY_DIM = colors.HexColor("#64748b")

PDF_GREEN = colors.HexColor("#10b981")
PDF_RED = colors.HexColor("#ef4444")
PDF_ORANGE = colors.HexColor("#f97316")

PDF_PAGE_W, PDF_PAGE_H = A4

PDF_MARGIN = 42


# ============================================================
# PDF HELPERS
# ============================================================

def pdf_threat_color(level):

    return {
        "HIGH": PDF_RED,
        "MEDIUM": PDF_ORANGE,
        "LOW": colors.HexColor("#eab308"),
        "NONE": PDF_GREEN
    }.get(
        level,
        PDF_GRAY
    )


def _pdf_background(c):

    c.setFillColor(PDF_BG)

    c.rect(
        0,
        0,
        PDF_PAGE_W,
        PDF_PAGE_H,
        fill=1,
        stroke=0
    )


def _pdf_header(c):

    y_top = PDF_PAGE_H - 40

    c.setFillColor(PDF_BLUE)

    c.roundRect(
        PDF_MARGIN,
        y_top - 6,
        22,
        22,
        5,
        fill=1,
        stroke=0
    )

    c.setFillColor(PDF_WHITE)

    c.setFont(
        "Helvetica-Bold",
        11
    )

    c.drawCentredString(
        PDF_MARGIN + 11,
        y_top + 1,
        "S"
    )

    c.setFillColor(PDF_WHITE)

    c.setFont(
        "Helvetica-Bold",
        13
    )

    c.drawString(
        PDF_MARGIN + 32,
        y_top + 1,
        "STEGSHIELD"
    )

    c.setFillColor(PDF_GRAY)

    c.setFont(
        "Helvetica",
        8
    )

    c.drawRightString(
        PDF_PAGE_W - PDF_MARGIN,
        y_top + 3,
        "AI IMAGE SECURITY ANALYSIS"
    )

    c.setStrokeColor(
        colors.HexColor("#1e293b")
    )

    c.setLineWidth(1)

    c.line(
        PDF_MARGIN,
        y_top - 14,
        PDF_PAGE_W - PDF_MARGIN,
        y_top - 14
    )

    c.setStrokeColor(PDF_BLUE)

    c.setLineWidth(1.4)

    c.line(
        PDF_MARGIN,
        y_top - 14,
        PDF_MARGIN +
        (
            PDF_PAGE_W -
            2 * PDF_MARGIN
        ) * 0.42,
        y_top - 14
    )


def _pdf_footer(c, page_num):

    y = 32

    c.setStrokeColor(
        colors.HexColor("#1e293b")
    )

    c.setLineWidth(0.7)

    c.line(
        PDF_MARGIN,
        y + 14,
        PDF_PAGE_W - PDF_MARGIN,
        y + 14
    )

    c.setFillColor(PDF_GRAY_DIM)

    c.setFont(
        "Helvetica",
        7.5
    )

    c.drawString(
        PDF_MARGIN,
        y,
        "StegShield • AI-Powered Image Security Analysis • ResNet18 Detection Engine"
    )

    c.drawRightString(
        PDF_PAGE_W - PDF_MARGIN,
        y,
        f"Page {page_num}"
    )


def _pdf_section_title(c, x, y, text):

    c.setFillColor(PDF_CYAN)

    c.setFont(
        "Helvetica-Bold",
        12.5
    )

    c.drawString(
        x,
        y,
        text
    )

    return y - 20


def _pdf_card(
    c,
    x,
    y,
    w,
    h,
    radius=10
):

    c.setFillColor(PDF_CARD_BG)

    c.setStrokeColor(PDF_CARD_BORDER)

    c.setLineWidth(1)

    c.roundRect(
        x,
        y,
        w,
        h,
        radius,
        fill=1,
        stroke=1
    )


def _pdf_paragraph(
    c,
    x,
    y,
    width,
    text,
    size=9.5,
    color=PDF_GRAY,
    leading=14
):

    style = ParagraphStyle(
        "StegShieldParagraph",
        fontName="Helvetica",
        fontSize=size,
        textColor=color,
        leading=leading,
        alignment=TA_LEFT
    )

    p = Paragraph(
        text,
        style
    )

    w, h = p.wrap(
        width,
        400
    )

    p.drawOn(
        c,
        x,
        y - h
    )

    return y - h


def _pdf_table_card(
    c,
    x,
    top_y,
    w,
    rows,
    row_h=28
):

    h = row_h * len(rows)

    y = top_y - h

    _pdf_card(
        c,
        x,
        y,
        w,
        h
    )

    for i, (label, value) in enumerate(rows):

        row_top = (
            top_y -
            i * row_h
        )

        row_bottom = (
            row_top -
            row_h
        )

        if i > 0:

            c.setStrokeColor(
                PDF_CARD_BORDER
            )

            c.setLineWidth(0.6)

            c.line(
                x + 14,
                row_top,
                x + w - 14,
                row_top
            )

        text_y = (
            row_bottom +
            row_h / 2 -
            3.5
        )

        c.setFillColor(PDF_CYAN)

        c.setFont(
            "Helvetica-Bold",
            9.5
        )

        c.drawString(
            x + 16,
            text_y,
            label
        )

        c.setFillColor(PDF_WHITE)

        c.setFont(
            "Helvetica",
            9.5
        )

        c.drawRightString(
            x + w - 16,
            text_y,
            str(value)
        )

    return y


def _pdf_stat_card(
    c,
    x,
    y,
    w,
    h,
    label,
    value,
    value_color
):

    _pdf_card(
        c,
        x,
        y - h,
        w,
        h,
        radius=8
    )

    c.setFillColor(PDF_GRAY)

    c.setFont(
        "Helvetica-Bold",
        7.5
    )

    c.drawCentredString(
        x + w / 2,
        y - 20,
        label
    )

    c.setFillColor(value_color)

    c.setFont(
        "Helvetica-Bold",
        13
    )

    c.drawCentredString(
        x + w / 2,
        y - h + h / 2 - 10,
        str(value)
    )


def _pdf_probability_bar(
    c,
    x,
    y,
    w,
    label,
    percent,
    color
):

    c.setFillColor(PDF_WHITE)

    c.setFont(
        "Helvetica-Bold",
        9.5
    )

    c.drawString(
        x,
        y,
        label
    )

    c.setFillColor(color)

    c.setFont(
        "Helvetica-Bold",
        9.5
    )

    c.drawRightString(
        x + w,
        y,
        f"{percent:.2f}%"
    )

    bar_y = y - 14

    bar_h = 8

    c.setFillColor(
        colors.HexColor("#1e293b")
    )

    c.roundRect(
        x,
        bar_y - bar_h,
        w,
        bar_h,
        bar_h / 2,
        fill=1,
        stroke=0
    )

    fill_w = max(
        6,
        w * (percent / 100)
    )

    c.setFillColor(color)

    c.roundRect(
        x,
        bar_y - bar_h,
        fill_w,
        bar_h,
        bar_h / 2,
        fill=1,
        stroke=0
    )

    tick_y = (
        bar_y -
        bar_h -
        12
    )

    c.setFillColor(
        PDF_GRAY_DIM
    )

    c.setFont(
        "Helvetica",
        7
    )

    for tick in [0, 25, 50, 75, 100]:

        tick_x = (
            x +
            w * (tick / 100)
        )

        c.drawCentredString(
            tick_x,
            tick_y,
            str(tick)
        )

    return tick_y - 20


# ============================================================
# PDF PAGE 1
# ============================================================

def _draw_cover_page(c, data):

    _pdf_background(c)

    _pdf_header(c)

    _pdf_footer(c, 1)

    content_w = (
        PDF_PAGE_W -
        2 * PDF_MARGIN
    )

    y = PDF_PAGE_H - 100

    c.setFillColor(PDF_WHITE)

    c.setFont(
        "Helvetica-Bold",
        26
    )

    c.drawCentredString(
        PDF_PAGE_W / 2,
        y,
        "STEGSHIELD"
    )

    y -= 20

    c.setFillColor(PDF_GRAY)

    c.setFont(
        "Helvetica",
        9
    )

    c.drawCentredString(
        PDF_PAGE_W / 2,
        y,
        "A I   I M A G E   S E C U R I T Y   A N A L Y S I S"
    )

    y -= 30

    c.setFillColor(PDF_BLUE)

    c.setFont(
        "Helvetica-Bold",
        16
    )

    c.drawCentredString(
        PDF_PAGE_W / 2,
        y,
        "Steganography Detection Report"
    )

    y -= 46

    y = _pdf_section_title(
        c,
        PDF_MARGIN,
        y,
        "Report Overview"
    )

    classification = (
        "STEGO"
        if data["is_stego"]
        else "CLEAN"
    )

    overview_rows = [
        (
            "Analyzed Image",
            data["filename"]
        ),
        (
            "Analysis Date / Time",
            data["timestamp"]
        ),
        (
            "AI Engine",
            data["model_name"]
        ),
        (
            "Classification",
            classification
        ),
        (
            "Threat Level",
            data["threat_level"]
        )
    ]

    y = _pdf_table_card(
        c,
        PDF_MARGIN,
        y,
        content_w,
        overview_rows,
        row_h=28
    ) - 26

    verdict_h = 66

    verdict_color = (
        PDF_RED
        if data["is_stego"]
        else PDF_GREEN
    )

    verdict_bg = (
        colors.HexColor("#2a1414")
        if data["is_stego"]
        else colors.HexColor("#0f2419")
    )

    c.setFillColor(verdict_bg)

    c.setStrokeColor(verdict_color)

    c.setLineWidth(1.2)

    c.roundRect(
        PDF_MARGIN,
        y - verdict_h,
        content_w,
        verdict_h,
        10,
        fill=1,
        stroke=1
    )

    c.setFillColor(PDF_GRAY)

    c.setFont(
        "Helvetica-Bold",
        8.5
    )

    c.drawCentredString(
        PDF_PAGE_W / 2,
        y - 22,
        "AI VERDICT"
    )

    c.setFillColor(verdict_color)

    c.setFont(
        "Helvetica-Bold",
        22
    )

    c.drawCentredString(
        PDF_PAGE_W / 2,
        y - 48,
        classification
    )

    y -= (
        verdict_h +
        24
    )

    desc = (
        "This report was generated automatically by "
        "StegShield's ResNet18 detection engine. It documents "
        "the image classification result, confidence, "
        "probability breakdown, and image metadata."
    )

    y = _pdf_paragraph(
        c,
        PDF_MARGIN,
        y,
        content_w,
        desc
    )

    y -= 26

    y = _pdf_section_title(
        c,
        PDF_MARGIN,
        y,
        "Contents"
    )

    c.setFillColor(PDF_GRAY)

    c.setFont(
        "Helvetica",
        9.5
    )

    c.drawString(
        PDF_MARGIN,
        y,
        "02   Analyzed Image & Image Details"
    )

    y -= 16

    c.drawString(
        PDF_MARGIN,
        y,
        "03   AI Analysis Summary, Model Probabilities & Interpretation"
    )


# ============================================================
# PDF PAGE 2
# ============================================================

def _draw_image_page(c, data):

    _pdf_background(c)

    _pdf_header(c)

    _pdf_footer(c, 2)

    content_w = (
        PDF_PAGE_W -
        2 * PDF_MARGIN
    )

    y = PDF_PAGE_H - 90

    y = _pdf_section_title(
        c,
        PDF_MARGIN,
        y,
        "Analyzed Image"
    )

    y -= 10

    img_card_h = 300

    _pdf_card(
        c,
        PDF_MARGIN,
        y - img_card_h,
        content_w,
        img_card_h
    )

    image_drawn = False

    if (
        data.get("image_path")
        and os.path.exists(
            data["image_path"]
        )
    ):

        try:

            img_reader = ImageReader(
                data["image_path"]
            )

            iw, ih = img_reader.getSize()

            max_w = (
                content_w -
                60
            )

            max_h = (
                img_card_h -
                40
            )

            scale = min(
                max_w / iw,
                max_h / ih
            )

            draw_w = iw * scale

            draw_h = ih * scale

            img_x = (
                PDF_MARGIN +
                (
                    content_w -
                    draw_w
                ) / 2
            )

            img_y = (
                y -
                img_card_h +
                (
                    img_card_h -
                    draw_h
                ) / 2
            )

            c.drawImage(
                img_reader,
                img_x,
                img_y,
                width=draw_w,
                height=draw_h,
                preserveAspectRatio=True,
                mask="auto"
            )

            image_drawn = True

        except Exception as e:

            print(
                "PDF image error:",
                e
            )

    if not image_drawn:

        c.setFillColor(PDF_GRAY)

        c.setFont(
            "Helvetica",
            10
        )

        c.drawCentredString(
            PDF_PAGE_W / 2,
            y - img_card_h / 2,
            "Image preview unavailable"
        )

    y -= (
        img_card_h +
        18
    )

    c.setFillColor(
        PDF_GRAY_DIM
    )

    c.setFont(
        "Helvetica",
        9
    )

    c.drawCentredString(
        PDF_PAGE_W / 2,
        y,
        data["filename"]
    )

    y -= 34

    y = _pdf_section_title(
        c,
        PDF_MARGIN,
        y,
        "Image Details"
    )

    detail_rows = [
        (
            "Image Dimensions",
            data["dimensions"]
        ),
        (
            "File Size",
            data["file_size"]
        ),
        (
            "Input Format",
            "RGB Image"
        ),
        (
            "AI Model",
            data["model_name"]
        ),
        (
            "Detection Confidence",
            f"{data['confidence']:.2f}%"
        ),
        (
            "Detection Threshold",
            "50%"
        )
    ]

    _pdf_table_card(
        c,
        PDF_MARGIN,
        y,
        content_w,
        detail_rows,
        row_h=28
    )


# ============================================================
# PDF PAGE 3
# ============================================================

def _draw_summary_page(c, data):

    _pdf_background(c)

    _pdf_header(c)

    _pdf_footer(c, 3)

    content_w = (
        PDF_PAGE_W -
        2 * PDF_MARGIN
    )

    y = PDF_PAGE_H - 90

    y = _pdf_section_title(
        c,
        PDF_MARGIN,
        y,
        "AI Analysis Summary"
    )

    y -= 6

    is_stego = data["is_stego"]

    classification = (
        "STEGO"
        if is_stego
        else "CLEAN"
    )

    verdict_color = (
        PDF_RED
        if is_stego
        else PDF_GREEN
    )

    threat_color = pdf_threat_color(
        data["threat_level"]
    )

    cards = [
        (
            "AI VERDICT",
            classification,
            verdict_color
        ),
        (
            "THREAT LEVEL",
            data["threat_level"],
            threat_color
        ),
        (
            "MODEL CONFIDENCE",
            f"{data['confidence']:.2f}%",
            PDF_CYAN
        ),
        (
            "AI MODEL",
            data["model_name"],
            PDF_WHITE
        ),
        (
            "MODEL TYPE",
            "CNN",
            PDF_WHITE
        )
    ]

    card_w = (
        content_w -
        2 * 12
    ) / 3

    card_h = 58

    row1 = cards[:3]

    row2 = cards[3:]

    for i, (
        label,
        value,
        value_color
    ) in enumerate(row1):

        cx = (
            PDF_MARGIN +
            i * (card_w + 12)
        )

        _pdf_stat_card(
            c,
            cx,
            y,
            card_w,
            card_h,
            label,
            value,
            value_color
        )

    y -= (
        card_h +
        12
    )

    for i, (
        label,
        value,
        value_color
    ) in enumerate(row2):

        cx = (
            PDF_MARGIN +
            i * (card_w + 12)
        )

        _pdf_stat_card(
            c,
            cx,
            y,
            card_w,
            card_h,
            label,
            value,
            value_color
        )

    y -= (
        card_h +
        30
    )

    y = _pdf_section_title(
        c,
        PDF_MARGIN,
        y,
        "Model Probabilities"
    )

    y -= 10

    y = _pdf_probability_bar(
        c,
        PDF_MARGIN,
        y,
        content_w,
        "STEGANOGRAPHY",
        data["stego_prob"],
        PDF_ORANGE
    )

    y -= 10

    y = _pdf_probability_bar(
        c,
        PDF_MARGIN,
        y,
        content_w,
        "CLEAN IMAGE",
        data["clean_prob"],
        PDF_GREEN
    )

    y -= 14

    y = _pdf_section_title(
        c,
        PDF_MARGIN,
        y,
        "Security Interpretation"
    )

    if is_stego:

        interpretation = (
            "The AI model detected strong indicators associated "
            "with potential steganographic content. Further "
            "forensic investigation is recommended."
        )

    else:

        interpretation = (
            "The AI model did not detect indicators associated "
            "with steganographic content in this image based "
            "on the current classification threshold."
        )

    y = _pdf_paragraph(
        c,
        PDF_MARGIN,
        y,
        content_w,
        interpretation
    )

    y -= 8

    threshold_note = (
        "A probability below 50% is classified as CLEAN. "
        "Values from 50% to 69% are treated as LOW/UNCERTAIN, "
        "70% to 89% as MEDIUM, and 90% or above as HIGH."
    )

    y = _pdf_paragraph(
        c,
        PDF_MARGIN,
        y,
        content_w,
        threshold_note
    )

    y -= 22

    y = _pdf_section_title(
        c,
        PDF_MARGIN,
        y,
        "Disclaimer"
    )

    disclaimer = (
        "<i>AI-based detection provides an assessment of "
        "image characteristics and should not be treated as "
        "definitive forensic proof without additional "
        "investigation.</i>"
    )

    _pdf_paragraph(
        c,
        PDF_MARGIN,
        y,
        content_w,
        disclaimer,
        size=8.5,
        color=PDF_GRAY_DIM,
        leading=13
    )


# ============================================================
# BUILD PDF
# ============================================================

def build_report_pdf(data):

    buffer = io.BytesIO()

    c = pdfcanvas.Canvas(
        buffer,
        pagesize=A4
    )

    c.setTitle(
        "StegShield Detection Report"
    )

    _draw_cover_page(
        c,
        data
    )

    c.showPage()

    _draw_image_page(
        c,
        data
    )

    c.showPage()

    _draw_summary_page(
        c,
        data
    )

    c.showPage()

    c.save()

    buffer.seek(0)

    return buffer


# ============================================================
# HOME PAGE
# ============================================================

@app.route(
    "/",
    methods=["GET", "POST"]
)
def index():

    error = None

    if request.method == "POST":

        if "image" not in request.files:

            error = "No image selected."

            return render_template(
                "index.html",
                error=error
            )

        file = request.files["image"]

        if file.filename == "":

            error = "Please select an image."

            return render_template(
                "index.html",
                error=error
            )

        try:

            filename = secure_filename(
                file.filename
            )

            if not filename:

                error = "Invalid file name."

                return render_template(
                    "index.html",
                    error=error
                )

            save_path = os.path.join(
                app.config["UPLOAD_FOLDER"],
                filename
            )

            file.save(
                save_path
            )

            (
                result,
                confidence,
                class_probabilities,
                (img_w, img_h)
            ) = predict_image(
                save_path
            )

            (
                clean_prob,
                stego_prob
            ) = split_clean_stego(
                class_probabilities
            )

            is_stego = (
                "stego"
                in result.lower()
            )

            threat_level = (
                compute_threat_level(
                    stego_prob
                )
            )

            # ------------------------------------------------
            # STORE RESULT IN SESSION
            # ------------------------------------------------

            session["prediction"] = result

            session["confidence"] = confidence

            session["filename"] = filename

            session["stored_filename"] = filename

            session["image_url"] = url_for(
                "static",
                filename=(
                    "uploads/" +
                    filename
                )
            )

            session["report_id"] = (
                uuid.uuid4()
                .hex[:8]
                .upper()
            )

            session["timestamp"] = (
                datetime.now()
                .strftime(
                    "%d %B %Y, %I:%M:%S %p"
                )
            )

            session["file_size"] = (
                format_file_size(
                    os.path.getsize(
                        save_path
                    )
                )
            )

            session["image_width"] = img_w

            session["image_height"] = img_h

            session["clean_prob"] = clean_prob

            session["stego_prob"] = stego_prob

            session["threat_level"] = (
                threat_level
            )

            session["classification"] = (
                "STEGO"
                if is_stego
                else "CLEAN"
            )

            return redirect(
                url_for(
                    "result_page"
                )
            )

        except Exception as e:

            print(
                "Prediction error:",
                e
            )

            error = str(e)

    return render_template(
        "index.html",
        error=error
    )


# ============================================================
# RESULT PAGE
# ============================================================

@app.route(
    "/result",
    methods=["GET"]
)
def result_page():

    prediction = session.get(
        "prediction"
    )

    if prediction is None:

        return redirect(
            url_for("index")
        )

    return render_template(
        "result.html",

        prediction=prediction,

        confidence=session.get(
            "confidence"
        ),

        filename=session.get(
            "filename"
        ),

        stored_filename=session.get(
            "stored_filename"
        ),

        image_url=session.get(
            "image_url"
        ),

        report_id=session.get(
            "report_id"
        ),

        timestamp=session.get(
            "timestamp"
        ),

        file_size=session.get(
            "file_size"
        ),

        image_width=session.get(
            "image_width"
        ),

        image_height=session.get(
            "image_height"
        ),

        clean_prob=session.get(
            "clean_prob"
        ),

        stego_prob=session.get(
            "stego_prob"
        ),

        threat_level=session.get(
            "threat_level"
        ),

        classification=session.get(
            "classification"
        )
    )


# ============================================================
# FULL WEB REPORT PAGE
# ============================================================

@app.route(
    "/report",
    methods=["GET"]
)
def report_page():

    prediction = session.get(
        "prediction"
    )

    if prediction is None:

        return redirect(
            url_for("index")
        )

    return render_template(
        "report.html",

        prediction=prediction,

        confidence=session.get(
            "confidence"
        ),

        filename=session.get(
            "filename"
        ),

        image_url=session.get(
            "image_url"
        ),

        report_id=session.get(
            "report_id"
        ),

        timestamp=session.get(
            "timestamp"
        ),

        file_size=session.get(
            "file_size"
        ),

        image_width=session.get(
            "image_width"
        ),

        image_height=session.get(
            "image_height"
        ),

        clean_prob=session.get(
            "clean_prob"
        ),

        stego_prob=session.get(
            "stego_prob"
        ),

        threat_level=session.get(
            "threat_level"
        ),

        classification=session.get(
            "classification"
        )
    )


# ============================================================
# DOWNLOAD PDF REPORT
# ============================================================

@app.route(
    "/download-report",
    methods=["GET"]
)
def download_report():

    prediction = session.get(
        "prediction"
    )

    if prediction is None:

        return redirect(
            url_for("index")
        )

    filename = session.get(
        "filename"
    )

    image_path = None

    if filename:

        candidate = os.path.join(
            app.config["UPLOAD_FOLDER"],
            filename
        )

        if os.path.exists(candidate):

            image_path = candidate

    img_w = session.get(
        "image_width"
    )

    img_h = session.get(
        "image_height"
    )

    dimensions = (
        f"{img_w} × {img_h}"
        if img_w and img_h
        else "N/A"
    )

    data = {

        "filename":
            filename or "N/A",

        "timestamp":
            session.get(
                "timestamp",
                ""
            ),

        "model_name":
            "ResNet18",

        "is_stego":
            "stego"
            in prediction.lower(),

        "threat_level":
            session.get(
                "threat_level",
                "NONE"
            ),

        "confidence":
            session.get(
                "confidence"
            ) or 0.0,

        "stego_prob":
            session.get(
                "stego_prob"
            ) or 0.0,

        "clean_prob":
            session.get(
                "clean_prob"
            ) or 0.0,

        "dimensions":
            dimensions,

        "file_size":
            session.get(
                "file_size",
                "N/A"
            ),

        "image_path":
            image_path
    }

    buffer = build_report_pdf(
        data
    )

    report_id = session.get(
        "report_id",
        "report"
    )

    return send_file(
        buffer,
        mimetype="application/pdf",
        as_attachment=True,
        download_name=(
            f"StegShield_Report_"
            f"{report_id}.pdf"
        )
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("        STEGSHIELD IMAGE CLASSIFIER")
    print("=" * 60)

    print()
    print(
        "Open this in your browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print()
    print(
        "Press CTRL+C to stop the server."
    )

    app.run(
        debug=True,
        host="127.0.0.1",
        port=5000
    )