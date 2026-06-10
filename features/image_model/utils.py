import io
import cv2
import torch
import numpy as np
import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from albumentations.pytorch import ToTensorV2
from pytorch_grad_cam.utils.image import show_cam_on_image

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, HRFlowable
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT

# -----------------------------------------------------
# IMPORT MODEL CONFIG
# -----------------------------------------------------
try:
    from .model import device, IMG_SIZE
except ImportError:
    from model import device, IMG_SIZE


# -----------------------------------------------------
# IMAGE DECODING
# -----------------------------------------------------
def _decode_image_bytes(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# -----------------------------------------------------
# TRANSFORM
# -----------------------------------------------------
val_tfms = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(),
    ToTensorV2()
])


# -----------------------------------------------------
# FRACTAL MAP
# -----------------------------------------------------
def fractal_map(gray, patch=16):

    h, w = gray.shape

    H = (h // patch) * patch
    W = (w // patch) * patch

    gray = gray[:H, :W]

    out = np.zeros((H, W), dtype=np.float32)

    thresholds = [64, 96, 128, 160, 192]

    for y in range(0, H, patch):
        for x in range(0, W, patch):

            p = gray[y:y+patch, x:x+patch]

            counts = [(p > t).sum() for t in thresholds]

            out[y:y+patch, x:x+patch] = np.std(counts)

    out = (out - out.min()) / (out.max() - out.min() + 1e-6)

    return out


# -----------------------------------------------------
# FIXED 4-CHANNEL INPUT (IMPORTANT FIX HERE)
# -----------------------------------------------------
def _build_4ch_input(rgb, patch=16):

    H, W = rgb.shape[:2]

    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    f_map = fractal_map(gray, patch=patch)

    # 🔥 FIX: FORCE SAME SIZE AS RGB
    f_map = cv2.resize(f_map, (W, H), interpolation=cv2.INTER_LINEAR)

    f_map = np.nan_to_num(f_map)

    f_u8 = (f_map * 255).astype(np.uint8)

    f_u8 = f_u8[:, :, None]

    return np.concatenate([rgb, f_u8], axis=2)


# -----------------------------------------------------
# MODEL PREDICTION
# -----------------------------------------------------
@torch.no_grad()
def _predict(img4, model):

    x = val_tfms(image=img4)["image"].unsqueeze(0).to(device)

    logits = model(x)

    probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

    return float(probs[0]), float(probs[1])


# -----------------------------------------------------
# VISUAL HELPERS
# -----------------------------------------------------
def _render_heatmap(rgb, prob_map, thr):
    """Render a 3-panel heatmap strip: original | prob map | blended overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor("#0f0f1a")

    # Panel 1 – original
    axes[0].imshow(rgb)
    axes[0].set_title("Original Image", color="white", fontsize=10, fontweight="bold")
    axes[0].axis("off")

    # Panel 2 – probability heatmap
    im = axes[1].imshow(prob_map, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Probability Heatmap", color="white", fontsize=10, fontweight="bold")
    axes[1].axis("off")
    cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=7)

    # Panel 3 – blended overlay with contour
    overlay = rgb.copy().astype(np.float32) / 255.0
    heatmap_colored = matplotlib.colormaps["jet"](prob_map)[:, :, :3]
    blended = np.clip(0.55 * overlay + 0.45 * heatmap_colored, 0, 1)
    axes[2].imshow(blended)
    hot_contour = (prob_map >= thr).astype(np.uint8)
    axes[2].contour(hot_contour, levels=[0.5], colors=["red"], linewidths=1.5)
    axes[2].set_title("Suspicious Regions", color="white", fontsize=10, fontweight="bold")
    axes[2].axis("off")

    for ax in axes:
        ax.set_facecolor("#0f0f1a")

    plt.tight_layout(pad=1.5)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _render_bar_chart(prob_real, prob_fake):
    """Render a horizontal confidence bar chart."""
    fig, ax = plt.subplots(figsize=(6, 2.2))
    fig.patch.set_facecolor("#0f0f1a")
    ax.set_facecolor("#0f0f1a")

    bars = ax.barh(
        ["Real", "Fake"],
        [prob_real, prob_fake],
        color=["#2ecc71", "#e74c3c"],
        height=0.5,
        edgecolor="white",
        linewidth=0.6,
    )

    for bar, val in zip(bars, [prob_real, prob_fake]):
        ax.text(
            min(val + 0.02, 0.95),
            bar.get_y() + bar.get_height() / 2,
            f"{val * 100:.1f}%",
            va="center", ha="left", color="white",
            fontsize=11, fontweight="bold",
        )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence", color="white", fontsize=9)
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")
    ax.axvline(0.5, color="#888", linestyle="--", linewidth=0.8)
    ax.xaxis.label.set_color("white")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# -----------------------------------------------------
# MAIN DETECTOR
# -----------------------------------------------------
def detect_deepfake_best_localization(
    img_bytes,
    model,
    patch_for_fractal=16,
    grid=8,
    thr=0.6,
):

    rgb = _decode_image_bytes(img_bytes)

    rgb = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))

    img4 = _build_4ch_input(rgb, patch_for_fractal)

    prob_real, prob_fake = _predict(img4, model)

    label = "Fake" if prob_fake >= prob_real else "Real"

    # -------------------------
    # REAL CASE
    # -------------------------
    if label == "Real":
        bar_bytes = _render_bar_chart(prob_real, prob_fake)
        return {
            "prediction": "Real",
            "prob_real": prob_real,
            "prob_fake": prob_fake,
            "suspicious_area_percent": 0.0,
            "heatmap_strip_bytes": b"",
            "bar_chart_bytes": bar_bytes,
        }

    # -------------------------
    # FAKE CASE — patch scoring
    # -------------------------
    H = IMG_SIZE
    patch = H // grid

    prob_map = np.zeros((H, H))
    count    = np.zeros((H, H))

    blur = cv2.GaussianBlur(rgb, (0, 0), 7)

    for y in range(0, H - patch, patch):
        for x in range(0, H - patch, patch):

            temp = blur.copy()
            temp[y:y+patch, x:x+patch] = rgb[y:y+patch, x:x+patch]

            img4_patch = _build_4ch_input(temp, patch_for_fractal)
            _, pf = _predict(img4_patch, model)

            prob_map[y:y+patch, x:x+patch] += pf
            count[y:y+patch, x:x+patch]    += 1

    prob_map /= np.maximum(count, 1e-6)

    hot     = (prob_map >= thr).astype(np.uint8)
    percent = float(hot.mean() * 100)

    # -------------------------
    # GENERATE VISUALS
    # -------------------------
    heatmap_bytes = _render_heatmap(rgb, prob_map, thr)
    bar_bytes     = _render_bar_chart(prob_real, prob_fake)

    return {
        "prediction": "Fake",
        "prob_real": prob_real,
        "prob_fake": prob_fake,
        "suspicious_area_percent": percent,
        "heatmap_strip_bytes": heatmap_bytes,
        "bar_chart_bytes": bar_bytes,
    }


# -----------------------------------------------------
# PDF GENERATION — Professional TemperEye Report
# -----------------------------------------------------
def generate_pdf_report(result, image_name,
                         heatmap_strip_bytes,
                         bar_chart_bytes,
                         out_path):

    # ── Colour palette ──────────────────────────────────────────────────────
    DARK_BG    = colors.HexColor("#ffffff")
    ACCENT     = colors.HexColor("#173B6C")   # Dark Blue Header
    ACCENT2    = colors.HexColor("#000000")
    FAKE_RED   = colors.HexColor("#e74c3c")
    REAL_GREEN = colors.HexColor("#2ecc71")
    WHITE      = colors.white
    LIGHT_GRAY = colors.black
    MID_GRAY   = colors.HexColor("#c0c0c0")
    ROW_ALT    = colors.HexColor("#f2f2f2")

    PAGE_W, PAGE_H = A4
    MARGIN = 18 * mm

    # ── Paragraph styles ────────────────────────────────────────────────────
    title_style = ParagraphStyle(
        "TETitle",
        fontSize=26, fontName="Helvetica-Bold",
        textColor=LIGHT_GRAY, alignment=TA_CENTER,
        spaceAfter=2,
    )
    subtitle_style = ParagraphStyle(
        "TESubtitle",
        fontSize=10, fontName="Helvetica",
        textColor=ACCENT2, alignment=TA_CENTER,
        spaceAfter=14,
    )
    section_style = ParagraphStyle(
        "TESection",
        fontSize=12, fontName="Helvetica-Bold",
        textColor=ACCENT2, spaceBefore=14, spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "TEBody",
        fontSize=9, fontName="Helvetica",
        textColor=LIGHT_GRAY, spaceAfter=4,
    )
    caption_style = ParagraphStyle(
        "TECaption",
        fontSize=8, fontName="Helvetica-Oblique",
        textColor=colors.HexColor("#9ca3af"), alignment=TA_CENTER,
        spaceAfter=8,
    )
    disclaimer_style = ParagraphStyle(
        "TEDisclaimer",
        fontSize=8, fontName="Helvetica",
        textColor=colors.HexColor("#6b7280"), spaceAfter=4,
    )

    # ── Per-page canvas callback (dark background + branded header/footer) ──
    def on_page(canvas, doc):
        canvas.saveState()
        # Dark background fill
        #canvas.setFillColor(DARK_BG)
        #canvas.rect(0, 0, PAGE_W, PAGE_H, fill=1, stroke=0)
        # Top accent bar
        canvas.setFillColor(ACCENT)
        canvas.rect(0, PAGE_H - 14 * mm, PAGE_W, 14 * mm, fill=1, stroke=0)
        # Brand in header
        canvas.setFillColor(WHITE)
        canvas.setFont("Helvetica-Bold", 13)
        canvas.drawString(MARGIN, PAGE_H - 9.5 * mm, "TemperEye")
        canvas.setFillColor(ACCENT2)
        canvas.setFont("Helvetica", 9)
        canvas.drawString(MARGIN + 78, PAGE_H - 9.5 * mm, "AI Deepfake Detection Platform")
        # Footer line
        canvas.setStrokeColor(ACCENT)
        canvas.setLineWidth(1.2)
        canvas.line(MARGIN, 15 * mm, PAGE_W - MARGIN, 15 * mm)
        # Footer text
        canvas.setFillColor(colors.HexColor("#6b7280"))
        canvas.setFont("Helvetica", 8)
        canvas.drawRightString(PAGE_W - MARGIN, 8 * mm,
                               f"Page {doc.page}  |  Confidential")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        out_path, pagesize=A4,
        leftMargin=MARGIN, rightMargin=MARGIN,
        topMargin=22 * mm, bottomMargin=22 * mm,
    )

    story = []

    # ── Title block ─────────────────────────────────────────────────────────
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("Deepfake Analysis Report", title_style))
    story.append(Spacer(1, 6 * mm))
    story.append(Spacer(1, 6 * mm))
    story.append(Paragraph("Powered by TemperEye Neural Forensics Engine", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=1, color=ACCENT, spaceAfter=10))

    # ── Verdict banner ───────────────────────────────────────────────────────
    is_fake = result["prediction"] == "Fake"
    verdict_color = FAKE_RED if is_fake else REAL_GREEN
    verdict_label = "&#9888;  FAKE DETECTED" if is_fake else "&#10004;  AUTHENTIC"

    verdict_style = ParagraphStyle(
        "Verdict",
        fontSize=18, fontName="Helvetica-Bold",
        textColor=verdict_color, alignment=TA_CENTER,
        spaceBefore=4, spaceAfter=4,
    )
    story.append(Paragraph(verdict_label, verdict_style))
    story.append(Spacer(1, 4 * mm))

    # ── Summary table ────────────────────────────────────────────────────────
    story.append(Paragraph("Detection Summary", section_style))

    susp = result.get("suspicious_area_percent", 0.0)
    summary_data = [
        ["Metric", "Value"],
        ["File Analysed", image_name],
        ["Verdict", result["prediction"]],
        ["Authenticity Confidence", f"{result['prob_real'] * 100:.2f}%"],
        ["Manipulation Confidence", f"{result['prob_fake'] * 100:.2f}%"],
        ["Suspicious Area", f"{susp:.2f}%" if is_fake else "N/A"],
    ]

    col_w = [(PAGE_W - 2 * MARGIN) * 0.45, (PAGE_W - 2 * MARGIN) * 0.55]
    tbl = Table(summary_data, colWidths=col_w)
    tbl.setStyle(TableStyle([
        # Header
        ("BACKGROUND",    (0, 0), (-1, 0), ACCENT),
        ("TEXTCOLOR",     (0, 0), (-1, 0), WHITE),
        ("FONTNAME",      (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE",      (0, 0), (-1, 0), 10),
        ("ALIGN",         (0, 0), (-1, 0), "CENTER"),
        # Data rows
        ("FONTNAME",      (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE",      (0, 1), (-1, -1), 9),
        ("TEXTCOLOR",     (0, 1), (0, -1), ACCENT2),
        ("TEXTCOLOR",     (1, 1), (1, -1), LIGHT_GRAY),
        # Alternating row shading
        *[("BACKGROUND",  (0, i), (-1, i), ROW_ALT)  for i in range(2, len(summary_data), 2)],
        *[("BACKGROUND",  (0, i), (-1, i), DARK_BG)  for i in range(1, len(summary_data), 2)],
        # Grid
        ("GRID",          (0, 0), (-1, -1), 0.5, MID_GRAY),
        ("TOPPADDING",    (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING",   (0, 0), (-1, -1), 10),
    ]))
    story.append(tbl)
    story.append(Spacer(1, 6 * mm))

    # ── Confidence bar chart ─────────────────────────────────────────────────
    if bar_chart_bytes:
        story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GRAY, spaceAfter=6))
        story.append(Paragraph("Confidence Scores", section_style))
        buf = io.BytesIO(bar_chart_bytes)
        chart_w = PAGE_W - 2 * MARGIN
        img = RLImage(buf, width=chart_w * 0.7, height=chart_w * 0.7 * (2.2 / 6))
        story.append(img)
        story.append(Paragraph(
            "Horizontal bars show the model's confidence for Real vs Fake classification.",
            caption_style,
        ))

    # ── Heatmap section (Fake only) ──────────────────────────────────────────
    if is_fake and heatmap_strip_bytes:
        story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GRAY, spaceAfter=6))
        story.append(Paragraph("Spatial Manipulation Analysis", section_style))
        story.append(Paragraph(
            "The heatmap visualises regions of the image that contributed most strongly "
            "to the fake classification. Red contours mark areas exceeding the detection threshold.",
            body_style,
        ))
        story.append(Spacer(1, 3 * mm))
        buf = io.BytesIO(heatmap_strip_bytes)
        img_w = PAGE_W - 2 * MARGIN
        img_h = img_w * (4 / 12)   # preserves the 12×4 figsize ratio
        story.append(RLImage(buf, width=img_w, height=img_h))
        story.append(Paragraph(
            "Left: Original  |  Centre: Per-pixel fake probability  |  "
            "Right: Blended overlay with suspicious region contours",
            caption_style,
        ))
        story.append(Spacer(1, 4 * mm))

    # ── Methodology note ─────────────────────────────────────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=MID_GRAY, spaceAfter=6))
    story.append(Paragraph("Methodology", section_style))
    story.append(Paragraph(
        "TemperEye uses a 4-channel neural network combining RGB image data with a "
        "fractal complexity map computed via multi-threshold box-counting. Spatial "
        "localisation is performed by occlusion-based patch scoring: each image region "
        "is selectively revealed against a Gaussian-blurred background and the per-patch "
        "fake probability is recorded to construct the probability heatmap above.",
        body_style,
    ))
    story.append(Spacer(1, 3 * mm))
    story.append(Paragraph(
        "<b>Disclaimer:</b> This report is generated automatically by an AI system and "
        "should be treated as a forensic aid, not a definitive legal determination. "
        "Results should be interpreted in conjunction with other evidence.",
        disclaimer_style,
    ))

    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)