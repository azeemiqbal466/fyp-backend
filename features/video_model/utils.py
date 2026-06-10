import cv2
import numpy as np
import torch
import base64
import io
import datetime
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for FastAPI
import matplotlib.pyplot as plt
from PIL import Image

# ── ReportLab ────────────────────────────────────────────────────
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether,
)

# ─────────────────────────── CONSTANTS ───────────────────────────
IMG_SIZE   = 160
NUM_FRAMES = 4
PATCH_SIZE = 16

MEAN4 = [0.485, 0.456, 0.406, 0.5]
STD4  = [0.229, 0.224, 0.225, 0.25]

LABELS = ["Unedited", "Manipulated"]


# ─────────────────────────── VIDEO VALIDATION ────────────────────

def validate_video_duration(video_path: str, min_seconds: float = 5,
                             max_seconds: float = 20):
    """Returns (ok: bool, message: str)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False, "Could not open video file."

    fps    = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    if fps <= 0:
        return False, "Could not read FPS from video."

    duration = frames / fps

    if duration < min_seconds:
        return False, f"Video too short: {duration:.2f}s (minimum {min_seconds}s)."
    if duration > max_seconds:
        return False, f"Video too long: {duration:.2f}s (maximum {max_seconds}s)."

    return True, f"Video OK: {duration:.2f}s"


# ─────────────────────────── FRAME SAMPLING ──────────────────────

def sample_video_frames(video_path: str, num_frames: int = 8,
                        target_size: int = 160):
    """Uniformly samples `num_frames` RGB frames from a video file."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return []

    indices = np.linspace(0, total - 1, num_frames, dtype=int)
    frames  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret or frame is None:
            continue
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (target_size, target_size))
        frames.append(frame)

    cap.release()
    return frames


# ─────────────────────────── FRACTAL MAP ─────────────────────────

def fractal_map_fast(gray: np.ndarray, patch: int = 16) -> np.ndarray:
    """Computes a normalised fractal-complexity map from a grayscale image."""
    h, w = gray.shape
    H = (h // patch) * patch
    W = (w // patch) * patch

    gray_crop = gray[:H, :W]
    blocks    = gray_crop.reshape(H // patch, patch, W // patch, patch)
    blocks    = blocks.transpose(0, 2, 1, 3)

    thresholds  = np.array([64, 96, 128, 160, 192], dtype=np.uint8)
    counts      = (blocks[..., None] > thresholds).sum(axis=(2, 3))
    fmap_small  = counts.std(axis=2).astype(np.float32)
    fmap        = cv2.resize(fmap_small, (W, H), interpolation=cv2.INTER_CUBIC)

    fmap -= fmap.min()
    fmap /= fmap.max() + 1e-6
    fmap  = cv2.resize(fmap, (w, h))
    return np.clip(fmap, 0, 1)


# ─────────────────────────── 4-CHANNEL BUILD ─────────────────────

def build_4ch_frame(rgb_frame: np.ndarray, patch: int = 16) -> np.ndarray:
    """Returns a (H, W, 4) float32 array: RGB + fractal channel."""
    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    fmap = fractal_map_fast(gray, patch=patch)
    rgb  = rgb_frame.astype(np.float32) / 255.0
    return np.dstack([rgb, fmap.astype(np.float32)])


def transform_infer(frame4: np.ndarray) -> torch.Tensor:
    """Normalises a (H, W, 4) frame → (4, H, W) tensor."""
    tensor = torch.from_numpy(frame4).permute(2, 0, 1).float()
    for i in range(4):
        tensor[i] = (tensor[i] - MEAN4[i]) / STD4[i]
    return tensor


def prepare_video_clip(frames: list, num_frames: int = 8,
                       patch: int = 16) -> torch.Tensor:
    """Converts a list of RGB frames into a (1, T, 4, H, W) tensor."""
    tensors = [transform_infer(build_4ch_frame(f, patch)) for f in frames]
    while len(tensors) < num_frames:
        tensors.append(tensors[-1].clone())
    clip = torch.stack(tensors[:num_frames], dim=0).unsqueeze(0)
    return clip


# ─────────────────────────── GLOBAL PREDICTION ───────────────────

def predict_full_video(frames: list, model, device,
                       num_frames: int = 8, patch: int = 16):
    """Returns (prob_unedited, prob_manipulated) as floats."""
    clip = prepare_video_clip(frames, num_frames=num_frames,
                               patch=patch).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(clip)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    return float(probs[0]), float(probs[1])


# ─────────────────────────── ANOMALY MAPS ────────────────────────

def normalize_map(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x -= x.min()
    x /= x.max() + 1e-6
    return np.clip(x, 0, 1)


def texture_anomaly_map(rgb_frame: np.ndarray, patch: int = 16) -> np.ndarray:
    gray     = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    fmap     = fractal_map_fast(gray, patch=patch)
    edges    = cv2.Canny(gray, 80, 160).astype(np.float32) / 255.0
    blur     = cv2.GaussianBlur(gray, (9, 9), 0)
    residual = cv2.absdiff(gray, blur).astype(np.float32) / 255.0
    combined = (0.50 * normalize_map(fmap)
                + 0.25 * normalize_map(edges)
                + 0.25 * normalize_map(residual))
    return normalize_map(combined)


def temporal_difference_map(frame: np.ndarray,
                             all_frames: list) -> np.ndarray:
    stack        = np.stack(all_frames, axis=0).astype(np.float32)
    median_frame = np.median(stack, axis=0).astype(np.uint8)
    diff         = cv2.absdiff(frame, median_frame)
    diff_gray    = cv2.cvtColor(diff, cv2.COLOR_RGB2GRAY).astype(
                       np.float32) / 255.0
    return normalize_map(diff_gray)


def patch_model_probability_map(rgb_frame: np.ndarray, model, device,
                                 num_frames: int = 8,
                                 patch_for_fractal: int = 16,
                                 grid: int = 8):
    """Occlusion-based spatial manipulation probability map."""
    H, W       = rgb_frame.shape[:2]
    patch_size = H // grid

    prob_map  = np.zeros((H, W), dtype=np.float32)
    count_map = np.zeros((H, W), dtype=np.float32)
    blurred   = cv2.GaussianBlur(rgb_frame, (0, 0), sigmaX=7, sigmaY=7)

    model.eval()
    for y in range(0, H - patch_size + 1, patch_size):
        for x in range(0, W - patch_size + 1, patch_size):
            masked = blurred.copy()
            masked[y:y+patch_size, x:x+patch_size] = (
                rgb_frame[y:y+patch_size, x:x+patch_size]
            )
            frames = [masked.copy() for _ in range(num_frames)]
            clip   = prepare_video_clip(frames, num_frames=num_frames,
                                         patch=patch_for_fractal).to(device)
            with torch.no_grad():
                logits = model(clip)
                probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
            prob_map[y:y+patch_size, x:x+patch_size]  += float(probs[1])
            count_map[y:y+patch_size, x:x+patch_size] += 1.0

    prob_map = prob_map / np.maximum(count_map, 1e-6)
    return normalize_map(prob_map), prob_map


def combined_manipulation_map(frame: np.ndarray, all_frames: list,
                               model, device,
                               num_frames: int = 8,
                               patch: int = 16,
                               grid: int = 8):
    model_norm, model_raw = patch_model_probability_map(
        rgb_frame=frame, model=model, device=device,
        num_frames=num_frames, patch_for_fractal=patch, grid=grid,
    )
    texture_map  = texture_anomaly_map(frame, patch=patch)
    temporal_map = temporal_difference_map(frame, all_frames)
    combined     = (0.50 * model_norm
                    + 0.30 * texture_map
                    + 0.20 * temporal_map)
    return normalize_map(combined), model_raw, texture_map, temporal_map


def select_best_suspicious_frame(frames: list, model, device,
                                  num_frames: int = 8,
                                  patch: int = 16,
                                  grid: int = 8):
    """Returns the frame with the highest top-90th-percentile anomaly score."""
    best_score = -1.0
    best_data  = None

    for idx, frame in enumerate(frames):
        combined, model_raw, texture_map, temporal_map = (
            combined_manipulation_map(
                frame=frame, all_frames=frames, model=model, device=device,
                num_frames=num_frames, patch=patch, grid=grid,
            )
        )
        top_pixels = np.percentile(combined, 90)
        score      = float(combined[combined >= top_pixels].mean())

        if score > best_score:
            best_score = score
            best_data  = {
                "frame_index":  idx,
                "frame":        frame,
                "combined_map": combined,
                "model_raw":    model_raw,
                "texture_map":  texture_map,
                "temporal_map": temporal_map,
                "score":        score,
            }

    return best_data


# ─────────────────────────── OVERLAY HELPER ──────────────────────

def create_overlay(rgb_frame: np.ndarray,
                   anomaly_map: np.ndarray) -> np.ndarray:
    heat_u8 = np.uint8(anomaly_map * 255)
    heatmap = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(rgb_frame, 0.60, heatmap, 0.40, 0)


# ─────────────────────────── MATPLOTLIB → PIL ────────────────────

def _fig_to_pil(fig) -> Image.Image:
    """Convert a matplotlib Figure to a PIL Image (in-memory, no file)."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=120)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    img.load()          # force decode before closing buf
    plt.close(fig)
    return img


def _pil_to_rl_image(pil_img: Image.Image,
                     max_width_cm: float = 17.0) -> RLImage:
    """Wrap a PIL image as a ReportLab Image flowable, respecting page width."""
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    buf.seek(0)

    w_px, h_px    = pil_img.size
    max_width_pt  = max_width_cm * cm
    scale         = max_width_pt / w_px
    rl_w          = w_px * scale
    rl_h          = h_px * scale
    return RLImage(buf, width=rl_w, height=rl_h)


# ─────────────────────────── PDF REPORT BUILDER ──────────────────

_BRAND_DARK  = colors.HexColor("#1A1A2E")
_BRAND_MID   = colors.HexColor("#16213E")
_BRAND_BLUE  = colors.HexColor("#0F3460")
_BRAND_RED   = colors.HexColor("#E94560")
_BRAND_GREEN = colors.HexColor("#4CAF50")
_BRAND_GRAY  = colors.HexColor("#EEEEEE")
_WHITE       = colors.white


def _build_pdf_report(result: dict, pil_overview: Image.Image | None,
                      pil_components: Image.Image | None) -> bytes:
    """
    Constructs a professional multi-page PDF report.
    Returns raw PDF bytes.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=2*cm, rightMargin=2*cm,
        topMargin=2.5*cm, bottomMargin=2.5*cm,
    )

    W_PAGE = A4[0] - 4*cm     # usable width

    styles = getSampleStyleSheet()
    style_title   = ParagraphStyle("rpt_title",
                                   fontSize=22, leading=26,
                                   alignment=TA_CENTER,
                                   textColor=_WHITE,
                                   fontName="Helvetica-Bold")
    style_h1      = ParagraphStyle("rpt_h1",
                                   fontSize=14, leading=18,
                                   textColor=_BRAND_BLUE,
                                   fontName="Helvetica-Bold",
                                   spaceBefore=12, spaceAfter=6)
    style_body    = ParagraphStyle("rpt_body",
                                   fontSize=10, leading=14,
                                   textColor=colors.HexColor("#333333"),
                                   spaceAfter=4)
    style_caption = ParagraphStyle("rpt_caption",
                                   fontSize=9, leading=11,
                                   alignment=TA_CENTER,
                                   textColor=colors.HexColor("#666666"),
                                   spaceAfter=8)
    style_footer  = ParagraphStyle("rpt_footer",
                                   fontSize=8, leading=10,
                                   alignment=TA_CENTER,
                                   textColor=colors.HexColor("#999999"))

    prediction      = result.get("prediction", "N/A")
    prob_unedited   = result.get("prob_unedited", 0.0)
    prob_manip      = result.get("prob_manipulated", 0.0)
    confidence      = result.get("confidence", 0.0)
    best_frame      = result.get("best_frame_index")
    suspicious_area = result.get("suspicious_area_percent", 0.0)
    dur_msg         = result.get("duration_message", "N/A")
    timestamp       = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    is_manip        = prediction == "Manipulated"
    verdict_color   = _BRAND_RED if is_manip else _BRAND_GREEN

    story = []

    # ── HEADER BANNER ─────────────────────────────────────────────
    header_data = [[Paragraph(
        "VIDEO MANIPULATION DETECTION REPORT",
        style_title,
    )]]
    header_tbl = Table(header_data, colWidths=[W_PAGE])
    header_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), _BRAND_DARK),
        ("TOPPADDING",    (0,0), (-1,-1), 18),
        ("BOTTOMPADDING", (0,0), (-1,-1), 18),
        ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ("RIGHTPADDING",  (0,0), (-1,-1), 12),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(header_tbl)
    story.append(Spacer(1, 0.4*cm))

    # Sub-header: timestamp
    story.append(Paragraph(
        f"Generated: {timestamp}",
        style_caption,
    ))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=_BRAND_GRAY, spaceAfter=10))

    # ── VERDICT BANNER ────────────────────────────────────────────
    verdict_style = ParagraphStyle("verdict",
                                   fontSize=18, leading=22,
                                   alignment=TA_CENTER,
                                   textColor=_WHITE,
                                   fontName="Helvetica-Bold")
    verdict_data = [[Paragraph(
        f"VERDICT: {prediction.upper()}",
        verdict_style,
    )]]
    verdict_tbl = Table(verdict_data, colWidths=[W_PAGE])
    verdict_tbl.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), verdict_color),
        ("TOPPADDING",    (0,0), (-1,-1), 14),
        ("BOTTOMPADDING", (0,0), (-1,-1), 14),
        ("ROUNDEDCORNERS", [4]),
    ]))
    story.append(verdict_tbl)
    story.append(Spacer(1, 0.5*cm))

    # ── PREDICTION METRICS TABLE ──────────────────────────────────
    story.append(Paragraph("1. Prediction Metrics", style_h1))

    def pct(v: float) -> str:
        return f"{v * 100:.2f}%"

    metrics_rows = [
        ["Metric", "Value", "Notes"],
        ["Prediction",            prediction,           "Final model verdict"],
        ["Prob. Unedited",        pct(prob_unedited),   "Probability video is authentic"],
        ["Prob. Manipulated",     pct(prob_manip),      "Probability video is manipulated"],
        ["Confidence",            pct(confidence),      "max(prob_unedited, prob_manipulated)"],
        ["Suspicious Area",       f"{suspicious_area:.2f}%",
                                                         "% of best-frame flagged as anomalous"],
        ["Best Suspicious Frame", str(best_frame) if best_frame is not None else "N/A",
                                                         "Frame index with highest anomaly score"],
        ["Duration Status",       dur_msg,              "Video length validation result"],
    ]

    col_w = [W_PAGE * 0.32, W_PAGE * 0.22, W_PAGE * 0.46]
    mt    = Table(metrics_rows, colWidths=col_w, repeatRows=1)
    mt.setStyle(TableStyle([
        # Header row
        ("BACKGROUND",  (0,0), (-1,0), _BRAND_BLUE),
        ("TEXTCOLOR",   (0,0), (-1,0), _WHITE),
        ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
        ("FONTSIZE",    (0,0), (-1,0), 10),
        ("TOPPADDING",  (0,0), (-1,0), 8),
        ("BOTTOMPADDING",(0,0),(-1,0), 8),
        # Data rows
        ("FONTNAME",    (0,1), (-1,-1), "Helvetica"),
        ("FONTSIZE",    (0,1), (-1,-1), 9),
        ("TOPPADDING",  (0,1), (-1,-1), 6),
        ("BOTTOMPADDING",(0,1),(-1,-1), 6),
        ("LEFTPADDING", (0,0), (-1,-1), 8),
        ("RIGHTPADDING",(0,0), (-1,-1), 8),
        # Alternating row colours
        *[("BACKGROUND", (0,i), (-1,i), _BRAND_GRAY)
          for i in range(2, len(metrics_rows), 2)],
        # Grid
        ("GRID",        (0,0), (-1,-1), 0.5, colors.HexColor("#CCCCCC")),
        ("ROWBACKGROUNDS", (0,1), (-1,-1),
         [colors.white, _BRAND_GRAY]),
        # Highlight verdict row
        ("TEXTCOLOR",   (1,1), (1,1), verdict_color),
        ("FONTNAME",    (1,1), (1,1), "Helvetica-Bold"),
    ]))
    story.append(mt)
    story.append(Spacer(1, 0.5*cm))

    # ── PROBABILITY BAR CHART ─────────────────────────────────────
    story.append(Paragraph("2. Probability Distribution", style_h1))

    fig_bar, ax = plt.subplots(figsize=(6, 2.8))
    bar_colors  = ["#4CAF50", "#E94560"]
    bars        = ax.bar(["Unedited", "Manipulated"],
                          [prob_unedited, prob_manip],
                          color=bar_colors, width=0.45, edgecolor="white",
                          linewidth=1.2)
    for bar, val in zip(bars, [prob_unedited, prob_manip]):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.02,
                f"{val*100:.2f}%", ha="center", va="bottom",
                fontsize=11, fontweight="bold",
                color=bar.get_facecolor())
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Probability", fontsize=10)
    ax.set_title("Global Class Probability", fontsize=11, fontweight="bold")
    ax.axhline(0.60, color="grey", linestyle="--", linewidth=0.8,
               label="Decision threshold (0.60)")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig_bar.tight_layout()

    pil_bar = _fig_to_pil(fig_bar)
    story.append(_pil_to_rl_image(pil_bar, max_width_cm=10))
    story.append(Spacer(1, 0.3*cm))

    # ── ANOMALY ANALYSIS (Manipulated only) ──────────────────────
    if is_manip and pil_overview is not None:
        story.append(HRFlowable(width="100%", thickness=1,
                                color=_BRAND_GRAY, spaceAfter=10))
        story.append(Paragraph("3. Anomaly Overview", style_h1))
        story.append(Paragraph(
            "The panel below shows (left to right): the most suspicious "
            "sampled frame, the heatmap overlay, the binary hot-mask, "
            "the combined manipulation map, and the class probability bar.",
            style_body,
        ))
        story.append(Spacer(1, 0.2*cm))
        story.append(_pil_to_rl_image(pil_overview, max_width_cm=17.0))
        story.append(Paragraph("Figure 1 — Anomaly overview panel.", style_caption))

        # ── Additional numeric stats for the anomaly map ──────────
        story.append(Spacer(1, 0.3*cm))
        story.append(Paragraph("3.1  Anomaly Map Statistics", style_h1))

        amap = result.get("_anomaly_map_raw")  # injected by predict fn
        if amap is not None:
            mean_v = float(np.mean(amap))
            std_v  = float(np.std(amap))
            max_v  = float(np.max(amap))
            min_v  = float(np.min(amap))
            p90    = float(np.percentile(amap, 90))
            p95    = float(np.percentile(amap, 95))
            thr    = min(max(mean_v + 1.2 * std_v, 0.35), 0.80)

            amap_rows = [
                ["Statistic", "Value"],
                ["Mean anomaly score",         f"{mean_v:.4f}"],
                ["Std-dev anomaly score",       f"{std_v:.4f}"],
                ["Min anomaly score",           f"{min_v:.4f}"],
                ["Max anomaly score",           f"{max_v:.4f}"],
                ["90th-percentile score",       f"{p90:.4f}"],
                ["95th-percentile score",       f"{p95:.4f}"],
                ["Adaptive threshold used",     f"{thr:.4f}"],
                ["Suspicious area (% pixels)",  f"{suspicious_area:.2f}%"],
            ]
            cw2 = [W_PAGE * 0.55, W_PAGE * 0.45]
            at  = Table(amap_rows, colWidths=cw2, repeatRows=1)
            at.setStyle(TableStyle([
                ("BACKGROUND",   (0,0), (-1,0), _BRAND_MID),
                ("TEXTCOLOR",    (0,0), (-1,0), _WHITE),
                ("FONTNAME",     (0,0), (-1,0), "Helvetica-Bold"),
                ("FONTSIZE",     (0,0), (-1,0), 10),
                ("TOPPADDING",   (0,0), (-1,-1), 6),
                ("BOTTOMPADDING",(0,0), (-1,-1), 6),
                ("LEFTPADDING",  (0,0), (-1,-1), 8),
                ("RIGHTPADDING", (0,0), (-1,-1), 8),
                ("FONTNAME",     (0,1), (-1,-1), "Helvetica"),
                ("FONTSIZE",     (0,1), (-1,-1), 9),
                ("GRID",         (0,0), (-1,-1), 0.5,
                 colors.HexColor("#CCCCCC")),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),
                 [colors.white, _BRAND_GRAY]),
            ]))
            story.append(at)

    if is_manip and pil_components is not None:
        story.append(Spacer(1, 0.5*cm))
        story.append(HRFlowable(width="100%", thickness=1,
                                color=_BRAND_GRAY, spaceAfter=10))
        story.append(Paragraph("4. Component Anomaly Maps", style_h1))
        story.append(Paragraph(
            "Three individual signal maps that contribute to the combined "
            "score: model patch probability (weight 0.50), fractal/texture "
            "difference (weight 0.30), and temporal frame difference "
            "(weight 0.20).",
            style_body,
        ))
        story.append(Spacer(1, 0.2*cm))
        story.append(_pil_to_rl_image(pil_components, max_width_cm=17.0))
        story.append(Paragraph(
            "Figure 2 — Component anomaly maps: model patch probability | "
            "texture/fractal | temporal.",
            style_caption,
        ))

    elif not is_manip and pil_overview is not None:
        story.append(HRFlowable(width="100%", thickness=1,
                                color=_BRAND_GRAY, spaceAfter=10))
        story.append(Paragraph("3. Unedited Video Overview", style_h1))
        story.append(Paragraph(
            "No significant manipulation detected. The panel below shows a "
            "representative input frame, the global probability distribution, "
            "and a blank anomaly map (no suspicious region).",
            style_body,
        ))
        story.append(Spacer(1, 0.2*cm))
        story.append(_pil_to_rl_image(pil_overview, max_width_cm=17.0))
        story.append(Paragraph("Figure 1 — Unedited video overview.", style_caption))

    # ── INTERPRETATION NOTES ──────────────────────────────────────
    story.append(Spacer(1, 0.5*cm))
    story.append(HRFlowable(width="100%", thickness=1,
                            color=_BRAND_GRAY, spaceAfter=10))
    next_sec = "5." if is_manip else "4."
    story.append(Paragraph(f"{next_sec} Interpretation Notes", style_h1))

    notes = [
        f"<b>Decision threshold:</b> 0.60 — probabilities above this "
        f"level trigger a Manipulated verdict.",
        f"<b>Anomaly map weights:</b> Model patch (0.50) + "
        f"Fractal/Texture (0.30) + Temporal (0.20).",
        f"<b>Suspicious area threshold:</b> Adaptive — "
        f"min(max(mean + 1.2 × std, 0.35), 0.80) of the combined map.",
        f"<b>Backbone:</b> EfficientNet-B0 with 4-channel input "
        f"(RGB + fractal complexity channel).",
        f"<b>Frame sampling:</b> {NUM_FRAMES} frames sampled uniformly; "
        f"frame size {IMG_SIZE}×{IMG_SIZE} px.",
    ]
    for n in notes:
        story.append(Paragraph(f"• {n}", style_body))

    # ── FOOTER ────────────────────────────────────────────────────
    story.append(Spacer(1, 1.0*cm))
    story.append(HRFlowable(width="100%", thickness=0.5,
                            color=_BRAND_GRAY, spaceAfter=6))
    story.append(Paragraph(
        "Video Manipulation Detection System  ·  "
        "EfficientNet-B0 Backbone  ·  "
        f"Report generated {timestamp}",
        style_footer,
    ))

    doc.build(story)
    return buf.getvalue()


# ─────────────────────────── MAIN PREDICTION ─────────────────────

def predict_video_visual_manipulation(
    video_path: str,
    model,
    device,
    num_frames: int  = NUM_FRAMES,
    img_size: int    = IMG_SIZE,
    patch: int       = PATCH_SIZE,
    threshold: float = 0.60,
    grid: int        = 8,
) -> dict:
    """
    Full pipeline: validates → samples → predicts → generates anomaly maps
    → builds a professional PDF report.

    Returns a dict with:
        prediction              : "Manipulated" | "Unedited"
        prob_unedited           : float
        prob_manipulated        : float
        confidence              : float
        best_frame_index        : int | None
        suspicious_area_percent : float
        duration_message        : str
        report_pdf_b64          : str  (base64-encoded PDF bytes)
        error                   : str | None
    """
    # 1. Validate duration
    ok, msg = validate_video_duration(video_path, min_seconds=5,
                                       max_seconds=20)
    if not ok:
        return {"error": msg}

    # 2. Sample frames
    frames = sample_video_frames(video_path, num_frames=num_frames,
                                  target_size=img_size)
    if not frames:
        return {"error": "Could not read any frames from the video."}

    # 3. Global prediction
    prob_unedited, prob_manipulated = predict_full_video(
        frames=frames, model=model, device=device,
        num_frames=num_frames, patch=patch,
    )
    prediction = "Manipulated" if prob_manipulated >= threshold else "Unedited"

    result = {
        "prediction":              prediction,
        "prob_unedited":           round(prob_unedited,  4),
        "prob_manipulated":        round(prob_manipulated, 4),
        "confidence":              round(max(prob_unedited, prob_manipulated), 4),
        "best_frame_index":        None,
        "suspicious_area_percent": 0.0,
        "duration_message":        msg,
        "report_pdf_b64":          None,
        "error":                   None,
        "_anomaly_map_raw":        None,   # internal; stripped before API response
    }

    pil_overview   = None
    pil_components = None

    # 4. Manipulation-specific analysis
    if prediction == "Manipulated":
        best = select_best_suspicious_frame(
            frames=frames, model=model, device=device,
            num_frames=num_frames, patch=patch, grid=grid,
        )

        frame       = best["frame"]
        anomaly_map = best["combined_map"]

        mean_val     = float(np.mean(anomaly_map))
        std_val      = float(np.std(anomaly_map))
        adaptive_thr = min(max(mean_val + 1.2 * std_val, 0.35), 0.80)

        hot_mask = (anomaly_map >= adaptive_thr).astype(np.uint8)
        kernel   = np.ones((3, 3), np.uint8)
        hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_OPEN,  kernel)
        hot_mask = cv2.morphologyEx(hot_mask, cv2.MORPH_CLOSE, kernel)

        suspicious_area = float(hot_mask.mean() * 100)
        overlay         = create_overlay(frame, anomaly_map)

        result["best_frame_index"]        = int(best["frame_index"])
        result["suspicious_area_percent"] = round(suspicious_area, 2)
        result["_anomaly_map_raw"]        = anomaly_map  # for PDF stats

        # ── Overview plot (5 panels) ─────────────────────────────
        fig1, axes = plt.subplots(1, 5, figsize=(18, 4))
        axes[0].imshow(frame)
        axes[0].set_title(f"Suspicious Frame\nIndex: {best['frame_index']}")
        axes[0].axis("off")

        axes[1].imshow(overlay)
        axes[1].set_title("Heatmap Overlay")
        axes[1].axis("off")

        axes[2].imshow(hot_mask, cmap="gray")
        axes[2].set_title(f"Hot Mask\nSuspicious: {suspicious_area:.2f}%")
        axes[2].axis("off")

        axes[3].imshow(anomaly_map, cmap="inferno")
        axes[3].set_title("Combined Manipulation Map")
        axes[3].axis("off")

        axes[4].bar(["Unedited", "Manipulated"],
                    [prob_unedited, prob_manipulated],
                    color=["steelblue", "tomato"])
        axes[4].set_ylim(0, 1)
        axes[4].set_title("Global Probability")
        axes[4].set_ylabel("Probability")
        fig1.tight_layout()
        pil_overview = _fig_to_pil(fig1)

        # ── Component maps (3 panels) ────────────────────────────
        fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
        axes2[0].imshow(best["model_raw"],    cmap="inferno")
        axes2[0].set_title("Model Patch Probability");  axes2[0].axis("off")
        axes2[1].imshow(best["texture_map"],  cmap="inferno")
        axes2[1].set_title("Fractal / Texture Difference"); axes2[1].axis("off")
        axes2[2].imshow(best["temporal_map"], cmap="inferno")
        axes2[2].set_title("Temporal Frame Difference"); axes2[2].axis("off")
        fig2.tight_layout()
        pil_components = _fig_to_pil(fig2)

    else:
        # ── Unedited overview (3 panels) ─────────────────────────
        frame = frames[len(frames) // 2]
        black = np.zeros((img_size, img_size), dtype=np.uint8)

        fig3, axes3 = plt.subplots(1, 3, figsize=(14, 4))
        axes3[0].imshow(frame)
        axes3[0].set_title("Input Frame"); axes3[0].axis("off")
        axes3[1].bar(["Unedited", "Manipulated"],
                     [prob_unedited, prob_manipulated],
                     color=["steelblue", "tomato"])
        axes3[1].set_ylim(0, 1)
        axes3[1].set_title("Global Probability")
        axes3[1].set_ylabel("Probability")
        axes3[2].imshow(black, cmap="gray")
        axes3[2].set_title("No Suspicious Area"); axes3[2].axis("off")
        fig3.tight_layout()
        pil_overview = _fig_to_pil(fig3)

    # 5. Build PDF and encode as base64
    pdf_bytes = _build_pdf_report(result, pil_overview, pil_components)
    result["report_pdf_b64"] = base64.b64encode(pdf_bytes).decode("utf-8")

    # Strip internal key before returning
    del result["_anomaly_map_raw"]

    return result