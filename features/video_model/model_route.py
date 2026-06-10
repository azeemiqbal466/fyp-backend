import os
import shutil
import tempfile

import torch
from fastapi import APIRouter, File, UploadFile, HTTPException
from pydantic import BaseModel
from typing import Optional

from .model import video_manipulation_model, device as _device
from .utils import (
    predict_video_visual_manipulation,
    IMG_SIZE,
    NUM_FRAMES,
    PATCH_SIZE,
)

# ─────────────────────────── ROUTER ─────────────────────────────
router = APIRouter(prefix="/api/v1", tags=["Visual Manipulation Detection"])

# ─────────────────────────── MODEL (singleton) ──────────────────
# The model is loaded eagerly at import time inside model.py.
# We just retrieve the singleton here — no repeated disk reads.
_model = None


def get_model():
    global _model
    if _model is None:
        _model = video_manipulation_model()
        _model.eval()
    return _model


# ─────────────────────────── RESPONSE SCHEMA ────────────────────

class PredictionResult(BaseModel):
    """
    All numeric values and the final prediction verdict.
    Graphs and visualisation images are embedded in `report_pdf_b64`
    as a base64-encoded PDF — decode and save as a .pdf file.
    """
    # ── Verdict ──────────────────────────────────────────────────
    prediction: str
    """'Manipulated' or 'Unedited'"""

    # ── Class probabilities ───────────────────────────────────────
    prob_unedited: float
    """Probability [0–1] that the video is authentic."""

    prob_manipulated: float
    """Probability [0–1] that the video has been manipulated."""

    confidence: float
    """max(prob_unedited, prob_manipulated) — strength of the verdict."""

    # ── Spatial analysis ─────────────────────────────────────────
    best_frame_index: Optional[int]
    """0-based index of the sampled frame with the highest anomaly score.
    None when the video is classified as Unedited."""

    suspicious_area_percent: float
    """Percentage of pixels in the best frame that exceed the adaptive
    anomaly threshold.  0.0 for Unedited videos."""

    # ── Meta ──────────────────────────────────────────────────────
    duration_message: Optional[str]
    """Human-readable video-duration validation result."""

    error: Optional[str]
    """Non-null only when the pipeline failed before producing a verdict."""

    # ── PDF report ────────────────────────────────────────────────
    report_pdf_b64: Optional[str]
    """Base64-encoded bytes of a professional PDF report containing all
    charts, heatmaps, and anomaly maps.  Decode and write to a .pdf file:

        import base64
        with open("report.pdf", "wb") as f:
            f.write(base64.b64decode(response["report_pdf_b64"]))
    """


# ─────────────────────────── ENDPOINTS ──────────────────────────

@router.get("/health", summary="Health check")
def health():
    """Returns service status, device info, and whether the model is loaded."""
    return {
        "status":       "ok",
        "device":       str(_device),
        "model_loaded": _model is not None,
    }


@router.post(
    "/predict",
    response_model=PredictionResult,
    summary="Detect visual manipulation in a video",
    description=(
        "Upload a video file (MP4, AVI, MOV, MKV, WEBM — 5–20 s). "
        "Returns all numeric predictions plus a base64-encoded PDF report "
        "containing all visualisation charts and anomaly maps."
    ),
)
async def predict_video(
    file: UploadFile = File(..., description="Video file to analyse"),
) -> PredictionResult:

    # ── 1. Validate extension ────────────────────────────────────
    ALLOWED = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in ALLOWED:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED)}",
        )

    # ── 2. Save to temp file ─────────────────────────────────────
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name

        # ── 3. Run full pipeline ─────────────────────────────────
        model = get_model()
        raw   = predict_video_visual_manipulation(
            video_path=tmp_path,
            model=model,
            device=_device,
            num_frames=NUM_FRAMES,
            img_size=IMG_SIZE,
            patch=PATCH_SIZE,
            threshold=0.60,
            grid=8,
        )

        # ── 4. Surface pipeline errors ───────────────────────────
        if raw.get("error"):
            raise HTTPException(status_code=422, detail=raw["error"])

        # ── 5. Build and return response (no image fields) ───────
        return PredictionResult(
            prediction              = raw["prediction"],
            prob_unedited           = raw["prob_unedited"],
            prob_manipulated        = raw["prob_manipulated"],
            confidence              = raw["confidence"],
            best_frame_index        = raw.get("best_frame_index"),
            suspicious_area_percent = raw.get("suspicious_area_percent", 0.0),
            duration_message        = raw.get("duration_message"),
            error                   = None,
            report_pdf_b64          = raw.get("report_pdf_b64"),
        )

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)