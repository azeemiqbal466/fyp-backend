import io
import base64

from fastapi import APIRouter, UploadFile, File, Query, HTTPException
from fastapi.responses import JSONResponse

from .model import model
from .utils import (
    detect_deepfake_best_localization,
    generate_pdf_report,
)

router = APIRouter()


# -----------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------
@router.get("/health")
def health():
    return {"status": "ok", "service": "TemperEYE"}


# -----------------------------------------------------
# MAIN DETECTION ENDPOINT
# ONLY IMAGE INPUT
# RETURNS: JSON + PDF (base64)
# -----------------------------------------------------
@router.post("/detect")
async def detect(
    file: UploadFile = File(...),
    grid: int = Query(default=8, ge=2, le=32),
    thr: float = Query(default=0.60, ge=0.0, le=1.0),
    patch_for_fractal: int = Query(default=16, ge=4, le=64),
):

    allowed = {"image/jpeg", "image/png", "image/jpg", "image/webp"}

    if file.content_type not in allowed:
        raise HTTPException(415, "Only image files allowed")

    img_bytes = await file.read()

    # ---------------------------
    # RUN MODEL
    # ---------------------------
    result = detect_deepfake_best_localization(
        img_bytes=img_bytes,
        model=model,
        patch_for_fractal=patch_for_fractal,
        grid=grid,
        thr=thr,
    )

    # ---------------------------
    # EXTRACT PDF VISUALS ONLY
    # ---------------------------
    heatmap_strip_bytes = result.pop("heatmap_strip_bytes")
    bar_chart_bytes = result.pop("bar_chart_bytes")

    # ---------------------------
    # CREATE PDF
    # ---------------------------
    pdf_buffer = io.BytesIO()

    generate_pdf_report(
        result=result,
        image_name=file.filename or "image",
        heatmap_strip_bytes=heatmap_strip_bytes,
        bar_chart_bytes=bar_chart_bytes,
        out_path=pdf_buffer,
    )

    pdf_buffer.seek(0)
    pdf_bytes = pdf_buffer.read()

    # ---------------------------
    # ENCODE PDF
    # ---------------------------
    pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")

    # ---------------------------
    # RETURN ONLY VALUES + PDF
    # ---------------------------
    return JSONResponse({
        "prediction": result["prediction"],
        "prob_real": round(result["prob_real"], 4),
        "prob_fake": round(result["prob_fake"], 4),
        "suspicious_area_percent": round(result["suspicious_area_percent"], 2),
        "pdf_base64": pdf_base64
    })