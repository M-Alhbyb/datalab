import os
import io
from typing import List, Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException
from PIL import Image
import torch

from surya.foundation import FoundationPredictor
from surya.recognition import RecognitionPredictor
from surya.detection import DetectionPredictor

# -----------------------------
# Performance setup
# -----------------------------
torch.set_num_threads(os.cpu_count() or 4)

# -----------------------------
# Load models once (startup)
# -----------------------------
foundation_predictor = None
recognition_predictor = None
detection_predictor = None

app = FastAPI(title="Surya OCR API", version="1.0.0")


def preprocess_pil(img: Image.Image) -> Image.Image:
    img = img.convert("RGB")

    # Surya usually works better when width is not too large
    max_width = 1600
    if img.width > max_width:
        ratio = max_width / img.width
        img = img.resize((max_width, int(img.height * ratio)), Image.LANCZOS)

    return img


@torch.inference_mode()
def run_ocr_on_pil(img: Image.Image):
    img = preprocess_pil(img)

    preds = recognition_predictor(
        [img],
        det_predictor=detection_predictor
    )
    return preds[0]


def extract_lines(result) -> List[str]:
    lines = []
    for line in result.text_lines:
        t = (line.text or "").strip()
        if t:
            lines.append(t)
    return lines


@app.on_event("startup")
def load_models():
    global foundation_predictor, recognition_predictor, detection_predictor

    # Load once
    foundation_predictor = FoundationPredictor()
    recognition_predictor = RecognitionPredictor(foundation_predictor)
    detection_predictor = DetectionPredictor()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ocr")
async def ocr_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    # Basic validation
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        raw = await file.read()
        img = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image file")

    try:
        result = run_ocr_on_pil(img)
        lines = extract_lines(result)
        full_text = "\n".join(lines)

        return {
            "filename": file.filename,
            "lines": lines,
            "text": full_text,
            "num_lines": len(lines),
        }
    except Exception as e:
        # Avoid leaking internal info in production, لكن مفيد أثناء التطوير
        raise HTTPException(status_code=500, detail=f"OCR failed: {str(e)}")
