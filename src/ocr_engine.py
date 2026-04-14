"""PaddleOCR wrapper for text recognition."""

from __future__ import annotations

import os
os.environ.setdefault("FLAGS_use_mkldnn", "0")

import cv2
import numpy as np

from src.models import OCRResult

# Singleton OCR instance
_ocr_instance = None


def init_ocr():
    """Create (or return cached) PaddleOCR instance."""
    global _ocr_instance
    if _ocr_instance is None:
        from paddleocr import PaddleOCR

        _ocr_instance = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            device="cpu",
            enable_mkldnn=False,
        )
    return _ocr_instance


def recognize_text(
    ocr, image: np.ndarray, *, padding: int = 20
) -> OCRResult:
    """Run OCR on a single image region.

    Small images are padded with a white border to help detection.
    Multiple text lines are concatenated with newlines.
    """
    if image.size == 0:
        return OCRResult(text="", confidence=0.0)

    h, w = image.shape[:2]

    # Pad small images so PaddleOCR's detector can find text
    if h < 32 or w < 32:
        padded = cv2.copyMakeBorder(
            image, padding, padding, padding, padding,
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
    else:
        padded = image

    # PaddleOCR.predict() expects a file path or numpy array
    # Convert BGR -> RGB for PaddleOCR
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

    try:
        raw_results = list(ocr.predict(rgb))
    except Exception:
        return OCRResult(text="", confidence=0.0)

    if not raw_results:
        return OCRResult(text="", confidence=0.0)

    texts: list[str] = []
    scores: list[float] = []

    for item in raw_results:
        rec_texts = item.get("rec_texts", [])
        rec_scores = item.get("rec_scores", [])
        for t, s in zip(rec_texts, rec_scores):
            t = str(t).strip()
            if t:
                texts.append(t)
                scores.append(float(s))

    if not texts:
        return OCRResult(text="", confidence=0.0)

    combined_text = "\n".join(texts) if len(texts) > 1 else texts[0]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return OCRResult(text=combined_text, confidence=avg_score)
