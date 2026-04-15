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
        return _recognize_text_from_rgb(
            ocr,
            rgb,
            allow_line_recrop=True,
        )
    except Exception:
        return OCRResult(text="", confidence=0.0)


def _recognize_text_from_rgb(
    ocr,
    rgb: np.ndarray,
    *,
    allow_line_recrop: bool,
) -> OCRResult:
    raw_results = list(ocr.predict(rgb))
    if not raw_results:
        return OCRResult(text="", confidence=0.0)

    detections = _extract_detections(raw_results)
    if not detections:
        return OCRResult(text="", confidence=0.0)

    if any(det["has_box"] for det in detections):
        lines = _group_detections_into_lines(detections)
        line_texts: list[str] = []
        line_scores: list[float] = []
        for line in lines:
            text, score = _recognize_line(
                ocr,
                rgb,
                line,
                allow_line_recrop=allow_line_recrop,
            )
            text = text.strip()
            if text:
                line_texts.append(text)
                line_scores.append(score)
        if line_texts:
            combined_text = "\n".join(line_texts)
            avg_score = (
                sum(line_scores) / len(line_scores) if line_scores else 0.0
            )
            return OCRResult(text=combined_text, confidence=avg_score)

    texts = [det["text"] for det in detections if det["text"]]
    scores = [det["score"] for det in detections if det["text"]]
    if not texts:
        return OCRResult(text="", confidence=0.0)
    combined_text = "\n".join(texts) if len(texts) > 1 else texts[0]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    return OCRResult(text=combined_text, confidence=avg_score)


def _extract_detections(raw_results: list[dict]) -> list[dict]:
    detections: list[dict] = []
    for item in raw_results:
        rec_texts = item.get("rec_texts", [])
        rec_scores = item.get("rec_scores", [])
        polys = item.get("dt_polys", [])
        for idx, (t, s) in enumerate(zip(rec_texts, rec_scores)):
            text = str(t).strip()
            if not text:
                continue

            poly = polys[idx] if idx < len(polys) else None
            det = {
                "text": text,
                "score": float(s),
                "has_box": False,
            }
            if poly is not None:
                arr = np.asarray(poly)
                if arr.size >= 8:
                    xs = arr[:, 0]
                    ys = arr[:, 1]
                    x0 = int(np.floor(xs.min()))
                    y0 = int(np.floor(ys.min()))
                    x1 = int(np.ceil(xs.max()))
                    y1 = int(np.ceil(ys.max()))
                    det.update({
                        "has_box": True,
                        "x0": x0,
                        "y0": y0,
                        "x1": x1,
                        "y1": y1,
                        "cx": (x0 + x1) / 2.0,
                        "cy": (y0 + y1) / 2.0,
                        "w": max(x1 - x0, 1),
                        "h": max(y1 - y0, 1),
                    })
            detections.append(det)
    return detections


def _group_detections_into_lines(detections: list[dict]) -> list[list[dict]]:
    boxed = [det for det in detections if det.get("has_box")]
    if not boxed:
        return []

    boxed.sort(key=lambda det: (det["cy"], det["x0"]))
    lines: list[dict] = []

    for det in boxed:
        assigned = None
        for line in lines:
            line_h = max(line["y1"] - line["y0"], 1)
            ref_h = max(det["h"], line_h)
            vertical_overlap = min(det["y1"], line["y1"]) - max(det["y0"], line["y0"])
            center_delta = abs(det["cy"] - line["cy"])
            if (
                vertical_overlap >= ref_h * 0.35
                or center_delta <= ref_h * 0.45
            ):
                assigned = line
                break

        if assigned is None:
            lines.append({
                "items": [det],
                "x0": det["x0"],
                "y0": det["y0"],
                "x1": det["x1"],
                "y1": det["y1"],
                "cy": det["cy"],
            })
        else:
            assigned["items"].append(det)
            assigned["x0"] = min(assigned["x0"], det["x0"])
            assigned["y0"] = min(assigned["y0"], det["y0"])
            assigned["x1"] = max(assigned["x1"], det["x1"])
            assigned["y1"] = max(assigned["y1"], det["y1"])
            assigned["cy"] = (
                sum(item["cy"] for item in assigned["items"]) /
                len(assigned["items"])
            )

    lines.sort(key=lambda line: (line["cy"], line["x0"]))
    return [
        sorted(line["items"], key=lambda det: det["x0"])
        for line in lines
    ]


def _recognize_line(
    ocr,
    rgb: np.ndarray,
    line: list[dict],
    *,
    allow_line_recrop: bool,
) -> tuple[str, float]:
    if len(line) == 1:
        return line[0]["text"], line[0]["score"]

    if allow_line_recrop and _line_has_joinable_boxes(line):
        x0 = max(min(det["x0"] for det in line) - 4, 0)
        y0 = max(min(det["y0"] for det in line) - 4, 0)
        x1 = min(max(det["x1"] for det in line) + 4, rgb.shape[1])
        y1 = min(max(det["y1"] for det in line) + 4, rgb.shape[0])
        union = rgb[y0:y1, x0:x1]
        if union.size > 0:
            rerun = _recognize_text_from_rgb(
                ocr,
                union,
                allow_line_recrop=False,
            )
            if rerun.text:
                return rerun.text, rerun.confidence

    return _join_line_segments(line)


def _line_has_joinable_boxes(line: list[dict]) -> bool:
    for left, right in zip(line, line[1:]):
        gap = right["x0"] - left["x1"]
        ref_h = max(left["h"], right["h"], 1)
        if gap <= ref_h * 0.6:
            return True
    return False


def _join_line_segments(line: list[dict]) -> tuple[str, float]:
    parts: list[str] = []
    scores: list[float] = []
    prev = None
    for det in line:
        if prev is not None:
            gap = det["x0"] - prev["x1"]
            ref_h = max(prev["h"], det["h"], 1)
            if gap > ref_h * 0.75:
                parts.append(" ")
        parts.append(det["text"])
        scores.append(det["score"])
        prev = det
    text = "".join(parts).strip()
    score = sum(scores) / len(scores) if scores else 0.0
    return text, score
