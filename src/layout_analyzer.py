"""Layout analysis using PaddleX PP-DocLayoutV3."""

from __future__ import annotations

import cv2
import numpy as np

from src.models import BBox, LayoutRegion


# Lazy-loaded model singleton
_model = None


def _get_model():
    global _model
    if _model is None:
        from paddlex import create_model
        _model = create_model("PP-DocLayoutV3")
    return _model


def analyze_layout(
    image: np.ndarray, *, threshold: float = 0.5
) -> list[LayoutRegion]:
    """Detect layout regions (table, text, title, etc.) in a page image.

    Parameters
    ----------
    image : BGR numpy array of one page.
    threshold : minimum confidence score to keep a detection.

    Returns
    -------
    List of LayoutRegion with bbox, label, and confidence.
    """
    model = _get_model()

    # PaddleX expects RGB or file path; convert BGR -> RGB
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = list(model.predict(rgb))

    regions: list[LayoutRegion] = []
    if not results:
        return regions

    res = results[0]  # single image -> single result dict

    # PaddleX layout detection returns a dict-like object with 'boxes' key.
    # Each box: {"cls_id": int, "label": str, "score": float,
    #            "coordinate": [x1, y1, x2, y2]}
    boxes = res.get("boxes", []) if isinstance(res, dict) else []

    # If result is not a plain dict, try attribute access (PaddleX result obj)
    if not boxes and hasattr(res, "boxes"):
        boxes = res.boxes if res.boxes is not None else []

    for box in boxes:
        if isinstance(box, dict):
            score = float(box.get("score", 0))
            label = str(box.get("label", "unknown"))
            coord = box.get("coordinate", [0, 0, 0, 0])
        else:
            # numpy row: [cls_id, score, x1, y1, x2, y2] or similar
            score = float(box[1]) if len(box) > 1 else 0.0
            label = "unknown"
            coord = box[2:6] if len(box) >= 6 else [0, 0, 0, 0]

        if score < threshold:
            continue

        x1, y1, x2, y2 = (int(c) for c in coord[:4])
        bbox = BBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1)
        regions.append(LayoutRegion(bbox=bbox, label=label, confidence=score))

    return regions
