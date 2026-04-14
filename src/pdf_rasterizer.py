"""PDF to page images using PyMuPDF."""

from __future__ import annotations

from pathlib import Path

import cv2
import fitz  # PyMuPDF
import numpy as np


def rasterize_pdf(pdf_path: Path, dpi: int = 300) -> list[np.ndarray]:
    """Rasterize each page of *pdf_path* into a BGR numpy array.

    Returns one image per page, in page order.
    """
    doc = fitz.open(str(pdf_path))
    images: list[np.ndarray] = []

    for page in doc:
        pix = page.get_pixmap(dpi=dpi)
        # pixmap.samples is raw bytes in RGB or RGBA order
        arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)

        if pix.n == 4:  # RGBA
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        elif pix.n == 3:  # RGB
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        else:  # grayscale
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)

        images.append(bgr)

    doc.close()
    return images
