"""Shape column normalization: split false merges and validate cells.

Two main functions:

1. ``repair_shape_column`` — if the shape column has an abnormally large
   rowspan (caused by faint / missing horizontal lines), use adjacent
   columns' per-row boundaries to split it back into individual cells.

2. ``is_likely_shape`` — before sending a cell to the shape classifier,
   check whether it actually contains a graphical shape (as opposed to
   a number, text label, or summary-row value).
"""

from __future__ import annotations

import cv2
import numpy as np

from src.models import BBox, GridCell

# Generic indicators for summary / totals rows (not domain-specific).
_SUMMARY_INDICATORS = {"合計", "小計", "總計", "total", "subtotal", "sum"}


# ------------------------------------------------------------------
# 1. Repair: split false merges in the shape column
# ------------------------------------------------------------------


def repair_shape_column(
    grid_cells: list[GridCell],
    shape_col_idx: int,
    max_rowspan: int = 2,
) -> tuple[list[GridCell], list[dict]]:
    """Split shape-column cells whose rowspan exceeds *max_rowspan*.

    Uses y-coordinates of ``rowspan == 1`` cells in **other** columns as
    reference boundaries for splitting.

    Returns
    -------
    repaired_cells : list[GridCell]
        Updated cell list (other columns untouched).
    repair_log : list[dict]
        One entry per split, with ``"original"`` and ``"split_cells"`` keys.
    """

    # Build row -> (y_start, y_end) from non-shape columns
    row_y_start: dict[int, int] = {}
    row_y_end: dict[int, int] = {}

    for gc in grid_cells:
        # Skip cells that are IN the shape column
        if gc.col == shape_col_idx and gc.colspan == 1:
            continue
        if gc.rowspan != 1:
            continue
        if gc.row not in row_y_start:
            row_y_start[gc.row] = gc.bbox.y
            row_y_end[gc.row] = gc.bbox.y + gc.bbox.h

    result: list[GridCell] = []
    repair_log: list[dict] = []

    for gc in grid_cells:
        is_shape_col = gc.col == shape_col_idx and gc.colspan == 1

        if not is_shape_col or gc.rowspan <= max_rowspan:
            result.append(gc)
            continue

        # Collect reference boundaries for the covered rows
        covered = range(gc.row, gc.row + gc.rowspan)
        split_cells: list[GridCell] = []

        for r in covered:
            if r not in row_y_start:
                continue
            y_s = max(row_y_start[r], gc.bbox.y)
            y_e = min(row_y_end[r], gc.bbox.y + gc.bbox.h)
            if y_e - y_s < 5:
                continue
            split_cells.append(
                GridCell(
                    bbox=BBox(x=gc.bbox.x, y=y_s, w=gc.bbox.w, h=y_e - y_s),
                    row=r,
                    col=shape_col_idx,
                    rowspan=1,
                    colspan=1,
                )
            )

        if len(split_cells) >= 2:
            result.extend(split_cells)
            repair_log.append(
                {
                    "original": gc,
                    "split_cells": split_cells,
                }
            )
        else:
            # Cannot split reliably — keep the original cell
            result.append(gc)

    result.sort(key=lambda g: (g.row, g.col))
    return result, repair_log


# ------------------------------------------------------------------
# 2. Validity check: is this cell actually a shape?
# ------------------------------------------------------------------


def is_likely_shape(
    cell_crop: np.ndarray,
    ocr_text: str,
    ocr_confidence: float,
    adjacent_texts: list[str] | None = None,
) -> bool:
    """Return ``True`` only if the cell likely contains a graphical shape.

    Checks (in order):
    1. Generic summary indicators in adjacent cells → reject.
    2. High-confidence short numeric / alphanumeric OCR → reject.
    3. Foreground pixel ratio too low → reject (empty cell).
    4. Connected-component spatial spread too small → reject (text-sized).

    Note: header-row skipping is handled by the caller (pipeline)
    via row-type classification, not by this function.
    """
    text = ocr_text.strip()

    # --- 1. Summary-row indicators in adjacent cells ---
    if adjacent_texts:
        for at in adjacent_texts:
            at_lower = at.lower()
            for kw in _SUMMARY_INDICATORS:
                if kw.lower() in at_lower:
                    return False

    # --- 2. Confident numeric text ---
    # Only reject pure numbers — shapes can look like letters (C, O, L)
    # to OCR, so we must not reject alphabetic OCR hits.
    if text and ocr_confidence > 0.5:
        cleaned = (
            text.replace(".", "", 1)
            .replace(",", "")
            .replace("-", "", 1)
            .replace(" ", "")
        )
        if cleaned.isdigit():
            return False

    # --- 3. Pixel-level checks ---
    h, w = cell_crop.shape[:2]
    if h < 10 or w < 10:
        return False

    gray = (
        cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)
        if len(cell_crop.shape) == 3
        else cell_crop
    )
    _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    total = h * w
    fg = int(np.count_nonzero(bw))
    if fg / total < 0.005:
        return False  # nearly empty

    # --- 4. Component spatial spread ---
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    min_area = max(total * 0.003, 8)
    sig = [i for i in range(1, n_labels) if stats[i, cv2.CC_STAT_AREA] >= min_area]
    if not sig:
        return False

    max_cw = max(stats[i, cv2.CC_STAT_WIDTH] for i in sig)
    max_ch = max(stats[i, cv2.CC_STAT_HEIGHT] for i in sig)

    # Shape strokes should span a meaningful fraction of the cell
    if max_cw < w * 0.12 and max_ch < h * 0.12:
        return False

    return True
