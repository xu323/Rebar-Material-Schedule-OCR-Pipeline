"""Table cell segmentation via OpenCV line detection.

Supports detection of merged cells (rowspan/colspan) by checking whether
internal borders between candidate 1×1 cells actually exist in the
extracted horizontal/vertical line masks.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.config import PipelineConfig
from src.models import BBox, GridCell


def segment_table_cells(
    table_image: np.ndarray, cfg: PipelineConfig | None = None
) -> list[GridCell]:
    """Detect grid lines in *table_image* and return cell bounding boxes.

    Internal borders that are absent in the line mask cause adjacent
    candidate cells to be merged via union-find, producing GridCells with
    ``rowspan`` / ``colspan`` > 1.
    """
    if cfg is None:
        cfg = PipelineConfig()

    h, w = table_image.shape[:2]
    gray = cv2.cvtColor(table_image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold — invert so lines are white
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 5
    )

    # --- Isolate horizontal lines ---
    h_kernel_len = max(w // cfg.morph_h_scale, 10)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_len, 1))
    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel, iterations=2)

    # --- Isolate vertical lines ---
    v_kernel_len = max(h // cfg.morph_v_scale, 10)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_len))
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel, iterations=2)

    # --- Extract line coordinates ---
    row_ys = _extract_line_positions(
        h_lines, axis="horizontal", tolerance=cfg.line_cluster_tolerance
    )
    col_xs = _extract_line_positions(
        v_lines, axis="vertical", tolerance=cfg.line_cluster_tolerance
    )

    # Ensure outer borders are present. Snap lines that sit near the edge
    # to the exact edge, otherwise insert a new boundary. Using a larger
    # tolerance prevents sliver rows/columns when the top/bottom border
    # was detected a few pixels inside the edge.
    tol = cfg.line_cluster_tolerance
    edge_tol = max(tol * 3, 20)
    row_ys = _ensure_edges(row_ys, 0, h, edge_tol)
    col_xs = _ensure_edges(col_xs, 0, w, edge_tol)

    n_rows = len(row_ys) - 1
    n_cols = len(col_xs) - 1
    if n_rows <= 0 or n_cols <= 0:
        return []

    # --- Detect merged cells via missing-border test ---
    merged_cells = _detect_merges(
        row_ys,
        col_xs,
        h_lines,
        v_lines,
        tol=tol,
        min_coverage=cfg.merge_border_coverage,
    )

    # --- Build GridCells, filtering tiny cells ---
    cells: list[GridCell] = []
    for r, c, rowspan, colspan in merged_cells:
        x = col_xs[c]
        y = row_ys[r]
        cell_w = col_xs[c + colspan] - x
        cell_h = row_ys[r + rowspan] - y
        if cell_w < 5 or cell_h < 5:
            continue
        cells.append(
            GridCell(
                bbox=BBox(x=x, y=y, w=cell_w, h=cell_h),
                row=r,
                col=c,
                rowspan=rowspan,
                colspan=colspan,
            )
        )

    return cells


def _detect_merges(
    row_ys: list[int],
    col_xs: list[int],
    h_lines: np.ndarray,
    v_lines: np.ndarray,
    *,
    tol: int,
    min_coverage: float,
) -> list[tuple[int, int, int, int]]:
    """Run union-find over the candidate 1×1 grid, merging cells that
    share a missing internal border.

    Returns a list of ``(row, col, rowspan, colspan)`` for each merged cell.
    Non-rectangular merges are split back into their 1×1 components to
    avoid producing wrong bounding boxes.
    """
    n_rows = len(row_ys) - 1
    n_cols = len(col_xs) - 1
    H, W = h_lines.shape[:2]

    def has_right_border(r: int, c: int) -> bool:
        """True if the vertical border on the right side of cell (r,c) exists."""
        if c == n_cols - 1:
            return True  # outer border
        x = col_xs[c + 1]
        y0 = row_ys[r]
        y1 = row_ys[r + 1]
        x0 = max(0, x - tol)
        x1 = min(W, x + tol + 1)
        if y1 <= y0 or x1 <= x0:
            return True
        slab = v_lines[y0:y1, x0:x1]
        # For each row in the slab, check if any pixel is set (line present)
        per_row = (slab > 0).any(axis=1)
        return float(per_row.mean()) >= min_coverage

    def has_bottom_border(r: int, c: int) -> bool:
        if r == n_rows - 1:
            return True
        y = row_ys[r + 1]
        x0 = col_xs[c]
        x1 = col_xs[c + 1]
        y0 = max(0, y - tol)
        y1 = min(H, y + tol + 1)
        if x1 <= x0 or y1 <= y0:
            return True
        slab = h_lines[y0:y1, x0:x1]
        per_col = (slab > 0).any(axis=0)
        return float(per_col.mean()) >= min_coverage

    # Union-find
    parent: dict[tuple[int, int], tuple[int, int]] = {
        (r, c): (r, c) for r in range(n_rows) for c in range(n_cols)
    }

    def find(x: tuple[int, int]) -> tuple[int, int]:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for r in range(n_rows):
        for c in range(n_cols):
            if c + 1 < n_cols and not has_right_border(r, c):
                union((r, c), (r, c + 1))
            if r + 1 < n_rows and not has_bottom_border(r, c):
                union((r, c), (r + 1, c))

    # Collect groups
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for pos in parent:
        root = find(pos)
        groups.setdefault(root, []).append(pos)

    out: list[tuple[int, int, int, int]] = []
    for group in groups.values():
        rows = [p[0] for p in group]
        cols = [p[1] for p in group]
        r_min, r_max = min(rows), max(rows)
        c_min, c_max = min(cols), max(cols)
        rowspan = r_max - r_min + 1
        colspan = c_max - c_min + 1
        expected = rowspan * colspan
        if len(group) == expected:
            # Rectangular merge — safe
            out.append((r_min, c_min, rowspan, colspan))
        else:
            # Non-rectangular (L-shape etc.) — fall back to 1×1 cells
            for r, c in group:
                out.append((r, c, 1, 1))

    # Sort top-to-bottom, left-to-right for deterministic output
    out.sort(key=lambda t: (t[0], t[1]))
    return out


def _ensure_edges(positions: list[int], lo: int, hi: int, edge_tol: int) -> list[int]:
    """Snap near-edge positions to the exact edge, or insert edge if missing."""
    positions = sorted(positions)
    if not positions:
        return [lo, hi]
    # Top/left edge
    if positions[0] <= lo + edge_tol:
        positions[0] = lo
    else:
        positions.insert(0, lo)
    # Bottom/right edge
    if positions[-1] >= hi - edge_tol:
        positions[-1] = hi
    else:
        positions.append(hi)
    return positions


def _extract_line_positions(
    line_mask: np.ndarray, axis: str, tolerance: int
) -> list[int]:
    """Extract clustered line positions from a binary mask.

    For horizontal lines, returns sorted list of y-coordinates.
    For vertical lines, returns sorted list of x-coordinates.
    """
    if axis == "horizontal":
        projection = np.sum(line_mask, axis=1)
    else:
        projection = np.sum(line_mask, axis=0)

    threshold = line_mask.shape[1 if axis == "horizontal" else 0] * 0.3
    positions = np.where(projection > threshold * 255)[0].tolist()

    if not positions:
        return []

    clusters: list[list[int]] = [[positions[0]]]
    for pos in positions[1:]:
        if pos - clusters[-1][-1] <= tolerance:
            clusters[-1].append(pos)
        else:
            clusters.append([pos])

    return [int(np.median(c)) for c in clusters]
