"""Debug overlay image generation and faithful table rendering.

Image 01: Layout regions overlay on full page (OpenCV).
Image 02: Cell grid overlay on table crop (OpenCV).
Image 03: Faithful table reconstruction from raw grid cells + OCR results
          (PIL), preserving metadata rows, column headers, unit rows, and
          data rows exactly as they appear in the original document.
Image 04: Repair verification overlay (OpenCV), only when repairs occurred.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.models import BBox, GridCell, LayoutRegion, NonTableRegion

# Colour palette (BGR) — for OpenCV overlays (images 01 & 02)
COLOR_TABLE = (0, 200, 0)  # green
COLOR_TEXT = (200, 0, 0)  # blue
COLOR_TITLE = (0, 0, 200)  # red
COLOR_OTHER = (0, 200, 200)  # yellow
COLOR_CELL = (200, 200, 0)  # cyan
COLOR_SHAPE = (0, 140, 255)  # orange

_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "table": COLOR_TABLE,
    "text": COLOR_TEXT,
    "doc_title": COLOR_TITLE,
}

# --- CJK font loading ---
_CJK_FONT_PATHS = [
    "C:/Windows/Fonts/msjh.ttc",  # Microsoft JhengHei (繁體)
    "C:/Windows/Fonts/msyh.ttc",  # Microsoft YaHei
    "C:/Windows/Fonts/simsun.ttc",  # SimSun
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/System/Library/Fonts/PingFang.ttc",
]

_font_cache: dict[int, ImageFont.FreeTypeFont | ImageFont.ImageFont] = {}


def _load_cjk_font(size: int = 14) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a CJK-capable font, falling back to Pillow default."""
    if size in _font_cache:
        return _font_cache[size]
    for p in _CJK_FONT_PATHS:
        if Path(p).exists():
            try:
                font = ImageFont.truetype(p, size)
                _font_cache[size] = font
                return font
            except Exception:
                continue
    try:
        font = ImageFont.load_default(size=size)  # type: ignore[assignment]
    except TypeError:
        font = ImageFont.load_default()  # type: ignore[assignment]
    _font_cache[size] = font
    return font


# ================================================================
# Public API
# ================================================================


def save_debug_page(
    page_image: np.ndarray,
    regions: list[LayoutRegion],
    table_crop: np.ndarray,
    grid_cells: list[GridCell],
    cell_contents: dict[tuple[int, int], dict],
    shape_col_idx: int | None,
    output_dir: Path,
    page_idx: int,
    table_idx: int,
    *,
    table=None,
    shapes_dir: Path | None = None,
    repair_log: list[dict] | None = None,
    non_table_regions: list[NonTableRegion] | None = None,
) -> None:
    """Save all debug images for one table on one page."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract row_types from assembled Table (if available)
    row_types: dict[int, str] | None = None
    if table is not None and hasattr(table, "rows"):
        row_types = {row.index: row.row_type for row in table.rows}

    # 1. Layout overlay
    layout_vis = _draw_layout(page_image, regions)
    cv2.imwrite(str(output_dir / "01_layout.png"), layout_vis)

    # 2. Cell grid overlay
    grid_vis = _draw_grid(table_crop, grid_cells, shape_col_idx)
    cv2.imwrite(str(output_dir / f"02_table_{table_idx}_grid.png"), grid_vis)

    # 3. Faithful table rendering from raw grid cells
    if shapes_dir is not None:
        rendered = render_faithful_table(
            grid_cells,
            cell_contents,
            shape_col_idx,
            shapes_dir,
            non_table_regions=non_table_regions,
            table_bbox=table.bbox if table is not None else None,
            row_types=row_types,
        )
        cv2.imwrite(str(output_dir / f"03_table_{table_idx}_rendered.png"), rendered)

    # 4. Repair verification (only if repairs were made)
    if repair_log:
        repair_vis = _draw_repair_debug(
            table_crop,
            repair_log,
            cell_contents,
            shapes_dir,
        )
        cv2.imwrite(str(output_dir / f"04_table_{table_idx}_repair.png"), repair_vis)


# ================================================================
# Image 03 — Faithful table reconstruction from raw grid cells
# ================================================================


def render_faithful_table(
    grid_cells: list[GridCell],
    cell_contents: dict[tuple[int, int], dict],
    shape_col_idx: int | None,
    shapes_dir: Path,
    non_table_regions: list[NonTableRegion] | None = None,
    table_bbox: BBox | None = None,
    row_types: dict[int, str] | None = None,
) -> np.ndarray:
    """Render a faithful reconstruction of the original table layout.

    Renders ALL grid rows — metadata headers, column headers, unit rows,
    and data rows — using proportional column widths and row heights
    derived from the original cell bounding boxes.  Non-table regions
    (document title, etc.) are rendered above the table.

    Returns a BGR numpy array suitable for ``cv2.imwrite``.
    """
    if not grid_cells:
        img = Image.new("RGB", (300, 60), "white")
        ImageDraw.Draw(img).text((10, 20), "Empty table", fill="gray")
        return np.array(img)[..., ::-1]

    # --- Grid dimensions ---
    n_rows = max(gc.row + gc.rowspan for gc in grid_cells)
    n_cols = max(gc.col + gc.colspan for gc in grid_cells)

    # --- Build cell lookup: (row, col) -> GridCell that covers it ---
    cell_lookup: dict[tuple[int, int], GridCell] = {}
    for gc in grid_cells:
        for rr in range(gc.row, gc.row + gc.rowspan):
            for cc in range(gc.col, gc.col + gc.colspan):
                cell_lookup[(rr, cc)] = gc

    # --- Trim leading/trailing rows that are completely empty ---
    trim_start, trim_end = _find_nonempty_row_range(
        n_rows,
        n_cols,
        cell_lookup,
        cell_contents,
    )
    full_n_rows = n_rows
    n_rows = trim_end - trim_start + 1

    # --- Derive column widths & row heights from original bboxes ---
    orig_col_w: dict[int, int] = {}
    orig_row_h: dict[int, int] = {}

    for gc in grid_cells:
        if gc.colspan == 1:
            orig_col_w[gc.col] = max(orig_col_w.get(gc.col, 0), gc.bbox.w)
        if gc.rowspan == 1:
            orig_row_h[gc.row] = max(orig_row_h.get(gc.row, 0), gc.bbox.h)

    # Fill missing entries from merged cells (approximate per-cell share)
    for gc in grid_cells:
        if gc.colspan > 1:
            per_col = gc.bbox.w // gc.colspan
            for cc in range(gc.col, gc.col + gc.colspan):
                if cc not in orig_col_w:
                    orig_col_w[cc] = per_col
        if gc.rowspan > 1:
            per_row = gc.bbox.h // gc.rowspan
            for rr in range(gc.row, gc.row + gc.rowspan):
                if rr not in orig_row_h:
                    orig_row_h[rr] = per_row

    avg_w = sum(orig_col_w.values()) // max(len(orig_col_w), 1) or 80
    avg_h = sum(orig_row_h.values()) // max(len(orig_row_h), 1) or 40
    col_widths_orig = [orig_col_w.get(c, avg_w) for c in range(n_cols)]
    row_heights_all = [orig_row_h.get(r, avg_h) for r in range(full_n_rows)]
    row_heights_orig = row_heights_all[trim_start : trim_end + 1]

    # --- Scale to target width ---
    target_w = 1400
    total_orig_w = sum(col_widths_orig)
    scale = target_w / total_orig_w if total_orig_w > 0 else 1.0

    col_widths = [max(int(w * scale), 30) for w in col_widths_orig]
    row_heights = [max(int(h * scale), 28) for h in row_heights_orig]

    # Ensure shape column is wide enough for template images
    if shape_col_idx is not None and 0 <= shape_col_idx < n_cols:
        col_widths[shape_col_idx] = max(col_widths[shape_col_idx], 160)

    # --- Compute grid positions ---
    col_xs = [0] * (n_cols + 1)
    for c in range(n_cols):
        col_xs[c + 1] = col_xs[c] + col_widths[c]

    row_ys = [0] * (n_rows + 1)
    for r in range(n_rows):
        row_ys[r + 1] = row_ys[r] + row_heights[r]

    table_w = col_xs[n_cols]
    table_h = row_ys[n_rows]

    # --- Resolve row types (prefer caller-supplied, else detect generically) ---
    if row_types is None:
        row_types = _detect_row_types(full_n_rows, n_cols, cell_contents, grid_cells)
    row_types = {
        r - trim_start: rt for r, rt in row_types.items() if trim_start <= r <= trim_end
    }

    # --- Font sizing based on the rendered grid scale ---
    fonts = _build_render_fonts(row_heights, row_types, shape_col_idx)

    # --- Non-table region space (document title, etc.) ---
    pad = 14
    title_lines = _collect_render_title_lines(non_table_regions, table_bbox=table_bbox)

    title_line_h = _font_line_height(fonts["title"]) + 8
    title_h = len(title_lines) * title_line_h + (10 if title_lines else 0)

    # --- Canvas ---
    canvas_w = table_w + pad * 2
    canvas_h = title_h + table_h + pad * 2

    img = Image.new("RGB", (canvas_w, canvas_h), "white")
    draw = ImageDraw.Draw(img)

    # --- Render non-table regions above the table ---
    ty = pad
    for label, text in title_lines:
        f = fonts["title"] if "title" in label else fonts["meta"]
        draw.text((pad, ty), text, fill="#000000", font=f)
        ty += title_line_h

    ox = pad  # table x-offset on canvas
    oy = title_h + pad  # table y-offset on canvas

    # --- Render every cell ---
    drawn: set[tuple[int, int]] = set()  # canonical positions already rendered

    for r in range(trim_start, trim_end + 1):
        for c in range(n_cols):
            if (r, c) in drawn:
                continue

            gc_or_none = cell_lookup.get((r, c))
            if gc_or_none is None:
                drawn.add((r, c))
                continue
            gc = gc_or_none

            canonical = (gc.row, gc.col)
            if canonical in drawn:
                drawn.add((r, c))
                continue

            # Mark all positions covered by this cell
            for rr in range(gc.row, gc.row + gc.rowspan):
                for cc in range(gc.col, gc.col + gc.colspan):
                    drawn.add((rr, cc))

            # Cell rectangle in canvas coordinates
            render_row = gc.row - trim_start
            x1 = ox + col_xs[gc.col]
            y1 = oy + row_ys[render_row]
            x2 = ox + col_xs[min(gc.col + gc.colspan, n_cols)]
            y2 = oy + row_ys[min(gc.row + gc.rowspan - trim_start, n_rows)]
            cell_w = x2 - x1
            cell_h = y2 - y1

            # Background colour based on row type
            bg = _row_background(render_row, row_types)
            draw.rectangle([x1 + 1, y1 + 1, x2 - 1, y2 - 1], fill=bg)
            draw.rectangle([x1, y1, x2, y2], outline="#333333")

            # Cell content
            content = cell_contents.get(canonical, {})

            if "shape_id" in content:
                _draw_shape_cell(
                    draw,
                    img,
                    x1,
                    y1,
                    cell_w,
                    cell_h,
                    content,
                    shapes_dir,
                    fonts["shape"],
                )
            else:
                text = content.get("text", "")
                if text:
                    rt = row_types.get(render_row, "data")
                    f = _font_for_row_type(rt, fonts)
                    _draw_fitted_wrapped_text(
                        draw, text, x1 + 4, y1 + 3, cell_w - 8, cell_h - 6, f
                    )

    return np.array(img)[..., ::-1]  # RGB -> BGR


def _collect_render_title_lines(
    non_table_regions: list[NonTableRegion] | None,
    *,
    table_bbox: BBox | None = None,
) -> list[tuple[str, str]]:
    """Return only header/title lines that belong above the rendered table.

    The layout model may detect shape images as ``image`` regions, and OCR on
    those regions can produce short garbage strings such as ``A\\nC``. Those
    must never be rendered as page titles.
    """
    if not non_table_regions:
        return []

    allowed_labels = {"figure_title", "doc_title", "text"}
    max_gap_above = 250
    overlap_tolerance = 10

    filtered: list[tuple[str, str, int, int]] = []
    for ntr in non_table_regions:
        label = (ntr.label or "").strip()
        text = (ntr.text or "").strip()
        if not text or label not in allowed_labels:
            continue

        if table_bbox is not None:
            region_bottom = ntr.bbox.y + ntr.bbox.h
            gap_above = table_bbox.y - region_bottom

            # Keep only regions that sit above the table and close to its top.
            if region_bottom > table_bbox.y + overlap_tolerance:
                continue
            if gap_above < -overlap_tolerance or gap_above > max_gap_above:
                continue

        filtered.append((label, text, ntr.bbox.y, ntr.bbox.x))

    filtered.sort(key=lambda item: (item[2], item[3]))
    return [(label, text) for label, text, _, _ in filtered]


def _find_nonempty_row_range(
    n_rows: int,
    n_cols: int,
    cell_lookup: dict[tuple[int, int], GridCell],
    cell_contents: dict[tuple[int, int], dict],
) -> tuple[int, int]:
    """Trim only leading/trailing rows that contain no text or shape data."""
    nonempty_rows: list[int] = []

    for r in range(n_rows):
        canonical_cells: set[tuple[int, int]] = set()
        has_content = False
        for c in range(n_cols):
            gc = cell_lookup.get((r, c))
            if gc is None:
                continue
            canonical = (gc.row, gc.col)
            if canonical in canonical_cells:
                continue
            canonical_cells.add(canonical)

            content = cell_contents.get(canonical, {})
            text = (content.get("text") or "").strip()
            shape_id = content.get("shape_id")
            components = content.get("components") or []
            if text or shape_id or components:
                has_content = True
                break

        if has_content:
            nonempty_rows.append(r)

    if not nonempty_rows:
        return 0, n_rows - 1
    return nonempty_rows[0], nonempty_rows[-1]


def _build_render_fonts(
    row_heights: list[int],
    row_types: dict[int, str],
    shape_col_idx: int | None,
) -> dict[str, ImageFont.FreeTypeFont | ImageFont.ImageFont]:
    """Choose readable font sizes that follow the rendered table scale."""
    meta_rows = [row_heights[r] for r, rt in row_types.items() if rt == "meta"]
    header_rows = [row_heights[r] for r, rt in row_types.items() if rt == "col_header"]
    unit_rows = [row_heights[r] for r, rt in row_types.items() if rt == "unit"]
    data_rows = [row_heights[r] for r, rt in row_types.items() if rt == "data"]

    default_row = int(np.median(row_heights)) if row_heights else 40
    meta_base = int(np.median(meta_rows)) if meta_rows else default_row
    header_base = int(np.median(header_rows)) if header_rows else default_row
    unit_base = int(np.median(unit_rows)) if unit_rows else max(default_row // 2, 28)
    data_base = int(np.median(data_rows)) if data_rows else default_row

    meta_size = _clamp(int(meta_base * 0.28), 16, 34)
    header_size = _clamp(int(header_base * 0.30), 18, 38)
    unit_size = _clamp(int(unit_base * 0.34), 14, 26)
    data_size = _clamp(int(data_base * 0.24), 16, 30)
    title_size = _clamp(max(meta_size + 6, int(meta_base * 0.34)), 22, 44)
    shape_size = _clamp(data_size - 2, 13, 24)

    return {
        "title": _load_cjk_font(title_size),
        "meta": _load_cjk_font(meta_size),
        "header": _load_cjk_font(header_size),
        "unit": _load_cjk_font(unit_size),
        "data": _load_cjk_font(data_size),
        "shape": _load_cjk_font(shape_size),
    }


def _font_for_row_type(
    row_type: str,
    fonts: dict[str, ImageFont.FreeTypeFont | ImageFont.ImageFont],
) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    if row_type == "meta":
        return fonts["meta"]
    if row_type == "col_header":
        return fonts["header"]
    if row_type == "unit":
        return fonts["unit"]
    return fonts["data"]


def _font_line_height(
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> int:
    bbox = font.getbbox("測試Ag")
    return int(bbox[3] - bbox[1])


def _clamp(value: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, value))


# ================================================================
# Image 04 — Repair verification
# ================================================================


def _draw_repair_debug(
    table_crop: np.ndarray,
    repair_log: list[dict],
    cell_contents: dict[tuple[int, int], dict],
    shapes_dir: Path | None,
) -> np.ndarray:
    """Overlay showing original merged cells (red) and split cells (green).

    Each split cell is labelled with its shape_id or OCR text.
    """
    vis = table_crop.copy()

    for entry in repair_log:
        orig = entry["original"]
        splits = entry["split_cells"]

        # Original merged cell — thick red rectangle
        ob = orig.bbox
        cv2.rectangle(
            vis,
            (ob.x, ob.y),
            (ob.x + ob.w, ob.y + ob.h),
            (0, 0, 255),
            3,
        )
        cv2.putText(
            vis,
            f"ORIG rs={orig.rowspan}",
            (ob.x + 4, ob.y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
        )

        # Split cells — green rectangles with classification labels
        for sc in splits:
            sb = sc.bbox
            cv2.rectangle(
                vis,
                (sb.x, sb.y),
                (sb.x + sb.w, sb.y + sb.h),
                (0, 200, 0),
                2,
            )

            content = cell_contents.get((sc.row, sc.col), {})
            if "shape_id" in content:
                sid = content.get("shape_id") or "None"
                conf = content.get("confidence", 0)
                label = f"{sid} ({conf:.2f})"
                color = (0, 140, 255)  # orange
            else:
                text = content.get("text", "")[:12]
                label = f'"{text}"' if text else "(empty)"
                color = (200, 200, 0)  # cyan

            cv2.putText(
                vis,
                label,
                (sb.x + 4, sb.y + sb.h // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
            )

            # Paste a small template preview if shape was identified
            if shapes_dir and content.get("shape_id"):
                tmpl = resolve_template_image(content["shape_id"], shapes_dir)
                if tmpl is not None:
                    max_h = max(sb.h - 20, 20)
                    max_w = min(sb.w // 3, 80)
                    fitted = _fit_image(tmpl, max_w, max_h)
                    arr = np.array(fitted)[..., ::-1]  # RGB->BGR
                    px = sb.x + sb.w - fitted.width - 4
                    py = sb.y + 4
                    if (
                        py + arr.shape[0] <= vis.shape[0]
                        and px + arr.shape[1] <= vis.shape[1]
                        and px >= 0
                        and py >= 0
                    ):
                        vis[py : py + arr.shape[0], px : px + arr.shape[1]] = arr

    return vis


# ================================================================
# Rendered-table helpers
# ================================================================


def _detect_row_types(
    n_rows: int,
    n_cols: int,
    cell_contents: dict[tuple[int, int], dict],
    grid_cells: list[GridCell],
) -> dict[int, str]:
    """Fallback: detect row types using generic heuristics (no keywords).

    Used only when the caller does not supply row_types from assembly.
    """
    cell_lookup: dict[tuple[int, int], GridCell] = {}
    for gc in grid_cells:
        for rr in range(gc.row, gc.row + gc.rowspan):
            for cc in range(gc.col, gc.col + gc.colspan):
                cell_lookup[(rr, cc)] = gc

    def get_text(r: int, c: int) -> str:
        gc = cell_lookup.get((r, c))
        if gc is None:
            return ""
        return cell_contents.get((gc.row, gc.col), {}).get("text", "").strip()

    # --- Find column-header row (generic scoring) ---
    best_row = 0
    best_score = -1
    for r in range(min(n_rows, 8)):
        score = 0
        non_empty = 0
        has_full_span = False
        for c in range(n_cols):
            gc_or_none = cell_lookup.get((r, c))
            if gc_or_none is None:
                continue
            gc = gc_or_none
            if gc.colspan >= n_cols - 1 and n_cols > 2:
                has_full_span = True
                break
            if (gc.row, gc.col) != (r, c):
                continue
            text = get_text(r, c)
            if not text:
                continue
            non_empty += 1
            if text.startswith("(") or text.startswith("（"):
                score -= 2
                continue
            cleaned = (
                text.replace(".", "", 1)
                .replace(",", "")
                .replace("-", "", 1)
                .replace(" ", "")
            )
            if cleaned.isdigit():
                score -= 1
                continue
            score += 2 if len(text) <= 8 else 1
        if has_full_span:
            continue
        score += non_empty
        if score > best_score:
            best_score = score
            best_row = r

    col_header_row = best_row

    # --- Detect unit row (parenthesis heuristic) ---
    unit_row = -1
    candidate = col_header_row + 1
    if candidate < n_rows:
        for c in range(n_cols):
            gc_or_none = cell_lookup.get((candidate, c))
            if gc_or_none is None:
                continue
            gc = gc_or_none
            if gc.row != candidate:
                continue
            text = get_text(candidate, c)
            if text and (text.startswith("(") or text.startswith("（")):
                unit_row = candidate
                break

    # --- Build classifications ---
    row_types: dict[int, str] = {}
    for r in range(n_rows):
        if r < col_header_row:
            row_types[r] = "meta"
        elif r == col_header_row:
            row_types[r] = "col_header"
        elif r == unit_row:
            row_types[r] = "unit"
        else:
            row_types[r] = "data"
    return row_types


def _row_background(row: int, row_types: dict[int, str]) -> str:
    """Return background colour based on row type."""
    rt = row_types.get(row, "data")
    if rt == "meta":
        return "#E8EEF4"  # metadata rows — light blue
    if rt == "col_header":
        return "#D0D8E0"  # column headers — gray
    if rt == "unit":
        return "#E0E4E8"  # unit row — lighter gray
    if rt == "summary":
        return "#FFF3E0"  # summary rows — light orange
    return "#F7F9FB" if row % 2 == 0 else "#FFFFFF"


def _draw_shape_cell(
    draw: ImageDraw.ImageDraw,
    canvas: Image.Image,
    x: int,
    y: int,
    w: int,
    h: int,
    cell: dict,
    shapes_dir: Path,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Render shape template image(s) and label inside a cell."""
    shape_id = cell.get("shape_id")
    is_composite = cell.get("is_composite", False)
    components = cell.get("components", [])

    label_h = _clamp(int(h * 0.16), 16, 30)
    inner_pad = _clamp(int(min(w, h) * 0.03), 3, 8)
    img_area_h = h - label_h - inner_pad * 3

    if is_composite and len(components) >= 2:
        # --- Composite: multiple templates side-by-side ---
        n = len(components)
        comp_w = max((w - inner_pad * 2) // n, 20)
        cx = x + inner_pad
        for comp in components:
            cid = comp.get("shape_id")
            tmpl = resolve_template_image(cid, shapes_dir)
            if tmpl is not None:
                fitted = _fit_image(tmpl, comp_w - 4, img_area_h)
                px = cx + (comp_w - fitted.width) // 2
                py = y + inner_pad + (img_area_h - fitted.height) // 2
                canvas.paste(fitted, (px, py))
            else:
                draw.text(
                    (cx + 2, y + img_area_h // 2), "N/A", fill="#999999", font=font
                )
            cx += comp_w

        ids = [c.get("shape_id", "?") for c in components]
        draw.text(
            (x + inner_pad, y + h - label_h - inner_pad),
            " + ".join(ids),
            fill="#E65100",
            font=font,
        )
    else:
        # --- Single shape ---
        tmpl = resolve_template_image(shape_id, shapes_dir)
        if tmpl is not None:
            fitted = _fit_image(tmpl, w - inner_pad * 2, img_area_h)
            px = x + (w - fitted.width) // 2
            py = y + inner_pad + (img_area_h - fitted.height) // 2
            canvas.paste(fitted, (px, py))
        elif shape_id:
            draw.text(
                (x + inner_pad, y + img_area_h // 2),
                f"missing: {shape_id}",
                fill="#CC0000",
                font=font,
            )
        else:
            _draw_centered_text(draw, "N/A", x, y, w, h - label_h, font, fill="#999999")

        label = shape_id if shape_id else "N/A"
        draw.text(
            (x + inner_pad, y + h - label_h - inner_pad),
            label,
            fill="#E65100",
            font=font,
        )


def resolve_template_image(
    shape_id: str | None,
    shapes_dir: Path,
) -> Image.Image | None:
    """Load the template PNG for *shape_id*, or ``None`` if unavailable."""
    if not shape_id:
        return None
    path = shapes_dir / f"{shape_id}.png"
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None


def _fit_image(img: Image.Image, max_w: int, max_h: int) -> Image.Image:
    """Resize *img* to fit within *max_w* x *max_h*, keeping aspect ratio."""
    w, h = img.size
    if w <= 0 or h <= 0:
        return img
    scale = min(max_w / w, max_h / h, 1.0)
    if scale >= 1.0:
        return img
    new_w = max(int(w * scale), 1)
    new_h = max(int(h * scale), 1)
    return img.resize((new_w, new_h), Image.LANCZOS)  # type: ignore[attr-defined]


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    w: int,
    h: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str = "black",
) -> None:
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text((x + (w - tw) // 2, y + (h - th) // 2), text, fill=fill, font=font)


def _draw_right_aligned(
    draw: ImageDraw.ImageDraw,
    text: str,
    right_x: int,
    y: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str = "black",
) -> None:
    tw = draw.textbbox((0, 0), text, font=font)[2]
    draw.text((right_x - tw, y), text, fill=fill, font=font)


def _draw_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    max_w: int,
    max_h: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str = "black",
) -> None:
    """Draw text with basic new-line splitting and truncation."""
    sample = draw.textbbox((0, 0), "中A", font=font)
    line_h = sample[3] - sample[1] + 3
    y_start = y

    for line in text.replace("\r", "").split("\n"):
        if y - y_start + line_h > max_h:
            break
        display = line
        tw = draw.textbbox((0, 0), display, font=font)[2]
        if tw > max_w and len(display) > 1:
            while tw > max_w and len(display) > 1:
                display = display[:-1]
                tw = draw.textbbox((0, 0), display + "…", font=font)[2]
            display += "…"
        draw.text((x, y), display, fill=fill, font=font)
        y += int(line_h)


# ================================================================
# OpenCV overlays (images 01 & 02 — unchanged)
# ================================================================


def _draw_layout(
    image: np.ndarray,
    regions: list[LayoutRegion],
) -> np.ndarray:
    vis = image.copy()
    for r in regions:
        color = _LABEL_COLORS.get(r.label, COLOR_OTHER)
        b = r.bbox
        cv2.rectangle(vis, (b.x, b.y), (b.x + b.w, b.y + b.h), color, 3)
        label = f"{r.label} {r.confidence:.2f}"
        cv2.putText(
            vis,
            label,
            (b.x, max(b.y - 6, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
    return vis


def _draw_grid(
    table_img: np.ndarray,
    cells: list[GridCell],
    shape_col: int | None,
) -> np.ndarray:
    vis = table_img.copy()
    for gc in cells:
        b = gc.bbox
        covers_shape = (
            shape_col is not None and gc.col <= shape_col < gc.col + gc.colspan
        )
        is_merged = gc.rowspan > 1 or gc.colspan > 1
        color = COLOR_SHAPE if covers_shape else COLOR_CELL
        thickness = 2 if is_merged else 1
        cv2.rectangle(vis, (b.x, b.y), (b.x + b.w, b.y + b.h), color, thickness)
        label = f"{gc.row},{gc.col}"
        if is_merged:
            label += f" ({gc.rowspan}x{gc.colspan})"
        cv2.putText(
            vis,
            label,
            (b.x + 2, b.y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
        )
    return vis


def _draw_fitted_wrapped_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    x: int,
    y: int,
    max_w: int,
    max_h: int,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    fill: str = "black",
) -> None:
    """Draw text per cell using only OCR-provided lines, auto-shrunk to fit."""
    if not text or max_w <= 4 or max_h <= 4:
        return

    fitted_font, lines, line_h, line_spacing = _fit_text_block_to_cell(
        text=text,
        base_font=font,
        max_w=max_w,
        max_h=max_h,
    )

    total_h = len(lines) * line_h + max(0, len(lines) - 1) * line_spacing
    start_y = y + max((max_h - total_h) // 2, 0)

    for idx, line in enumerate(lines):
        draw.text((x, start_y), line, fill=fill, font=fitted_font)
        start_y += line_h
        if idx < len(lines) - 1:
            start_y += line_spacing


def _fit_text_block_to_cell(
    *,
    text: str,
    base_font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
    max_w: int,
    max_h: int,
    min_font_size: int = 8,
) -> tuple[
    ImageFont.FreeTypeFont | ImageFont.ImageFont,
    list[str],
    int,
    int,
]:
    """Keep OCR line breaks, shrink font until all lines fit inside a cell."""
    base_size = _font_pixel_size(base_font)
    last_font = base_font
    last_lines = _split_explicit_lines(text)
    last_line_h = _font_line_height(base_font)
    last_spacing = _line_spacing(last_line_h)

    for size in range(base_size, min_font_size - 1, -1):
        fitted_font = _load_cjk_font(size)
        lines = _split_explicit_lines(text)
        line_h = _font_line_height(fitted_font)
        spacing = _line_spacing(line_h)
        max_line_w = max((_text_width(line, fitted_font) for line in lines), default=0)
        total_h = len(lines) * line_h + max(0, len(lines) - 1) * spacing
        last_font, last_lines, last_line_h, last_spacing = (
            fitted_font,
            lines,
            line_h,
            spacing,
        )
        if total_h <= max_h and max_line_w <= max_w:
            return fitted_font, lines, line_h, spacing

    return last_font, last_lines, last_line_h, last_spacing


def _split_explicit_lines(
    text: str,
) -> list[str]:
    """Preserve OCR-provided lines; do not auto-wrap a single OCR line."""
    return text.replace("\r", "").split("\n") or [""]


def _text_width(
    text: str,
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> int:
    if not text:
        return 0
    bbox = font.getbbox(text)
    return int(bbox[2] - bbox[0])


def _font_pixel_size(
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> int:
    return getattr(font, "size", 14)


def _line_spacing(line_h: int) -> int:
    return _clamp(int(line_h * 0.18), 2, 8)
