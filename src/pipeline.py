"""Top-level pipeline orchestrator."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2
import numpy as np

from src.config import PipelineConfig, DEFAULT_OUTPUT_DIR
from src.models import (
    DocumentResult,
    PageResult,
)


def run_pipeline(
    pdf_path: Path,
    output_path: Path,
    *,
    debug: bool = False,
    dpi: int = 300,
    classifier_name: str = "template",
) -> DocumentResult:
    """Execute the full OCR pipeline: PDF -> JSON."""
    from src.pdf_rasterizer import rasterize_pdf
    from src.layout_analyzer import analyze_layout
    from src.table_cell_segmenter import segment_table_cells
    from src.ocr_engine import init_ocr, recognize_text
    from src.shape_classifier import get_classifier
    from src.assembler import (
        assemble_page, document_to_dict,
        find_column_header_row, classify_rows_from_grid,
    )
    from src.debug_visualizer import save_debug_page
    from src.shape_column_repair import repair_shape_column, is_likely_shape

    start = time.time()
    cfg = PipelineConfig(pdf_dpi=dpi, debug=debug)

    # --- 1. Rasterize PDF ---
    print(f"[1/5] Rasterizing PDF ({dpi} DPI) ...")
    page_images = rasterize_pdf(pdf_path, dpi=dpi)
    print(f"       {len(page_images)} page(s) rasterized.")

    # --- 2. Init OCR + shape classifier ---
    print("[2/5] Initializing OCR engine & shape classifier ...")
    ocr = init_ocr()
    classifier_kwargs: dict = {}
    if classifier_name == "template":
        classifier_kwargs = {
            "threshold": cfg.template_match_threshold,
            "resize_heights": cfg.template_resize_heights,
            "composite_min_area_ratio": cfg.composite_min_area_ratio,
            "composite_min_dim": cfg.composite_min_dim,
        }
    elif classifier_name == "cnn":
        classifier_kwargs = {"threshold": cfg.cnn_match_threshold}
    classifier = get_classifier(classifier_name, **classifier_kwargs)
    classifier.load_templates(cfg.shapes_dir)

    # --- 3. Process each page ---
    pages: list[PageResult] = []
    for page_idx, page_img in enumerate(page_images):
        print(f"[3/5] Processing page {page_idx + 1}/{len(page_images)} ...")

        # Layout analysis
        regions = analyze_layout(page_img, threshold=cfg.layout_score_threshold)
        table_regions = [r for r in regions if r.label == "table"]
        non_table_regions = [r for r in regions if r.label != "table"]

        # Process non-table regions
        non_table_results = []
        for region in non_table_regions:
            b = region.bbox
            crop = page_img[b.y : b.y + b.h, b.x : b.x + b.w]
            ocr_res = recognize_text(ocr, crop)
            from src.models import NonTableRegion
            non_table_results.append(
                NonTableRegion(bbox=region.bbox, label=region.label, text=ocr_res.text)
            )

        # Process table regions
        tables = []
        for t_idx, table_region in enumerate(table_regions):
            b = table_region.bbox
            table_crop = page_img[b.y : b.y + b.h, b.x : b.x + b.w]

            # Segment cells
            grid_cells = segment_table_cells(table_crop, cfg)

            # Phase 1: OCR all cells first
            cell_contents: dict[tuple[int, int], dict] = {}
            for gc in grid_cells:
                cb = gc.bbox
                cell_crop = table_crop[cb.y : cb.y + cb.h, cb.x : cb.x + cb.w]
                ocr_res = recognize_text(ocr, cell_crop)
                cell_contents[(gc.row, gc.col)] = {
                    "text": ocr_res.text,
                    "confidence": round(ocr_res.confidence, 3),
                }

            # Phase 2: Lightweight row classification
            col_header_row = find_column_header_row(grid_cells, cell_contents)
            row_types = classify_rows_from_grid(
                grid_cells, cell_contents, col_header_row,
            )

            # Determine data row start from row classifications
            data_start_row = col_header_row + 1
            if row_types.get(data_start_row) == "unit":
                data_start_row += 1

            # Phase 2b: Infer shape column (generic strategy)
            shape_col_idx = _infer_shape_column(
                grid_cells, cell_contents, table_crop,
                data_start_row=data_start_row,
            )

            # Phase 2.5: Repair shape column — split false merges
            repair_log: list[dict] = []
            if shape_col_idx is not None:
                grid_cells, repair_log = repair_shape_column(
                    grid_cells, shape_col_idx,
                )

                # Phase 2.6: Re-OCR repaired (split) cells
                for entry in repair_log:
                    for sc in entry["split_cells"]:
                        cb = sc.bbox
                        scrop = table_crop[cb.y : cb.y + cb.h, cb.x : cb.x + cb.w]
                        ocr_res = recognize_text(ocr, scrop)
                        cell_contents[(sc.row, sc.col)] = {
                            "text": ocr_res.text,
                            "confidence": round(ocr_res.confidence, 3),
                        }

            # Phase 3: Classify valid shape-column cells
            if shape_col_idx is not None:
                for gc in grid_cells:
                    covers_shape_col = (
                        gc.col <= shape_col_idx < gc.col + gc.colspan
                    )
                    if not covers_shape_col:
                        continue
                    # Skip non-data rows using classification
                    rt = row_types.get(gc.row, "data")
                    if rt in ("meta", "col_header", "unit"):
                        continue

                    cb = gc.bbox
                    cell_crop = table_crop[cb.y : cb.y + cb.h, cb.x : cb.x + cb.w]

                    # Validity check — skip cells that are text/numbers
                    ocr_data = cell_contents.get((gc.row, gc.col), {})
                    adj_texts = [
                        cell_contents.get((oc.row, oc.col), {}).get("text", "")
                        for oc in grid_cells
                        if oc.row == gc.row and oc.col != gc.col
                    ]
                    if not is_likely_shape(
                        cell_crop,
                        ocr_data.get("text", ""),
                        ocr_data.get("confidence", 0),
                        [t for t in adj_texts if t],
                    ):
                        continue  # keep OCR text as-is

                    matches = classifier.classify(cell_crop)
                    if matches:
                        is_composite = len(matches) > 1
                        if is_composite:
                            cell_contents[(gc.row, gc.col)] = {
                                "shape_id": matches[0].shape_id,
                                "confidence": round(matches[0].confidence, 3),
                                "is_composite": True,
                                "components": [
                                    {
                                        "shape_id": m.shape_id,
                                        "confidence": round(m.confidence, 3),
                                    }
                                    for m in matches
                                ],
                            }
                        else:
                            m = matches[0]
                            cell_contents[(gc.row, gc.col)] = {
                                "shape_id": m.shape_id,
                                "confidence": round(m.confidence, 3),
                                "is_composite": False,
                            }
                    else:
                        cell_contents[(gc.row, gc.col)] = {
                            "shape_id": None,
                            "confidence": 0.0,
                            "is_composite": False,
                        }

            table = assemble_page(
                table_region.bbox, grid_cells, cell_contents, cfg,
                shape_col_idx=shape_col_idx,
            )
            tables.append(table)

            # Debug overlays
            if debug:
                debug_dir = output_path.parent / "debug" / f"page_{page_idx}"
                save_debug_page(
                    page_img, regions, table_crop, grid_cells,
                    cell_contents, shape_col_idx, debug_dir, page_idx, t_idx,
                    table=table, shapes_dir=cfg.shapes_dir,
                    repair_log=repair_log,
                    non_table_regions=non_table_results,
                )

        pages.append(PageResult(
            page_index=page_idx,
            tables=tables,
            non_table_regions=non_table_results,
        ))

    # --- 4. Assemble document ---
    elapsed = round(time.time() - start, 2)
    doc = DocumentResult(
        source_pdf=str(pdf_path),
        total_pages=len(page_images),
        pages=pages,
        processing_time_seconds=elapsed,
    )

    # --- 5. Write JSON ---
    print(f"[4/5] Writing JSON to {output_path} ...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(document_to_dict(doc), f, ensure_ascii=False, indent=2)

    print(f"[5/5] Done in {elapsed}s.")
    return doc


# ===================================================================
# Shape column inference — generic, content-based
# ===================================================================

# Keyword hints (used as one signal, never as sole fallback)
_SHAPE_KEYWORDS = {"形狀", "shape", "形状"}


def _infer_shape_column(
    grid_cells: list[GridCell],
    cell_contents: dict[tuple[int, int], dict],
    table_crop: np.ndarray,
    *,
    data_start_row: int = 0,
) -> int | None:
    """Infer which column contains shape images using generic strategies.

    Combines:
    1. Header keyword hints (形狀, shape, etc.) — one signal among many.
    2. Content analysis: columns where data cells have low OCR confidence
       and image-like pixel characteristics.

    Returns ``None`` if no shape column can be confidently identified.
    """
    if not grid_cells:
        return None

    from src.models import GridCell as _GC  # type hint only

    n_cols = max(gc.col + gc.colspan for gc in grid_cells)

    # --- Strategy 1: keyword hint in header rows ---
    for (r, c), content in cell_contents.items():
        if r >= data_start_row:
            continue
        text = (content.get("text", "") or "").strip()
        for kw in _SHAPE_KEYWORDS:
            if kw in text:
                return c

    # --- Strategy 2: content-based scoring on data rows ---
    col_scores: dict[int, float] = {c: 0.0 for c in range(n_cols)}
    col_counts: dict[int, int] = {c: 0 for c in range(n_cols)}

    for gc in grid_cells:
        if gc.row < data_start_row:
            continue
        if gc.colspan > 1:
            continue  # skip merged cells for scoring

        c = gc.col
        col_counts[c] += 1

        content = cell_contents.get((gc.row, gc.col), {})
        text = (content.get("text", "") or "").strip()
        confidence = content.get("confidence", 0.0)

        # Low / no OCR result → shape-like
        if not text:
            col_scores[c] += 1.0
        elif confidence < 0.3:
            col_scores[c] += 0.8
        elif confidence < 0.5:
            col_scores[c] += 0.3
        else:
            # High-confidence text → not shape
            cleaned = (
                text.replace(".", "", 1)
                    .replace(",", "")
                    .replace("-", "", 1)
                    .replace(" ", "")
            )
            if cleaned.isdigit():
                col_scores[c] -= 1.0
            elif len(text) > 3:
                col_scores[c] -= 0.5

        # Pixel analysis: check if cell looks like a line drawing
        cb = gc.bbox
        cell_crop = table_crop[cb.y : cb.y + cb.h, cb.x : cb.x + cb.w]
        if _cell_has_drawing_characteristics(cell_crop):
            col_scores[c] += 0.5

    # Normalise and pick best
    best_col = None
    best_norm = 0.0
    for c in range(n_cols):
        count = col_counts[c]
        if count < 2:
            continue  # need enough data cells
        norm = col_scores[c] / count
        if norm > best_norm:
            best_norm = norm
            best_col = c

    # Require a meaningful signal
    if best_norm >= 0.4:
        return best_col

    return None


def _cell_has_drawing_characteristics(cell_crop: np.ndarray) -> bool:
    """Check if a cell crop has pixel characteristics of a line drawing.

    Line drawings typically have 1–15% foreground pixels with significant
    spatial spread across the cell.
    """
    h, w = cell_crop.shape[:2]
    if h < 15 or w < 15:
        return False

    gray = (
        cv2.cvtColor(cell_crop, cv2.COLOR_BGR2GRAY)
        if len(cell_crop.shape) == 3
        else cell_crop
    )
    _, bw = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)

    fg_ratio = np.count_nonzero(bw) / (h * w)
    if fg_ratio < 0.005 or fg_ratio > 0.35:
        return False

    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    min_area = max(h * w * 0.003, 8)
    sig = [
        i for i in range(1, n_labels)
        if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]
    if not sig:
        return False

    max_cw = max(stats[i, cv2.CC_STAT_WIDTH] for i in sig)
    max_ch = max(stats[i, cv2.CC_STAT_HEIGHT] for i in sig)
    return max_cw > w * 0.2 or max_ch > h * 0.2
