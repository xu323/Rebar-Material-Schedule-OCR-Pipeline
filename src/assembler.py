"""Assemble OCR + shape results into structured JSON output.

Layer 1 — Generic table extraction:
    Preserves the full grid (all rows including meta, header, unit, data,
    summary) with soft row-type classifications and OCR'd column names.

Layer 2 — Optional semantic interpretation:
    Best-effort domain hints (header fields, shape column) stored in
    ``Table.semantic`` without overriding the raw structure.
"""

from __future__ import annotations

from src.config import PipelineConfig
from src.models import (
    BBox,
    DocumentResult,
    GridCell,
    PageResult,
    Table,
    TableRow,
    TableSemantic,
)

# ===================================================================
# Public API
# ===================================================================


def assemble_page(
    table_bbox: BBox,
    grid_cells: list[GridCell],
    cell_contents: dict[tuple[int, int], dict],
    cfg: PipelineConfig,
    *,
    shape_col_idx: int | None = None,
) -> Table:
    """Build a :class:`Table` from grid cells and their recognised contents.

    Produces a generic table including **all** rows (meta, header, unit,
    data, summary) with soft row-type classifications.  An optional
    semantic layer records domain-specific interpretation.
    """
    if not grid_cells:
        return Table(bbox=table_bbox, columns=[], rows=[])

    # --- Canonical-position lookup: (rr, cc) → GridCell ---
    cell_to_grid: dict[tuple[int, int], GridCell] = {}
    for gc in grid_cells:
        for rr in range(gc.row, gc.row + gc.rowspan):
            for cc in range(gc.col, gc.col + gc.colspan):
                cell_to_grid[(rr, cc)] = gc

    max_row = max(rr for (rr, _) in cell_to_grid)
    max_col = max(cc for (_, cc) in cell_to_grid)
    n_cols = max_col + 1

    def get_text(r: int, c: int) -> str:
        gc = cell_to_grid.get((r, c))
        if gc is None:
            return ""
        content = cell_contents.get((gc.row, gc.col), {})
        return content.get("text", "") or ""

    def get_content(r: int, c: int) -> dict:
        gc = cell_to_grid.get((r, c))
        if gc is None:
            return {}
        return cell_contents.get((gc.row, gc.col), {})

    # --- Find the column-header row (generic) ---
    col_header_row = _find_column_header_row(max_row, n_cols, get_text, cell_to_grid)

    # --- Read column names (OCR'd text, no reconciliation) ---
    column_names: list[str] = []
    for c in range(n_cols):
        text = get_text(col_header_row, c).replace("\n", "").strip()
        column_names.append(text if text else f"col_{c}")

    # --- Classify all rows ---
    row_types = _classify_rows(max_row, n_cols, col_header_row, get_text, cell_to_grid)

    # --- Extract generic meta fields ---
    meta_fields = _extract_meta_fields(
        col_header_row,
        n_cols,
        get_text,
        get_content,
        cell_to_grid,
    )

    # --- Build ALL rows ---
    rows: list[TableRow] = []
    for r in range(max_row + 1):
        row_cells: dict[str, dict] = {}
        row_merge_info: dict[str, dict] = {}

        for c in range(n_cols):
            col_name = column_names[c] if c < len(column_names) else f"col_{c}"
            gc_or_none = cell_to_grid.get((r, c))
            if gc_or_none is None:
                row_cells[col_name] = {}
                continue
            gc = gc_or_none

            canonical = (gc.row, gc.col)
            content = dict(cell_contents.get(canonical, {}))

            if gc.rowspan > 1 or gc.colspan > 1:
                is_continuation = (r, c) != canonical
                content["rowspan"] = gc.rowspan
                content["colspan"] = gc.colspan
                content["is_continuation"] = is_continuation
                row_merge_info[col_name] = {
                    "rowspan": gc.rowspan,
                    "colspan": gc.colspan,
                    "is_continuation": is_continuation,
                    "anchor_row": gc.row,
                    "anchor_col": gc.col,
                }

            row_cells[col_name] = content

        rows.append(
            TableRow(
                index=r,
                row_type=row_types.get(r, "unknown"),
                cells=row_cells,
                merge_info=row_merge_info or None,
            )
        )

    # --- Optional semantic layer ---
    semantic = _build_semantic(meta_fields, column_names, shape_col_idx)

    return Table(
        bbox=table_bbox,
        columns=column_names,
        rows=rows,
        col_header_row_index=col_header_row,
        meta_fields=meta_fields,
        semantic=semantic,
    )


# ===================================================================
# Public helpers (used by pipeline before full assembly)
# ===================================================================


def find_column_header_row(
    grid_cells: list[GridCell],
    cell_contents: dict[tuple[int, int], dict],
) -> int:
    """Find column-header row from grid cells + OCR results."""
    if not grid_cells:
        return 0

    cell_to_grid: dict[tuple[int, int], GridCell] = {}
    for gc in grid_cells:
        for rr in range(gc.row, gc.row + gc.rowspan):
            for cc in range(gc.col, gc.col + gc.colspan):
                cell_to_grid[(rr, cc)] = gc

    max_row = max(rr for (rr, _) in cell_to_grid)
    n_cols = max(cc for (_, cc) in cell_to_grid) + 1

    def get_text(r: int, c: int) -> str:
        gc = cell_to_grid.get((r, c))
        if gc is None:
            return ""
        return cell_contents.get((gc.row, gc.col), {}).get("text", "") or ""

    return _find_column_header_row(max_row, n_cols, get_text, cell_to_grid)


def classify_rows_from_grid(
    grid_cells: list[GridCell],
    cell_contents: dict[tuple[int, int], dict],
    col_header_row: int,
) -> dict[int, str]:
    """Classify all rows from grid cells + OCR results."""
    if not grid_cells:
        return {}

    cell_to_grid: dict[tuple[int, int], GridCell] = {}
    for gc in grid_cells:
        for rr in range(gc.row, gc.row + gc.rowspan):
            for cc in range(gc.col, gc.col + gc.colspan):
                cell_to_grid[(rr, cc)] = gc

    max_row = max(rr for (rr, _) in cell_to_grid)
    n_cols = max(cc for (_, cc) in cell_to_grid) + 1

    def get_text(r: int, c: int) -> str:
        gc = cell_to_grid.get((r, c))
        if gc is None:
            return ""
        return cell_contents.get((gc.row, gc.col), {}).get("text", "") or ""

    return _classify_rows(max_row, n_cols, col_header_row, get_text, cell_to_grid)


# ===================================================================
# Internal — row classification
# ===================================================================


def _find_column_header_row(
    max_row: int,
    n_cols: int,
    get_text,
    cell_to_grid: dict,
) -> int:
    """Find the column-header row using generic heuristics.

    Scores each candidate row (first 8 rows) on:
    - Number of non-empty, non-numeric, short-text cells (column names)
    - Penalises unit-like text (parenthesised values)
    - Penalises pure-number cells (data, not headers)
    - Skips full-width merged cells (title / meta rows)
    """
    best_row = 0
    best_score = -1

    for r in range(min(max_row + 1, 8)):
        score = 0
        non_empty = 0
        has_full_span = False

        for c in range(n_cols):
            gc = cell_to_grid.get((r, c))
            if gc is None:
                continue
            # Full-width merged cell → title / meta row
            if gc.colspan >= n_cols - 1 and n_cols > 2:
                has_full_span = True
                break
            # Only count canonical positions
            if (gc.row, gc.col) != (r, c):
                continue

            text = get_text(r, c).strip()
            if not text:
                continue
            non_empty += 1

            # Parenthesised text → unit row, not header
            if text.startswith("(") or text.startswith("（"):
                score -= 2
                continue

            # Pure numbers → data row
            cleaned = (
                text.replace(".", "", 1)
                .replace(",", "")
                .replace("-", "", 1)
                .replace(" ", "")
            )
            if cleaned.isdigit():
                score -= 1
                continue

            # Short text → good header candidate
            if len(text) <= 8:
                score += 2
            else:
                score += 1

        if has_full_span:
            continue

        # Bonus for filling many columns
        score += non_empty

        if score > best_score:
            best_score = score
            best_row = r

    return best_row


def _classify_rows(
    max_row: int,
    n_cols: int,
    col_header_row: int,
    get_text,
    cell_to_grid: dict,
) -> dict[int, str]:
    """Classify each row as meta / col_header / unit / data / summary."""
    classifications: dict[int, str] = {}

    for r in range(max_row + 1):
        if r < col_header_row:
            classifications[r] = "meta"
        elif r == col_header_row:
            classifications[r] = "col_header"
        elif r == col_header_row + 1 and _is_unit_row(
            r, n_cols, get_text, cell_to_grid
        ):
            classifications[r] = "unit"
        elif _is_summary_row(r, n_cols, get_text, cell_to_grid):
            classifications[r] = "summary"
        else:
            classifications[r] = "data"

    return classifications


def _is_unit_row(
    r: int,
    n_cols: int,
    get_text,
    cell_to_grid: dict,
) -> bool:
    """Detect if row *r* is a unit row (e.g. (#), (kg/m), (cm))."""
    for c in range(n_cols):
        gc = cell_to_grid.get((r, c))
        if gc is None:
            continue
        if gc.row != r:
            continue  # continuation from rowspan above
        text = get_text(r, c).strip()
        if text and (text.startswith("(") or text.startswith("（")):
            return True
    return False


# Universal summary keywords — not tied to any specific document type.
_GENERIC_SUMMARY_KW = {"合計", "小計", "總計", "total", "subtotal", "sum"}


def _is_summary_row(
    r: int,
    n_cols: int,
    get_text,
    cell_to_grid: dict,
) -> bool:
    """Detect summary / totals rows using generic patterns."""
    for c in range(n_cols):
        gc = cell_to_grid.get((r, c))
        if gc is None:
            continue
        if gc.row != r or gc.col != c:
            continue
        text = get_text(r, c).strip().lower()
        for kw in _GENERIC_SUMMARY_KW:
            if kw.lower() in text:
                return True
    return False


# ===================================================================
# Internal — meta field extraction
# ===================================================================


def _extract_meta_fields(
    col_header_row: int,
    n_cols: int,
    get_text,
    get_content,
    cell_to_grid: dict,
) -> list[dict[str, str]]:
    """Extract key-value pairs from meta rows above the column headers.

    Recognises ``Label：Value`` and ``Label: Value`` patterns.
    Falls back to checking the adjacent cell when the value is empty.
    Skips full-width cells (titles) and tracks consumed value cells.
    """
    fields: list[dict[str, str]] = []
    seen_keys: set[str] = set()
    consumed: set[tuple[int, int]] = set()  # (row, col) already used as values

    for r in range(col_header_row):
        for c in range(n_cols):
            if (r, c) in consumed:
                continue

            # Skip continuation positions of merged cells
            gc = cell_to_grid.get((r, c))
            if gc is None:
                continue
            if (gc.row, gc.col) != (r, c):
                continue  # this is a continuation cell

            # Skip full-width merged cells (title rows)
            if gc.colspan >= n_cols - 1 and n_cols > 2:
                continue

            content = get_content(r, c)
            text = (content.get("text", "") or "").strip()
            if not text:
                continue

            extracted = False
            for sep in ("：", ":"):
                if sep in text:
                    parts = text.split(sep, 1)
                    key = parts[0].strip()
                    value = parts[1].strip() if len(parts) > 1 else ""
                    if key and key not in seen_keys:
                        if not value:
                            right = get_content(r, c + 1)
                            value = (right.get("text", "") or "").strip()
                            if value:
                                consumed.add((r, c + 1))
                        fields.append({"key": key, "value": value})
                        seen_keys.add(key)
                    extracted = True
                    break

            # Label in one cell, value in the next
            if not extracted and len(text) <= 20 and text not in seen_keys:
                right = get_content(r, c + 1)
                right_text = (right.get("text", "") or "").strip()
                if right_text:
                    fields.append({"key": text, "value": right_text})
                    seen_keys.add(text)
                    consumed.add((r, c + 1))

    return fields


# ===================================================================
# Internal — semantic layer
# ===================================================================

_SHAPE_KEYWORDS = {"形狀", "shape", "形状"}


def _build_semantic(
    meta_fields: list[dict[str, str]],
    column_names: list[str],
    shape_col_idx: int | None,
) -> TableSemantic | None:
    """Build the optional semantic interpretation layer."""
    header_fields: dict[str, str] = {f["key"]: f["value"] for f in meta_fields}

    shape_col_name = None
    if shape_col_idx is not None and 0 <= shape_col_idx < len(column_names):
        shape_col_name = column_names[shape_col_idx]

    # Also try to identify shape column from names (if not passed in)
    if shape_col_idx is None:
        for i, name in enumerate(column_names):
            for kw in _SHAPE_KEYWORDS:
                if kw in name.lower():
                    shape_col_idx = i
                    shape_col_name = name
                    break
            if shape_col_idx is not None:
                break

    if not header_fields and shape_col_idx is None:
        return None

    return TableSemantic(
        header_fields=header_fields,
        shape_column_index=shape_col_idx,
        shape_column_name=shape_col_name,
    )


# ===================================================================
# Serialization
# ===================================================================

# Cells with confidence below this threshold are flagged for human review.
REVIEW_CONFIDENCE_THRESHOLD = 0.9


def _mark_needs_review(cell: dict) -> dict:
    """Return *cell* with ``needs_review`` set if confidence < threshold.

    Continuation cells of merged regions are not flagged directly;
    reviewers edit only the anchor cell.
    """
    if not isinstance(cell, dict) or not cell:
        return cell
    if cell.get("is_continuation"):
        return cell
    has_text = bool((cell.get("text") or "").strip())
    has_shape = bool(cell.get("shape_id"))
    has_components = bool(cell.get("components"))
    if not (has_text or has_shape or has_components):
        return cell
    conf = cell.get("confidence")
    if conf is None:
        return cell
    try:
        needs = float(conf) < REVIEW_CONFIDENCE_THRESHOLD
    except (TypeError, ValueError):
        return cell
    if needs:
        cell = dict(cell)
        cell["needs_review"] = True
    return cell


def document_to_dict(doc: DocumentResult) -> dict:
    return {
        "source_pdf": doc.source_pdf,
        "total_pages": doc.total_pages,
        "pipeline_version": doc.pipeline_version,
        "processing_time_seconds": doc.processing_time_seconds,
        "pages": [_page_to_dict(p) for p in doc.pages],
    }


def _page_to_dict(page: PageResult) -> dict:
    return {
        "page_index": page.page_index,
        "non_table_regions": [
            {
                "label": r.label,
                "text": r.text,
                "bbox": r.bbox.to_dict(),
            }
            for r in page.non_table_regions
        ],
        "tables": [_table_to_dict(t) for t in page.tables],
    }


def _table_to_dict(table: Table) -> dict:
    result: dict = {
        "bbox": table.bbox.to_dict(),
        "col_header_row_index": table.col_header_row_index,
        "meta_fields": table.meta_fields,
        "columns": table.columns,
        "rows": [
            {
                "index": row.index,
                "row_type": row.row_type,
                "cells": {
                    col: _mark_needs_review(cell) for col, cell in row.cells.items()
                },
                "merge_info": row.merge_info,
            }
            for row in table.rows
        ],
    }
    if table.semantic is not None:
        result["semantic"] = {
            "header_fields": table.semantic.header_fields,
            "shape_column_index": table.semantic.shape_column_index,
            "shape_column_name": table.semantic.shape_column_name,
        }
    else:
        result["semantic"] = None
    return result
