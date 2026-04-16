"""Re-render a corrected table for the human-in-the-loop review UI.

Reads the grid-cell sidecar produced by the pipeline plus a corrected
``Table`` dict (as stored in ``result.json``) and reproduces the same
image that ``debug_visualizer.render_faithful_table`` would have drawn.

The renderer never re-runs OCR — it only uses text / shape data already
present in the JSON, so edits made in the UI are reflected directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.debug_visualizer import render_faithful_table
from src.models import BBox, GridCell, NonTableRegion


def load_grid_sidecar(path: Path) -> tuple[list[GridCell], BBox, int | None]:
    """Load grid metadata previously written by the pipeline."""
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    table_bbox = BBox(**data["table_bbox"])
    shape_col_idx = data.get("shape_col_idx")
    cells = [
        GridCell(
            bbox=BBox(**c["bbox"]),
            row=c["row"],
            col=c["col"],
            rowspan=c.get("rowspan", 1),
            colspan=c.get("colspan", 1),
        )
        for c in data["cells"]
    ]
    return cells, table_bbox, shape_col_idx


def cell_contents_from_table_dict(
    table_dict: dict,
    grid_cells: list[GridCell],
) -> dict[tuple[int, int], dict]:
    """Reconstruct the ``cell_contents`` lookup used by the renderer.

    The JSON ``Table`` stores cells keyed by column name.  The renderer
    needs them keyed by canonical ``(row, col)`` of the owning GridCell.
    We iterate grid cells and pull the matching entry from the JSON row
    using the column name at index ``gc.col``.
    """
    rows_by_index: dict[int, dict] = {r["index"]: r for r in table_dict.get("rows", [])}

    cell_contents: dict[tuple[int, int], dict] = {}
    for gc in grid_cells:
        r = gc.row
        c = gc.col
        row = rows_by_index.get(r)
        if row is None:
            continue

        cell = _resolve_cell_by_column_index(table_dict, row, c)

        if not cell:
            continue
        clean = {k: v for k, v in cell.items() if k != "needs_review"}
        cell_contents[(r, c)] = clean
    return cell_contents


def _resolve_cell_by_column_index(
    table_dict: dict,
    row_dict: dict,
    column_index: int,
) -> dict | None:
    """Resolve a cell using the table's canonical column order."""
    columns = table_dict.get("columns", []) or []
    cells_dict = row_dict.get("cells", {}) or {}
    if 0 <= column_index < len(columns):
        cell = cells_dict.get(columns[column_index])
        if cell is not None:
            return cell

    keys = list(cells_dict.keys())
    if 0 <= column_index < len(keys):
        return cells_dict[keys[column_index]]
    return None


def non_table_regions_from_dict(
    page_dict: dict,
) -> list[NonTableRegion]:
    out: list[NonTableRegion] = []
    for ntr in page_dict.get("non_table_regions", []) or []:
        b = ntr.get("bbox") or {}
        out.append(
            NonTableRegion(
                bbox=BBox(
                    x=int(b.get("x", 0)),
                    y=int(b.get("y", 0)),
                    w=int(b.get("w", 0)),
                    h=int(b.get("h", 0)),
                ),
                label=ntr.get("label", ""),
                text=ntr.get("text", ""),
            )
        )
    return out


def rerender_table(
    page_dict: dict,
    table_dict: dict,
    sidecar_path: Path,
    shapes_dir: Path,
) -> np.ndarray:
    """Produce a BGR image reconstructing *table_dict* after edits."""
    grid_cells, table_bbox, shape_col_idx = load_grid_sidecar(sidecar_path)
    cell_contents = cell_contents_from_table_dict(table_dict, grid_cells)
    non_table_regions = non_table_regions_from_dict(page_dict)

    row_types = {
        row["index"]: row.get("row_type", "data") for row in table_dict.get("rows", [])
    }

    return render_faithful_table(
        grid_cells=grid_cells,
        cell_contents=cell_contents,
        shape_col_idx=shape_col_idx,
        shapes_dir=shapes_dir,
        non_table_regions=non_table_regions,
        table_bbox=table_bbox,
        row_types=row_types,
    )


def rerender_document(
    reviewed_json_path: Path,
    review_assets_dir: Path,
    shapes_dir: Path,
    output_dir: Path,
) -> list[Path]:
    """Re-render every table in the reviewed JSON.

    Outputs one PNG per table at
    ``output_dir/page_<i>/table_<j>_rendered.png`` and returns the list
    of written paths.
    """
    with open(reviewed_json_path, encoding="utf-8") as f:
        doc = json.load(f)

    written: list[Path] = []
    for page in doc.get("pages", []):
        page_idx = page.get("page_index", 0)
        page_assets = review_assets_dir / f"page_{page_idx}"
        page_out = output_dir / f"page_{page_idx}"
        page_out.mkdir(parents=True, exist_ok=True)

        for t_idx, table in enumerate(page.get("tables", [])):
            sidecar = page_assets / f"table_{t_idx}_grid.json"
            if not sidecar.exists():
                continue
            img = rerender_table(
                page_dict=page,
                table_dict=table,
                sidecar_path=sidecar,
                shapes_dir=shapes_dir,
            )
            out_path = page_out / f"table_{t_idx}_rendered.png"
            cv2.imwrite(str(out_path), img)
            written.append(out_path)

    return written
