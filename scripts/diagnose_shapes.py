"""Diagnostic: dump every shape-column cell's crop, is_likely_shape result,
top-3 template scores, and final decision across all pages.

Usage:
    .venv/Scripts/python scripts/diagnose_shapes.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import PipelineConfig, SHAPES_DIR
from src.models import BBox, GridCell
from src.shape_column_repair import is_likely_shape
from src.shape_classifier.template_matcher import TemplateMatcherClassifier


def load_grid_sidecar(path: Path):
    with open(path, "r", encoding="utf-8") as f:
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


def main():
    cfg = PipelineConfig()
    classifier = TemplateMatcherClassifier(
        threshold=cfg.template_match_threshold,
        resize_heights=cfg.template_resize_heights,
    )
    classifier.load_templates(cfg.shapes_dir)
    print(f"Loaded {len(classifier.templates)} templates from {cfg.shapes_dir}")
    print(f"Template IDs: {sorted(classifier.templates.keys())}")
    print(f"Threshold: {classifier.threshold}")
    print(f"Resize heights: {classifier.resize_heights}")
    print()

    # Print template dimensions for reference
    print("Template dimensions (after tight crop):")
    for sid in sorted(classifier.templates.keys()):
        t = classifier.templates[sid]
        print(f"  {sid}: {t.shape[1]}x{t.shape[0]} (WxH), fg_pixels={np.count_nonzero(t)}")
    print()

    result_path = ROOT / "output" / "result.json"
    with open(result_path, "r", encoding="utf-8") as f:
        doc = json.load(f)

    review_assets = ROOT / "output" / "review_assets"
    diag_dir = ROOT / "output" / "diag_shapes"
    diag_dir.mkdir(parents=True, exist_ok=True)

    for page in doc.get("pages", []):
        page_idx = page["page_index"]
        page_img_path = review_assets / f"page_{page_idx}" / "page.png"
        if not page_img_path.exists():
            print(f"[page {page_idx}] page image not found, skipping")
            continue
        page_img = cv2.imread(str(page_img_path))

        for t_idx, table in enumerate(page.get("tables", [])):
            sidecar_path = review_assets / f"page_{page_idx}" / f"table_{t_idx}_grid.json"
            if not sidecar_path.exists():
                print(f"[page {page_idx} table {t_idx}] no grid sidecar, skipping")
                continue

            grid_cells, table_bbox, shape_col_idx = load_grid_sidecar(sidecar_path)
            if shape_col_idx is None:
                print(f"[page {page_idx} table {t_idx}] shape_col_idx=None, skipping")
                continue

            print(f"{'='*80}")
            print(f"PAGE {page_idx}  TABLE {t_idx}  shape_col_idx={shape_col_idx}")
            print(f"table_bbox: x={table_bbox.x} y={table_bbox.y} w={table_bbox.w} h={table_bbox.h}")
            print(f"{'='*80}")

            # Get table crop
            b = table_bbox
            table_crop = page_img[b.y : b.y + b.h, b.x : b.x + b.w]

            # Build row types from result.json
            row_types = {}
            for row in table.get("rows", []):
                row_types[row["index"]] = row.get("row_type", "data")

            # Get expected shape_ids from result.json
            columns = table.get("columns", [])
            expected_shapes: dict[int, str | None] = {}
            for row in table.get("rows", []):
                ri = row["index"]
                cells = row.get("cells", {})
                # Find the shape column cell
                vals = list(cells.values())
                if shape_col_idx < len(vals):
                    cell = vals[shape_col_idx]
                    expected_shapes[ri] = cell.get("shape_id")

            # Save diag page dir
            page_diag = diag_dir / f"page_{page_idx}_table_{t_idx}"
            page_diag.mkdir(parents=True, exist_ok=True)

            # Analyze each shape-column cell
            shape_cells = [
                gc for gc in grid_cells
                if gc.col <= shape_col_idx < gc.col + gc.colspan
            ]
            shape_cells.sort(key=lambda g: g.row)

            for gc in shape_cells:
                rt = row_types.get(gc.row, "data")
                cb = gc.bbox
                cell_crop = table_crop[cb.y : cb.y + cb.h, cb.x : cb.x + cb.w]

                # Skip header/meta/unit as pipeline does
                if rt in ("meta", "col_header", "unit"):
                    print(f"  row={gc.row:3d} [{rt:10s}] SKIPPED (non-data row)")
                    continue

                # Save cell crop
                crop_path = page_diag / f"row_{gc.row:03d}_crop.png"
                cv2.imwrite(str(crop_path), cell_crop)

                # OCR data from result.json
                expected_sid = expected_shapes.get(gc.row)

                # Run is_likely_shape
                # We need OCR text for this cell - get from result.json
                ocr_text = ""
                ocr_conf = 0.0
                adj_texts = []
                for row in table.get("rows", []):
                    if row["index"] == gc.row:
                        vals = list(row.get("cells", {}).values())
                        if shape_col_idx < len(vals):
                            sc = vals[shape_col_idx]
                            ocr_text = sc.get("text", "") or ""
                            ocr_conf = sc.get("confidence", 0.0)
                        for ci, cell in enumerate(vals):
                            if ci != shape_col_idx:
                                t = cell.get("text", "") or ""
                                if t:
                                    adj_texts.append(t)
                        break

                likely = is_likely_shape(cell_crop, ocr_text, ocr_conf, adj_texts)

                # Prepare cleaned binary for analysis
                prepared = classifier._prepare_for_matching(cell_crop)
                cell_bw = prepared["primary"] if prepared is not None else None
                fg_ratio = 0.0
                if cell_bw is not None:
                    fg_ratio = np.count_nonzero(cell_bw) / max(cell_bw.size, 1)

                    # Save binarized crop
                    bw_path = page_diag / f"row_{gc.row:03d}_bw.png"
                    cv2.imwrite(str(bw_path), cell_bw)

                # Top-5 template scores
                top5 = []
                if prepared is not None:
                    top5 = classifier.top_matches(cell_crop, limit=5)

                # Classifier decision
                matches = classifier.classify(cell_crop)
                decided_id = matches[0].shape_id if matches else None
                decided_conf = matches[0].confidence if matches else 0.0

                # Print
                mark = ""
                if expected_sid and decided_id != expected_sid:
                    mark = " *** MISMATCH"
                elif not expected_sid and decided_id:
                    mark = " (unexpected match)"
                elif expected_sid and decided_id == expected_sid:
                    mark = " OK"

                print(
                    f"  row={gc.row:3d} [{rt:10s}] "
                    f"crop={cb.w}x{cb.h} "
                    f"rspan={gc.rowspan} "
                    f"fg={fg_ratio:.3f} "
                    f"likely={likely} "
                    f"ocr=\"{ocr_text[:20]}\"/{ocr_conf:.2f} "
                    f"expected={expected_sid} "
                    f"decided={decided_id}/{decided_conf:.3f}"
                    f"{mark}"
                )
                if top5:
                    scores_str = " | ".join(f"{sid}={sc:.3f}" for sid, sc in top5)
                    print(f"           top5: {scores_str}")

    print(f"\nDiagnostic crops saved to {diag_dir}")


if __name__ == "__main__":
    main()
