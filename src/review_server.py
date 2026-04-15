"""Flask server for the human-in-the-loop OCR review UI.

Launch with::

    python scripts/review.py --result output/result.json

The server reads an existing pipeline JSON output, serves an editable
review UI at http://localhost:5000, and on save writes a corrected
``result_reviewed.json`` plus re-rendered table images into
``output/reviewed/``.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from src.config import SHAPES_DIR
from src.review_renderer import rerender_document


# ------------------------------------------------------------------
# App factory
# ------------------------------------------------------------------

def create_app(
    result_path: Path,
    *,
    review_assets_dir: Path | None = None,
    output_dir: Path | None = None,
    shapes_dir: Path = SHAPES_DIR,
) -> Flask:
    """Create a Flask app bound to a specific result JSON file."""
    result_path = result_path.resolve()
    base_dir = result_path.parent
    review_assets_dir = (
        review_assets_dir or base_dir / "review_assets"
    ).resolve()
    output_dir = (output_dir or base_dir / "reviewed").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    template_dir = Path(__file__).resolve().parent / "review_templates"
    static_dir = template_dir  # serve CSS/JS from same folder

    app = Flask(
        __name__,
        template_folder=str(template_dir),
        static_folder=str(static_dir),
        static_url_path="/static",
    )

    # Keep the current JSON in-memory so that edits applied through the
    # UI can be merged against the originally loaded document.
    state: dict = {
        "result_path": result_path,
        "review_assets_dir": review_assets_dir,
        "output_dir": output_dir,
        "shapes_dir": Path(shapes_dir),
        "reviewed_path": base_dir / "result_reviewed.json",
    }

    # ------------------------------------------------------------------
    # Routes
    # ------------------------------------------------------------------

    def current_document_path() -> Path:
        reviewed = state["reviewed_path"]
        original = state["result_path"]
        if not reviewed.exists():
            return original
        if not original.exists():
            return reviewed
        return reviewed if reviewed.stat().st_mtime >= original.stat().st_mtime else original

    def current_rendered_image_path(page_idx: int, table_idx: int) -> Path | None:
        """Return the rendered image that matches the current document source."""
        current_doc = current_document_path()
        reviewed_img = (
            state["output_dir"]
            / f"page_{page_idx}"
            / f"table_{table_idx}_rendered.png"
        )
        debug_img = (
            state["result_path"].parent
            / "debug"
            / f"page_{page_idx}"
            / f"03_table_{table_idx}_rendered.png"
        )

        # If the reviewed JSON is the active source, prefer the reviewed render,
        # but only if it is at least as new as the reviewed JSON it represents.
        if current_doc == state["reviewed_path"]:
            if (
                reviewed_img.exists()
                and reviewed_img.stat().st_mtime >= current_doc.stat().st_mtime
            ):
                return reviewed_img
            if debug_img.exists():
                return debug_img
            return reviewed_img if reviewed_img.exists() else None

        # If the original JSON is newer/current, ignore stale reviewed renders.
        if debug_img.exists():
            return debug_img
        return None

    @app.route("/")
    def index():
        return send_from_directory(template_dir, "index.html")

    @app.route("/api/result")
    def api_result():
        """Return the current result JSON (reviewed version if it exists)."""
        path = current_document_path()
        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)
        return jsonify({
            "source": str(path),
            "is_reviewed": path == state["reviewed_path"],
            "document": doc,
        })

    @app.route("/api/original")
    def api_original():
        """Return the un-edited original JSON (for 'reset' in the UI)."""
        with open(state["result_path"], "r", encoding="utf-8") as f:
            return jsonify(json.load(f))

    @app.route("/api/page_image/<int:page_idx>")
    def api_page_image(page_idx: int):
        """Serve the raw page PNG written by the pipeline."""
        png = state["review_assets_dir"] / f"page_{page_idx}" / "page.png"
        if not png.exists():
            return jsonify({"error": f"page image not found: {png}"}), 404
        return send_file(png, mimetype="image/png")

    @app.route("/api/rendered/<int:page_idx>/<int:table_idx>")
    def api_rendered(page_idx: int, table_idx: int):
        """Serve the rendered image matching the active document source."""
        img_path = current_rendered_image_path(page_idx, table_idx)
        if img_path and img_path.exists():
            return send_file(img_path, mimetype="image/png")

        return jsonify({"error": "rendered image not found"}), 404

    @app.route("/api/shape/<shape_id>")
    def api_shape(shape_id: str):
        path = state["shapes_dir"] / f"{shape_id}.png"
        if not path.exists():
            return jsonify({"error": "shape not found"}), 404
        return send_file(path, mimetype="image/png")

    @app.route("/api/grid/<int:page_idx>/<int:table_idx>")
    def api_grid(page_idx: int, table_idx: int):
        """Serve grid sidecar JSON for a specific table."""
        path = (
            state["review_assets_dir"]
            / f"page_{page_idx}"
            / f"table_{table_idx}_grid.json"
        )
        if not path.exists():
            return jsonify({"error": "grid sidecar not found"}), 404
        return send_file(path, mimetype="application/json")

    @app.route("/api/save", methods=["POST"])
    def api_save():
        """Accept edited cells, write reviewed JSON, and re-render."""
        payload = request.get_json(force=True, silent=True) or {}
        edits = payload.get("edits", [])

        # Load the most recent document (reviewed if it exists)
        source_path = current_document_path()
        with open(source_path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        applied = _apply_edits(doc, edits)

        # Write the reviewed JSON alongside the original
        with open(state["reviewed_path"], "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=2)

        # Re-render every table from the corrected JSON
        rendered_paths: list[Path] = []
        try:
            rendered_paths = rerender_document(
                reviewed_json_path=state["reviewed_path"],
                review_assets_dir=state["review_assets_dir"],
                shapes_dir=state["shapes_dir"],
                output_dir=state["output_dir"],
            )
        except Exception as exc:  # noqa: BLE001
            return jsonify({
                "ok": False,
                "applied_edits": applied,
                "reviewed_path": str(state["reviewed_path"]),
                "error": f"re-render failed: {exc}",
            }), 500

        return jsonify({
            "ok": True,
            "applied_edits": applied,
            "reviewed_path": str(state["reviewed_path"]),
            "rendered": [str(p) for p in rendered_paths],
        })

    return app


# ------------------------------------------------------------------
# Edit application
# ------------------------------------------------------------------

def _resolve_cell_by_column_index(
    table: dict,
    row: dict,
    column_index: int,
) -> dict | None:
    """Resolve a row cell using the table's canonical column order."""
    columns = table.get("columns", []) or []
    cells_dict = row.get("cells", {}) or {}
    if 0 <= column_index < len(columns):
        cell = cells_dict.get(columns[column_index])
        if cell is not None:
            return cell

    # Fallback for legacy / malformed rows where columns and cells drifted.
    keys = list(cells_dict.keys())
    if 0 <= column_index < len(keys):
        return cells_dict[keys[column_index]]
    return None


def _apply_edits(doc: dict, edits: list[dict]) -> int:
    """Apply UI edits to an in-memory document.

    Each edit is a dict of the form::

        {
          "page_index": 0,
          "table_index": 0,
          "row_index": 5,
          "column": "號數",
          "text": "#5"
        }

    Returns the number of edits that were successfully applied.
    """
    applied = 0
    for edit in edits:
        try:
            page = doc["pages"][int(edit["page_index"])]
        except (KeyError, ValueError, IndexError, TypeError):
            continue

        if "non_table_region_index" in edit:
            try:
                region_idx = int(edit["non_table_region_index"])
                regions = page.get("non_table_regions", []) or []
                region = regions[region_idx]
            except (ValueError, IndexError, TypeError):
                continue

            new_text = edit.get("text")
            if new_text is None:
                continue

            region["text"] = new_text
            region["human_reviewed"] = True
            region["needs_review"] = False
            if "confidence" in region:
                region["confidence"] = 1.0
            applied += 1
            continue

        try:
            table = page["tables"][int(edit["table_index"])]
            rows = table.get("rows", [])
            row = next(
                (r for r in rows if r.get("index") == int(edit["row_index"])),
                None,
            )
            if row is None:
                continue

            cell = None

            # Primary: canonical table column order
            if "column_index" in edit:
                ci = int(edit["column_index"])
                cell = _resolve_cell_by_column_index(table, row, ci)

            # Fallback: column name (backward compat)
            if cell is None and "column" in edit:
                cell = (row.get("cells", {}) or {}).get(edit["column"])

            if cell is None:
                continue
        except (KeyError, ValueError, IndexError, TypeError):
            continue

        new_text = edit.get("text")
        if new_text is not None:
            cell["text"] = new_text
            # Human confirmation overrides model confidence
            cell["confidence"] = 1.0
            cell["needs_review"] = False
            cell["human_reviewed"] = True

        new_shape = edit.get("shape_id")
        if new_shape is not None:
            cell["shape_id"] = new_shape or None
            cell["confidence"] = 1.0
            cell["needs_review"] = False
            cell["human_reviewed"] = True

        applied += 1
    return applied
