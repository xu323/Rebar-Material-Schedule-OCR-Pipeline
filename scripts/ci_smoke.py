"""CI smoke checks for pipeline output and review UI endpoints."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.review_server import create_app


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def validate_result_json(result_path: Path) -> dict:
    _require(result_path.exists(), f"missing result JSON: {result_path}")
    with result_path.open("r", encoding="utf-8") as f:
        doc = json.load(f)

    _require(isinstance(doc, dict), "result JSON must be an object")
    _require("pages" in doc and isinstance(doc["pages"], list), "missing pages list")
    _require(
        int(doc.get("total_pages", len(doc["pages"]))) == len(doc["pages"]),
        "total_pages does not match pages length",
    )
    _require(doc["pages"], "document must contain at least one page")

    saw_table = False
    for page_idx, page in enumerate(doc["pages"]):
        _require(
            int(page.get("page_index", -1)) == page_idx,
            f"page_index mismatch on page {page_idx}",
        )
        _require(isinstance(page.get("tables", []), list), "tables must be a list")
        _require(
            isinstance(page.get("non_table_regions", []), list),
            "non_table_regions must be a list",
        )
        for table_idx, table in enumerate(page.get("tables", [])):
            saw_table = True
            _require("bbox" in table, f"missing table bbox on page {page_idx}/{table_idx}")
            _require(
                isinstance(table.get("columns", []), list),
                f"table columns must be a list on page {page_idx}/{table_idx}",
            )
            _require(
                isinstance(table.get("rows", []), list),
                f"table rows must be a list on page {page_idx}/{table_idx}",
            )
    _require(saw_table, "document must contain at least one table")
    return doc


def validate_debug_artifacts(result_path: Path, doc: dict) -> None:
    base_dir = result_path.parent
    debug_dir = base_dir / "debug"
    review_assets_dir = base_dir / "review_assets"

    _require(debug_dir.exists(), f"missing debug directory: {debug_dir}")
    _require(review_assets_dir.exists(), f"missing review_assets directory: {review_assets_dir}")

    for page in doc["pages"]:
        page_idx = int(page["page_index"])
        page_debug_dir = debug_dir / f"page_{page_idx}"
        page_assets_dir = review_assets_dir / f"page_{page_idx}"
        _require(page_debug_dir.exists(), f"missing page debug dir: {page_debug_dir}")
        _require(page_assets_dir.exists(), f"missing page assets dir: {page_assets_dir}")
        _require(
            (page_debug_dir / "01_layout.png").exists(),
            f"missing layout debug image for page {page_idx}",
        )
        _require(
            (page_assets_dir / "page.png").exists(),
            f"missing page image asset for page {page_idx}",
        )

        for table_idx, _table in enumerate(page.get("tables", [])):
            _require(
                (page_debug_dir / f"03_table_{table_idx}_rendered.png").exists(),
                f"missing rendered debug image for page {page_idx}/{table_idx}",
            )
            _require(
                (page_assets_dir / f"table_{table_idx}_grid.json").exists(),
                f"missing grid sidecar for page {page_idx}/{table_idx}",
            )


def validate_review_api(result_path: Path) -> None:
    app = create_app(result_path=result_path)
    client = app.test_client()

    response = client.get("/")
    _require(response.status_code == 200, "GET / must return 200")

    response = client.get("/api/result")
    _require(response.status_code == 200, "GET /api/result must return 200")
    payload = response.get_json()
    _require(isinstance(payload, dict), "/api/result must return JSON object")
    _require("document" in payload, "/api/result missing document payload")

    doc = payload["document"]
    first_page_idx = int(doc["pages"][0]["page_index"])
    response = client.get(f"/api/page_image/{first_page_idx}")
    _require(response.status_code == 200, "GET /api/page_image/<page> must return 200")

    if doc["pages"][0].get("tables"):
        response = client.get(f"/api/grid/{first_page_idx}/0")
        _require(response.status_code == 200, "GET /api/grid/<page>/<table> must return 200")
        response = client.get(f"/api/rendered/{first_page_idx}/0")
        _require(
            response.status_code == 200,
            "GET /api/rendered/<page>/<table> must return 200",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CI smoke checks for TSG OCR")
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("output/result.json"),
        help="Pipeline output JSON to validate",
    )
    parser.add_argument(
        "--check-review-api",
        action="store_true",
        help="Also validate review server endpoints using Flask test_client",
    )
    args = parser.parse_args()

    doc = validate_result_json(args.result)
    validate_debug_artifacts(args.result, doc)
    if args.check_review_api:
        validate_review_api(args.result)
    print("CI smoke checks passed.")


if __name__ == "__main__":
    main()
