"""CLI entry point for the rebar material schedule OCR pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/run_pipeline.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebar Material Schedule OCR Pipeline (鋼筋施工圖料單 OCR)"
    )
    parser.add_argument("--pdf", type=Path, required=True, help="Input PDF path")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("output/result.json"),
        help="Output JSON path (default: output/result.json)",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Save debug overlay images"
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="PDF rasterization DPI (default: 300)"
    )
    parser.add_argument(
        "--shape-classifier",
        default="template",
        help="Shape classifier backend (default: template)",
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        print(f"Error: PDF not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    run_pipeline(
        pdf_path=args.pdf,
        output_path=args.out,
        debug=args.debug,
        dpi=args.dpi,
        classifier_name=args.shape_classifier,
    )
    print(f"Done. Output saved to {args.out}")


if __name__ == "__main__":
    main()
