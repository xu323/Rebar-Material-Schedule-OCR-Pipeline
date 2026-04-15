"""CLI launcher for the human-in-the-loop review server.

Usage::

    python scripts/review.py --result output/result.json

Then open http://localhost:5000 in a browser.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as `python scripts/review.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import SHAPES_DIR
from src.review_server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the TSG OCR human-in-the-loop review UI"
    )
    parser.add_argument(
        "--result",
        type=Path,
        default=Path("output/result.json"),
        help="Path to the pipeline JSON output (default: output/result.json)",
    )
    parser.add_argument(
        "--shapes-dir",
        type=Path,
        default=SHAPES_DIR,
        help="Directory containing shape template PNGs",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if not args.result.exists():
        print(f"Error: result JSON not found: {args.result}", file=sys.stderr)
        print(
            "Run `python scripts/run_pipeline.py --pdf <pdf> --debug` first.",
            file=sys.stderr,
        )
        sys.exit(1)

    review_assets = args.result.parent / "review_assets"
    if not review_assets.exists():
        print(
            f"Warning: review assets not found at {review_assets}.\n"
            "Re-run the pipeline to regenerate them:\n"
            "    python scripts/run_pipeline.py --pdf <pdf> --debug",
            file=sys.stderr,
        )

    app = create_app(
        result_path=args.result,
        shapes_dir=args.shapes_dir,
    )

    print(f"Review UI: http://{args.host}:{args.port}")
    print(f"Reading:   {args.result.resolve()}")
    print(
        "Reviewed JSON will be written to "
        f"{(args.result.parent / 'result_reviewed.json').resolve()}"
    )
    print(
        "Re-rendered images: "
        f"{(args.result.parent / 'reviewed').resolve()}"
    )
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
