"""Central configuration for the OCR pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SHAPES_DIR = DATA_DIR / "shapes"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output"


@dataclass(frozen=True)
class PipelineConfig:
    # PDF rasterization
    pdf_dpi: int = 300

    # Layout analysis
    layout_score_threshold: float = 0.5

    # Table cell segmentation (OpenCV line detection)
    line_cluster_tolerance: int = 8
    morph_h_scale: int = 30  # horizontal kernel = (width // this, 1)
    morph_v_scale: int = 30  # vertical kernel   = (1, height // this)
    # Fraction of an internal border that must contain line pixels for
    # the border to be considered "present". If below this value the two
    # adjacent cells are merged (rowspan/colspan > 1).
    merge_border_coverage: float = 0.25

    # OCR
    ocr_score_threshold: float = 0.5
    ocr_padding: int = 20  # white-border padding for small cells (px)

    # Shape classifier (template matching)
    template_match_threshold: float = 0.6
    template_resize_heights: tuple[int, ...] = (40, 60, 80, 100)

    # Composite shape splitting (connected components)
    composite_min_area_ratio: float = 0.01  # comp area / cell area
    composite_min_dim: int = 8  # minimum comp bbox side in px

    # CNN shape classifier
    cnn_match_threshold: float = 0.75

    # Paths
    shapes_dir: Path = SHAPES_DIR

    # Debug
    debug: bool = False
