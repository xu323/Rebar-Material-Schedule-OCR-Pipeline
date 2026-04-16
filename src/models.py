"""Data models used throughout the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class BBox:
    x: int
    y: int
    w: int
    h: int

    def to_dict(self) -> dict:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass
class LayoutRegion:
    bbox: BBox
    label: str  # "table", "text", "doc_title", "header", etc.
    confidence: float


@dataclass
class GridCell:
    bbox: BBox
    row: int
    col: int
    rowspan: int = 1
    colspan: int = 1


@dataclass
class OCRResult:
    text: str
    confidence: float


@dataclass
class ShapeMatch:
    shape_id: str  # e.g. "shape_55"
    confidence: float


@dataclass
class CellContent:
    cell_type: Literal["text", "shape", "empty"]
    text: str | None = None
    ocr_confidence: float = 0.0
    shape_id: str | None = None
    shape_confidence: float = 0.0
    is_composite: bool = False


@dataclass
class TableSemantic:
    """Optional domain-specific semantic interpretation.

    Best-effort layer that does not override the raw grid data.
    Consumers may ignore it entirely.
    """

    header_fields: dict[str, str] = field(default_factory=dict)
    shape_column_index: int | None = None
    shape_column_name: str | None = None


@dataclass
class TableRow:
    index: int
    row_type: str = "data"  # "meta", "col_header", "unit", "data", "summary", "unknown"
    cells: dict[str, dict] = field(default_factory=dict)  # column_name -> cell dict
    merge_info: dict | None = None


@dataclass
class Table:
    bbox: BBox
    columns: list[str]  # OCR'd column names (from detected header row)
    rows: list[TableRow]  # ALL rows (meta, header, unit, data, summary)
    col_header_row_index: int | None = None
    meta_fields: list[dict[str, str]] = field(default_factory=list)
    semantic: TableSemantic | None = None


@dataclass
class NonTableRegion:
    bbox: BBox
    label: str
    text: str


@dataclass
class PageResult:
    page_index: int
    tables: list[Table]
    non_table_regions: list[NonTableRegion]


@dataclass
class DocumentResult:
    source_pdf: str
    total_pages: int
    pages: list[PageResult]
    pipeline_version: str = "0.2.0"
    processing_time_seconds: float = 0.0
