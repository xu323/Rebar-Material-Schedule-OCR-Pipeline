"""Template matching shape classifier using cv2.matchTemplate.

Supports composite-shape splitting: the cell image is analysed for
connected components, and if two or more significant components are
found they are each classified independently and returned as a list
of ShapeMatch instances.  The pipeline then marks the result as
``is_composite=True``.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.models import BBox, ShapeMatch
from .base import ShapeClassifier


class TemplateMatcherClassifier(ShapeClassifier):
    """Classify rebar shapes via multi-scale template matching."""

    def __init__(
        self,
        threshold: float = 0.6,
        resize_heights: tuple[int, ...] | None = None,
        composite_min_area_ratio: float = 0.01,
        composite_min_dim: int = 8,
    ) -> None:
        self.threshold = threshold
        self.resize_heights = resize_heights or (40, 60, 80, 100)
        self.composite_min_area_ratio = composite_min_area_ratio
        self.composite_min_dim = composite_min_dim
        self.templates: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def load_templates(self, shapes_dir: Path) -> None:
        self.templates.clear()
        for p in sorted(shapes_dir.glob("shape_*.png")):
            shape_id = p.stem
            img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            _, bw = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY_INV)
            coords = cv2.findNonZero(bw)
            if coords is None:
                continue
            x, y, w, h = cv2.boundingRect(coords)
            self.templates[shape_id] = bw[y : y + h, x : x + w]

    # ------------------------------------------------------------------
    def classify(self, cell_image: np.ndarray) -> list[ShapeMatch]:
        """Match the cell image against all loaded templates.

        Returns 0, 1, or more matches. Multiple matches indicate a
        composite cell (e.g. ``shape_87`` = circle + zigzag).
        """
        if not self.templates:
            return []

        cell_bw = self._binarize(cell_image)
        if cell_bw is None:
            return []

        cell_h, cell_w = cell_bw.shape[:2]
        if cell_h < 5 or cell_w < 5:
            return []

        # --- Component-based composite detection ---
        components = self._find_components(cell_bw)

        # Always compute the full-cell match as a fallback / single-shape answer
        full_match = self._best_match_in(cell_bw)

        if len(components) >= 2:
            comp_matches: list[ShapeMatch] = []
            for comp_bbox in components:
                sub = cell_bw[
                    comp_bbox.y : comp_bbox.y + comp_bbox.h,
                    comp_bbox.x : comp_bbox.x + comp_bbox.w,
                ]
                m = self._best_match_in(sub)
                if m is not None:
                    comp_matches.append(m)

            # Only treat as composite if we got >=2 confident sub-matches AND
            # the combined sub-match score is at least as good as a single.
            if len(comp_matches) >= 2:
                sub_avg = sum(m.confidence for m in comp_matches) / len(comp_matches)
                full_score = full_match.confidence if full_match else 0.0
                if sub_avg >= full_score * 0.9:
                    return comp_matches

        return [full_match] if full_match is not None else []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _binarize(cell_image: np.ndarray) -> np.ndarray | None:
        if cell_image is None or cell_image.size == 0:
            return None
        if len(cell_image.shape) == 3:
            gray = cv2.cvtColor(cell_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_image
        _, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        return bw

    def _best_match_in(self, cell_bw: np.ndarray) -> ShapeMatch | None:
        """Run multi-scale template match and return the best hit above threshold."""
        if cell_bw is None:
            return None
        cell_h, cell_w = cell_bw.shape[:2]
        if cell_h < 5 or cell_w < 5:
            return None

        best_id: str | None = None
        best_score: float = -1.0

        for shape_id, tmpl in self.templates.items():
            t_h, t_w = tmpl.shape[:2]
            if t_h == 0:
                continue
            for target_h in self.resize_heights:
                scale = target_h / t_h
                target_w = max(int(t_w * scale), 1)
                if target_h > cell_h or target_w > cell_w:
                    continue
                resized = cv2.resize(
                    tmpl, (target_w, target_h), interpolation=cv2.INTER_AREA
                )
                result = cv2.matchTemplate(
                    cell_bw, resized, cv2.TM_CCOEFF_NORMED
                )
                _, max_val, _, _ = cv2.minMaxLoc(result)
                if max_val > best_score:
                    best_score = max_val
                    best_id = shape_id

        if best_id is not None and best_score >= self.threshold:
            return ShapeMatch(shape_id=best_id, confidence=float(best_score))
        return None

    def _find_components(self, cell_bw: np.ndarray) -> list[BBox]:
        """Return bounding boxes of the significant foreground components.

        Merges components whose bboxes overlap along the vertical axis (rebar
        shapes often break into several strokes that belong to the same symbol).
        """
        h, w = cell_bw.shape[:2]
        total_area = h * w

        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cell_bw, connectivity=8)
        raw: list[BBox] = []
        min_area = max(int(total_area * self.composite_min_area_ratio), 20)

        for i in range(1, n_labels):  # skip background
            x = stats[i, cv2.CC_STAT_LEFT]
            y = stats[i, cv2.CC_STAT_TOP]
            cw = stats[i, cv2.CC_STAT_WIDTH]
            ch = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if area < min_area:
                continue
            if cw < self.composite_min_dim or ch < self.composite_min_dim:
                continue
            raw.append(BBox(x=int(x), y=int(y), w=int(cw), h=int(ch)))

        if len(raw) < 2:
            return raw

        # Merge components whose x-ranges overlap — they likely belong to the same
        # visual shape (e.g. rebar drawn with two disconnected strokes).
        raw.sort(key=lambda b: b.x)
        merged: list[BBox] = []
        for b in raw:
            if not merged:
                merged.append(b)
                continue
            last = merged[-1]
            last_x2 = last.x + last.w
            b_x2 = b.x + b.w
            # Overlap or touch within a small gap
            gap = b.x - last_x2
            if gap <= max(w * 0.03, 4):
                new_x = min(last.x, b.x)
                new_y = min(last.y, b.y)
                new_x2 = max(last_x2, b_x2)
                new_y2 = max(last.y + last.h, b.y + b.h)
                merged[-1] = BBox(
                    x=new_x, y=new_y, w=new_x2 - new_x, h=new_y2 - new_y
                )
            else:
                merged.append(b)

        return merged
