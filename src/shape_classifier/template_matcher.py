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
        self.resize_heights = resize_heights or (20, 30, 40, 60, 80, 100, 140)
        self.composite_min_area_ratio = composite_min_area_ratio
        self.composite_min_dim = composite_min_dim
        self.templates: dict[str, np.ndarray] = {}
        self.template_canvases: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    def load_templates(self, shapes_dir: Path) -> None:
        self.templates.clear()
        self.template_canvases.clear()
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
            tight = bw[y : y + h, x : x + w]
            self.templates[shape_id] = tight
            self.template_canvases[shape_id] = self._normalize_to_canvas(tight)

    # ------------------------------------------------------------------
    def classify(self, cell_image: np.ndarray) -> list[ShapeMatch]:
        """Match the cell image against all loaded templates.

        Returns 0, 1, or more matches. Multiple matches indicate a
        composite cell (e.g. ``shape_87`` = circle + zigzag).
        """
        if not self.templates:
            return []

        prepared = self._prepare_for_matching(cell_image)
        if prepared is None:
            return []

        primary = prepared["primary"]
        regions = prepared["regions"]

        cell_h, cell_w = primary.shape[:2]
        if cell_h < 5 or cell_w < 5:
            return []

        # --- Component-based composite detection ---
        components = self._find_components(primary)

        # Always compute the full-cell match as a fallback / single-shape answer
        full_match = self._best_match_in_regions(regions)

        if len(components) >= 2:
            comp_matches: list[ShapeMatch] = []
            for comp_bbox in components:
                sub = primary[
                    comp_bbox.y : comp_bbox.y + comp_bbox.h,
                    comp_bbox.x : comp_bbox.x + comp_bbox.w,
                ]
                m = self._best_match_in_regions(self._regions_from_binary(sub))
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

    def top_matches(
        self,
        cell_image: np.ndarray,
        *,
        limit: int = 5,
    ) -> list[tuple[str, float]]:
        """Return the best-scoring template ids for a cell image."""
        prepared = self._prepare_for_matching(cell_image)
        if prepared is None:
            return []
        scores = self._score_templates(prepared["regions"])
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:limit]

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
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, bw = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return bw

    def _prepare_for_matching(
        self,
        cell_image: np.ndarray,
    ) -> dict[str, np.ndarray | list[np.ndarray]] | None:
        cell_bw = self._binarize(cell_image)
        if cell_bw is None:
            return None

        cleaned = self._strip_cell_borders(cell_bw)
        regions = self._regions_from_binary(cleaned)
        if not regions:
            return None
        return {"primary": regions[0], "regions": regions}

    def _regions_from_binary(self, cell_bw: np.ndarray) -> list[np.ndarray]:
        tight = self._tight_foreground_crop(cell_bw)
        regions: list[np.ndarray] = []
        if tight is not None:
            regions.append(tight)
        if (
            cell_bw is not None
            and cell_bw.size > 0
            and (not regions or cell_bw.shape != regions[0].shape)
        ):
            regions.append(cell_bw)
        return regions

    def _best_match_in_regions(
        self,
        regions: list[np.ndarray],
    ) -> ShapeMatch | None:
        """Run template match over prepared regions and return the best hit."""
        scores = self._score_templates(regions)
        if not scores:
            return None
        best_id, best_score = max(scores.items(), key=lambda item: item[1])
        if best_score >= self.threshold:
            return ShapeMatch(shape_id=best_id, confidence=float(best_score))
        return None

    def _score_templates(
        self,
        regions: list[np.ndarray],
    ) -> dict[str, float]:
        if not regions:
            return {}

        valid_regions = [
            region
            for region in regions
            if region is not None and region.size > 0
        ]
        if not valid_regions:
            return {}

        scores: dict[str, float] = {}
        for shape_id, tmpl in self.templates.items():
            best_corr = -1.0
            best_overlap = -1.0
            best_canvas_corr = -1.0
            best_projection = -1.0
            t_h, t_w = tmpl.shape[:2]
            if t_h == 0 or t_w == 0:
                continue
            for region in valid_regions:
                r_h, r_w = region.shape[:2]
                if r_h < 5 or r_w < 5:
                    continue
                template_canvas = self.template_canvases.get(shape_id)
                canvas_corr = self._canvas_correlation(region, template_canvas)
                if canvas_corr > best_canvas_corr:
                    best_canvas_corr = canvas_corr
                overlap = self._overlap_score(
                    region,
                    template_canvas,
                )
                if overlap > best_overlap:
                    best_overlap = overlap
                projection = self._projection_similarity(region, template_canvas)
                if projection > best_projection:
                    best_projection = projection
                for target_h in self._candidate_heights(r_h):
                    scale = target_h / t_h
                    target_w = max(int(round(t_w * scale)), 1)
                    if target_h > r_h or target_w > r_w:
                        continue
                    resized = cv2.resize(
                        tmpl,
                        (target_w, target_h),
                        interpolation=cv2.INTER_AREA,
                    )
                    result = cv2.matchTemplate(
                        region, resized, cv2.TM_CCOEFF_NORMED
                    )
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    if max_val > best_corr:
                        best_corr = max_val
            combined = self._combine_scores(
                corr=best_corr,
                canvas_corr=best_canvas_corr,
                overlap=best_overlap,
                projection=best_projection,
            )
            if combined >= 0.0:
                scores[shape_id] = combined
        return scores

    def _candidate_heights(self, region_h: int) -> tuple[int, ...]:
        dynamic = {
            max(int(round(region_h * frac)), 12)
            for frac in (0.35, 0.5, 0.65, 0.8, 0.95)
        }
        return tuple(
            h for h in sorted(set(self.resize_heights).union(dynamic))
            if 10 <= h <= region_h
        )

    @staticmethod
    def _strip_cell_borders(cell_bw: np.ndarray) -> np.ndarray:
        cleaned = cell_bw.copy()
        h, w = cleaned.shape[:2]
        edge_band_x = max(int(round(w * 0.05)), 4)
        edge_band_y = max(int(round(h * 0.05)), 4)
        hard_trim_x = min(max(int(round(w * 0.015)), 2), w)
        hard_trim_y = min(max(int(round(h * 0.015)), 2), h)

        cleaned[:, :hard_trim_x] = 0
        cleaned[:, max(w - hard_trim_x, 0):] = 0
        cleaned[:hard_trim_y, :] = 0
        cleaned[max(h - hard_trim_y, 0):, :] = 0

        for x in range(min(edge_band_x, w)):
            if np.count_nonzero(cleaned[:, x]) >= h * 0.4:
                cleaned[:, x] = 0
        for x in range(max(w - edge_band_x, 0), w):
            if np.count_nonzero(cleaned[:, x]) >= h * 0.4:
                cleaned[:, x] = 0
        for y in range(min(edge_band_y, h)):
            if np.count_nonzero(cleaned[y, :]) >= w * 0.4:
                cleaned[y, :] = 0
        for y in range(max(h - edge_band_y, 0), h):
            if np.count_nonzero(cleaned[y, :]) >= w * 0.4:
                cleaned[y, :] = 0
        return cleaned

    @staticmethod
    def _tight_foreground_crop(cell_bw: np.ndarray) -> np.ndarray | None:
        if cell_bw is None or cell_bw.size == 0:
            return None
        h, w = cell_bw.shape[:2]
        total_area = h * w
        min_area = max(int(total_area * 0.001), 12)

        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            cell_bw, connectivity=8
        )
        boxes: list[tuple[int, int, int, int]] = []
        for i in range(1, n_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[i, cv2.CC_STAT_LEFT])
            y = int(stats[i, cv2.CC_STAT_TOP])
            cw = int(stats[i, cv2.CC_STAT_WIDTH])
            ch = int(stats[i, cv2.CC_STAT_HEIGHT])
            boxes.append((x, y, x + cw, y + ch))

        if not boxes:
            coords = cv2.findNonZero(cell_bw)
            if coords is None:
                return None
            x, y, w, h = cv2.boundingRect(coords)
            boxes = [(x, y, x + w, y + h)]

        x = min(box[0] for box in boxes)
        y = min(box[1] for box in boxes)
        x2 = max(box[2] for box in boxes)
        y2 = max(box[3] for box in boxes)

        if x2 <= x or y2 <= y:
            return None
        pad = max(int(round(min(cell_bw.shape[:2]) * 0.02)), 2)
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x2 + pad, cell_bw.shape[1])
        y1 = min(y2 + pad, cell_bw.shape[0])
        return cell_bw[y0:y1, x0:x1]

    @staticmethod
    def _normalize_to_canvas(
        cell_bw: np.ndarray,
        *,
        canvas_size: tuple[int, int] = (224, 224),
        pad_ratio: float = 0.12,
    ) -> np.ndarray:
        tight = TemplateMatcherClassifier._tight_foreground_crop(cell_bw)
        canvas_h, canvas_w = canvas_size
        canvas = np.zeros((canvas_h, canvas_w), dtype=np.uint8)
        if tight is None or tight.size == 0:
            return canvas

        src_h, src_w = tight.shape[:2]
        usable_h = max(int(round(canvas_h * (1.0 - 2.0 * pad_ratio))), 8)
        usable_w = max(int(round(canvas_w * (1.0 - 2.0 * pad_ratio))), 8)
        scale = min(usable_h / src_h, usable_w / src_w)
        dst_w = max(int(round(src_w * scale)), 1)
        dst_h = max(int(round(src_h * scale)), 1)
        resized = cv2.resize(
            tight,
            (dst_w, dst_h),
            interpolation=cv2.INTER_NEAREST,
        )
        x0 = (canvas_w - dst_w) // 2
        y0 = (canvas_h - dst_h) // 2
        canvas[y0:y0 + dst_h, x0:x0 + dst_w] = resized
        return canvas

    @staticmethod
    def _overlap_score(
        region: np.ndarray,
        template_canvas: np.ndarray | None,
    ) -> float:
        if template_canvas is None:
            return -1.0
        region_canvas = TemplateMatcherClassifier._normalize_to_canvas(
            region,
            canvas_size=template_canvas.shape[:2],
        )
        region_fg = region_canvas > 0
        template_fg = template_canvas > 0
        denom = int(region_fg.sum() + template_fg.sum())
        if denom == 0:
            return -1.0
        intersection = int(np.logical_and(region_fg, template_fg).sum())
        return (2.0 * intersection) / denom

    @staticmethod
    def _canvas_correlation(
        region: np.ndarray,
        template_canvas: np.ndarray | None,
    ) -> float:
        if template_canvas is None:
            return -1.0
        region_canvas = TemplateMatcherClassifier._normalize_to_canvas(
            region,
            canvas_size=template_canvas.shape[:2],
        )
        result = cv2.matchTemplate(
            region_canvas,
            template_canvas,
            cv2.TM_CCOEFF_NORMED,
        )
        _, max_val, _, _ = cv2.minMaxLoc(result)
        return float(max_val)

    @staticmethod
    def _projection_similarity(
        region: np.ndarray,
        template_canvas: np.ndarray | None,
    ) -> float:
        if template_canvas is None:
            return -1.0
        region_canvas = TemplateMatcherClassifier._normalize_to_canvas(
            region,
            canvas_size=template_canvas.shape[:2],
        )
        region_fg = region_canvas > 0
        template_fg = template_canvas > 0

        row_score = TemplateMatcherClassifier._cosine_similarity(
            region_fg.sum(axis=1).astype(np.float32),
            template_fg.sum(axis=1).astype(np.float32),
        )
        col_score = TemplateMatcherClassifier._cosine_similarity(
            region_fg.sum(axis=0).astype(np.float32),
            template_fg.sum(axis=0).astype(np.float32),
        )
        return (row_score + col_score) / 2.0

    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0.0:
            return 0.0
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _combine_scores(
        *,
        corr: float,
        canvas_corr: float,
        overlap: float,
        projection: float,
    ) -> float:
        global_weighted: list[tuple[float, float]] = []
        if canvas_corr >= 0.0:
            global_weighted.append((0.45, canvas_corr))
        if overlap >= 0.0:
            global_weighted.append((0.35, overlap))
        if projection >= 0.0:
            global_weighted.append((0.20, projection))
        if not global_weighted and corr < 0.0:
            return -1.0

        global_score = -1.0
        if global_weighted:
            total_weight = sum(weight for weight, _ in global_weighted)
            global_score = (
                sum(weight * value for weight, value in global_weighted)
                / total_weight
            )

        # Raw template correlation is useful, but it is only a local match:
        # for similar symbols it can over-score a partial hit. Clamp it by the
        # global shape fit so that a local fragment cannot overturn a stronger
        # full-shape match.
        if corr >= 0.0 and global_score >= 0.0:
            corr_support = min(corr, global_score)
            return (global_score * 0.8) + (corr_support * 0.2)
        if global_score >= 0.0:
            return global_score
        return corr

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
