import unittest

from src.debug_visualizer import (
    _find_nonempty_row_range,
    _fit_text_block_to_cell,
    _load_cjk_font,
    _split_explicit_lines,
)
from src.models import BBox, GridCell


class TextLayoutTests(unittest.TestCase):
    def test_fit_text_block_shrinks_and_wraps_to_cell(self) -> None:
        font = _load_cjk_font(20)
        fitted_font, lines, line_h, spacing = _fit_text_block_to_cell(
            text="CG1、G1、G2\n下部箍筋",
            base_font=font,
            max_w=90,
            max_h=60,
        )

        total_h = len(lines) * line_h + max(0, len(lines) - 1) * spacing
        self.assertLessEqual(total_h, 60)
        self.assertGreaterEqual(len(lines), 2)
        self.assertLessEqual(getattr(fitted_font, "size", 20), 20)

    def test_single_line_mode_preserves_only_explicit_newlines(self) -> None:
        lines = _split_explicit_lines("上層主筋")
        self.assertEqual(lines, ["上層主筋"])

        lines = _split_explicit_lines("CG1、G1、G2\n下部箍筋")
        self.assertEqual(lines, ["CG1、G1、G2", "下部箍筋"])

    def test_find_nonempty_row_range_trims_leading_blank_rows(self) -> None:
        cell_lookup = {
            (0, 0): GridCell(bbox=BBox(0, 0, 100, 20), row=0, col=0),
            (1, 0): GridCell(bbox=BBox(0, 20, 100, 30), row=1, col=0),
        }
        cell_contents = {
            (0, 0): {"text": ""},
            (1, 0): {"text": "鋼筋施工圖料單明細表"},
        }

        start, end = _find_nonempty_row_range(2, 1, cell_lookup, cell_contents)
        self.assertEqual((start, end), (1, 1))


if __name__ == "__main__":
    unittest.main()
