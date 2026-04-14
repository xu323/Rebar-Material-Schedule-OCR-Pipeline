import unittest

from src.debug_visualizer import _collect_render_title_lines
from src.models import BBox, NonTableRegion


class CollectRenderTitleLinesTests(unittest.TestCase):
    def test_keeps_figure_title_and_excludes_image_noise(self) -> None:
        regions = [
            NonTableRegion(
                bbox=BBox(x=1024, y=95, w=435, h=60),
                label="figure_title",
                text="表1鋼筋料單明細表",
            ),
            NonTableRegion(
                bbox=BBox(x=769, y=2618, w=133, h=145),
                label="image",
                text="A\nC",
            ),
        ]

        lines = _collect_render_title_lines(
            regions,
            table_bbox=BBox(x=115, y=204, w=2252, h=3160),
        )

        self.assertEqual(lines, [("figure_title", "表1鋼筋料單明細表")])

    def test_excludes_text_below_table_top(self) -> None:
        regions = [
            NonTableRegion(
                bbox=BBox(x=100, y=250, w=120, h=30),
                label="text",
                text="不應顯示",
            )
        ]

        lines = _collect_render_title_lines(
            regions,
            table_bbox=BBox(x=115, y=204, w=2252, h=3160),
        )

        self.assertEqual(lines, [])


if __name__ == "__main__":
    unittest.main()
