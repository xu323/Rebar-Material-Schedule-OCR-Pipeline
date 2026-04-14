# Rebar Material Schedule OCR Pipeline / 鋼筋施工圖料單 OCR

Convert rebar construction drawing material schedule PDFs into structured JSON.

將鋼筋施工圖料單明細表 PDF 轉換為結構化 JSON。

## Features / 功能

- PDF rasterization via PyMuPDF / 使用 PyMuPDF 光柵化 PDF
- Layout detection via PP-DocLayoutV3 / PP-DocLayoutV3 版面偵測
- Table cell segmentation via OpenCV line detection / OpenCV 線條偵測切割表格儲存格
- Text recognition via PaddleOCR PP-OCRv5 / PaddleOCR PP-OCRv5 文字辨識
- Rebar shape classification via template matching (27 BS 4466 shapes) / 模板比對鋼筋形狀分類（27 種 BS 4466 形狀）
- Debug overlay images / Debug 視覺化疊圖
- Pluggable shape classifier interface (future CNN/YOLO) / 可替換的形狀分類器介面

## Setup / 安裝

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Usage / 使用

```bash
python scripts/run_pipeline.py --pdf data/鋼筋施工圖料單明細表單.pdf --out output/result.json

# With debug overlay images / 含 debug 疊圖
python scripts/run_pipeline.py --pdf data/鋼筋施工圖料單明細表單.pdf --out output/result.json --debug
```

### CLI Options / CLI 參數

| Flag | Description | Default |
|------|-------------|---------|
| `--pdf` | Input PDF path / 輸入 PDF 路徑 | (required) |
| `--out` | Output JSON path / 輸出 JSON 路徑 | `output/result.json` |
| `--debug` | Save debug overlay images / 儲存 debug 疊圖 | off |
| `--dpi` | PDF rasterization DPI | 300 |
| `--shape-classifier` | Shape classifier backend / 形狀分類器 | `template` |

## Pipeline / 流程

```
PDF → Rasterize (PyMuPDF)
    → Layout Analysis (PP-DocLayoutV3)
    → Table Region:
        → Line Detection → Cell Grid (OpenCV)
        → Text Cells → PaddleOCR
        → Shape Cells → Template Matching
    → Non-Table Region → PaddleOCR
    → Assemble JSON
```

## Project Structure / 專案結構

```
TSG_OCR/
├── data/
│   ├── shapes/          # 27 BS 4466 shape templates
│   └── *.pdf            # Input PDFs
├── src/
│   ├── config.py        # Configuration
│   ├── models.py        # Data models
│   ├── pipeline.py      # Orchestrator
│   ├── pdf_rasterizer.py
│   ├── layout_analyzer.py
│   ├── table_cell_segmenter.py
│   ├── ocr_engine.py
│   ├── shape_classifier/
│   │   ├── base.py      # ABC interface
│   │   ├── template_matcher.py
│   │   └── registry.py
│   ├── assembler.py
│   └── debug_visualizer.py
├── scripts/
│   └── run_pipeline.py  # CLI entry point
└── output/              # Generated results
```

## TODO

- [ ] Merged cell detection (rowspan/colspan)
- [ ] Composite shape splitting
- [ ] CNN/YOLO shape classifier backend
- [ ] Cross-row merged shape cells
