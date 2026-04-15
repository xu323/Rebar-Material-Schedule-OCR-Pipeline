# OCR

English | [中文](#中文說明)

This OCR converts rebar material schedule PDFs into structured JSON, classifies
rebar shapes, generates debug renders, and provides a browser-based review UI.
The project now includes two independent shape-classification paths:

- `template`: the original template-matching backend
- `cnn`: a trainable supervised CNN backend for CPU training/inference
- `embed`: the older experimental embedding-based backend

`template` remains the default and is not modified by the CNN workflow.

## Features

- PDF rasterization with PyMuPDF
- Layout detection for table and non-table regions
- Table grid extraction with merged-cell support
- OCR for table cells and non-table regions
- Template-matching shape classification
- Supervised CNN shape classification with PyTorch checkpoints
- Faithful table re-rendering and review sidecars
- Browser review UI for editing OCR text, shape IDs, and non-table OCR regions
- Debug outputs for layout, grid, render, and shape diagnostics

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Optional CNN dependencies:

```bash
pip install torch torchvision
```

## Quick Start

Run the default template-based pipeline:

```bash
python scripts/run_pipeline.py --pdf data/鋼筋施工圖料單明細表單.pdf --out output/result.json --debug
```

Launch the review UI:

```bash
python scripts/review.py --result output/result.json
```

Open:

```text
http://127.0.0.1:5000
```

## CLI

### OCR Pipeline

```bash
python scripts/run_pipeline.py --pdf <input.pdf> [--out output/result.json] [--debug] [--dpi 300] [--shape-classifier template]
```

Arguments:

- `--pdf`: input PDF path
- `--out`: output JSON path, default `output/result.json`
- `--debug`: save debug images under `output/debug/`
- `--dpi`: rasterization DPI, default `300`
- `--shape-classifier`: `template`, `cnn`, or `embed`
- `--cnn-checkpoint`: required when `--shape-classifier cnn`

### Review UI

```bash
python scripts/review.py --result output/result.json
```

### Shape Diagnostics

```bash
python scripts/diagnose_shapes.py
```

## CNN Workflow

The supervised CNN backend is intentionally isolated from the template backend.
It uses its own dataset builder, checkpoint format, and inference path.

### 1. Build a synthetic dataset from templates

```bash
python scripts/build_shape_dataset.py --out-dir artifacts/shape_cnn/datasets/generated
```

### 2. Train on CPU

```bash
python scripts/train_shape_cnn.py ^
  --manifest artifacts/shape_cnn/datasets/generated/manifest.json ^
  --out-checkpoint artifacts/shape_cnn/checkpoints/shape_cnn.pt ^
  --device cpu
```

### 3. Evaluate

```bash
python scripts/eval_shape_cnn.py ^
  --manifest artifacts/shape_cnn/datasets/generated/manifest.json ^
  --checkpoint artifacts/shape_cnn/checkpoints/shape_cnn.pt ^
  --split test
```

### 4. Run OCR pipeline with CNN backend

```bash
python scripts/run_pipeline.py ^
  --pdf data/鋼筋施工圖料單明細表單.pdf ^
  --out output/result.json ^
  --debug ^
  --shape-classifier cnn ^
  --cnn-checkpoint artifacts/shape_cnn/checkpoints/shape_cnn.pt

```

## Project Structure

```text
TSG_OCR/
├─ data/
│  ├─ shapes/
│  └─ *.pdf
├─ scripts/
│  ├─ build_shape_dataset.py
│  ├─ diagnose_shapes.py
│  ├─ eval_shape_cnn.py
│  ├─ review.py
│  ├─ run_pipeline.py
│  └─ train_shape_cnn.py
├─ src/
│  ├─ pipeline.py
│  ├─ review_server.py
│  ├─ review_renderer.py
│  ├─ shape_column_repair.py
│  └─ shape_classifier/
│     ├─ base.py
│     ├─ cnn_classifier.py
│     ├─ cnn_dataset.py
│     ├─ cnn_dataset_builder.py
│     ├─ cnn_infer.py
│     ├─ cnn_model.py
│     ├─ cnn_train.py
│     ├─ cnn_transforms.py
│     ├─ embed_classifier.py
│     ├─ registry.py
│     └─ template_matcher.py
├─ tests/
├─ output/
└─ artifacts/shape_cnn/         # generated at runtime, ignored by git
```

## Output Files

Main outputs:

- `output/result.json`
- `output/debug/page_<i>/01_layout.png`
- `output/debug/page_<i>/02_table_<j>_grid.png`
- `output/debug/page_<i>/03_table_<j>_rendered.png`

Review outputs:

- `output/review_assets/page_<i>/page.png`
- `output/review_assets/page_<i>/table_<j>_grid.json`
- `output/result_reviewed.json`
- `output/reviewed/page_<i>/table_<j>_rendered.png`

CNN artifacts:

- `artifacts/shape_cnn/datasets/generated/manifest.json`
- `artifacts/shape_cnn/checkpoints/*.pt`
- `artifacts/shape_cnn/checkpoints/*.metrics.json`

## Notes

- `template` remains the default backend and is intentionally isolated from `cnn`
- `cnn` v1 is a single-shape classifier and does not auto-fallback to template
- `embed` is preserved as the older experimental nearest-neighbor backend
- CNN checkpoints store `class_to_idx`, `idx_to_class`, `num_classes`,
  preprocess config, and training metadata for future expansion beyond 27 classes

## 中文說明

這是一個完整的鋼筋料單 OCR 專案，能把 PDF 圖紙轉成結構化 JSON，
辨識鋼筋 shape，輸出 debug 圖，並提供瀏覽器人工複核介面。

目前 shape classifier 有三條路徑，而且彼此獨立：

- `template`：原本的 template matching 流程
- `cnn`：新增的可訓練 supervised CNN 流程
- `embed`：舊的實驗性 embedding 最近鄰流程

其中 `template` 仍然是預設值，CNN 的導入不會改壞原本 template 的行為。

## 功能

- 使用 PyMuPDF 將 PDF 轉成影像
- 版面分析，分出表格區與非表格區
- 表格切格與 merged-cell 保留
- 表格文字與非表格 OCR
- Template matching shape 辨識
- 可訓練的 CNN shape 辨識
- review sidecar 與忠實重繪表格
- 人工複核 UI，可編輯文字、shape ID、非表格 OCR 區塊
- debug 輸出與 shape 診斷工具

## 安裝

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

若要使用 CNN / embed backend，請另外安裝：

```bash
pip install torch torchvision
```

## 基本使用

先跑預設 template pipeline：

```bash
python scripts/run_pipeline.py --pdf data/鋼筋施工圖料單明細表單.pdf --out output/result.json --debug
```

再啟動人工複核 UI：

```bash
python scripts/review.py --result output/result.json
```

瀏覽器開啟：

```text
http://127.0.0.1:5000
```

## 指令

### OCR Pipeline

```bash
python scripts/run_pipeline.py --pdf <輸入.pdf> [--out output/result.json] [--debug] [--dpi 300] [--shape-classifier template]
```

參數：

- `--pdf`：輸入 PDF 路徑
- `--out`：輸出 JSON，預設 `output/result.json`
- `--debug`：輸出 debug 圖
- `--dpi`：PDF rasterize DPI，預設 `300`
- `--shape-classifier`：`template`、`cnn`、`embed`
- `--cnn-checkpoint`：當 `--shape-classifier cnn` 時必填

### Review UI

```bash
python scripts/review.py --result output/result.json
```

### Shape 診斷

```bash
python scripts/diagnose_shapes.py
```

## CNN 使用流程

CNN 流程與 template 流程完全並存，資料集、checkpoint、推論邏輯都分開。

### 1. 由模板自動建立合成資料集

```bash
python scripts/build_shape_dataset.py --out-dir artifacts/shape_cnn/datasets/generated
```

### 2. 用 CPU 訓練

```bash
python scripts/train_shape_cnn.py ^
  --manifest artifacts/shape_cnn/datasets/generated/manifest.json ^
  --out-checkpoint artifacts/shape_cnn/checkpoints/shape_cnn.pt ^
  --device cpu
```

### 3. 評估

```bash
python scripts/eval_shape_cnn.py ^
  --manifest artifacts/shape_cnn/datasets/generated/manifest.json ^
  --checkpoint artifacts/shape_cnn/checkpoints/shape_cnn.pt ^
  --split test
```

### 4. 用 CNN backend 跑 OCR pipeline

```bash
python scripts/run_pipeline.py ^
  --pdf data/鋼筋施工圖料單明細表單.pdf ^
  --out output/result.json ^
  --debug ^
  --shape-classifier cnn ^
  --cnn-checkpoint artifacts/shape_cnn/checkpoints/shape_cnn.pt
```

## 專案結構

```text
TSG_OCR/
├─ data/
├─ scripts/
├─ src/
│  └─ shape_classifier/
│     ├─ template_matcher.py
│     ├─ cnn_classifier.py
│     ├─ embed_classifier.py
│     ├─ cnn_dataset.py
│     ├─ cnn_dataset_builder.py
│     ├─ cnn_infer.py
│     ├─ cnn_model.py
│     ├─ cnn_train.py
│     └─ cnn_transforms.py
├─ tests/
├─ output/
└─ artifacts/shape_cnn/
```

## 重要說明

- `template` 仍是預設 backend，不受 `cnn` 影響
- `cnn` v1 只處理單一 shape 分類，不會偷偷 fallback 到 template
- `embed` 保留作為舊的實驗性最近鄰 backend
- CNN checkpoint 會保存 `class_to_idx`、`idx_to_class`、`num_classes`、
  preprocess 設定與訓練 metadata，方便未來擴展到 200 類別以上
