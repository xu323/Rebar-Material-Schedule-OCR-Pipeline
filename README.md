# TSG OCR — Rebar Material Schedule Extract

[中文文檔](./README_zh.md) | **English**

A complete OCR pipeline for converting rebar material schedule PDFs into structured JSON, with intelligent table detection, merged-cell handling, shape classification, and a browser-based human-in-the-loop review interface.

## Overview

This project combines multiple approaches to extract and classify information from construction rebar material schedules:

- **Template Matching**: Original robust shape classifier (default backend)
- **CNN**: Trainable supervised CNN for improved shape recognition

## Key Features

### OCR & Table Extraction
- PDF rasterization at configurable DPI (default 300)
- Automatic layout detection (tables vs. non-table regions)
- Table grid extraction with full merged-cell (`rowspan`/`colspan`) support
- Cell-level OCR with confidence scoring
- Multi-page document handling

### Shape Classification
- **Template matching** (default): Robust pixel-based correlation with composite detection
- **CNN**: PyTorch-based supervised learning with CPU training/inference

### Review & Validation
- Faithful table re-rendering using grid metadata
- Browser-based review UI for human-in-the-loop editing
- Edit tracking with anchor-based cell identity (stable across column renames)
- Per-table rendered previews with per-edit refresh
- Batch editing and confidence visualization

### Debug & Diagnostics
- Layout region overlays (01_layout.png)
- Cell grid visualization with merged-cell markers (02_table_<j>_grid.png)
- Faithful rendered table reconstruction (03_table_<j>_rendered.png)
- Shape repair verification overlays (04_table_<j>_repair.png)
- Shape diagnostic CLI tools

## Installation

### Basic Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Optional CNN/Embedding Support

```bash
pip install torch torchvision
```

## Quick Start

### 1. Run the Default Pipeline

```bash
python scripts/run_pipeline.py \
  --pdf data/鋼筋施工圖料單明細表單.pdf \
  --debug
```

Outputs:
- `output/result.json` — structured OCR results
- `output/debug/page_<i>/` — layout, grid, render, and repair visualizations
- `output/review_assets/page_<i>/` — review sidecar and original page image

### 2. Launch the Review UI

```bash
python scripts/review.py --result output/result.json
```

Then open:
```
http://127.0.0.1:5000
```

**Features:**
- View page images with per-table rendered reconstructions
- Edit cell text and shape IDs directly in the browser
- See confidence scores and needs-review flags
- Batch save edits with automatic re-rendering
- Track edits with visual highlights

## CLI Reference

### OCR Pipeline

```bash
python scripts/run_pipeline.py \
  --pdf <input.pdf> \
  [--out output/result.json] \
  [--debug] \
  [--dpi 300] \
  [--shape-classifier template|cnn|embed] \
  [--cnn-checkpoint artifacts/shape_cnn/checkpoints/shape_cnn.pt]
```

**Options:**
- `--pdf` — Input PDF path (required)
- `--out` — Output JSON path (default: `output/result.json`)
- `--debug` — Save debug visualizations (layout, grid, render)
- `--dpi` — Rasterization DPI (default: 300)
- `--shape-classifier` — Which backend to use (default: `template`)
- `--cnn-checkpoint` — Required when `--shape-classifier cnn`

### Review UI

```bash
python scripts/review.py \
  --result output/result.json \
  [--host 127.0.0.1] \
  [--port 5000]
```

### Shape Classification

**Diagnose shape classification:**
```bash
python scripts/diagnose_shapes.py
```

## CNN Workflow

The supervised CNN backend is fully isolated from the template backend. You can train, evaluate, and deploy it independently.

### 1. Generate Synthetic Dataset

```bash
python scripts/build_shape_dataset.py \
  --out-dir artifacts/shape_cnn/datasets/generated
```

Creates `manifest.json` linking each shape template to synthetic training examples.

### 2. Train on CPU

```bash
python scripts/train_shape_cnn.py \
  --manifest artifacts/shape_cnn/datasets/generated/manifest.json \
  --out-checkpoint artifacts/shape_cnn/checkpoints/shape_cnn.pt \
  [--device cpu] \
  [--epochs 50]
```

Training includes per-epoch loss/accuracy reporting and tqdm progress bars.

### 3. Evaluate & Calibrate Threshold

```bash
python scripts/eval_shape_cnn.py \
  --manifest artifacts/shape_cnn/datasets/generated/manifest.json \
  --checkpoint artifacts/shape_cnn/checkpoints/shape_cnn.pt \
  --split val
```

Output shows:
- Per-shape precision/recall
- Current checkpoint threshold
- **Recommended threshold** (optimized for F1 on validation set)

To write the recommended threshold back to the checkpoint:

```bash
python scripts/eval_shape_cnn.py \
  --manifest artifacts/shape_cnn/datasets/generated/manifest.json \
  --checkpoint artifacts/shape_cnn/checkpoints/shape_cnn.pt \
  --split val \
  --write-threshold-to-checkpoint
```

This saves the auto-calibrated threshold so future inference uses it automatically.

### 4. Run OCR with CNN Backend

```bash
python scripts/run_pipeline.py \
  --pdf data/鋼筋施工圖料單明細表單.pdf \
  --debug \
  --shape-classifier cnn \
  --cnn-checkpoint artifacts/shape_cnn/checkpoints/shape_cnn.pt
```

## Project Structure

```
TSG_OCR/
├─ data/                          # Input PDFs and shape template images
│  ├─ 鋼筋施工圖料單明細表單.pdf   # Sample rebar schedule PDF
│  └─ shapes/                     # Shape template PNG files (shape_00.png, etc.)
│
├─ scripts/                        # CLI entry points
│  ├─ run_pipeline.py             # Main OCR pipeline
│  ├─ review.py                   # Browser review UI launcher
│  ├─ train_shape_cnn.py          # CNN training entrypoint
│  ├─ eval_shape_cnn.py           # CNN evaluation & threshold calibration
│  ├─ build_shape_dataset.py      # Synthetic dataset generator
│  └─ diagnose_shapes.py          # Shape classification diagnostics
│
├─ src/
│  ├─ pipeline.py                 # Main OCR orchestrator
│  ├─ review_server.py            # Flask review UI backend
│  ├─ review_renderer.py          # Table re-rendering from edits
│  ├─ ocr_engine.py               # PaddleOCR wrapper
│  ├─ pdf_rasterizer.py           # PyMuPDF PDF→image
│  ├─ layout_analyzer.py          # Layout detection
│  ├─ table_cell_segmenter.py     # Table grid extraction
│  ├─ shape_column_repair.py      # Fix split merged cells in shape columns
│  ├─ assembler.py                # Assemble OCR results into JSON
│  ├─ config.py                   # Configuration & paths
│  ├─ debug_visualizer.py         # Debug image generation
│  ├─ models.py                   # Data classes (BBox, GridCell, etc.)
│  └─ shape_classifier/           # Shape classification backends
│     ├─ base.py                  # Abstract classifier interface
│     ├─ template_matcher.py      # Template matching (default)
│     ├─ cnn_classifier.py        # CNN inference wrapper
│     ├─ cnn_infer.py             # Low-level CNN inference
│     ├─ cnn_model.py             # PyTorch CNN model definition
│     ├─ cnn_train.py             # CNN training logic
│     ├─ cnn_dataset.py           # CNN training dataset class
│     ├─ cnn_dataset_builder.py   # Synthetic dataset generator
│     ├─ cnn_transforms.py        # Image augmentations
│     ├─ embed_classifier.py      # Embedding-based (legacy)
│     └─ registry.py              # Classifier factory
│
├─ tests/                          # Unit tests
├─ output/                         # Pipeline outputs (created at runtime)
├─ artifacts/                      # CNN artifacts (created at runtime)
└─ requirements.txt
```

## Output Files

### Main Pipeline Output
- `output/result.json` — Structured OCR results (pages, tables, cells)
- `output/debug/page_<i>/01_layout.png` — Layout regions overlay
- `output/debug/page_<i>/02_table_<j>_grid.png` — Cell grid visualization
- `output/debug/page_<i>/03_table_<j>_rendered.png` — Faithful table render
- `output/debug/page_<i>/04_table_<j>_repair.png` — Repair verification (if needed)

### Review Workflow Outputs
- `output/review_assets/page_<i>/page.png` — Original rasterized page
- `output/review_assets/page_<i>/table_<j>_grid.json` — Grid metadata (columns, row_types, cells)
- `output/result_reviewed.json` — Edited JSON after review
- `output/reviewed/page_<i>/table_<j>_rendered.png` — Re-rendered table after edits

### CNN Artifacts
- `artifacts/shape_cnn/datasets/generated/manifest.json` — Training dataset manifest
- `artifacts/shape_cnn/checkpoints/shape_cnn.pt` — Trained model checkpoint
- `artifacts/shape_cnn/checkpoints/*.metrics.json` — Training metrics

## Key Concepts

### Merged Cells & Grid Representation
Tables often include merged cells (`rowspan` > 1 or `colspan` > 1). The pipeline preserves this structure:
- **GridCell**: A cell in the canonical grid (row, col) with its actual span.
- **Continuation cells**: Positions covered by a merged cell that is not its anchor (marked `is_continuation: true`).
- **JSON serialization**: Each row's `cells` dict stores only anchor cells; continuation positions are marked in `merge_info` for reference.

### Shape Column Detection
The pipeline automatically detects the shape column using:
1. Keyword hints ("形狀", "shape", etc.) in headers
2. Content analysis: columns with low OCR confidence and image-like pixel characteristics

A detected shape column index is stored in:
- Pipeline JSON: `table.semantic.shape_column_index`
- Sidecar (review assets): `grid.shape_col_idx`

### Review UI Stability
The review UI uses **anchor-based editing** (`anchor_row`, `anchor_col`) to remain stable across column renames:
- Edits reference the canonical grid position, not column names
- If a user edits JSON column names, re-rendering still locates cells correctly
- The sidecar captures column order and row types at pipeline time, acting as a fallback

### Needs Review Flags
Cells with OCR confidence < 0.9 are automatically flagged in the JSON:
```json
{
  "text": "鋼筋號",
  "confidence": 0.85,
  "needs_review": true
}
```
The review UI highlights these cells with an amber border for easy batch review.

## Troubleshooting

### CNN Threshold Issues
If CNNmissing shapes after training, the threshold may be too high. Run evaluation:
```bash
python scripts/eval_shape_cnn.py --checkpoint ... --split val --write-threshold-to-checkpoint
```
This auto-calibrates the threshold and saves it to the checkpoint.

### Missing Grid Sidecar
If the review UI shows "sidecar not found", re-run the pipeline with `--debug`:
```bash
python scripts/run_pipeline.py --pdf data/file.pdf --debug
```
This generates the required `review_assets/` directory.

### Rendered Image Mismatch
If the rendered table doesn't match the original:
1. Check that all shape templates exist in `data/shapes/`
2. Verify the grid sidecar is present (`output/review_assets/page_<i>/table_<j>_grid.json`)
3. Re-run debug mode to regenerate renders