# TSG OCR — 鋼筋料單智能提取系統

**[English](./README_en.md) | [中文](./README.md)**

一套完整的 OCR 流程，可將鋼筋施工圖料單 PDF 轉成結構化 JSON，具備智能表格檢測、合併格保留、shape 辨識、以及瀏覽器人工審核介面。

## DEMO

https://github.com/user-attachments/assets/fd1ed9a8-c4c3-4523-af9b-4f10069d3a2b

## 核心功能

### OCR 與表格提取
- 可配置 DPI 的 PDF 光柵化（預設 300 DPI）
- 自動版面分析（表格 vs. 非表格區域）
- 完整的表格切格支持（包含合併格的 `rowspan`/`colspan`）
- 逐格 OCR 與信心分數
- 多頁文件處理

### Shape 辨識
- **Template Matching**（預設）：基於像素的相關性匹配，支援複合 shape 檢測
- **CNN**：PyTorch 監督式學習，支援 CPU 訓練/推論

### 人工審核與驗證
- 基於網格元數據的忠實表格重繪
- 瀏覽器型人工審核 UI
- 基於 anchor 的穩定格單位定位（免受欄名更改影響）
- 逐表渲染預覽與編輯即時刷新
- 批量編輯與信心可視化

## 安裝

### 基本安裝

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 選項：CNN/Embedding 支持

```bash
pip install torch torchvision
```

## 快速開始

### 1. 執行預設 Pipeline

```bash
python scripts/run_pipeline.py \
  --pdf data/鋼筋施工圖料單明細表單.pdf \
  --debug
```

輸出：
- `output/result.json` — 結構化 OCR 結果
- `output/debug/page_<i>/` — 版面、格線、重繪、修復視覺化
- `output/review_assets/page_<i>/` — 審核伴侶檔案與原始頁影像

### 2. 啟動人工審核 UI

```bash
python scripts/review.py --result output/result.json
```

瀏覽器開啟：
```
http://127.0.0.1:5000
```

**功能：**
- 查看頁面影像與逐表重繪結果
- 直接在瀏覽器中編輯表格文字與 shape ID
- 檢視信心分數與需要審核的標記
- 批量保存編輯與自動重繪
- 視覺化追蹤編輯狀態

## CLI 參考

### OCR Pipeline

```bash
python scripts/run_pipeline.py \
  --pdf <輸入.pdf> \
  [--out output/result.json] \
  [--debug] \
  [--dpi 300] \
  [--shape-classifier template|cnn|embed] \
  [--cnn-checkpoint artifacts/shape_cnn/checkpoints/shape_cnn.pt]
```

**選項：**
- `--pdf` — 輸入 PDF 路徑（必填）
- `--out` — 輸出 JSON 路徑（預設：`output/result.json`）
- `--debug` — 保存除錯視覺化（版面、格線、重繪）
- `--dpi` — 光柵化 DPI（預設：300）
- `--shape-classifier` — 選擇後端（預設：`template`）
- `--cnn-checkpoint` — 當 `--shape-classifier cnn` 時必填

### 人工審核 UI

```bash
python scripts/review.py \
  --result output/result.json \
  [--host 127.0.0.1] \
  [--port 5000]
```

### Shape 分類診斷

```bash
python scripts/diagnose_shapes.py
```

## CNN 工作流

監督式 CNN 後端與 template 後端完全隔離。你可以獨立訓練、評估、部署 CNN。

### 1. 生成合成資料集

```bash
python scripts/build_shape_dataset.py \
  --out-dir artifacts/shape_cnn/datasets/generated
```

建立 `manifest.json`，連結每個 shape 模板與合成訓練範例。

### 2. 用 CPU 訓練

```bash
python scripts/train_shape_cnn.py \
  --manifest artifacts/shape_cnn/datasets/generated/manifest.json \
  --out-checkpoint artifacts/shape_cnn/checkpoints/shape_cnn.pt \
  [--device cpu] \
  [--epochs 50]
```

訓練過程包括按 epoch 報告損失/準確率與 tqdm 進度條。

### 3. 評估與校準閾值

```bash
python scripts/eval_shape_cnn.py \
  --manifest artifacts/shape_cnn/datasets/generated/manifest.json \
  --checkpoint artifacts/shape_cnn/checkpoints/shape_cnn.pt \
  --split val
```

輸出展示：
- 逐 shape 的準確率/召回率
- 當前 checkpoint 閾值
- **推薦閾值**（在驗證集上最優化 F1）

若要把推薦閾值寫回 checkpoint：

```bash
python scripts/eval_shape_cnn.py \
  --manifest artifacts/shape_cnn/datasets/generated/manifest.json \
  --checkpoint artifacts/shape_cnn/checkpoints/shape_cnn.pt \
  --split val \
  --write-threshold-to-checkpoint
```

這會保存自動校準的閾值，往後推論會自動使用。

### 4. 用 CNN 後端執行 OCR

```bash
python scripts/run_pipeline.py \
  --pdf data/鋼筋施工圖料單明細表單.pdf \
  --debug \
  --shape-classifier cnn \
  --cnn-checkpoint artifacts/shape_cnn/checkpoints/shape_cnn.pt
```

## 專案結構

```
TSG_OCR/
├─ data/                          # 輸入 PDF 與 shape 模板圖
│  ├─ 鋼筋施工圖料單明細表單.pdf   # 樣本鋼筋料單 PDF
│  └─ shapes/                     # Shape 模板 PNG （shape_00.png 等）
│
├─ scripts/                        # CLI 進入點
│  ├─ run_pipeline.py             # 主 OCR pipeline
│  ├─ review.py                   # 瀏覽器審核 UI 啟動器
│  ├─ train_shape_cnn.py          # CNN 訓練進入點
│  ├─ eval_shape_cnn.py           # CNN 評估與閾值校準
│  ├─ build_shape_dataset.py      # 合成資料集生成器
│  └─ diagnose_shapes.py          # Shape 分類診斷工具
│
├─ src/
│  ├─ pipeline.py                 # 主 OCR 協調者
│  ├─ review_server.py            # Flask 審核 UI 後端
│  ├─ review_renderer.py          # 根據編輯重繪表格
│  ├─ ocr_engine.py               # PaddleOCR 包裝器
│  ├─ pdf_rasterizer.py           # PyMuPDF PDF→影像
│  ├─ layout_analyzer.py          # 版面檢測
│  ├─ table_cell_segmenter.py     # 表格切格
│  ├─ shape_column_repair.py      # 修復 shape 欄中的分割合併格
│  ├─ assembler.py                # 組裝 OCR 結果為 JSON
│  ├─ config.py                   # 配置與路徑
│  ├─ debug_visualizer.py         # 除錯圖像生成
│  ├─ models.py                   # 數據類別（BBox、GridCell等）
│  └─ shape_classifier/           # Shape 分類後端
│     ├─ base.py                  # 抽象分類器介面
│     ├─ template_matcher.py      # Template matching（預設）
│     ├─ cnn_classifier.py        # CNN 推論包裝器
│     ├─ cnn_infer.py             # 低階 CNN 推論
│     ├─ cnn_model.py             # PyTorch CNN 模型定義
│     ├─ cnn_train.py             # CNN 訓練邏輯
│     ├─ cnn_dataset.py           # CNN 訓練資料集類
│     ├─ cnn_dataset_builder.py   # 合成資料集生成器
│     ├─ cnn_transforms.py        # 影像增強
│     ├─ embed_classifier.py      # Embedding 版本（遺留）
│     └─ registry.py              # 分類器工廠
│
├─ tests/                          # 單元測試
├─ output/                         # Pipeline 輸出（執行時建立）
├─ artifacts/                      # CNN 製品（執行時建立）
└─ requirements.txt
```

## 輸出檔案

### 主 Pipeline 輸出
- `output/result.json` — 結構化 OCR 結果（頁面、表格、格單位）
- `output/debug/page_<i>/01_layout.png` — 版面區域疊加
- `output/debug/page_<i>/02_table_<j>_grid.png` — 格線視覺化
- `output/debug/page_<i>/03_table_<j>_rendered.png` — 忠實重繪表格
- `output/debug/page_<i>/04_table_<j>_repair.png` — 修復驗證（如需要）

### 人工審核工作流輸出
- `output/review_assets/page_<i>/page.png` — 原始光柵化頁面
- `output/review_assets/page_<i>/table_<j>_grid.json` — 格線元數據（欄名、行類型、格單位）
- `output/result_reviewed.json` — 審核後編輯的 JSON
- `output/reviewed/page_<i>/table_<j>_rendered.png` — 編輯後重繪的表格

### CNN 製品
- `artifacts/shape_cnn/datasets/generated/manifest.json` — 訓練資料集清單
- `artifacts/shape_cnn/checkpoints/shape_cnn.pt` — 訓練好的模型 checkpoint
- `artifacts/shape_cnn/checkpoints/*.metrics.json` — 訓練指標

## 關鍵概念

### 合併格與網格表表示
表格經常包含合併格（`rowspan` > 1 或 `colspan` > 1）。Pipeline 保留這個結構：
- **GridCell**：規範網格中的一格（row、col）及其實際跨度
- **延續格**：被合併格覆蓋但非其錨點的位置（標記 `is_continuation: true`）
- **JSON 序列化**：每行的 `cells` 字典只存儲錨點格；延續位置在 `merge_info` 中記錄供參考

### Shape 欄檢測
Pipeline 使用以下方法自動偵測 shape 欄：
1. 表頭中的關鍵字提示（「形狀」、「shape」等）
2. 內容分析：低 OCR 信心度、具有類似影像像素特徵的欄位

偵測到的 shape 欄索引存儲於：
- Pipeline JSON：`table.semantic.shape_column_index`
- Sidecar（審核資產）：`grid.shape_col_idx`

### 審核 UI 穩定性
審核 UI 使用**基於 anchor 的編輯**（`anchor_row`、`anchor_col`）以在欄名重命名時保持穩定：
- 編輯參考規範網格位置，而非欄名
- 若用戶編輯 JSON 欄名，重繪仍能正確定位格單位
- Sidecar 在 pipeline 時捕獲欄順序與行類型，作為降級備份

### 需要審核標記
OCR 信心度 < 0.9 的格單位會自動在 JSON 中被標記：
```json
{
  "text": "鋼筋號",
  "confidence": 0.85,
  "needs_review": true
}
```
審核 UI 會用琥珀色邊框突出這些格單位，方便批量審核。

## 故障排除

### CNN 閾值問題
若訓練後 CNN 遺漏某些 shape，閾值可能過高。執行評估：
```bash
python scripts/eval_shape_cnn.py --checkpoint ... --split val --write-threshold-to-checkpoint
```
這會自動校準閾值並保存到 checkpoint。

### 缺少 Sidecar
若審核 UI 顯示「sidecar not found」，用 `--debug` 重新執行 pipeline：
```bash
python scripts/run_pipeline.py --pdf data/file.pdf --debug
```
這會生成所需的 `review_assets/` 目錄。

### 重繪不匹配
若重繪表格與原始不符：
1. 檢查所有 shape 模板是否存在於 `data/shapes/`
2. 驗證 sidecar 存在（`output/review_assets/page_<i>/table_<j>_grid.json`）
3. 重新執行 debug 模式重新生成重繪
