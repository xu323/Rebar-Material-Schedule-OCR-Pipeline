"""Preprocessing and augmentation for supervised shape CNNs."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ShapePreprocessConfig:
    image_size: int = 128
    pad_ratio: float = 0.14
    border_trim_ratio: float = 0.015
    edge_band_ratio: float = 0.05
    min_component_area_ratio: float = 0.01
    min_component_dim: int = 8


def preprocess_shape_image(
    image: np.ndarray,
    *,
    config: ShapePreprocessConfig,
    augment: bool = False,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Convert a raw crop into a normalized grayscale tensor image."""
    if image is None or image.size == 0:
        return np.ones((config.image_size, config.image_size), dtype=np.uint8) * 255

    rng = rng or np.random.default_rng()
    bw = binarize_shape_image(image)
    bw = strip_cell_borders(bw, config=config)
    cropped = tight_foreground_crop(bw)
    if cropped is None:
        cropped = bw

    if augment:
        cropped = augment_binary_shape(cropped, rng=rng)

    canvas = normalize_binary_to_canvas(
        cropped,
        image_size=config.image_size,
        pad_ratio=config.pad_ratio,
    )
    return binary_to_grayscale(canvas)


def binarize_shape_image(image: np.ndarray) -> np.ndarray:
    if image is None or image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return bw


def strip_cell_borders(
    cell_bw: np.ndarray,
    *,
    config: ShapePreprocessConfig,
) -> np.ndarray:
    cleaned = cell_bw.copy()
    h, w = cleaned.shape[:2]
    edge_band_x = max(int(round(w * config.edge_band_ratio)), 4)
    edge_band_y = max(int(round(h * config.edge_band_ratio)), 4)
    hard_trim_x = min(max(int(round(w * config.border_trim_ratio)), 2), w)
    hard_trim_y = min(max(int(round(h * config.border_trim_ratio)), 2), h)

    cleaned[:, :hard_trim_x] = 0
    cleaned[:, max(w - hard_trim_x, 0) :] = 0
    cleaned[:hard_trim_y, :] = 0
    cleaned[max(h - hard_trim_y, 0) :, :] = 0

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


def tight_foreground_crop(cell_bw: np.ndarray) -> np.ndarray | None:
    if cell_bw is None or cell_bw.size == 0:
        return None
    h, w = cell_bw.shape[:2]
    min_area = max(int(h * w * 0.001), 12)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(cell_bw, connectivity=8)

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
        x, y, bw, bh = cv2.boundingRect(coords)
        boxes = [(x, y, x + bw, y + bh)]

    x0 = min(box[0] for box in boxes)
    y0 = min(box[1] for box in boxes)
    x1 = max(box[2] for box in boxes)
    y1 = max(box[3] for box in boxes)
    if x1 <= x0 or y1 <= y0:
        return None
    pad = max(int(round(min(h, w) * 0.02)), 2)
    return cell_bw[
        max(y0 - pad, 0) : min(y1 + pad, h), max(x0 - pad, 0) : min(x1 + pad, w)
    ]


def normalize_binary_to_canvas(
    cell_bw: np.ndarray,
    *,
    image_size: int,
    pad_ratio: float,
) -> np.ndarray:
    canvas = np.zeros((image_size, image_size), dtype=np.uint8)
    if cell_bw is None or cell_bw.size == 0:
        return canvas
    src_h, src_w = cell_bw.shape[:2]
    usable = max(int(round(image_size * (1.0 - 2.0 * pad_ratio))), 8)
    scale = min(usable / max(src_h, 1), usable / max(src_w, 1))
    dst_w = max(int(round(src_w * scale)), 1)
    dst_h = max(int(round(src_h * scale)), 1)
    resized = cv2.resize(cell_bw, (dst_w, dst_h), interpolation=cv2.INTER_NEAREST)
    x0 = (image_size - dst_w) // 2
    y0 = (image_size - dst_h) // 2
    canvas[y0 : y0 + dst_h, x0 : x0 + dst_w] = resized
    return canvas


def binary_to_grayscale(binary: np.ndarray) -> np.ndarray:
    return np.where(binary > 0, 0, 255).astype(np.uint8)


def augment_binary_shape(
    binary: np.ndarray,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    if binary is None or binary.size == 0:
        return binary

    img = binary.copy()
    h, w = img.shape[:2]

    scale = float(rng.uniform(0.88, 1.08))
    angle = float(rng.uniform(-7.0, 7.0))
    tx = float(rng.uniform(-0.08, 0.08) * w)
    ty = float(rng.uniform(-0.08, 0.08) * h)
    matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, scale)
    matrix[0, 2] += tx
    matrix[1, 2] += ty
    img = cv2.warpAffine(
        img,
        matrix,
        (w, h),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0,),
    )

    morph_roll = float(rng.random())
    if morph_roll < 0.2:
        img = cv2.dilate(img, np.ones((2, 2), dtype=np.uint8), iterations=1)
    elif morph_roll < 0.4:
        img = cv2.erode(img, np.ones((2, 2), dtype=np.uint8), iterations=1)

    if rng.random() < 0.25:
        ksize = int(rng.choice([3, 5]))
        img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        _, img = cv2.threshold(img, 32, 255, cv2.THRESH_BINARY)

    if rng.random() < 0.3:
        noise = np.asarray(rng.normal(0, 12, size=img.shape)).astype(np.int16)
        noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        _, img = cv2.threshold(noisy, 32, 255, cv2.THRESH_BINARY)

    if rng.random() < 0.25:
        crop_ratio = float(rng.uniform(0.0, 0.05))
        trim_x = int(round(w * crop_ratio))
        trim_y = int(round(h * crop_ratio))
        if trim_x > 0:
            img[:, :trim_x] = 0
            img[:, max(w - trim_x, 0) :] = 0
        if trim_y > 0:
            img[:trim_y, :] = 0
            img[max(h - trim_y, 0) :, :] = 0

    return img


def count_significant_components(
    binary: np.ndarray,
    *,
    min_area_ratio: float,
    min_dim: int,
) -> int:
    if binary is None or binary.size == 0:
        return 0
    h, w = binary.shape[:2]
    total_area = max(h * w, 1)
    min_area = max(int(total_area * min_area_ratio), 20)
    n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    count = 0
    for i in range(1, n_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_area:
            continue
        if bw < min_dim or bh < min_dim:
            continue
        count += 1
    return count
