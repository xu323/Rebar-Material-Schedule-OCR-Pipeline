"""Synthetic dataset builder for supervised shape CNNs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .cnn_transforms import (
    ShapePreprocessConfig,
    augment_binary_shape,
    binarize_shape_image,
    normalize_binary_to_canvas,
    tight_foreground_crop,
)


@dataclass(frozen=True)
class BuildShapeDatasetConfig:
    shapes_dir: Path
    output_dir: Path
    train_per_class: int = 240
    val_per_class: int = 48
    test_per_class: int = 48
    image_size: int = 128
    seed: int = 1234


def build_shape_dataset(config: BuildShapeDatasetConfig) -> dict:
    rng = np.random.default_rng(config.seed)
    templates = sorted(config.shapes_dir.glob("shape_*.png"))
    if not templates:
        raise FileNotFoundError(f"no shape templates found in {config.shapes_dir}")

    output_dir = config.output_dir
    images_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    class_names = [p.stem for p in templates]
    class_to_idx = {shape_id: idx for idx, shape_id in enumerate(class_names)}
    idx_to_class = {idx: shape_id for shape_id, idx in class_to_idx.items()}

    split_counts = {
        "train": config.train_per_class,
        "val": config.val_per_class,
        "test": config.test_per_class,
    }
    preprocess_config = ShapePreprocessConfig(image_size=config.image_size)

    samples: list[dict] = []
    for template_path in templates:
        shape_id = template_path.stem
        template = cv2.imread(str(template_path), cv2.IMREAD_GRAYSCALE)
        if template is None:
            continue
        binary = binarize_shape_image(template)
        cropped = tight_foreground_crop(binary)
        if cropped is None:
            cropped = binary

        for split, count in split_counts.items():
            split_dir = images_dir / split / shape_id
            split_dir.mkdir(parents=True, exist_ok=True)
            for i in range(count):
                augmented = augment_binary_shape(cropped, rng=rng)
                canvas = normalize_binary_to_canvas(
                    augmented,
                    image_size=preprocess_config.image_size,
                    pad_ratio=preprocess_config.pad_ratio,
                )
                image = np.where(canvas > 0, 0, 255).astype(np.uint8)
                filename = f"{shape_id}_{split}_{i:04d}.png"
                path = split_dir / filename
                cv2.imwrite(str(path), image)
                samples.append(
                    {
                        "image_path": str(path.relative_to(output_dir)),
                        "shape_id": shape_id,
                        "split": split,
                    }
                )

    manifest = {
        "version": 1,
        "source_shapes_dir": str(config.shapes_dir),
        "class_to_idx": class_to_idx,
        "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
        "samples": samples,
        "dataset_metadata": {
            "image_size": config.image_size,
            "train_per_class": config.train_per_class,
            "val_per_class": config.val_per_class,
            "test_per_class": config.test_per_class,
            "seed": config.seed,
        },
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return {
        "manifest_path": str(manifest_path),
        "num_classes": len(class_names),
        "num_samples": len(samples),
    }
