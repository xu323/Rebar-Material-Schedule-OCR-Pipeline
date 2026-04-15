"""Dataset helpers for supervised shape CNN training."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .cnn_transforms import ShapePreprocessConfig, preprocess_shape_image


@dataclass(frozen=True)
class ShapeSample:
    image_path: str
    shape_id: str
    split: str


def load_manifest(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    required = {"samples", "class_to_idx", "idx_to_class"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"manifest missing keys: {sorted(missing)}")
    return data


class ShapeManifestDataset:
    """Map-style dataset backed by a JSON manifest."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        split: str,
        preprocess_config: ShapePreprocessConfig,
        augment: bool = False,
        seed: int = 1234,
    ) -> None:
        self.manifest_path = manifest_path
        self.root = manifest_path.parent
        self.manifest = load_manifest(manifest_path)
        self.class_to_idx = {
            str(k): int(v)
            for k, v in self.manifest["class_to_idx"].items()
        }
        self.idx_to_class = {
            int(k): str(v)
            for k, v in self.manifest["idx_to_class"].items()
        }
        self.preprocess_config = preprocess_config
        self.augment = augment
        self.rng = np.random.default_rng(seed)
        self.samples = [
            ShapeSample(
                image_path=str(sample["image_path"]),
                shape_id=str(sample["shape_id"]),
                split=str(sample["split"]),
            )
            for sample in self.manifest["samples"]
            if str(sample["split"]) == split
        ]
        if not self.samples:
            raise ValueError(f"no samples found for split={split!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        import torch

        sample = self.samples[index]
        image_path = (self.root / sample.image_path).resolve()
        image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"shape image not found: {image_path}")
        processed = preprocess_shape_image(
            image,
            config=self.preprocess_config,
            augment=self.augment,
            rng=self.rng,
        )
        tensor = torch.from_numpy(processed.astype(np.float32) / 255.0).unsqueeze(0)
        target = torch.tensor(self.class_to_idx[sample.shape_id], dtype=torch.long)
        return tensor, target
