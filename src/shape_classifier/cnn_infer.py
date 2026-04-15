"""Inference utilities for supervised shape CNN checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .cnn_model import ShapeCNN
from .cnn_transforms import (
    ShapePreprocessConfig,
    preprocess_shape_image,
)


@dataclass(frozen=True)
class CNNPredictResult:
    shape_id: str | None
    confidence: float
    probabilities: dict[str, float]


class LoadedShapeCNN:
    """Loaded CNN checkpoint ready for CPU/optional-device inference."""

    def __init__(
        self,
        checkpoint_path: Path,
        *,
        device: str = "cpu",
    ) -> None:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ImportError(
                "Supervised CNN backend requires torch. Install CPU build of torch."
            ) from exc

        self._torch = torch
        self.device = device
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model_config = dict(checkpoint.get("model_config", {}))
        num_classes = int(checkpoint["num_classes"])
        model = ShapeCNN(num_classes=num_classes, **model_config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        model.eval()

        preprocess_config = checkpoint.get("preprocess_config", {})
        self.preprocess_config = ShapePreprocessConfig(**preprocess_config)
        self.threshold = float(checkpoint.get("threshold", 0.0))
        self.class_to_idx = {
            str(k): int(v) for k, v in checkpoint["class_to_idx"].items()
        }
        self.idx_to_class = {
            int(k): str(v) for k, v in checkpoint["idx_to_class"].items()
        }
        self.model = model
        self.model_config = model_config
        self.metadata = dict(checkpoint.get("training_metadata", {}))

    def predict(
        self,
        image: np.ndarray,
        *,
        threshold: float | None = None,
        reject_multi_component: bool = False,
    ) -> CNNPredictResult:
        processed = preprocess_shape_image(
            image,
            config=self.preprocess_config,
            augment=False,
        )
        torch = self._torch
        tensor = (
            torch.from_numpy(processed.astype(np.float32) / 255.0)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        best_idx = int(np.argmax(probs))
        best_score = float(probs[best_idx])
        best_sid = self.idx_to_class[best_idx]
        cutoff = self.threshold if threshold is None else float(threshold)
        # Legitimate single-label shapes can contain disconnected strokes,
        # so confidence is a safer generic rejection criterion than
        # component-count rejection.
        if best_score < cutoff:
            return CNNPredictResult(
                shape_id=None,
                confidence=best_score,
                probabilities={self.idx_to_class[i]: float(p) for i, p in enumerate(probs)},
            )
        return CNNPredictResult(
            shape_id=best_sid,
            confidence=best_score,
            probabilities={self.idx_to_class[i]: float(p) for i, p in enumerate(probs)},
        )
