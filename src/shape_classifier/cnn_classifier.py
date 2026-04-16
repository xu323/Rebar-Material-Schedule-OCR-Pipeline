"""Supervised CNN shape classifier backed by a PyTorch checkpoint."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.models import ShapeMatch

from .base import ShapeClassifier
from .cnn_infer import LoadedShapeCNN


class CNNShapeClassifier(ShapeClassifier):
    """Checkpoint-backed single-shape CNN classifier."""

    def __init__(
        self,
        *,
        checkpoint_path: str | Path | None = None,
        threshold: float | None = None,
        device: str = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self.threshold = None if threshold is None else float(threshold)
        self.device = device
        self._model: LoadedShapeCNN | None = None

    def load_templates(self, shapes_dir: Path) -> None:
        if self.checkpoint_path is None:
            raise ValueError(
                "CNNShapeClassifier requires a checkpoint_path. "
                "Pass --cnn-checkpoint when using --shape-classifier cnn."
            )
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"CNN checkpoint not found: {self.checkpoint_path}")
        self._model = LoadedShapeCNN(self.checkpoint_path, device=self.device)

    def classify(self, cell_image: np.ndarray) -> list[ShapeMatch]:
        if self._model is None:
            raise RuntimeError(
                "CNNShapeClassifier.load_templates() must be called first"
            )
        result = self._model.predict(
            cell_image,
            threshold=self.threshold,
            reject_multi_component=False,
        )
        if result.shape_id is None:
            return []
        return [ShapeMatch(shape_id=result.shape_id, confidence=result.confidence)]
