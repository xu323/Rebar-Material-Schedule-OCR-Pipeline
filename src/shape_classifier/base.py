"""Abstract base class for shape classifiers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.models import ShapeMatch


class ShapeClassifier(ABC):
    """Interface for rebar shape classifiers.

    Implementations:
    - TemplateMatcherClassifier: cv2.matchTemplate (v1)
    - TODO: CNNShapeClassifier, YOLOShapeClassifier (v2)
    """

    @abstractmethod
    def load_templates(self, shapes_dir: Path) -> None:
        """Load shape templates or model weights from *shapes_dir*."""

    @abstractmethod
    def classify(self, cell_image: np.ndarray) -> list[ShapeMatch]:
        """Classify shape(s) in a single cell image.

        Returns a list because composite shapes (e.g. shape_87) may
        contain multiple sub-shapes.  For the happy path the list has
        length 0 (no match) or 1.
        """

    def classify_batch(self, cell_images: list[np.ndarray]) -> list[list[ShapeMatch]]:
        """Batch classification — default loops over *classify*."""
        return [self.classify(img) for img in cell_images]
