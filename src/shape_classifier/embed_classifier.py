"""Embedding-based shape classifier using a pretrained ResNet feature extractor."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.models import ShapeMatch
from .base import ShapeClassifier


class EmbedShapeClassifier(ShapeClassifier):
    """Pretrained-ResNet nearest-neighbour classifier."""

    def __init__(
        self,
        threshold: float = 0.75,
        device: str | None = None,
        model_name: str = "resnet18",
    ) -> None:
        try:
            import torch  # type: ignore
            import torchvision  # type: ignore
            import torchvision.transforms as T  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "EmbedShapeClassifier requires torch and torchvision. "
                "Install with: pip install torch torchvision"
            ) from e

        self._torch = torch
        self.threshold = threshold
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if model_name == "resnet18":
            weights = torchvision.models.ResNet18_Weights.DEFAULT
            model = torchvision.models.resnet18(weights=weights)
        elif model_name == "resnet34":
            weights = torchvision.models.ResNet34_Weights.DEFAULT
            model = torchvision.models.resnet34(weights=weights)
        else:
            raise ValueError(f"Unsupported model_name: {model_name!r}")

        model.fc = torch.nn.Identity()
        model.eval()
        model.to(self.device)
        self._model = model

        self._transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.Grayscale(num_output_channels=3),
            T.ToTensor(),
            T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

        self._template_embeddings: dict[str, "torch.Tensor"] = {}

    def load_templates(self, shapes_dir: Path) -> None:
        self._template_embeddings.clear()
        for p in sorted(shapes_dir.glob("shape_*.png")):
            shape_id = p.stem
            img = cv2.imread(str(p), cv2.IMREAD_COLOR)
            if img is None:
                continue
            img = self._pad_to_square(img, padding=12)
            emb = self._embed(img)
            self._template_embeddings[shape_id] = emb

    def classify(self, cell_image: np.ndarray) -> list[ShapeMatch]:
        if not self._template_embeddings:
            return []
        if cell_image is None or cell_image.size == 0:
            return []
        if cell_image.shape[0] < 10 or cell_image.shape[1] < 10:
            return []

        if len(cell_image.shape) == 2:
            cell_image = cv2.cvtColor(cell_image, cv2.COLOR_GRAY2BGR)

        cell_emb = self._embed(cell_image)

        best_id: str | None = None
        best_score: float = -1.0
        for sid, t_emb in self._template_embeddings.items():
            score = float((cell_emb * t_emb).sum().item())
            if score > best_score:
                best_score = score
                best_id = sid

        if best_id is not None and best_score >= self.threshold:
            return [ShapeMatch(shape_id=best_id, confidence=best_score)]
        return []

    def _embed(self, img_bgr: np.ndarray):
        torch = self._torch
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        x = self._transform(rgb).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self._model(x)
            feat = feat / feat.norm(dim=1, keepdim=True).clamp(min=1e-9)
        return feat.squeeze(0).cpu()

    @staticmethod
    def _pad_to_square(img: np.ndarray, padding: int = 0) -> np.ndarray:
        h, w = img.shape[:2]
        side = max(h, w) + 2 * padding
        top = (side - h) // 2
        bottom = side - h - top
        left = (side - w) // 2
        right = side - w - left
        return cv2.copyMakeBorder(
            img,
            top, bottom, left, right,
            cv2.BORDER_CONSTANT, value=(255, 255, 255),
        )
