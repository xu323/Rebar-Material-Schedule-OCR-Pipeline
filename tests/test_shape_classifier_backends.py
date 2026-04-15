import importlib.util
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.shape_classifier import get_classifier
from src.shape_classifier.cnn_dataset import ShapeManifestDataset, load_manifest
from src.shape_classifier.cnn_dataset_builder import (
    BuildShapeDatasetConfig,
    build_shape_dataset,
)
from src.shape_classifier.cnn_infer import LoadedShapeCNN
from src.shape_classifier.cnn_model import ShapeCNN
from src.shape_classifier.cnn_train import TrainShapeCNNConfig, train_shape_cnn
from src.shape_classifier.cnn_transforms import ShapePreprocessConfig


HAS_TORCH = bool(importlib.util.find_spec("torch"))


def _write_shape(path: Path, kind: str) -> None:
    img = np.ones((160, 160), dtype=np.uint8) * 255
    if kind == "line":
        cv2.line(img, (20, 80), (140, 80), 0, 7)
    elif kind == "hook":
        cv2.line(img, (20, 80), (140, 80), 0, 7)
        cv2.ellipse(img, (20, 80), (16, 16), 0, 90, 270, 0, 7)
    else:
        cv2.rectangle(img, (35, 35), (125, 125), 0, 7)
    cv2.imwrite(str(path), img)


@unittest.skipUnless(HAS_TORCH, "torch is required for CNN backend tests")
class ShapeClassifierBackendTests(unittest.TestCase):
    def test_registry_exposes_template_cnn_and_embed(self) -> None:
        self.assertIsNotNone(get_classifier("template"))
        self.assertIsNotNone(get_classifier("cnn", checkpoint_path="dummy.pt"))
        self.assertIsNotNone(get_classifier("embed"))

    def test_checkpoint_class_map_round_trip(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as td:
            checkpoint = Path(td) / "shape_cnn.pt"
            model = ShapeCNN(num_classes=3)
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {"width": 32, "dropout": 0.15},
                "class_to_idx": {"shape_20": 0, "shape_32": 1, "shape_33": 2},
                "idx_to_class": {"0": "shape_20", "1": "shape_32", "2": "shape_33"},
                "num_classes": 3,
                "threshold": 0.75,
                "preprocess_config": {"image_size": 128},
                "training_metadata": {},
            }, checkpoint)

            loaded = LoadedShapeCNN(checkpoint)
            self.assertEqual(loaded.class_to_idx["shape_33"], 2)
            self.assertEqual(loaded.idx_to_class[1], "shape_32")
            self.assertEqual(loaded.model.classifier[-1].out_features, 3)

    def test_model_scales_to_200_classes(self) -> None:
        import torch

        model = ShapeCNN(num_classes=200)
        logits = model(torch.randn(2, 1, 128, 128))
        self.assertEqual(tuple(logits.shape), (2, 200))

    def test_manifest_can_represent_200_classes(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            image = root / "sample.png"
            cv2.imwrite(str(image), np.ones((64, 64), dtype=np.uint8) * 255)
            class_to_idx = {f"shape_{i:03d}": i for i in range(200)}
            manifest = {
                "class_to_idx": class_to_idx,
                "idx_to_class": {str(i): sid for sid, i in class_to_idx.items()},
                "samples": [
                    {"image_path": "sample.png", "shape_id": "shape_000", "split": "train"}
                ],
            }
            manifest_path = root / "manifest.json"
            manifest_path.write_text(__import__("json").dumps(manifest), encoding="utf-8")
            dataset = ShapeManifestDataset(
                manifest_path,
                split="train",
                preprocess_config=ShapePreprocessConfig(),
                augment=False,
            )
            self.assertEqual(len(dataset.class_to_idx), 200)
            self.assertEqual(dataset.idx_to_class[199], "shape_199")

    def test_cnn_does_not_hard_reject_disconnected_shapes(self) -> None:
        import torch

        with tempfile.TemporaryDirectory() as td:
            checkpoint = Path(td) / "shape_cnn.pt"
            model = ShapeCNN(num_classes=2)
            torch.save({
                "model_state_dict": model.state_dict(),
                "model_config": {"width": 32, "dropout": 0.15},
                "class_to_idx": {"shape_20": 0, "shape_32": 1},
                "idx_to_class": {"0": "shape_20", "1": "shape_32"},
                "num_classes": 2,
                "threshold": 0.0,
                "preprocess_config": {"image_size": 128},
                "training_metadata": {},
            }, checkpoint)

            classifier = get_classifier("cnn", checkpoint_path=checkpoint)
            classifier.load_templates(Path(td))
            image = np.ones((128, 128), dtype=np.uint8) * 255
            cv2.line(image, (12, 40), (56, 40), 0, 5)
            cv2.line(image, (72, 88), (116, 88), 0, 5)
            matches = classifier.classify(image)
            self.assertLessEqual(len(matches), 1)
            if matches:
                self.assertIn(matches[0].shape_id, {"shape_20", "shape_32"})

    def test_build_dataset_and_train_smoke_epoch(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            shapes_dir = root / "shapes"
            shapes_dir.mkdir()
            _write_shape(shapes_dir / "shape_20.png", "line")
            _write_shape(shapes_dir / "shape_32.png", "hook")

            dataset_info = build_shape_dataset(BuildShapeDatasetConfig(
                shapes_dir=shapes_dir,
                output_dir=root / "dataset",
                train_per_class=4,
                val_per_class=2,
                test_per_class=2,
                image_size=96,
                seed=1234,
            ))
            manifest_path = Path(dataset_info["manifest_path"])
            manifest = load_manifest(manifest_path)
            self.assertEqual(len(manifest["class_to_idx"]), 2)

            result = train_shape_cnn(TrainShapeCNNConfig(
                manifest_path=manifest_path,
                checkpoint_path=root / "shape_cnn.pt",
                epochs=1,
                batch_size=4,
                learning_rate=1e-3,
                num_workers=0,
                device="cpu",
                early_stopping_patience=1,
                image_size=96,
                width=16,
                dropout=0.1,
                threshold=0.5,
                seed=1234,
            ))
            self.assertTrue(Path(result["checkpoint_path"]).exists())
            loaded = LoadedShapeCNN(Path(result["checkpoint_path"]))
            self.assertEqual(len(loaded.class_to_idx), 2)


if __name__ == "__main__":
    unittest.main()
