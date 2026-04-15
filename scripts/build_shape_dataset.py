"""Build a synthetic shape dataset from template PNGs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.config import SHAPES_DIR
from src.shape_classifier.cnn_dataset_builder import (
    BuildShapeDatasetConfig,
    build_shape_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build synthetic dataset for supervised shape CNN training"
    )
    parser.add_argument("--shapes-dir", type=Path, default=SHAPES_DIR)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("artifacts/shape_cnn/datasets/generated"),
    )
    parser.add_argument("--train-per-class", type=int, default=240)
    parser.add_argument("--val-per-class", type=int, default=48)
    parser.add_argument("--test-per-class", type=int, default=48)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    result = build_shape_dataset(BuildShapeDatasetConfig(
        shapes_dir=args.shapes_dir,
        output_dir=args.out_dir,
        train_per_class=args.train_per_class,
        val_per_class=args.val_per_class,
        test_per_class=args.test_per_class,
        image_size=args.image_size,
        seed=args.seed,
    ))
    print(f"Manifest: {result['manifest_path']}")
    print(f"Classes:  {result['num_classes']}")
    print(f"Samples:  {result['num_samples']}")


if __name__ == "__main__":
    main()
