"""Train a supervised CNN shape classifier on CPU."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.shape_classifier.cnn_train import TrainShapeCNNConfig, train_shape_cnn


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train supervised CNN shape classifier"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/shape_cnn/datasets/generated/manifest.json"),
    )
    parser.add_argument(
        "--out-checkpoint",
        type=Path,
        default=Path("artifacts/shape_cnn/checkpoints/shape_cnn.pt"),
    )
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    result = train_shape_cnn(
        TrainShapeCNNConfig(
            manifest_path=args.manifest,
            checkpoint_path=args.out_checkpoint,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            num_workers=args.num_workers,
            device=args.device,
            early_stopping_patience=args.patience,
            image_size=args.image_size,
            width=args.width,
            dropout=args.dropout,
            threshold=args.threshold,
            seed=args.seed,
            show_progress=not args.quiet,
        )
    )

    print(f"Checkpoint: {result['checkpoint_path']}")
    print(f"Best epoch: {result['best_epoch']}")
    print(f"Best val acc: {result['best_val_acc']:.4f}")
    print(f"Decision threshold: {result['threshold']:.4f}")
    print(f"Epochs ran: {result['epochs_ran']}")


if __name__ == "__main__":
    main()
