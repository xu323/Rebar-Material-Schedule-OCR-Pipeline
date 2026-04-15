"""Evaluate a supervised CNN shape classifier checkpoint."""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2

from src.shape_classifier.cnn_dataset import load_manifest
from src.shape_classifier.cnn_infer import LoadedShapeCNN

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate supervised CNN shape classifier"
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/shape_cnn/datasets/generated/manifest.json"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/shape_cnn/checkpoints/shape_cnn.pt"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    model = LoadedShapeCNN(args.checkpoint, device=args.device)
    manifest = load_manifest(args.manifest)
    root = args.manifest.parent
    samples = [s for s in manifest["samples"] if s["split"] == args.split]
    if not samples:
        raise ValueError(f"no samples found for split={args.split!r}")

    correct = 0
    total = 0
    rejected = 0
    confusions: Counter[tuple[str, str]] = Counter()
    threshold = model.threshold if args.threshold is None else float(args.threshold)
    iterator = samples
    if not args.quiet:
        try:
            from tqdm import tqdm
            iterator = tqdm(samples, desc=f"eval {args.split}", file=sys.stdout)
        except ImportError:
            iterator = samples

    for sample in iterator:
        image = cv2.imread(str((root / sample["image_path"]).resolve()), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(sample["image_path"])
        pred = model.predict(
            image,
            threshold=threshold,
            reject_multi_component=False,
        )
        expected = str(sample["shape_id"])
        total += 1
        if pred.shape_id is None:
            rejected += 1
            confusions[(expected, "<rejected>")] += 1
            continue
        if pred.shape_id == expected:
            correct += 1
        else:
            confusions[(expected, pred.shape_id)] += 1

    accuracy = correct / total if total else 0.0
    print(f"Threshold in use: {threshold:.4f}")
    print(f"Samples:   {total}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Rejected:  {rejected}")
    if threshold <= 1e-6:
        print(
            "Note: threshold is 0.0 by design for CNN v1, so predictions are "
            "not rejected by confidence. This avoids dropping valid shapes "
            "whose probabilities are not yet well-calibrated."
        )
    print("Top confusions:")
    for (expected, predicted), count in confusions.most_common(10):
        print(f"  {expected} -> {predicted}: {count}")


if __name__ == "__main__":
    main()
