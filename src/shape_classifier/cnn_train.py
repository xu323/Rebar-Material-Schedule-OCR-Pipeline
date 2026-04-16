"""Training helpers for supervised shape CNNs."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .cnn_dataset import ShapeManifestDataset, load_manifest
from .cnn_model import ShapeCNN
from .cnn_transforms import ShapePreprocessConfig


@dataclass(frozen=True)
class TrainShapeCNNConfig:
    manifest_path: Path
    checkpoint_path: Path
    epochs: int = 8
    batch_size: int = 16
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    num_workers: int = 0
    device: str = "cpu"
    early_stopping_patience: int = 3
    image_size: int = 128
    width: int = 32
    dropout: float = 0.15
    threshold: float = 0.0
    seed: int = 1234
    show_progress: bool = True


def train_shape_cnn(config: TrainShapeCNNConfig) -> dict:
    try:
        import torch
        from torch.utils.data import DataLoader
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Training shape CNN requires torch. Install CPU build of torch."
        ) from exc

    torch.manual_seed(config.seed)
    manifest = load_manifest(config.manifest_path)
    preprocess_config = ShapePreprocessConfig(image_size=config.image_size)

    train_ds = ShapeManifestDataset(
        config.manifest_path,
        split="train",
        preprocess_config=preprocess_config,
        augment=True,
        seed=config.seed,
    )
    val_ds = ShapeManifestDataset(
        config.manifest_path,
        split="val",
        preprocess_config=preprocess_config,
        augment=False,
        seed=config.seed,
    )

    train_loader: DataLoader[ShapeManifestDataset] = DataLoader(
        train_ds,  # type: ignore[arg-type]
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    val_loader: DataLoader[ShapeManifestDataset] = DataLoader(
        val_ds,  # type: ignore[arg-type]
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    num_classes = len(train_ds.class_to_idx)
    model = ShapeCNN(
        num_classes=num_classes,
        width=config.width,
        dropout=config.dropout,
    ).to(config.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = -1.0
    best_epoch = -1
    best_threshold = config.threshold
    patience_left = config.early_stopping_patience
    history: list[dict] = []
    started = time.time()

    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc, _, _ = _run_epoch(
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
            device=config.device,
            epoch=epoch,
            num_epochs=config.epochs,
            phase="train",
            show_progress=config.show_progress,
        )
        val_loss, val_acc, _, _ = _run_epoch(
            model,
            val_loader,
            criterion,
            optimizer=None,
            device=config.device,
            epoch=epoch,
            num_epochs=config.epochs,
            phase="val",
            show_progress=config.show_progress,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "threshold": config.threshold,
            }
        )
        if config.show_progress:
            print(
                f"[epoch {epoch}/{config.epochs}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"threshold={config.threshold:.4f}"
            )
            if config.threshold <= 1e-6:
                print(
                    "  note: threshold=0.0 means validation accuracy is best "
                    "for CNN v1 because we do not reject predictions by "
                    "confidence. This keeps low-confidence-but-correct shape "
                    "matches from being dropped."
                )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_threshold = config.threshold
            patience_left = config.early_stopping_patience
            _save_checkpoint(
                config=config,
                manifest=manifest,
                model=model,
                preprocess_config=preprocess_config,
                best_epoch=best_epoch,
                best_val_acc=best_val_acc,
                threshold=best_threshold,
                history=history,
                training_seconds=time.time() - started,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                break

    return {
        "checkpoint_path": str(config.checkpoint_path),
        "best_epoch": best_epoch,
        "best_val_acc": best_val_acc,
        "threshold": best_threshold,
        "epochs_ran": len(history),
        "history": history,
    }


def _run_epoch(
    model,
    loader,
    criterion,
    *,
    optimizer,
    device: str,
    epoch: int,
    num_epochs: int,
    phase: str,
    show_progress: bool,
) -> tuple[float, float, list[tuple[float, int]], list[int]]:
    try:
        import torch
    except ImportError:  # pragma: no cover
        raise
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        tqdm = None

    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total = 0
    correct = 0
    max_probs: list[tuple[float, int]] = []
    labels: list[int] = []

    iterator = loader
    if show_progress and tqdm is not None:
        iterator = tqdm(
            loader,
            desc=f"{phase} {epoch}/{num_epochs}",
            leave=False,
            file=sys.stdout,
        )

    for inputs, targets in iterator:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            logits = model(inputs)
            loss = criterion(logits, targets)
            if is_train:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * targets.size(0)
        probs = torch.softmax(logits, dim=1)
        max_conf, preds = probs.max(dim=1)
        correct += int((preds == targets).sum().item())
        total += int(targets.size(0))
        max_probs.extend(
            zip(
                max_conf.detach().cpu().tolist(),
                preds.detach().cpu().tolist(),
                strict=False,
            )
        )
        labels.extend(targets.detach().cpu().tolist())

        if show_progress and tqdm is not None:
            iterator.set_postfix(
                {
                    "loss": f"{(total_loss / max(total, 1)):.4f}",
                    "acc": f"{(correct / max(total, 1)):.4f}",
                }
            )

    if total == 0:
        return 0.0, 0.0, [], []
    return total_loss / total, correct / total, max_probs, labels


def _save_checkpoint(
    *,
    config: TrainShapeCNNConfig,
    manifest: dict,
    model,
    preprocess_config: ShapePreprocessConfig,
    best_epoch: int,
    best_val_acc: float,
    threshold: float,
    history: list[dict],
    training_seconds: float,
) -> None:
    import torch

    config.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "width": config.width,
            "dropout": config.dropout,
        },
        "class_to_idx": manifest["class_to_idx"],
        "idx_to_class": manifest["idx_to_class"],
        "num_classes": len(manifest["class_to_idx"]),
        "threshold": threshold,
        "preprocess_config": asdict(preprocess_config),
        "training_metadata": {
            "manifest_path": str(config.manifest_path),
            "epochs_requested": config.epochs,
            "best_epoch": best_epoch,
            "best_val_acc": best_val_acc,
            "decision_threshold": threshold,
            "training_seconds": round(training_seconds, 3),
            "history": history,
        },
    }
    torch.save(payload, config.checkpoint_path)

    metrics_path = config.checkpoint_path.with_suffix(".metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(payload["training_metadata"], f, ensure_ascii=False, indent=2)
