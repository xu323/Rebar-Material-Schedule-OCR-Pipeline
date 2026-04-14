from .base import ShapeClassifier, ShapeMatch
from .registry import get_classifier, register
from .template_matcher import TemplateMatcherClassifier

register("template", TemplateMatcherClassifier)

# Optional CNN backend: registered only if torch/torchvision import succeeds.
try:
    from .cnn_classifier import CNNShapeClassifier  # noqa: F401

    register("cnn", CNNShapeClassifier)
except ImportError:  # pragma: no cover — optional dep
    CNNShapeClassifier = None  # type: ignore

__all__ = [
    "ShapeClassifier",
    "ShapeMatch",
    "TemplateMatcherClassifier",
    "CNNShapeClassifier",
    "get_classifier",
    "register",
]
