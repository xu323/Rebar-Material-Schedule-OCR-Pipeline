from src.models import ShapeMatch

from .base import ShapeClassifier
from .cnn_classifier import CNNShapeClassifier
from .embed_classifier import EmbedShapeClassifier
from .registry import get_classifier, register
from .template_matcher import TemplateMatcherClassifier

register("template", TemplateMatcherClassifier)
register("cnn", CNNShapeClassifier)
register("embed", EmbedShapeClassifier)

__all__ = [
    "ShapeClassifier",
    "ShapeMatch",
    "TemplateMatcherClassifier",
    "CNNShapeClassifier",
    "EmbedShapeClassifier",
    "get_classifier",
    "register",
]
