"""DPCNN model factory."""

from __future__ import annotations

from pathlib import Path

from toxic_comments.config import EMBEDDINGS_DIR
from toxic_comments.embeddings.fasttext import FASTTEXT_VECTOR_FILE
from toxic_comments.models._optional import MissingDependencyEstimator


def build_dpcnn_classifier(
    embedding_dir: Path = EMBEDDINGS_DIR,
    embedding_file: str | None = FASTTEXT_VECTOR_FILE,
    device: str | None = None,
    **kwargs,
):
    """Build a Deep Pyramid CNN classifier."""

    try:
        from toxic_comments.models._dpcnn_torch import DPCNNMultiLabelClassifier
    except ImportError:
        return MissingDependencyEstimator("torch", "dpcnn")

    embedding_path = None if embedding_file is None else embedding_dir / embedding_file
    return DPCNNMultiLabelClassifier(embedding_path=embedding_path, device=device, **kwargs)
