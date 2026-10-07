"""Bi-LSTM + max/attention pooling model factory.

See ``_rnn_base.py`` for the shared architecture.
"""

from __future__ import annotations

from pathlib import Path

from toxic_comments.config import EMBEDDINGS_DIR
from toxic_comments.embeddings.fasttext import FASTTEXT_VECTOR_FILE
from toxic_comments.models._optional import MissingDependencyEstimator


def build_bilstm_max_attention_classifier(
    embedding_dir: Path = EMBEDDINGS_DIR,
    embedding_file: str | None = FASTTEXT_VECTOR_FILE,
    device: str | None = None,
    **kwargs,
):
    """Build a Bi-LSTM classifier initialised with pretrained vectors.

    Pass ``embedding_file=None`` to train the embedding layer from scratch.
    Extra keyword arguments go to ``RecurrentMultiLabelClassifier``.
    """

    try:
        from toxic_comments.models._rnn_base import RecurrentMultiLabelClassifier
    except ImportError:
        return MissingDependencyEstimator("torch", "bilstm_max_attention")

    embedding_path = None if embedding_file is None else embedding_dir / embedding_file
    return RecurrentMultiLabelClassifier(
        cell_type="lstm", embedding_path=embedding_path, device=device, **kwargs
    )
