"""Bi-GRU model factory.

The PyTorch implementation lives in ``_bigru_torch.py``; this module stays
importable without torch so the registry and tests keep working on machines
that only run the classical models.
"""

from __future__ import annotations

from pathlib import Path

from toxic_comments.models._optional import missing_dependency


def build_bigru_max_attention_classifier(
    embedding_path: str | Path | None = None,
    max_embedding_vectors: int | None = None,
    freeze_embeddings: bool = True,
    max_length: int = 200,
    hidden_size: int = 64,
    num_layers: int = 2,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    num_epochs: int = 4,
    use_pos_weight: bool = False,
    device: str | None = None,
    **kwargs,
):
    """Build Bi-GRU with max + attention pooling.

    ``embedding_path=None`` trains embeddings from scratch; otherwise a
    word-vector text file (FastText ``.vec`` / GloVe ``.txt``) is loaded at
    ``fit`` time, so building the estimator is cheap.
    """

    try:
        from toxic_comments.models._bigru_torch import BiGRUMaxAttentionClassifier
    except ImportError as exc:
        raise missing_dependency("torch", "bigru_max_attention_classifier") from exc

    return BiGRUMaxAttentionClassifier(
        embedding_path=str(embedding_path) if embedding_path is not None else None,
        max_embedding_vectors=max_embedding_vectors,
        freeze_embeddings=freeze_embeddings,
        max_length=max_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        use_pos_weight=use_pos_weight,
        device=device,
        **kwargs,
    )
