"""Bi-GRU model factory."""

from __future__ import annotations

from toxic_comments.models._optional import missing_dependency


def build_bigru_max_attention_classifier(*args, **kwargs):
    """Build Bi-GRU with max + attention pooling."""

    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:
        raise missing_dependency("tensorflow", "bigru_max_attention_classifier") from exc

    raise NotImplementedError(
        "Bi-GRU max+attention needs a sequence/tokenizer training pipeline before "
        "it can be used with train_model()."
    )
