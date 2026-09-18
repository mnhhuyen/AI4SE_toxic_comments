"""DPCNN model factory."""

from __future__ import annotations

from toxic_comments.models._optional import missing_dependency


def build_dpcnn_classifier(*args, **kwargs):
    """Build a Deep Pyramid CNN classifier."""

    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:
        raise missing_dependency("tensorflow", "dpcnn_classifier") from exc

    raise NotImplementedError(
        "DPCNN needs a sequence/tokenizer training pipeline before it can be used "
        "with train_model()."
    )
