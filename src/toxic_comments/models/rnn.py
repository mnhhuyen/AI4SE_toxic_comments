"""Plain RNN model factory.

This file is intentionally separate from the sklearn models because neural
models need a tokenization/sequence pipeline before they can replace the
current TF-IDF or embedding-average baselines.
"""

from __future__ import annotations

from toxic_comments.models._optional import missing_dependency


def build_rnn_classifier(*args, **kwargs):
    """Build a plain RNN classifier once the TensorFlow pipeline is added."""

    try:
        import tensorflow as tf  # noqa: F401
    except ImportError as exc:
        raise missing_dependency("tensorflow", "rnn_classifier") from exc

    raise NotImplementedError(
        "RNN needs a sequence/tokenizer training pipeline before it can be used "
        "with train_model()."
    )
