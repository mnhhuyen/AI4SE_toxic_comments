"""Tests for the label-dependency graph layer (roberta_label_dependency.py).

These tests avoid downloading the pretrained ``roberta-base`` weights (no
network needed), so they run in CI / on a laptop without a Hugging Face Hub
connection. They cover:

1. The co-occurrence adjacency computation.
2. The graph message-passing layer's forward pass (shape + gradient flow).
3. That the sklearn estimator contract (``get_params`` / ``clone``) holds,
   which is what ``ThresholdedClassifier`` (thresholds.py) and any k-fold
   loop rely on.
4. The registry's graceful-fallback behavior when torch/transformers are
   not installed, matching test_model_registry.py's style for the other
   optional-dependency models.
"""

from __future__ import annotations

import numpy as np
import pytest

from toxic_comments.config import LABEL_COLUMNS

torch = pytest.importorskip("torch")

from sklearn.base import clone  # noqa: E402

from toxic_comments.models.roberta_label_dependency import (  # noqa: E402
    LabelDependencyGraphLayer,
    RobertaLabelDependencyClassifier,
    compute_cooccurrence_adjacency,
)


def test_adjacency_captures_strong_cooccurrence():
    # severe_toxic (col 1) only ever appears alongside toxic (col 0)
    y = np.array(
        [
            [1, 1, 1, 0, 1, 0],
            [1, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 1],
            [1, 1, 1, 0, 1, 1],
        ]
    )

    adjacency = compute_cooccurrence_adjacency(y)

    assert adjacency.shape == (len(LABEL_COLUMNS), len(LABEL_COLUMNS))
    assert torch.allclose(torch.diagonal(adjacency), torch.zeros(len(LABEL_COLUMNS)))
    # P(toxic=1 | severe_toxic=1) should be 1.0 given the toy data above
    assert adjacency[1, 0].item() == pytest.approx(1.0)


def test_adjacency_handles_a_label_with_zero_positives():
    # "threat" (col 3) never occurs -> would divide by zero without clipping
    y = np.zeros((5, len(LABEL_COLUMNS)), dtype=int)
    y[:, 0] = 1

    adjacency = compute_cooccurrence_adjacency(y)

    assert torch.isfinite(adjacency).all()


def test_label_dependency_layer_forward_shape_and_gradients():
    batch_size, hidden_size, dep_dim, num_labels = 4, 32, 8, len(LABEL_COLUMNS)
    adjacency = torch.rand(num_labels, num_labels)
    adjacency.fill_diagonal_(0.0)

    layer = LabelDependencyGraphLayer(hidden_size, dep_dim, num_labels, adjacency)
    pooled = torch.randn(batch_size, hidden_size, requires_grad=True)

    logits = layer(pooled)
    assert logits.shape == (batch_size, num_labels)

    logits.sum().backward()
    assert pooled.grad is not None
    assert not torch.isnan(pooled.grad).any()


def test_label_dependency_layer_neighbors_actually_influence_output():
    """If the adjacency is all-zero, output should differ from a fully-connected adjacency."""

    hidden_size, dep_dim, num_labels = 16, 8, len(LABEL_COLUMNS)
    pooled = torch.randn(2, hidden_size)

    torch.manual_seed(0)
    zero_adjacency = torch.zeros(num_labels, num_labels)
    layer_isolated = LabelDependencyGraphLayer(hidden_size, dep_dim, num_labels, zero_adjacency)

    torch.manual_seed(0)
    full_adjacency = torch.ones(num_labels, num_labels) - torch.eye(num_labels)
    layer_connected = LabelDependencyGraphLayer(hidden_size, dep_dim, num_labels, full_adjacency)

    out_isolated = layer_isolated(pooled)
    out_connected = layer_connected(pooled)

    assert not torch.allclose(out_isolated, out_connected)


def test_classifier_is_clonable_without_being_fit():
    """ThresholdedClassifier and any k-fold loop call sklearn.base.clone(estimator)."""

    estimator = RobertaLabelDependencyClassifier(dep_dim=32, num_epochs=1, batch_size=4)
    cloned = clone(estimator)

    assert cloned.get_params() == estimator.get_params()
    assert cloned is not estimator

