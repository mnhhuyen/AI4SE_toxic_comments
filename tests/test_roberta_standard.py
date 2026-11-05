"""Tests for the standard RoBERTa baseline (roberta_standard.py).

No network: ``AutoModel.from_pretrained`` is patched to return a tiny,
randomly initialised RoBERTa built from a config.
"""

from __future__ import annotations

import numpy as np
import pytest

from toxic_comments.config import LABEL_COLUMNS
from toxic_comments.models._optional import MissingDependencyEstimator
from toxic_comments.models.registry import build_models

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from sklearn.base import clone  # noqa: E402

from toxic_comments.models import roberta_standard  # noqa: E402
from toxic_comments.models.roberta_standard import RobertaStandardClassifier  # noqa: E402


@pytest.fixture
def tiny_encoder(monkeypatch):
    config = transformers.RobertaConfig(
        vocab_size=100, hidden_size=16, num_hidden_layers=1, num_attention_heads=2,
        intermediate_size=32, max_position_embeddings=64,
    )
    monkeypatch.setattr(
        roberta_standard.AutoModel, "from_pretrained", lambda *a, **k: transformers.RobertaModel(config)
    )
    return config


def test_forward_shape_and_gradients(tiny_encoder):
    model = RobertaStandardClassifier()._build_model(np.zeros((1, len(LABEL_COLUMNS))))
    input_ids = torch.randint(5, 100, (3, 10))
    logits = model(input_ids=input_ids, attention_mask=torch.ones_like(input_ids))

    assert logits.shape == (3, len(LABEL_COLUMNS))
    logits.sum().backward()
    assert model.classifier.weight.grad is not None


def test_head_is_a_single_linear_layer(tiny_encoder):
    model = RobertaStandardClassifier()._build_model(np.zeros((1, len(LABEL_COLUMNS))))
    assert isinstance(model.classifier, torch.nn.Linear)
    assert model.classifier.in_features == tiny_encoder.hidden_size
    assert model.classifier.out_features == len(LABEL_COLUMNS)


def test_classifier_is_clonable_without_being_fit():
    estimator = RobertaStandardClassifier(num_epochs=1, batch_size=4)
    cloned = clone(estimator)
    assert cloned.get_params() == estimator.get_params()
    assert cloned is not estimator


def test_same_default_hyperparameters_as_label_dependency():
    lab_dep = pytest.importorskip("toxic_comments.models.roberta_label_dependency")
    standard = RobertaStandardClassifier().get_params()
    dependency = lab_dep.RobertaLabelDependencyClassifier().get_params()
    shared = set(standard) & set(dependency)
    assert {k: standard[k] for k in shared} == {k: dependency[k] for k in shared}


def test_registry_includes_roberta_standard():
    models = build_models(max_features=20)
    assert "roberta_standard" in models
    estimator = models["roberta_standard"]
    if isinstance(estimator, MissingDependencyEstimator):
        with pytest.raises(ImportError, match="transformers"):
            estimator.fit(["a"], [[0] * len(LABEL_COLUMNS)])
    else:
        assert hasattr(estimator, "fit")
