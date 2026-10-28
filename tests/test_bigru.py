"""Tests for the Bi-GRU max+attention classifier (skipped without torch)."""

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")

import joblib  # noqa: E402
from sklearn.base import clone  # noqa: E402

from toxic_comments.config import LABEL_COLUMNS  # noqa: E402
from toxic_comments.models.bigru import build_bigru_max_attention_classifier  # noqa: E402

TEXTS = [
    "you are a stupid idiot",
    "thank you for the helpful edit",
    "i will kill you",
    "great article, well sourced",
    "shut up you moron",
    "please add a citation here",
    "",
    "idiot idiot idiot",
] * 4
LABELS = np.array(
    [
        [1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0],
    ]
    * 4
)


def _small_model(**overrides):
    params = dict(
        embedding_path=None,
        max_length=16,
        hidden_size=8,
        num_layers=1,
        batch_size=8,
        num_epochs=2,
        device="cpu",
        verbose=False,
    )
    params.update(overrides)
    return build_bigru_max_attention_classifier(**params)


def test_fit_predict_shapes_and_ranges():
    model = _small_model().fit(pd.Series(TEXTS), LABELS)
    proba = model.predict_proba(TEXTS[:5])
    assert proba.shape == (5, len(LABEL_COLUMNS))
    assert np.all((proba >= 0) & (proba <= 1))
    assert set(np.unique(model.predict(TEXTS[:5]))) <= {0, 1}
    assert len(model.loss_history_) == 2


def test_handles_empty_and_unknown_text():
    model = _small_model().fit(TEXTS, LABELS)
    proba = model.predict_proba(["", "completely unseen tokens zzz"])
    assert proba.shape == (2, len(LABEL_COLUMNS))
    assert np.isfinite(proba).all()


def test_sklearn_clone_keeps_params_and_is_unfitted():
    model = _small_model(hidden_size=12)
    cloned = clone(model)
    assert cloned.get_params()["hidden_size"] == 12
    assert not cloned.__sklearn_is_fitted__()


def test_joblib_roundtrip_gives_same_predictions(tmp_path):
    model = _small_model().fit(TEXTS, LABELS)
    path = tmp_path / "bigru.joblib"
    joblib.dump(model, path)
    restored = joblib.load(path)
    np.testing.assert_allclose(model.predict_proba(TEXTS), restored.predict_proba(TEXTS), atol=1e-6)


def test_pretrained_embedding_file_is_used(tmp_path):
    vec = tmp_path / "tiny.vec"
    rng = np.random.default_rng(0)
    words = ["you", "stupid", "idiot", "thank", "kill", "article"]
    lines = ["%d 5" % len(words)] + [w + " " + " ".join(f"{v:.4f}" for v in rng.normal(size=5)) for w in words]
    vec.write_text("\n".join(lines) + "\n", encoding="utf-8")

    model = _small_model(embedding_path=str(vec)).fit(TEXTS, LABELS)
    assert model.embedding_dim_ == 5
    assert 0 < model.embedding_coverage_ <= 1
    assert not model.model_.embedding.weight.requires_grad  # frozen by default


def test_explain_returns_attention_per_token():
    model = _small_model().fit(TEXTS, LABELS)
    frame = model.explain("you are a stupid idiot")
    assert list(frame.columns) == ["token", "attention"]
    assert len(frame) == 5
    assert frame["attention"].sum() == pytest.approx(1.0, abs=1e-5)
