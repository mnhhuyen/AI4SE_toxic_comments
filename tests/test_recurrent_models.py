"""Tests for the Bi-RNN/Bi-LSTM/Bi-GRU classifiers (models/_rnn_base.py).

Trained on a tiny toy dataset on CPU, with a small hand-written embedding
file, so no pretrained vectors or GPU are needed.
"""

from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from toxic_comments.config import LABEL_COLUMNS

torch = pytest.importorskip("torch")

from sklearn.base import clone  # noqa: E402

from toxic_comments.models._rnn_base import (  # noqa: E402
    RecurrentMultiLabelClassifier,
    load_vocab_embeddings,
)
from toxic_comments.thresholds import ThresholdedClassifier  # noqa: E402

TEXTS = pd.Series(
    [
        "you are a stupid idiot",
        "thanks for the helpful edit",
        "i will kill you",
        "great article well written",
        "idiot stupid fool",
        "",
    ]
    * 4
)
Y = np.zeros((len(TEXTS), len(LABEL_COLUMNS)), dtype=int)
Y[TEXTS.str.contains("idiot|kill"), 0] = 1
Y[TEXTS.str.contains("kill"), 3] = 1


def _small_model(**kwargs) -> RecurrentMultiLabelClassifier:
    params = dict(embedding_dim=8, hidden_size=4, batch_size=8, num_epochs=1, device="cpu")
    params.update(kwargs)
    return RecurrentMultiLabelClassifier(**params)


@pytest.mark.parametrize("cell_type", ["rnn", "lstm", "gru"])
def test_fit_predict_shapes(cell_type):
    model = _small_model(cell_type=cell_type).fit(TEXTS, Y)

    scores = model.predict_proba(TEXTS)

    assert scores.shape == Y.shape
    assert ((scores >= 0) & (scores <= 1)).all()
    assert set(np.unique(model.predict(TEXTS))) <= {0, 1}
    assert len(model.history_) == 1


def test_unknown_cell_type_is_rejected():
    with pytest.raises(ValueError, match="cell_type"):
        _small_model(cell_type="transformer").fit(TEXTS, Y)


def test_clone_keeps_params_and_drops_fitted_state():
    model = _small_model(cell_type="lstm", hidden_size=7).fit(TEXTS, Y)

    cloned = clone(model)

    assert cloned.get_params() == model.get_params()
    assert not hasattr(cloned, "model_")


def test_joblib_round_trip_preserves_predictions(tmp_path):
    model = _small_model().fit(TEXTS, Y)
    path = tmp_path / "rnn.joblib"

    joblib.dump(model, path)
    loaded = joblib.load(path)

    np.testing.assert_allclose(loaded.predict_proba(TEXTS), model.predict_proba(TEXTS), rtol=1e-5)


def test_pretrained_embeddings_are_loaded_and_frozen(tmp_path):
    path = tmp_path / "vectors.vec"
    path.write_text("2 3\nidiot 1 2 3\nthanks 4 5 6\n", encoding="utf-8")

    model = _small_model(embedding_path=path).fit(TEXTS, Y)

    weight = model.model_.embedding.weight
    assert weight.shape[1] == 3
    assert not weight.requires_grad
    torch.testing.assert_close(weight[model.vocab_["idiot"]], torch.tensor([1.0, 2.0, 3.0]))
    assert 0 < model.embedding_coverage_ < 1


def test_load_vocab_embeddings_zeroes_padding_row(tmp_path):
    path = tmp_path / "vectors.vec"
    path.write_text("cat 1 1\ndog 2 2\n", encoding="utf-8")

    matrix, coverage = load_vocab_embeddings(path, {"cat": 2, "bird": 3})

    assert matrix.shape == (4, 2)
    assert (matrix[0] == 0).all()
    assert (matrix[2] == 1).all()
    assert coverage == 0.5


def test_works_inside_thresholded_classifier():
    model = ThresholdedClassifier(_small_model()).fit(TEXTS, Y)

    assert model.predict(TEXTS).shape == Y.shape


def test_token_attention_sums_to_one():
    model = _small_model().fit(TEXTS, Y)

    attention = model.token_attention("you stupid zzzunseen")

    assert list(attention["token"]) == ["you", "stupid", "zzzunseen"]
    assert attention["attention"].sum() == pytest.approx(1.0, abs=1e-3)
    assert not attention.loc[2, "in_vocab"]
