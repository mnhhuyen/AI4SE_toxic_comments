from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest

from toxic_comments.config import LABEL_COLUMNS

torch = pytest.importorskip("torch")

from sklearn.base import clone  # noqa: E402

from toxic_comments.models._dpcnn_torch import (  # noqa: E402
    DPCNNMultiLabelClassifier,
    _DPCNNNet,
    num_downsampling_blocks,
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


def _small_model(**kwargs) -> DPCNNMultiLabelClassifier:
    params = dict(
        embedding_dim=8, num_filters=6, max_length=16, batch_size=8, num_epochs=1, device="cpu"
    )
    params.update(kwargs)
    return DPCNNMultiLabelClassifier(**params)


@pytest.mark.parametrize(
    "max_length, expected", [(2, 1), (16, 4), (17, 5), (200, 8), (400, 9)]
)
def test_num_downsampling_blocks_reaches_length_one(max_length, expected):
    assert num_downsampling_blocks(max_length) == expected


@pytest.mark.parametrize("max_length", [2, 16, 17, 200])
def test_pyramid_shrinks_sequence_to_length_one(max_length):
    net = _DPCNNNet(
        vocab_size=10, embedding_dim=4, num_filters=3, kernel_size=3,
        num_blocks=num_downsampling_blocks(max_length), num_labels=6, dropout=0.0,
    )
    ids = torch.randint(0, 10, (2, max_length))

    x = net.first_block(net.region_embedding(net.embedding(ids).transpose(1, 2)))
    assert x.shape == (2, 3, max_length)
    for block in net.blocks:
        x = block(net.pool(x))
    assert x.shape == (2, 3, 1)
    assert net(ids).shape == (2, 6)


def test_fit_predict_shapes():
    model = _small_model().fit(TEXTS, Y)

    scores = model.predict_proba(TEXTS)

    assert scores.shape == Y.shape
    assert ((scores >= 0) & (scores <= 1)).all()
    assert set(np.unique(model.predict(TEXTS))) <= {0, 1}
    assert len(model.history_) == 1


def test_longer_texts_are_truncated_to_max_length():
    long_text = pd.Series([" ".join(["idiot"] * 100)])
    model = _small_model(max_length=8).fit(TEXTS, Y)

    assert model.predict_proba(long_text).shape == (1, len(LABEL_COLUMNS))


def test_training_reduces_loss():
    model = _small_model(num_epochs=15, learning_rate=1e-2, dropout=0.0).fit(TEXTS, Y)

    assert model.history_[-1] < model.history_[0]


def test_even_kernel_is_rejected():
    with pytest.raises(ValueError, match="kernel_size"):
        _small_model(kernel_size=4).fit(TEXTS, Y)


def test_clone_keeps_params_and_drops_fitted_state():
    model = _small_model(num_filters=7).fit(TEXTS, Y)

    cloned = clone(model)

    assert cloned.get_params() == model.get_params()
    assert not hasattr(cloned, "model_")


def test_joblib_round_trip_preserves_predictions(tmp_path):
    model = _small_model().fit(TEXTS, Y)
    path = tmp_path / "dpcnn.joblib"

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


def test_works_inside_thresholded_classifier():
    model = ThresholdedClassifier(_small_model()).fit(TEXTS, Y)

    assert model.predict(TEXTS).shape == Y.shape
