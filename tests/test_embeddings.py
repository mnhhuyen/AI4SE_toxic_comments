import numpy as np
import pandas as pd
import pytest

from toxic_comments.config import HEAVY_TEXT_COLUMN, LABEL_COLUMNS
from toxic_comments.embeddings.pooling import (
    AttentionEmbeddingVectorizer,
    MaxAttentionEmbeddingVectorizer,
    MaxEmbeddingVectorizer,
    MeanEmbeddingVectorizer,
)
from toxic_comments.train import train_model


def test_max_attention_empty_and_unknown_text(tmp_path):
    vectorizer = MaxAttentionEmbeddingVectorizer(_write_tiny_glove(tmp_path))
    vectors = vectorizer.fit_transform(["nice", "unknown", "", None])
    assert vectors.shape == (4, 6)
    assert np.all(vectors[1:] == 0)
    assert vectorizer.transform([]).shape == (0, 6)


@pytest.mark.parametrize("pooling", ["mean", "max", "attention", "max_attention"])
def test_bpemb_uses_subwords_and_supports_clone(tmp_path, monkeypatch, pooling):
    import sys
    from types import SimpleNamespace
    from sklearn.base import clone
    from toxic_comments.embeddings.bpemb import BPEmbVectorizer

    calls = []

    class FakeBPEmb:
        def __init__(self, **kwargs):
            self.dim = 3
            self.emb = object()

        def embed(self, text):
            calls.append(text)
            if not text:
                return np.empty((0, 3), dtype=np.float32)
            return np.array([[1, 2, 3], [3, 2, 1]], dtype=np.float32)

    monkeypatch.setitem(sys.modules, "bpemb", SimpleNamespace(BPEmb=FakeBPEmb))
    vectorizer = clone(BPEmbVectorizer(embedding_dir=tmp_path, pooling=pooling))
    result = vectorizer.fit_transform(["unseenword", ""])
    assert "unseenword" in calls
    assert result.shape == (2, 6 if pooling == "max_attention" else 3)
    assert np.all(result[1] == 0)
    expected = [3, 2, 3] if pooling == "max" else [2, 2, 2]
    if pooling == "max_attention":
        expected = [3, 2, 3, 2, 2, 2]
    np.testing.assert_allclose(result[0], expected, rtol=1e-6)


def _write_tiny_glove(embedding_dir):
    glove_dir = embedding_dir / "glove_twitter"
    glove_dir.mkdir(parents=True)
    path = glove_dir / "glove.twitter.27B.200d.txt"
    path.write_text(
        "\n".join(
            [
                "nice 0.1 0.2 0.3",
                "awful -0.1 -0.2 -0.3",
                "hate -0.2 -0.1 -0.4",
                "thanks 0.2 0.1 0.4",
                "you 0.0 0.1 0.0",
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_mean_embedding_vectorizer_loads_and_pools_vectors(tmp_path):
    embedding_path = _write_tiny_glove(tmp_path)
    vectorizer = MeanEmbeddingVectorizer(embedding_path)

    vectors = vectorizer.fit_transform(["nice thanks", "unknown"])

    assert vectors.shape == (2, 3)
    assert np.allclose(vectors[0], [0.15, 0.15, 0.35])
    assert np.allclose(vectors[1], [0.0, 0.0, 0.0])


def test_embedding_vectorizers_support_max_and_attention_pooling(tmp_path):
    embedding_path = _write_tiny_glove(tmp_path)

    max_vectors = MaxEmbeddingVectorizer(embedding_path).fit_transform(["nice thanks"])
    attention_vectors = AttentionEmbeddingVectorizer(embedding_path).fit_transform(["nice thanks"])
    max_attention_vectors = MaxAttentionEmbeddingVectorizer(embedding_path).fit_transform(["nice thanks"])

    assert max_vectors.shape == (1, 3)
    assert attention_vectors.shape == (1, 3)
    assert max_attention_vectors.shape == (1, 6)


def test_train_model_supports_glove_twitter_embeddings(tmp_path):
    _write_tiny_glove(tmp_path)
    data = pd.DataFrame(
        {
            HEAVY_TEXT_COLUMN: [
                "nice thanks",
                "you awful",
                "thanks nice",
                "hate you",
            ],
            **{label: [0, 1, 0, 1] for label in LABEL_COLUMNS},
        }
    )

    estimator = train_model(
        data,
        model_name="glove_twitter_logistic_regression",
        embedding_dir=tmp_path,
        embedding_pooling="max_attention",
    )

    assert hasattr(estimator, "predict")
