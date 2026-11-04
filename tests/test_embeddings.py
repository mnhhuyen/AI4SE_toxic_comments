import numpy as np
import pandas as pd
import pytest

from toxic_comments.config import HEAVY_TEXT_COLUMN, LABEL_COLUMNS
from toxic_comments.embeddings.pooling import (
    AttentionEmbeddingVectorizer,
    MaxAttentionEmbeddingVectorizer,
    MaxEmbeddingVectorizer,
    MeanEmbeddingVectorizer,
    SupervisedAttentionEmbeddingVectorizer,
    MaxSupervisedAttentionEmbeddingVectorizer,
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


@pytest.mark.parametrize(
    "vectorizer_class,expected_dim",
    [(SupervisedAttentionEmbeddingVectorizer, 3), (MaxSupervisedAttentionEmbeddingVectorizer, 6)],
)
def test_supervised_attention_pooling(tmp_path, vectorizer_class, expected_dim):
    pytest.importorskip("torch")
    from sklearn.base import clone
    import pickle

    path = _write_tiny_glove(tmp_path)
    texts = ["nice thanks", "awful hate", "nice", "hate thanks"]
    labels = np.array([[0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0],
                       [0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0]], dtype=np.float32)
    vectorizer = vectorizer_class(path, num_epochs=1, batch_size=2, random_state=7)
    result = vectorizer.fit_transform(texts, labels)

    assert result.shape == (4, expected_dim)
    assert vectorizer.loss_history_
    assert clone(vectorizer).num_epochs == 1
    restored = pickle.loads(pickle.dumps(vectorizer))
    np.testing.assert_allclose(restored.transform(["unknown"]), np.zeros((1, expected_dim)))
    np.testing.assert_allclose(restored.transform(texts), vectorizer.transform(texts))


@pytest.mark.parametrize("vectorizer_class", [
    SupervisedAttentionEmbeddingVectorizer, MaxSupervisedAttentionEmbeddingVectorizer,
])
def test_supervised_train_transform_parity(tmp_path, monkeypatch, vectorizer_class):
    torch = pytest.importorskip("torch")
    from toxic_comments.embeddings import pooling

    original = pooling._SupervisedAttentionNet
    captured = {}

    def capture(*args, **kwargs):
        captured["model"] = original(*args, **kwargs)
        return captured["model"]

    monkeypatch.setattr(pooling, "_SupervisedAttentionNet", capture)
    texts = ["awful hate", "nice thanks", "unknown", ""]
    vectorizer = vectorizer_class(
        _write_tiny_glove(tmp_path), max_length=2, num_epochs=1,
        batch_size=4, device="cpu",
    ).fit(texts, np.array([[1, 0], [0, 1], [0, 0], [0, 0]], dtype=np.float32))
    model = captured["model"].eval()
    ids = torch.tensor([[vectorizer.vocab_["awful"], vectorizer.vocab_["hate"], 0],
                        [0, 0, 0]])
    with torch.no_grad():
        expected = model.pool(ids).numpy()
    actual = vectorizer.transform(["unknown awful hate nice thanks", ""])
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    assert model.classifier.in_features == vectorizer.output_dim_
    assert np.isfinite(vectorizer.loss_history_).all()
    assert np.isfinite(expected).all()
    np.testing.assert_array_equal(expected[1], np.zeros(vectorizer.output_dim_))
    if vectorizer_class is MaxSupervisedAttentionEmbeddingVectorizer:
        # Zero padding must not beat negative real token values in the max branch.
        np.testing.assert_allclose(expected[0, :3], [-0.1, -0.1, -0.3])
    # A pretrained token absent from training is still available at inference.
    np.testing.assert_allclose(vectorizer.transform(["you"])[0, -3:], [0, 0.1, 0])


@pytest.mark.parametrize("max_length", [0, -1, 1.5])
def test_supervised_rejects_invalid_length(tmp_path, max_length):
    vectorizer = SupervisedAttentionEmbeddingVectorizer("unused", max_length=max_length)
    with pytest.raises(ValueError, match="max_length"):
        vectorizer.fit(["text"], [[1]])


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
