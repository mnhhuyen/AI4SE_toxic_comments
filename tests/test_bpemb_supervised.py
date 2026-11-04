"""Supervised BPEmb pooling without downloads."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from toxic_comments.embeddings.bpemb import BPEmbVectorizer


@pytest.mark.parametrize("pooling,width", [
    ("supervised_attention", 3),
    ("max_supervised_attention", 6),
])
def test_supervised_bpemb(monkeypatch, pooling, width):
    torch = pytest.importorskip("torch")
    from toxic_comments.embeddings import pooling as pooling_module
    original = pooling_module._SupervisedAttentionNet
    captured = {}

    def capture(*args, **kwargs):
        captured["model"] = original(*args, **kwargs)
        return captured["model"]

    monkeypatch.setattr(pooling_module, "_SupervisedAttentionNet", capture)

    class FakeBPEmb:
        def __init__(self, **kwargs):
            self.dim = 3
            self.vectors = np.eye(3, dtype=np.float32)
            self.emb = self.vectors

        def encode_ids(self, text):
            return [int(token) for token in text.split()]

        def embed(self, text):
            return self.vectors[self.encode_ids(text)]

    monkeypatch.setitem(sys.modules, "bpemb", SimpleNamespace(BPEmb=FakeBPEmb))
    texts = ["0", "1 2", "", "2 0"]
    targets = np.array([[1, 0], [0, 1], [0, 0], [1, 1]])
    vectorizer = BPEmbVectorizer(pooling=pooling).fit(texts, targets)
    result = vectorizer.transform(texts)
    assert result.shape == (4, width)
    assert np.isfinite(result).all()
    assert np.isfinite(vectorizer.loss_history_).all()
    np.testing.assert_array_equal(result[2], np.zeros(width))
    # ID zero must remain a real embedding, not be discarded as padding.
    np.testing.assert_allclose(result[0, -3:], [1, 0, 0])
    np.testing.assert_allclose(
        vectorizer.transform([" ".join(["0"] * 200 + ["1", "2"])])[0], result[0],
        atol=1e-6,
    )
    model = captured["model"].eval()
    assert model.classifier.in_features == width
    with torch.no_grad():
        expected = model.pool(torch.tensor([[1, 0, 0], [2, 3, 0], [0, 0, 0], [3, 1, 0]])).numpy()
    np.testing.assert_allclose(result, expected, atol=1e-6)
