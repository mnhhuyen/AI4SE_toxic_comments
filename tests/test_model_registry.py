import pytest

from toxic_comments.models.bigru import build_bigru_max_attention_classifier
from toxic_comments.models.bilstm import build_bilstm_max_attention_classifier
from toxic_comments.models.dpcnn import build_dpcnn_classifier
from toxic_comments.models.registry import build_models
from toxic_comments.models.rnn import build_rnn_classifier


def test_registry_includes_lightgbm_model_name():
    models = build_models(max_features=20)

    assert "tfidf_lightgbm" in models


@pytest.mark.parametrize(
    "builder",
    [
        build_rnn_classifier,
        build_bigru_max_attention_classifier,
        build_bilstm_max_attention_classifier,
        build_dpcnn_classifier,
    ],
)
def test_neural_model_scaffolds_are_importable(builder):
    try:
        builder()
    except ImportError as exc:
        assert "tensorflow" in str(exc)
    except NotImplementedError as exc:
        assert "pipeline" in str(exc)
