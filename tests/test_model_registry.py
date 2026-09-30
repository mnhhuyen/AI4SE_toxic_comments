import pytest

from toxic_comments.config import LABEL_COLUMNS
from toxic_comments.models._optional import MissingDependencyEstimator
from toxic_comments.models.bigru import build_bigru_max_attention_classifier
from toxic_comments.models.bilstm import build_bilstm_max_attention_classifier
from toxic_comments.models.dpcnn import build_dpcnn_classifier
from toxic_comments.models.registry import build_models
from toxic_comments.models.rnn import build_rnn_classifier


def test_registry_includes_lightgbm_model_name():
    models = build_models(max_features=20)

    assert "tfidf_lightgbm" in models


def test_registry_includes_roberta_label_dependency():
    """Tolerates torch/transformers being absent, like the neural stubs below."""

    models = build_models(max_features=20)
    assert "roberta_label_dependency" in models

    estimator = models["roberta_label_dependency"]
    if isinstance(estimator, MissingDependencyEstimator):
        with pytest.raises(ImportError, match="transformers"):
            estimator.fit(["a"], [[0] * len(LABEL_COLUMNS)])
    else:
        assert hasattr(estimator, "fit")


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

