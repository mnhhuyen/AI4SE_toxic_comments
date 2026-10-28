import pytest

from toxic_comments.config import LABEL_COLUMNS
from toxic_comments.models._optional import MissingDependencyEstimator
from toxic_comments.models.bigru import build_bigru_max_attention_classifier
from toxic_comments.models.dpcnn import build_dpcnn_classifier
from toxic_comments.models.registry import build_models


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


def test_dpcnn_scaffold_is_importable():
    try:
        build_dpcnn_classifier()
    except ImportError as exc:
        assert "tensorflow" in str(exc)
    except NotImplementedError as exc:
        assert "pipeline" in str(exc)


@pytest.mark.parametrize(
    "name", ["birnn_max_attention", "bilstm_max_attention", "bigru_shared_max_attention"]
)
def test_registry_includes_recurrent_models(name):
    estimator = build_models(max_features=20)[name]
    if isinstance(estimator, MissingDependencyEstimator):
        with pytest.raises(ImportError, match="torch"):
            estimator.fit(["a"], [[0] * len(LABEL_COLUMNS)])
    else:
        assert hasattr(estimator, "predict_proba")


@pytest.mark.parametrize(
    "name",
    ["bigru_max_attention", "bigru_max_attention_glove", "bigru_max_attention_random"],
)
def test_registry_includes_bigru_models(name):
    """Tolerates torch being absent, like roberta_label_dependency."""

    models = build_models(max_features=20)
    assert name in models

    estimator = models[name]
    if isinstance(estimator, MissingDependencyEstimator):
        with pytest.raises(ImportError, match="torch"):
            estimator.fit(["a"], [[0] * len(LABEL_COLUMNS)])
    else:
        assert hasattr(estimator, "predict_proba")


def test_bigru_builder_returns_estimator_or_clear_import_error():
    try:
        estimator = build_bigru_max_attention_classifier()
    except ImportError as exc:
        assert "torch" in str(exc)
    else:
        assert hasattr(estimator, "fit")
