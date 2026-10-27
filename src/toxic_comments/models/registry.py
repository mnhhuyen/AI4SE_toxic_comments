from pathlib import Path

from toxic_comments.config import EMBEDDINGS_DIR
from toxic_comments.embeddings.fasttext import FASTTEXT_VECTOR_FILE
from toxic_comments.embeddings.glove import GLOVE_TWITTER_VECTOR_FILE
from toxic_comments.models._optional import MissingDependencyEstimator
from toxic_comments.models.baseline import build_dummy_baseline
from toxic_comments.models.embedding_logreg import (
    build_bpemb_logistic_regression,
    build_fasttext_logistic_regression,
    build_glove_twitter_logistic_regression,
    build_lexvec_logistic_regression,
    build_word2vec_logistic_regression,
)
from toxic_comments.models.lightgbm import build_tfidf_lightgbm_classifier
from toxic_comments.models.linear import (
    build_raw_text_logistic_regression,
    build_tfidf_complement_nb,
    build_tfidf_linear_svc,
    build_tfidf_sgd_logistic,
    build_word_char_logistic_regression,
)
from toxic_comments.models.nbsvm import build_nbsvm
from toxic_comments.models.tfidf_logreg import build_tfidf_logistic_regression


def _build_roberta_label_dependency(device: str | None):
    """Deferred import: roberta_label_dependency.py defines torch.nn.Module
    subclasses, so — unlike models/dpcnn.py's deferred-import-inside-function
    pattern — the try/except has to wrap the whole module import, not just a
    call inside an already-imported module."""

    try:
        from toxic_comments.models.roberta_label_dependency import (
            build_roberta_label_dependency,
        )
    except ImportError:
        return MissingDependencyEstimator("transformers", "roberta_label_dependency")
    return build_roberta_label_dependency(device=device)


def _build_bigru(
    embedding_path: Path | None,
    max_embedding_vectors: int | None,
    device: str | None,
    model_name: str,
):
    """Deferred import like ``_build_roberta_label_dependency``: building the
    estimator is cheap (embeddings load inside ``fit``), and machines without
    torch get a ``MissingDependencyEstimator`` instead of an import error."""

    try:
        from toxic_comments.models.bigru import build_bigru_max_attention_classifier

        return build_bigru_max_attention_classifier(
            embedding_path=embedding_path,
            max_embedding_vectors=max_embedding_vectors,
            device=device,
        )
    except ImportError:
        return MissingDependencyEstimator("torch", model_name)


def build_models(
    max_features: int = 50_000,
    embedding_dir: Path = EMBEDDINGS_DIR,
    max_embedding_vectors: int | None = None,
    embedding_pooling: str = "mean",
    include_transformer_models: bool = True,
    device: str | None = None,
):
    """Return the registry of models to evaluate.

    ``roberta_label_dependency`` is registered unconditionally (like every
    other model here) so that ``train.py``'s ``train_model(model_name=...)``
    can find it by name — it doesn't pass ``include_transformer_models`` at
    all, so a boolean gate here would silently hide the model from that
    workflow. ``include_transformer_models`` is kept, defaulting to True,
    only for backward compatibility with ``experiment.py``'s call site,
    which uses it to decide which models to *iterate over*, not which ones
    exist in this dict; passing ``include_transformer_models=False`` no
    longer removes the model from here, it just means callers relying on
    the old gating behavior should filter on the returned dict themselves.
    """

    models = {
        "dummy_most_frequent": build_dummy_baseline(),
        "tfidf_logistic_regression": build_tfidf_logistic_regression(
            max_features=max_features
        ),
        "tfidf_lightgbm": build_tfidf_lightgbm_classifier(
            max_features=max_features
        ),
        "word_char_logistic_regression": build_word_char_logistic_regression(
            max_features=max_features
        ),
        "raw_text_logistic_regression": build_raw_text_logistic_regression(
            max_features=max_features
        ),
        "nbsvm": build_nbsvm(max_features=max_features),
        "tfidf_linear_svc": build_tfidf_linear_svc(max_features=max_features),
        "tfidf_sgd_logistic": build_tfidf_sgd_logistic(max_features=max_features),
        "tfidf_complement_nb": build_tfidf_complement_nb(max_features=max_features),
        "fasttext_logistic_regression": build_fasttext_logistic_regression(
            embedding_dir=embedding_dir,
            max_vectors=max_embedding_vectors,
            pooling=embedding_pooling,
        ),
        "glove_twitter_logistic_regression": build_glove_twitter_logistic_regression(
            embedding_dir=embedding_dir,
            max_vectors=max_embedding_vectors,
            pooling=embedding_pooling,
        ),
        "bpemb_logistic_regression": build_bpemb_logistic_regression(
            embedding_dir=embedding_dir,
            max_vectors=max_embedding_vectors,
            pooling=embedding_pooling,
        ),
        "word2vec_logistic_regression": build_word2vec_logistic_regression(
            embedding_dir=embedding_dir,
            max_vectors=max_embedding_vectors,
            pooling=embedding_pooling,
        ),
        "lexvec_logistic_regression": build_lexvec_logistic_regression(
            embedding_dir=embedding_dir,
            max_vectors=max_embedding_vectors,
            pooling=embedding_pooling,
        ),
        "roberta_label_dependency": _build_roberta_label_dependency(device=device),
        "bigru_max_attention": _build_bigru(
            embedding_path=embedding_dir / FASTTEXT_VECTOR_FILE,
            max_embedding_vectors=max_embedding_vectors,
            device=device,
            model_name="bigru_max_attention",
        ),
        "bigru_max_attention_glove": _build_bigru(
            embedding_path=embedding_dir / GLOVE_TWITTER_VECTOR_FILE,
            max_embedding_vectors=max_embedding_vectors,
            device=device,
            model_name="bigru_max_attention_glove",
        ),
        "bigru_max_attention_random": _build_bigru(
            embedding_path=None,
            max_embedding_vectors=None,
            device=device,
            model_name="bigru_max_attention_random",
        ),
    }
    return models

