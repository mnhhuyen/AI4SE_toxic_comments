from pathlib import Path

from toxic_comments.config import EMBEDDINGS_DIR
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


def build_models(
    max_features: int = 50_000,
    embedding_dir: Path = EMBEDDINGS_DIR,
    max_embedding_vectors: int | None = None,
    embedding_pooling: str = "mean",
):
    return {
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
    }
