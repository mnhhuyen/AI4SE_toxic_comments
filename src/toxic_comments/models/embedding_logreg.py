"""Embedding-average + logistic regression model factories."""

from __future__ import annotations

from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from toxic_comments.config import EMBEDDINGS_DIR
from toxic_comments.embeddings.bpemb import build_bpemb_vectorizer
from toxic_comments.embeddings.fasttext import build_fasttext_vectorizer
from toxic_comments.embeddings.glove import build_glove_twitter_vectorizer
from toxic_comments.embeddings.lexvec import build_lexvec_vectorizer
from toxic_comments.embeddings.pooling import MeanEmbeddingVectorizer
from toxic_comments.embeddings.word2vec import build_word2vec_vectorizer


def build_embedding_logistic_regression(
    vectorizer: MeanEmbeddingVectorizer,
    c_value: float = 4.0,
) -> Pipeline:
    """Return a mean-pooled embedding + one-vs-rest logistic regression classifier."""

    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=c_value,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
    )
    return Pipeline(
        steps=[
            ("embeddings", vectorizer),
            ("scaler", StandardScaler()),
            ("classifier", classifier),
        ]
    )


def build_fasttext_logistic_regression(
    embedding_dir: Path = EMBEDDINGS_DIR,
    max_vectors: int | None = None,
    pooling: str = "mean",
) -> Pipeline:
    return build_embedding_logistic_regression(
        build_fasttext_vectorizer(
            embedding_dir=embedding_dir,
            max_vectors=max_vectors,
            pooling=pooling,
        )
    )


def build_glove_twitter_logistic_regression(
    embedding_dir: Path = EMBEDDINGS_DIR,
    max_vectors: int | None = None,
    pooling: str = "mean",
) -> Pipeline:
    return build_embedding_logistic_regression(
        build_glove_twitter_vectorizer(
            embedding_dir=embedding_dir,
            max_vectors=max_vectors,
            pooling=pooling,
        )
    )


def build_bpemb_logistic_regression(
    embedding_dir: Path = EMBEDDINGS_DIR,
    max_vectors: int | None = None,
    pooling: str = "mean",
) -> Pipeline:
    return build_embedding_logistic_regression(
        build_bpemb_vectorizer(
            embedding_dir=embedding_dir,
            max_vectors=max_vectors,
            pooling=pooling,
        )
    )


def build_word2vec_logistic_regression(
    embedding_dir: Path = EMBEDDINGS_DIR,
    max_vectors: int | None = None,
    pooling: str = "mean",
) -> Pipeline:
    return build_embedding_logistic_regression(
        build_word2vec_vectorizer(
            embedding_dir=embedding_dir,
            max_vectors=max_vectors,
            pooling=pooling,
        )
    )


def build_lexvec_logistic_regression(
    embedding_dir: Path = EMBEDDINGS_DIR,
    max_vectors: int | None = None,
    pooling: str = "mean",
) -> Pipeline:
    return build_embedding_logistic_regression(
        build_lexvec_vectorizer(
            embedding_dir=embedding_dir,
            max_vectors=max_vectors,
            pooling=pooling,
        )
    )
