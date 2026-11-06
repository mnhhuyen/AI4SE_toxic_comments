"""LightGBM model factories."""

from __future__ import annotations

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

from toxic_comments.config import EMBEDDINGS_DIR
from toxic_comments.embeddings.fasttext import build_fasttext_vectorizer
from toxic_comments.models._optional import MissingDependencyEstimator


def _lightgbm_classifier(LGBMClassifier) -> MultiOutputClassifier:
    return MultiOutputClassifier(
        LGBMClassifier(
            objective="binary",
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            random_state=42,
            n_jobs=1,
            verbose=-1,
        ),
        n_jobs=-1,
    )


def build_tfidf_lightgbm_classifier(
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
) -> Pipeline:
    """Return TF-IDF + multi-output LightGBM classifier."""

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return MissingDependencyEstimator("lightgbm", "tfidf_lightgbm_classifier")

    return Pipeline(
        steps=[
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    token_pattern=r"(?u)\b\w\w+\b",
                    ngram_range=ngram_range,
                    max_features=max_features,
                    min_df=1,
                ),
            ),
            ("classifier", _lightgbm_classifier(LGBMClassifier)),
        ]
    )


def build_fasttext_lightgbm_classifier(
    embedding_dir: Path = EMBEDDINGS_DIR,
    max_vectors: int | None = None,
    pooling: str = "mean",
) -> Pipeline:
    """Return pooled FastText + multi-output LightGBM classifier."""

    try:
        from lightgbm import LGBMClassifier
    except ImportError:
        return MissingDependencyEstimator("lightgbm", "fasttext_lightgbm")

    return Pipeline(
        steps=[
            (
                "embeddings",
                build_fasttext_vectorizer(
                    embedding_dir=embedding_dir,
                    max_vectors=max_vectors,
                    pooling=pooling,
                ),
            ),
            ("classifier", _lightgbm_classifier(LGBMClassifier)),
        ]
    )
