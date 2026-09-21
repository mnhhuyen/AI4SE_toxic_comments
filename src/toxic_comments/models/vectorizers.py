"""Shared TF-IDF feature blocks."""

from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion


def build_word_vectorizer(
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    min_df: int = 1,
) -> TfidfVectorizer:
    """Return a word-level TF-IDF vectorizer."""

    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
    )


def build_char_vectorizer(
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (3, 5),
    min_df: int = 2,
) -> TfidfVectorizer:
    """Return a character n-gram TF-IDF vectorizer bounded to words."""

    return TfidfVectorizer(
        analyzer="char_wb",
        lowercase=True,
        strip_accents="unicode",
        ngram_range=ngram_range,
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
    )


def build_word_char_union(
    max_features: int = 50_000,
    char_max_features: int | None = None,
) -> FeatureUnion:
    """Return word and character TF-IDF blocks concatenated."""

    return FeatureUnion(
        [
            ("word", build_word_vectorizer(max_features=max_features)),
            ("char", build_char_vectorizer(max_features=char_max_features or max_features)),
        ]
    )
