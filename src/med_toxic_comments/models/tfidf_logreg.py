"""Model factories for the toxic comment classifiers."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline

def build_tfidf_logistic_regression(
    max_features: int = 50_000,
    ngram_range: tuple[int, int] = (1, 2),
    c_value: float = 4.0,
) -> Pipeline:
    """Return a TF-IDF + one-vs-rest logistic regression classifier."""

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
            ("classifier", classifier),
        ]
    )
