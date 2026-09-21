"""Linear and Naive Bayes model factories."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC

from toxic_comments.models.handcrafted import HandCraftedFeatures
from toxic_comments.models.vectorizers import build_word_char_union


def build_word_char_logistic_regression(
    max_features: int = 50_000,
    c_value: float = 4.0,
) -> Pipeline:
    """Return word + character TF-IDF with one-vs-rest logistic regression."""

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
            ("features", build_word_char_union(max_features=max_features)),
            ("classifier", classifier),
        ]
    )


def build_raw_text_logistic_regression(
    max_features: int = 50_000,
    c_value: float = 4.0,
) -> Pipeline:
    """Return word + character TF-IDF plus numeric features.

    Train this one with --text-column comment_text: the numeric features
    measure casing and punctuation, which the cleaned columns no longer have.
    """

    classifier = OneVsRestClassifier(
        LogisticRegression(
            C=c_value,
            solver="liblinear",
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        )
    )
    features = FeatureUnion(
        [
            ("tfidf", build_word_char_union(max_features=max_features)),
            ("handcrafted", HandCraftedFeatures()),
        ]
    )
    return Pipeline(steps=[("features", features), ("classifier", classifier)])


def build_tfidf_linear_svc(
    max_features: int = 50_000,
    c_value: float = 0.5,
) -> Pipeline:
    """Return word + character TF-IDF with a linear SVM.

    Exposes decision_function rather than predict_proba.
    """

    classifier = OneVsRestClassifier(
        LinearSVC(
            C=c_value,
            class_weight="balanced",
            dual=True,
            max_iter=5000,
            random_state=42,
        )
    )
    return Pipeline(
        steps=[
            ("features", build_word_char_union(max_features=max_features)),
            ("classifier", classifier),
        ]
    )


def build_tfidf_sgd_logistic(
    max_features: int = 50_000,
    alpha: float = 1e-6,
) -> Pipeline:
    """Return word + character TF-IDF with SGD-trained logistic regression."""

    classifier = OneVsRestClassifier(
        SGDClassifier(
            loss="log_loss",
            alpha=alpha,
            class_weight="balanced",
            max_iter=30,
            tol=1e-4,
            random_state=42,
        )
    )
    return Pipeline(
        steps=[
            ("features", build_word_char_union(max_features=max_features)),
            ("classifier", classifier),
        ]
    )


def build_tfidf_complement_nb(
    max_features: int = 50_000,
    alpha: float = 0.5,
) -> Pipeline:
    """Return word + character TF-IDF with Complement Naive Bayes."""

    return Pipeline(
        steps=[
            ("features", build_word_char_union(max_features=max_features)),
            ("classifier", OneVsRestClassifier(ComplementNB(alpha=alpha))),
        ]
    )
