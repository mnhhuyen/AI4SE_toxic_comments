"""NB-SVM model factory."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

from toxic_comments.models.vectorizers import build_word_char_union


class NbFeaturer(BaseEstimator, TransformerMixin):
    """Scale each feature by log(p(f|positive) / p(f|negative))."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def _rate(self, x, mask: np.ndarray) -> np.ndarray:
        matched = x[mask]
        counts = np.asarray(matched.sum(axis=0)).ravel()
        return (counts + self.alpha) / (matched.shape[0] + 2 * self.alpha)

    def fit(self, x, y):
        labels = np.asarray(y).ravel()
        if not sparse.issparse(x):
            x = sparse.csr_matrix(x)
        ratio = self._rate(x, labels == 1) / self._rate(x, labels == 0)
        self.log_ratio_ = sparse.csr_matrix(np.log(ratio))
        return self

    def transform(self, x) -> sparse.csr_matrix:
        if not sparse.issparse(x):
            x = sparse.csr_matrix(x)
        return x.multiply(self.log_ratio_).tocsr()


def build_nbsvm(
    max_features: int = 50_000,
    c_value: float = 4.0,
    alpha: float = 1.0,
) -> Pipeline:
    """Return TF-IDF with per-label NB weighting and logistic regression.

    The NB ratio is label specific, so it sits inside the one-vs-rest wrapper
    while the vectorizers above it are fitted once.
    """

    per_label = Pipeline(
        steps=[
            ("nb", NbFeaturer(alpha=alpha)),
            (
                "classifier",
                LogisticRegression(
                    C=c_value,
                    solver="liblinear",
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("features", build_word_char_union(max_features=max_features)),
            ("classifier", OneVsRestClassifier(per_label)),
        ]
    )
