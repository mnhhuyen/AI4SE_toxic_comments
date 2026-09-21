"""Per-label decision threshold tuning."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.metaestimators import available_if

from toxic_comments.config import LABEL_COLUMNS
from toxic_comments.evaluation import predict_scores

DEFAULT_THRESHOLD = 0.5


def candidate_thresholds(scores: np.ndarray, n_candidates: int = 100) -> np.ndarray:
    """Return threshold candidates drawn from the score distribution."""

    return np.unique(np.quantile(scores, np.linspace(0.01, 0.99, n_candidates)))


def tune_thresholds(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_candidates: int = 100,
) -> np.ndarray:
    """Return the F1-maximising threshold for each label."""

    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    thresholds = np.full(y_true.shape[1], DEFAULT_THRESHOLD, dtype=float)

    for index in range(y_true.shape[1]):
        truth = y_true[:, index]
        scores = y_score[:, index]
        if truth.sum() == 0 or np.unique(scores).size < 2:
            continue

        best_threshold, best_f1 = DEFAULT_THRESHOLD, -1.0
        for threshold in candidate_thresholds(scores, n_candidates):
            f1 = f1_score(truth, (scores >= threshold).astype(int), zero_division=0)
            if f1 > best_f1:
                best_threshold, best_f1 = float(threshold), f1

        thresholds[index] = best_threshold

    return thresholds


def apply_thresholds(y_score: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Binarise scores with one threshold per label."""

    return (np.asarray(y_score) >= np.asarray(thresholds)).astype(int)


def thresholds_to_frame(
    thresholds: np.ndarray,
    label_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Return tuned thresholds as a labeled dataframe."""

    return pd.DataFrame(
        {"label": label_columns or LABEL_COLUMNS, "threshold": np.asarray(thresholds)}
    )


def _wrapped_has(attribute: str):
    def check(self) -> bool:
        return hasattr(self.estimator, attribute)

    return check


class ThresholdedClassifier(BaseEstimator, ClassifierMixin):
    """Replace an estimator's 0.5 cut-off with tuned per-label thresholds."""

    def __init__(
        self,
        estimator,
        validation_size: float = 0.2,
        n_candidates: int = 100,
        random_state: int = 42,
    ):
        self.estimator = estimator
        self.validation_size = validation_size
        self.n_candidates = n_candidates
        self.random_state = random_state

    def _subset(self, x, index: np.ndarray):
        return x.iloc[index] if hasattr(x, "iloc") else x[index]

    def fit(self, x, y):
        y = np.asarray(y)
        train_index, validation_index = train_test_split(
            np.arange(y.shape[0]),
            test_size=self.validation_size,
            random_state=self.random_state,
            shuffle=True,
        )

        # tune on held-out scores, then refit on everything
        tuner = clone(self.estimator)
        tuner.fit(self._subset(x, train_index), y[train_index])
        scores = predict_scores(tuner, self._subset(x, validation_index))

        self.thresholds_ = (
            None
            if scores is None
            else tune_thresholds(y[validation_index], scores, self.n_candidates)
        )

        self.estimator_ = clone(self.estimator)
        self.estimator_.fit(x, y)
        return self

    def predict(self, x) -> np.ndarray:
        if self.thresholds_ is None:
            return np.asarray(self.estimator_.predict(x))

        scores = predict_scores(self.estimator_, x)
        if scores is None:
            return np.asarray(self.estimator_.predict(x))
        return apply_thresholds(scores, self.thresholds_)

    @available_if(_wrapped_has("predict_proba"))
    def predict_proba(self, x):
        return self.estimator_.predict_proba(x)

    @available_if(_wrapped_has("decision_function"))
    def decision_function(self, x):
        return self.estimator_.decision_function(x)

    def __sklearn_is_fitted__(self) -> bool:
        return hasattr(self, "estimator_")
