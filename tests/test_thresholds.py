import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score

from toxic_comments.config import HEAVY_TEXT_COLUMN, LABEL_COLUMNS
from toxic_comments.models.linear import build_tfidf_linear_svc
from toxic_comments.models.tfidf_logreg import build_tfidf_logistic_regression
from toxic_comments.thresholds import (
    ThresholdedClassifier,
    apply_thresholds,
    thresholds_to_frame,
    tune_thresholds,
)


def _training_data(n_rows: int = 20) -> pd.DataFrame:
    rows = []
    for index in range(n_rows):
        is_toxic = index % 2 == 0
        text = "you stupid idiot shut up" if is_toxic else "thanks for the helpful edit"
        rows.append(
            {
                HEAVY_TEXT_COLUMN: f"{text} {index}",
                **{label: int(is_toxic) for label in LABEL_COLUMNS},
            }
        )
    return pd.DataFrame(rows)


def test_tuning_beats_the_default_cut_off():
    y_true = np.array([[1]] * 10 + [[0]] * 90)
    y_score = np.array([[0.35]] * 10 + [[0.10]] * 90)

    thresholds = tune_thresholds(y_true, y_score)
    tuned = f1_score(y_true, apply_thresholds(y_score, thresholds), zero_division=0)

    assert f1_score(y_true, (y_score >= 0.5).astype(int), zero_division=0) == 0.0
    assert tuned > 0.0
    assert thresholds[0] <= 0.35


def test_label_without_positives_keeps_default():
    y_true = np.zeros((20, 1), dtype=int)
    y_score = np.linspace(0, 1, 20).reshape(-1, 1)

    assert tune_thresholds(y_true, y_score)[0] == 0.5


def test_apply_thresholds_is_per_label():
    y_score = np.array([[0.4, 0.4], [0.6, 0.6]])

    predictions = apply_thresholds(y_score, np.array([0.5, 0.3]))

    assert predictions.tolist() == [[0, 1], [1, 1]]


def test_thresholds_to_frame():
    frame = thresholds_to_frame(np.full(len(LABEL_COLUMNS), 0.5))

    assert frame["label"].tolist() == LABEL_COLUMNS


def test_wrapper_only_delegates_methods_that_exist():
    svc = ThresholdedClassifier(build_tfidf_linear_svc(max_features=200))
    logistic = ThresholdedClassifier(build_tfidf_logistic_regression(max_features=200))

    assert not hasattr(svc, "predict_proba")
    assert hasattr(svc, "decision_function")
    assert hasattr(logistic, "predict_proba")


def test_wrapper_fits_a_decision_function_model():
    data = _training_data()
    x = data[HEAVY_TEXT_COLUMN]
    y = data[LABEL_COLUMNS].to_numpy()

    model = ThresholdedClassifier(build_tfidf_linear_svc(max_features=200)).fit(x, y)

    assert len(model.thresholds_) == len(LABEL_COLUMNS)
    assert model.predict(x).shape == (len(data), len(LABEL_COLUMNS))


def test_wrapper_is_clonable():
    model = ThresholdedClassifier(build_tfidf_logistic_regression(max_features=200))

    copy = clone(model)

    assert copy.validation_size == model.validation_size
    assert not hasattr(copy, "estimator_")
