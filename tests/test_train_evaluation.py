import numpy as np
import pandas as pd

from toxic_comments.config import HEAVY_TEXT_COLUMN, LABEL_COLUMNS
from toxic_comments.evaluation import summarize_folds
from toxic_comments.thresholds import ThresholdedClassifier, tune_thresholds
from toxic_comments.train import evaluate_model_detailed, train_model


def _training_data(n_rows: int = 24) -> pd.DataFrame:
    rows = []
    for index in range(n_rows):
        is_toxic = index % 2 == 0
        text = "you stupid idiot shut up" if is_toxic else "thanks for the helpful edit"
        rows.append(
            {
                HEAVY_TEXT_COLUMN: f"{text} {index}",
                "toxic": int(is_toxic),
                "severe_toxic": int(index % 4 == 0),
                "obscene": int(index % 3 == 0),
                "threat": int(index % 5 == 0),
                "insult": int(index % 3 == 1),
                "identity_hate": int(index % 7 == 0),
            }
        )
    return pd.DataFrame(rows)


class CountingEstimator:
    """Records how many times the pipeline asked it to predict."""

    def __init__(self, n_labels: int):
        self.n_labels = n_labels
        self.predict_calls = 0
        self.proba_calls = 0

    def fit(self, x, y=None):
        return self

    def predict(self, x):
        self.predict_calls += 1
        return np.zeros((len(x), self.n_labels), dtype=int)

    def predict_proba(self, x):
        self.proba_calls += 1
        return np.full((len(x), self.n_labels), 0.25)


def test_detailed_evaluation_predicts_only_once():
    data = _training_data()
    estimator = CountingEstimator(len(LABEL_COLUMNS))

    aggregate, per_label = evaluate_model_detailed(estimator, data, HEAVY_TEXT_COLUMN)

    assert estimator.predict_calls == 1
    assert estimator.proba_calls == 1
    assert "micro_f1" in aggregate
    assert len(per_label) == len(LABEL_COLUMNS)


def test_constant_scores_keep_the_default_threshold():
    y_true = np.array([[1], [0], [1], [0]])
    y_score = np.full((4, 1), 0.25)

    assert tune_thresholds(y_true, y_score)[0] == 0.5


def test_dummy_baseline_is_unchanged_by_tuning():
    data = _training_data()
    plain = train_model(data, model_name="dummy_most_frequent", max_features=50)
    tuned = train_model(
        data, model_name="dummy_most_frequent", max_features=50, tune_thresholds=True
    )

    x = data[HEAVY_TEXT_COLUMN]

    assert np.array_equal(plain.predict(x), tuned.predict(x))


def test_train_model_can_wrap_with_tuned_thresholds():
    data = _training_data()

    estimator = train_model(
        data,
        model_name="tfidf_logistic_regression",
        max_features=50,
        tune_thresholds=True,
    )

    assert isinstance(estimator, ThresholdedClassifier)
    assert estimator.predict(data[HEAVY_TEXT_COLUMN]).shape == (len(data), len(LABEL_COLUMNS))


def test_summarize_folds_reports_mean_and_spread():
    metrics = pd.DataFrame(
        {
            "model_name": ["a", "a", "b", "b"],
            "fold": ["fold_1", "fold_2", "fold_1", "fold_2"],
            "micro_f1": [0.60, 0.70, 0.80, 0.90],
        }
    )

    summary = summarize_folds(metrics)

    assert summary.loc["a", ("micro_f1", "mean")] == 0.65
    assert summary.loc["b", ("micro_f1", "mean")] == 0.85
    assert summary.loc["a", ("micro_f1", "std")] > 0
