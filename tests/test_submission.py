import numpy as np
import pandas as pd
import pytest

from toxic_comments.config import HEAVY_TEXT_COLUMN, ID_COLUMN, LABEL_COLUMNS
from toxic_comments.models.linear import build_tfidf_linear_svc
from toxic_comments.submission import (
    UNLABELED_MARKER,
    build_submission_frame,
    make_submission,
    select_scored_rows,
    to_probability,
)


def _test_data(n_rows: int = 12) -> pd.DataFrame:
    return pd.DataFrame(
        {
            ID_COLUMN: [f"id_{index}" for index in range(n_rows)],
            HEAVY_TEXT_COLUMN: [
                f"you stupid idiot {index}" if index % 2 else f"thanks for the edit {index}"
                for index in range(n_rows)
            ],
        }
    )


def _labels_frame(rows):
    return pd.DataFrame(
        [{ID_COLUMN: key, **{label: value for label in LABEL_COLUMNS}} for key, value in rows]
    )


def test_probabilities_pass_through():
    scores = np.array([[0.0, 0.5], [1.0, 0.25]])

    assert np.allclose(to_probability(scores), scores)


def test_decision_values_are_squashed_monotonically():
    squashed = to_probability(np.array([[-4.0, 0.0, 7.0]]))

    assert squashed.min() > 0.0 and squashed.max() < 1.0
    assert list(np.argsort(squashed[0])) == [0, 1, 2]


def test_submission_frame_columns():
    frame = build_submission_frame(pd.Series(["a", "b"]), np.zeros((2, 6)))

    assert list(frame.columns) == [ID_COLUMN, *LABEL_COLUMNS]


def test_submission_frame_rejects_wrong_width():
    with pytest.raises(ValueError, match="score columns"):
        build_submission_frame(pd.Series(["a"]), np.zeros((1, 3)))


def test_unscored_rows_are_dropped():
    data = pd.DataFrame({ID_COLUMN: ["a", "b", "c"], HEAVY_TEXT_COLUMN: ["x", "y", "z"]})
    labels = _labels_frame([("a", 0), ("b", UNLABELED_MARKER), ("c", 1)])

    scored = select_scored_rows(data, labels)

    assert scored[ID_COLUMN].tolist() == ["a", "c"]


def test_submission_covers_every_row_without_predict_proba(tmp_path):
    data = _test_data()
    y = np.tile(np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]]), (len(data) // 2, 1))
    estimator = build_tfidf_linear_svc(max_features=50).fit(data[HEAVY_TEXT_COLUMN], y)

    path = make_submission(estimator, data, output_path=tmp_path / "svc.csv")
    written = pd.read_csv(path)

    assert written[ID_COLUMN].tolist() == data[ID_COLUMN].tolist()
    assert written[LABEL_COLUMNS].to_numpy().min() >= 0.0
    assert written[LABEL_COLUMNS].to_numpy().max() <= 1.0
