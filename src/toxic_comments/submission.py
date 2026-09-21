"""Kaggle submission files."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from toxic_comments.config import (
    HEAVY_TEXT_COLUMN,
    ID_COLUMN,
    LABEL_COLUMNS,
    RESULTS_DIR,
)
from toxic_comments.evaluation import predict_scores

UNLABELED_MARKER = -1


def to_probability(scores: np.ndarray) -> np.ndarray:
    """Squash decision values into [0, 1], leaving probabilities untouched."""

    scores = np.asarray(scores, dtype=float)
    if scores.size and scores.min() >= 0.0 and scores.max() <= 1.0:
        return scores
    return 1.0 / (1.0 + np.exp(-scores))


def build_submission_frame(ids, y_score: np.ndarray) -> pd.DataFrame:
    """Return the id column plus the six probability columns."""

    probabilities = to_probability(y_score)
    if probabilities.shape[1] != len(LABEL_COLUMNS):
        raise ValueError(
            f"Expected {len(LABEL_COLUMNS)} score columns, got {probabilities.shape[1]}"
        )

    submission = pd.DataFrame({ID_COLUMN: np.asarray(ids)})
    for index, label in enumerate(LABEL_COLUMNS):
        submission[label] = probabilities[:, index]
    return submission


def select_scored_rows(test_data: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Drop the test rows Kaggle marked -1 and never scored."""

    merged = test_data.merge(labels, on=ID_COLUMN, how="inner", suffixes=("", "_label"))
    scored = (merged[LABEL_COLUMNS] != UNLABELED_MARKER).all(axis=1)
    return merged[scored].reset_index(drop=True)


def make_submission(
    estimator,
    test_data: pd.DataFrame,
    text_column: str = HEAVY_TEXT_COLUMN,
    output_path: Path | None = None,
    model_name: str = "model",
) -> Path:
    """Write a submission CSV of probabilities for a fitted estimator."""

    x_test = test_data[text_column].fillna("").astype(str)
    scores = predict_scores(estimator, x_test)
    if scores is None:
        raise ValueError(f"Model '{model_name}' produces no rankable scores.")

    output_path = output_path or RESULTS_DIR / "submissions" / f"{model_name}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_submission_frame(test_data[ID_COLUMN], scores).to_csv(output_path, index=False)
    return output_path
