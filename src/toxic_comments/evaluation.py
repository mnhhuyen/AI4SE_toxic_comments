"""Evaluation utilities for multi-label toxic comment classification."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

from toxic_comments.config import LABEL_COLUMNS


def predict_scores(estimator, x) -> np.ndarray | None:
    """Return per-label continuous scores for ranking metrics."""

    if hasattr(estimator, "predict_proba"):
        try:
            probabilities = estimator.predict_proba(x)
        except (AttributeError, NotImplementedError):
            probabilities = None
        if probabilities is not None:
            return _normalize_proba(estimator, probabilities)

    if hasattr(estimator, "decision_function"):
        scores = np.asarray(estimator.decision_function(x))
        return scores.reshape(-1, 1) if scores.ndim == 1 else scores

    return None


def _normalize_proba(estimator, probabilities) -> np.ndarray:
    """Flatten the shapes scikit-learn uses for multi-label predict_proba."""

    if not isinstance(probabilities, list):
        probabilities = np.asarray(probabilities)
        return probabilities[:, :, 1].T if probabilities.ndim == 3 else probabilities

    estimators = getattr(estimator, "estimators_", [])
    if not estimators:
        steps = getattr(estimator, "steps", [])
        estimators = getattr(steps[-1][1], "estimators_", []) if steps else []

    scores = []
    for index, class_probability in enumerate(probabilities):
        classes = getattr(estimators[index], "classes_", None) if index < len(estimators) else None
        if classes is None or 1 not in list(classes):
            scores.append(np.zeros(class_probability.shape[0]))
            continue
        positive_index = int(np.where(np.asarray(classes) == 1)[0][0])
        scores.append(class_probability[:, positive_index])

    return np.column_stack(scores)


def _safe_auc(y_true, y_score, average: str) -> float | None:
    if y_score is None:
        return None
    try:
        return float(roc_auc_score(y_true, y_score, average=average))
    except ValueError:
        return None


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
) -> dict[str, float | None]:
    """Compute metrics suitable for multi-label classification."""

    return {
        "subset_accuracy": accuracy_score(y_true, y_pred),
        "hamming_loss": hamming_loss(y_true, y_pred),
        "micro_precision": precision_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_recall": recall_score(y_true, y_pred, average="micro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_roc_auc": _safe_auc(y_true, y_score, "micro"),
        "macro_roc_auc": _safe_auc(y_true, y_score, "macro"),
    }


def summarize_folds(metrics: pd.DataFrame, group_by: str = "model_name") -> pd.DataFrame:
    """Return mean and standard deviation of fold metrics per model."""

    numeric = metrics.select_dtypes("number").columns.difference([group_by])
    return metrics.groupby(group_by)[list(numeric)].agg(["mean", "std"]).round(4)


def evaluate_per_label(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray | None = None,
    label_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Compute precision, recall, F1 and ROC-AUC for each label."""

    label_columns = label_columns or LABEL_COLUMNS
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rows = []
    for index, label in enumerate(label_columns):
        truth = y_true[:, index]
        predicted = y_pred[:, index]
        auc = None
        if y_score is not None and len(np.unique(truth)) > 1:
            auc = float(roc_auc_score(truth, np.asarray(y_score)[:, index]))

        rows.append(
            {
                "label": label,
                "support": int(truth.sum()),
                "predicted_positive": int(predicted.sum()),
                "precision": float(precision_score(truth, predicted, zero_division=0)),
                "recall": float(recall_score(truth, predicted, zero_division=0)),
                "f1": float(f1_score(truth, predicted, zero_division=0)),
                "roc_auc": auc,
            }
        )

    return pd.DataFrame(rows)
