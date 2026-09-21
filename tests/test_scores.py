import numpy as np
import pandas as pd

from toxic_comments.config import HEAVY_TEXT_COLUMN, LABEL_COLUMNS
from toxic_comments.evaluation import (
    evaluate_per_label,
    evaluate_predictions,
    predict_scores,
)
from toxic_comments.models.baseline import build_dummy_baseline
from toxic_comments.models.linear import build_tfidf_linear_svc


def _training_data(n_rows: int = 20) -> pd.DataFrame:
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


def test_scores_fall_back_to_decision_function():
    data = _training_data()
    x = data[HEAVY_TEXT_COLUMN]
    y = data[LABEL_COLUMNS].to_numpy()
    model = build_tfidf_linear_svc(max_features=200).fit(x, y)

    scores = predict_scores(model, x)

    assert scores is not None
    assert scores.shape == (len(data), len(LABEL_COLUMNS))


def test_linear_svc_gets_a_roc_auc():
    data = _training_data()
    x = data[HEAVY_TEXT_COLUMN]
    y = data[LABEL_COLUMNS].to_numpy()
    model = build_tfidf_linear_svc(max_features=200).fit(x, y)

    metrics = evaluate_predictions(y, model.predict(x), predict_scores(model, x))

    assert metrics["micro_roc_auc"] is not None


def test_multi_output_proba_is_flattened():
    data = _training_data()
    x = data[HEAVY_TEXT_COLUMN]
    y = data[LABEL_COLUMNS].to_numpy()
    model = build_dummy_baseline().fit(x, y)

    assert predict_scores(model, x).shape == (len(data), len(LABEL_COLUMNS))


def test_per_label_covers_all_labels():
    y = _training_data()[LABEL_COLUMNS].to_numpy()

    per_label = evaluate_per_label(y, np.zeros_like(y))

    assert per_label["label"].tolist() == LABEL_COLUMNS
    assert per_label["support"].sum() == y.sum()
    assert (per_label["f1"] == 0.0).all()
