"""Tests for predict.py (joblib-based inference CLI/helper).

Uses the ``dummy_most_frequent`` model rather than
``roberta_label_dependency`` so these tests run without torch/transformers
installed — predict.py is generic across any registered model (it goes
through evaluation.predict_scores(), not a RoBERTa-specific code path), so
testing it against the dummy model still exercises the real logic.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pytest

from toxic_comments.config import LABEL_COLUMNS
from toxic_comments.models.registry import build_models
from toxic_comments.predict import main, predict


def _toy_data():
    X = ["nice article", "you awful idiot", "thanks for edit", "hate you"]
    y = np.array(
        [
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 1],
        ]
    )
    return X, y


def _save_dummy_model(tmp_path: Path) -> Path:
    models = build_models(max_features=20)
    model = models["dummy_most_frequent"]
    X, y = _toy_data()
    model.fit(X, y)
    model_path = tmp_path / "dummy_most_frequent.joblib"
    joblib.dump(model, model_path)
    return model_path


def test_predict_returns_one_row_per_text_with_prob_and_pred_columns(tmp_path):
    model_path = _save_dummy_model(tmp_path)

    result = predict(
        ["you are stupid", "thanks a lot"],
        model_path,
        threshold=0.5,
        already_clean=True,
    )

    assert len(result) == 2
    assert list(result["comment_text"]) == ["you are stupid", "thanks a lot"]
    for label in LABEL_COLUMNS:
        assert f"{label}_prob" in result.columns
        assert f"{label}_pred" in result.columns
        assert result[f"{label}_pred"].isin([0, 1]).all()


def test_predict_raw_skips_cleaning(monkeypatch, tmp_path):
    model_path = _save_dummy_model(tmp_path)

    calls: list[str] = []
    monkeypatch.setattr(
        "toxic_comments.predict.clean_light",
        lambda text: calls.append(text) or text,
    )

    predict(["Some Text"], model_path, already_clean=True)
    assert calls == []  # clean_light never called when already_clean=True

    predict(["Some Text"], model_path, already_clean=False)
    assert calls == ["Some Text"]  # clean_light called once when cleaning is needed


def test_predict_threshold_controls_pred_columns(tmp_path):
    model_path = _save_dummy_model(tmp_path)

    low = predict(["hello"], model_path, threshold=0.0, already_clean=True)
    high = predict(["hello"], model_path, threshold=1.01, already_clean=True)

    pred_columns = [f"{label}_pred" for label in LABEL_COLUMNS]
    # threshold 0.0 -> every label predicted positive (score is always >= 0);
    # threshold > 1 -> every label negative (score is never > 1)
    assert (low[pred_columns].to_numpy() == 1).all()
    assert (high[pred_columns].to_numpy() == 0).all()


def test_main_raises_clear_error_for_missing_model(tmp_path, monkeypatch):
    missing_path = tmp_path / "does_not_exist.joblib"
    monkeypatch.setattr(
        "sys.argv",
        ["predict.py", "--model", str(missing_path), "--text", "hello"],
    )

    with pytest.raises(FileNotFoundError, match="Không thấy model"):
        main()

