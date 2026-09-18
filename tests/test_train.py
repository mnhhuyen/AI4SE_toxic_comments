import pytest
import pandas as pd

from toxic_comments.config import HEAVY_TEXT_COLUMN, LABEL_COLUMNS
from toxic_comments.splits import save_kfold_datasets
from toxic_comments.train import train_from_folds, train_model


def _training_data() -> pd.DataFrame:
    return pd.DataFrame(
        {
            HEAVY_TEXT_COLUMN: [
                "nice article",
                "you awful idiot",
                "thanks for edit",
                "hate you",
            ],
            **{
                label: [0, 1, 0, 1]
                for label in LABEL_COLUMNS
            },
        }
    )


def test_train_model_fits_registered_model():
    estimator = train_model(
        _training_data(),
        model_name="tfidf_logistic_regression",
        max_features=20,
    )

    assert hasattr(estimator, "predict")


def test_train_model_rejects_unknown_model():
    with pytest.raises(ValueError, match="Unknown model"):
        train_model(_training_data(), model_name="missing_model")


def test_train_from_folds_saves_one_model_per_fold(tmp_path):
    data = pd.concat([_training_data()] * 3, ignore_index=True)
    folds_dir = tmp_path / "folds"
    models_dir = tmp_path / "models"
    results_dir = tmp_path / "results"
    save_kfold_datasets(data, output_dir=folds_dir, n_splits=3, strategy="kfold")

    model_paths, metrics = train_from_folds(
        folds_dir=folds_dir,
        model_name="tfidf_logistic_regression",
        output_dir=models_dir,
        results_dir=results_dir,
        max_features=20,
    )

    assert len(model_paths) == 3
    assert len(metrics) == 3
    assert all(path.exists() for path in model_paths)
    assert (results_dir / "tfidf_logistic_regression_fold_metrics.csv").exists()
