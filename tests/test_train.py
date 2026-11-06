import pytest
import pandas as pd

from toxic_comments.config import HEAVY_TEXT_COLUMN, LABEL_COLUMNS
from toxic_comments.splits import save_kfold_datasets
from toxic_comments.train import train_from_folds, train_model, train_pooled_from_folds


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


def _write_tiny_fasttext(embedding_dir):
    path = embedding_dir / "fasttext" / "wiki-news-300d-1M.vec"
    path.parent.mkdir(parents=True)
    path.write_text("4 3\nnice 0.1 0.2 0.3\nawful -0.4 0.5 0.1\nhate -0.2 0.6 -0.1\nyou 0.0 0.1 0.0\n")
    return embedding_dir


def test_train_pooled_from_folds_shares_embeddings(tmp_path):
    pytest.importorskip("torch")
    data = pd.concat([_training_data()] * 3, ignore_index=True)
    folds_dir = tmp_path / "folds"
    save_kfold_datasets(data, output_dir=folds_dir, n_splits=3, strategy="kfold")
    names = ["fasttext_logistic_regression"]
    try:
        import lightgbm  # noqa: F401
        names.append("fasttext_lightgbm")
    except (ImportError, OSError):
        pass

    frames = train_pooled_from_folds(
        folds_dir=folds_dir,
        model_names=names,
        output_dir=tmp_path / "models",
        results_dir=tmp_path / "results",
        embedding_dir=_write_tiny_fasttext(tmp_path),
        embedding_pooling="max_supervised_attention",
    )

    for name in names:
        assert len(frames[name]) == 3
        assert (tmp_path / "results" / f"{name}_fold_per_label.csv").exists()
        assert (tmp_path / "models" / "fold_1" / f"{name}.joblib").exists()


def test_train_pooled_from_folds_rejects_mixed_embeddings(tmp_path):
    save_kfold_datasets(pd.concat([_training_data()] * 3, ignore_index=True),
                        output_dir=tmp_path / "folds", n_splits=3, strategy="kfold")
    with pytest.raises(ValueError, match="same embeddings"):
        train_pooled_from_folds(
            folds_dir=tmp_path / "folds",
            model_names=["fasttext_logistic_regression", "glove_twitter_logistic_regression"],
            embedding_dir=tmp_path,
        )
