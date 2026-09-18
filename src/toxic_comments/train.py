"""Train and save toxic comment classification models."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from toxic_comments.config import (
    EMBEDDINGS_DIR,
    HEAVY_TEXT_COLUMN,
    K_FOLD_DATA_DIR,
    LABEL_COLUMNS,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RESULTS_DIR,
)
from toxic_comments.evaluation import evaluate_predictions
from toxic_comments.splits import train_validation_test_split
from toxic_comments.models.registry import build_models


def train_model(
    data: pd.DataFrame,
    model_name: str,
    text_column: str = HEAVY_TEXT_COLUMN,
    max_features: int = 50_000,
    embedding_dir: Path = EMBEDDINGS_DIR,
    max_embedding_vectors: int | None = None,
    embedding_pooling: str = "mean",
):
    """Fit one registered model on a labeled dataframe."""

    if text_column not in data.columns:
        raise ValueError(f"Missing text column: {text_column}")

    missing_labels = [label for label in LABEL_COLUMNS if label not in data.columns]
    if missing_labels:
        raise ValueError(f"Missing label columns: {', '.join(missing_labels)}")

    models = build_models(
        max_features=max_features,
        embedding_dir=embedding_dir,
        max_embedding_vectors=max_embedding_vectors,
        embedding_pooling=embedding_pooling,
    )
    if model_name not in models:
        available = ", ".join(sorted(models))
        raise ValueError(f"Unknown model '{model_name}'. Available models: {available}")

    estimator = models[model_name]
    x_train = data[text_column].fillna("").astype(str)
    y_train = data[LABEL_COLUMNS].to_numpy()
    estimator.fit(x_train, y_train)
    return estimator


def save_model(estimator, output_path: Path) -> Path:
    """Persist a trained estimator to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(estimator, output_path)
    return output_path


def evaluate_model(estimator, data: pd.DataFrame, text_column: str) -> dict[str, float | None]:
    """Evaluate one fitted model on a labeled dataframe."""

    x_test = data[text_column].fillna("").astype(str)
    y_true = data[LABEL_COLUMNS].to_numpy()
    y_pred = estimator.predict(x_test)
    y_score = _predict_scores(estimator, x_test)
    return evaluate_predictions(y_true, y_pred, y_score)


def train_holdout(
    data: pd.DataFrame,
    model_name: str,
    output_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
    text_column: str = HEAVY_TEXT_COLUMN,
    max_features: int = 50_000,
    embedding_dir: Path = EMBEDDINGS_DIR,
    max_embedding_vectors: int | None = None,
    embedding_pooling: str = "mean",
    validation_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
) -> tuple[Path, pd.DataFrame]:
    """Train one model with stratified train/validation/test holdout split."""

    train_data, validation_data, test_data = train_validation_test_split(
        data,
        validation_size=validation_size,
        test_size=test_size,
        random_state=random_state,
        stratified=True,
    )
    estimator = train_model(
        train_data,
        model_name=model_name,
        text_column=text_column,
        max_features=max_features,
        embedding_dir=embedding_dir,
        max_embedding_vectors=max_embedding_vectors,
        embedding_pooling=embedding_pooling,
    )

    metrics = []
    for split_name, split_data in [
        ("validation", validation_data),
        ("test", test_data),
    ]:
        row = {
            "model_name": model_name,
            "split": split_name,
            "rows": len(split_data),
        }
        row.update(evaluate_model(estimator, split_data, text_column=text_column))
        metrics.append(row)

    output_path = save_model(estimator, output_dir / f"{model_name}.joblib")
    metrics_frame = pd.DataFrame(metrics)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(results_dir / f"{model_name}_holdout_metrics.csv", index=False)
    return output_path, metrics_frame


def train_from_folds(
    folds_dir: Path,
    model_name: str,
    output_dir: Path = MODELS_DIR,
    results_dir: Path = RESULTS_DIR,
    text_column: str = HEAVY_TEXT_COLUMN,
    max_features: int = 50_000,
    embedding_dir: Path = EMBEDDINGS_DIR,
    max_embedding_vectors: int | None = None,
    embedding_pooling: str = "mean",
) -> tuple[list[Path], pd.DataFrame]:
    """Train one model on each fold_N/train.csv and evaluate fold_N/test.csv."""

    model_paths: list[Path] = []
    metrics = []

    for fold_dir in sorted(folds_dir.glob("fold_*")):
        train_path = fold_dir / "train.csv"
        test_path = fold_dir / "test.csv"
        if not train_path.exists() or not test_path.exists():
            continue

        fold_name = fold_dir.name
        train_data = pd.read_csv(train_path)
        test_data = pd.read_csv(test_path)
        estimator = train_model(
            train_data,
            model_name=model_name,
            text_column=text_column,
            max_features=max_features,
            embedding_dir=embedding_dir,
            max_embedding_vectors=max_embedding_vectors,
            embedding_pooling=embedding_pooling,
        )

        row = {
            "model_name": model_name,
            "fold": fold_name,
            "split": "test",
            "rows": len(test_data),
        }
        row.update(evaluate_model(estimator, test_data, text_column=text_column))
        metrics.append(row)

        model_paths.append(
            save_model(estimator, output_dir / fold_name / f"{model_name}.joblib")
        )

    if not model_paths:
        raise ValueError(f"No fold_N/train.csv and fold_N/test.csv files found in {folds_dir}")

    metrics_frame = pd.DataFrame(metrics)
    results_dir.mkdir(parents=True, exist_ok=True)
    metrics_frame.to_csv(results_dir / f"{model_name}_fold_metrics.csv", index=False)
    return model_paths, metrics_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train one toxic comment model.")
    parser.add_argument(
        "--mode",
        choices=["holdout", "folds", "full"],
        default="holdout",
        help="Training mode.",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PROCESSED_DATA_DIR / "train_clean.csv",
        help="Path to labeled training CSV.",
    )
    parser.add_argument(
        "--model",
        default="tfidf_logistic_regression",
        help="Registered model name from toxic_comments.models.registry.",
    )
    parser.add_argument(
        "--text-column",
        default=HEAVY_TEXT_COLUMN,
        help="Text column used as model input.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50_000,
        help="TF-IDF vocabulary size for models that use it.",
    )
    parser.add_argument(
        "--embedding-dir",
        type=Path,
        default=EMBEDDINGS_DIR,
        help="Directory containing pretrained embedding files.",
    )
    parser.add_argument(
        "--max-embedding-vectors",
        type=int,
        default=None,
        help="Optional cap for loaded embedding vectors, useful for smoke tests.",
    )
    parser.add_argument(
        "--embedding-pooling",
        choices=["mean", "max", "attention", "max_attention"],
        default="mean",
        help="Pooling strategy for pretrained embedding models.",
    )
    parser.add_argument(
        "--folds-dir",
        type=Path,
        default=K_FOLD_DATA_DIR,
        help="Directory containing fold_N/train.csv and fold_N/test.csv files.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Directory where metric CSV files are written.",
    )
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Directory for saved .joblib models. Defaults to models/.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output or MODELS_DIR

    if args.mode == "folds":
        model_paths, metrics = train_from_folds(
            folds_dir=args.folds_dir,
            model_name=args.model,
            output_dir=output_dir,
            results_dir=args.results_dir,
            text_column=args.text_column,
            max_features=args.max_features,
            embedding_dir=args.embedding_dir,
            max_embedding_vectors=args.max_embedding_vectors,
            embedding_pooling=args.embedding_pooling,
        )
        print(f"Saved {len(model_paths)} fold models to {output_dir}")
        print(metrics)
        return

    data = pd.read_csv(args.data)
    if args.mode == "holdout":
        output_path, metrics = train_holdout(
            data=data,
            model_name=args.model,
            output_dir=output_dir,
            results_dir=args.results_dir,
            text_column=args.text_column,
            max_features=args.max_features,
            embedding_dir=args.embedding_dir,
            max_embedding_vectors=args.max_embedding_vectors,
            embedding_pooling=args.embedding_pooling,
            validation_size=args.validation_size,
            test_size=args.test_size,
            random_state=args.random_state,
        )
        print(f"Saved model: {output_path}")
        print(metrics)
        return

    estimator = train_model(
        data=data,
        model_name=args.model,
        text_column=args.text_column,
        max_features=args.max_features,
        embedding_dir=args.embedding_dir,
        max_embedding_vectors=args.max_embedding_vectors,
        embedding_pooling=args.embedding_pooling,
    )
    output_path = save_model(estimator, output_dir / f"{args.model}.joblib")
    print(f"Saved model: {output_path}")


def _predict_scores(estimator, x_test: pd.Series):
    if not hasattr(estimator, "predict_proba"):
        return None

    probabilities = estimator.predict_proba(x_test)
    if isinstance(probabilities, list):
        scores = []
        for label_index, class_probability in enumerate(probabilities):
            estimators = getattr(estimator, "estimators_", [])
            classes = getattr(estimators[label_index], "classes_", None)
            if classes is None or 1 not in classes:
                scores.append([0.0] * class_probability.shape[0])
                continue

            positive_class_index = int((classes == 1).nonzero()[0][0])
            scores.append(class_probability[:, positive_class_index])
        return pd.DataFrame(scores).T.to_numpy()

    return probabilities


if __name__ == "__main__":
    main()
