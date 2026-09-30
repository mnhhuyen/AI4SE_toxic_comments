"""Run inference with a model saved via train.save_model() (joblib).

Works with any model saved by train.py's train_holdout()/train_from_folds()
— not just roberta_label_dependency — since joblib.load() reconstructs
whatever estimator was pickled and every registered model implements the
same predict_proba()/predict() contract.

Usage
-----
    python -m toxic_comments.predict --text "you are so stupid"
    python -m toxic_comments.predict --model models/fold_2/roberta_label_dependency.joblib --text "..."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd

from toxic_comments.cleaning import clean_heavy, clean_light
from toxic_comments.config import LABEL_COLUMNS, MODELS_DIR
from toxic_comments.evaluation import predict_scores


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify comment text with a saved model")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "fold_1" / "roberta_label_dependency.joblib",
        help="Path to a .joblib model saved by train.py (default: fold_1's roberta_label_dependency).",
    )
    parser.add_argument(
        "--text",
        type=str,
        action="append",
        required=True,
        help="Comment text to classify. Repeat --text for multiple comments.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Decision threshold applied to every label (default 0.5). If the "
        "model was trained with --tune-thresholds, its own predict() already "
        "applies per-label thresholds and this value is ignored for *_pred "
        "columns derived from predict_proba below.",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Treat --text as already-cleaned text and skip clean_light/clean_heavy "
        "(the model was trained on cleaned text, so leave this off for normal use).",
    )
    return parser.parse_args()


def predict(
    texts: list[str],
    model_path: Path,
    threshold: float = 0.5,
    already_clean: bool = False,
) -> pd.DataFrame:
    """Load a saved model and return a per-label probability/prediction table."""

    model = joblib.load(model_path)

    if already_clean:
        cleaned = pd.Series(texts)
    else:
        cleaned = pd.Series(texts).apply(clean_light).apply(clean_heavy)

    scores = predict_scores(model, cleaned)
    if scores is None:
        raise ValueError(
            f"Model at {model_path} produces no rankable scores (no predict_proba "
            "or decision_function)."
        )
    preds = (scores >= threshold).astype(int)

    result = pd.DataFrame({"comment_text": texts})
    for i, label in enumerate(LABEL_COLUMNS):
        result[f"{label}_prob"] = scores[:, i].round(4)
        result[f"{label}_pred"] = preds[:, i]
    return result


def main() -> None:
    args = parse_args()
    if not args.model.exists():
        raise FileNotFoundError(
            f"Không thấy model tại {args.model}. Model chỉ được tạo sau khi chạy "
            "train_from_folds()/train_holdout() — ví dụ qua notebook "
            "train_roberta_label_dependency_folds.ipynb."
        )

    result = predict(args.text, args.model, threshold=args.threshold, already_clean=args.raw)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

