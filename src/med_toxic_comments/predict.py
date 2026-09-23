"""Run inference with a saved toxic-comment classifier.

Loads a model saved via ``RobertaMultiLabelBase.save()`` — either the
automatic last-fold save from ``python -m toxic_comments --include-transformers``
(see ``experiment.run_experiment``), or a model saved from
``notebooks/train_and_save_roberta_label_dependency.ipynb`` — and predicts on
new comment text.

Usage
-----
    python -m toxic_comments.predict --text "you are so stupid" --text "have a nice day"
    python -m toxic_comments.predict --model models/roberta_label_dependency --text "..."
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from toxic_comments.cleaning import clean_heavy, clean_light
from toxic_comments.config import LABEL_COLUMNS, MODELS_DIR
from toxic_comments.models.roberta_label_dependency import RobertaLabelDependencyClassifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Classify comment text with a saved model")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "roberta_label_dependency",
        help="Directory a model was saved to via .save() (default: models/roberta_label_dependency).",
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
        help="Decision threshold applied to every label (default 0.5 — see the "
        "research design's note on per-label adaptive thresholds for why this "
        "is a simplification).",
    )
    parser.add_argument(
        "--raw",
        action="store_true",
        help="Treat --text as already-cleaned text and skip clean_light/clean_heavy "
        "(the model was trained on cleaned text, so leave this off for normal use).",
    )
    return parser.parse_args()


def predict(texts: list[str], model_dir: Path, threshold: float = 0.5, already_clean: bool = False) -> pd.DataFrame:
    """Load a saved model and return a per-label probability/prediction table."""

    model = RobertaLabelDependencyClassifier.load(model_dir)

    if already_clean:
        cleaned = pd.Series(texts)
    else:
        cleaned = pd.Series(texts).apply(clean_light).apply(clean_heavy)

    scores = model.predict_proba(cleaned)
    preds = (scores >= threshold).astype(int)

    result = pd.DataFrame({"comment_text": texts})
    for i, label in enumerate(LABEL_COLUMNS):
        result[f"{label}_prob"] = scores[:, i].round(4)
        result[f"{label}_pred"] = preds[:, i]
    return result


def main() -> None:
    args = parse_args()
    if not (args.model / "params.json").exists():
        raise FileNotFoundError(
            f"Không thấy model đã lưu tại {args.model} (thiếu params.json). "
            "Model chỉ được lưu khi chạy `--include-transformers` (mặc định "
            "auto-save fold cuối) hoặc qua notebook train_and_save_*.ipynb."
        )

    result = predict(args.text, args.model, threshold=args.threshold, already_clean=args.raw)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_colwidth", 40)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()

