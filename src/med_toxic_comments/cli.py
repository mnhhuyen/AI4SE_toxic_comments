"""Command line entry point for local runs and reproducible experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from toxic_comments.config import PROCESSED_DATA_DIR, RESULTS_DIR
from toxic_comments.experiment import run_experiment
from toxic_comments.repositories import CsvFileDatasetRepository


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Toxic comment classification experiment")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROCESSED_DATA_DIR / "train_clean.csv",
        help=(
            "Path to the training CSV. Can be the raw Kaggle train.csv, or an "
            "already-cleaned file (with comment_heavy/is_empty_heavy columns, "
            "like data/processed/train_clean.csv) — cleaning is auto-skipped "
            "when those columns are already present."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=RESULTS_DIR,
        help="Directory where result CSV files will be written.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of k-fold splits.")
    parser.add_argument("--max-features", type=int, default=50_000, help="TF-IDF vocabulary size.")
    parser.add_argument(
        "--include-transformers",
        action="store_true",
        help=(
            "Also fine-tune the RoBERTa-based models (e.g. Method 3, label "
            "dependency). Off by default because it fine-tunes a transformer "
            "once per fold — consider a smaller --folds value (e.g. 3) when "
            "this is set."
        ),
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help=(
            "Force the transformer models onto this device. Default (unset) "
            "auto-detects GPU vs CPU. Pass --device cuda to fail fast if no "
            "GPU is actually available, instead of silently falling back to "
            "a very slow CPU run."
        ),
    )
    parser.add_argument(
        "--no-save-models",
        dest="save_models",
        action="store_false",
        help=(
            "Don't save the last fold's fitted transformer model after "
            "training (saved to models/<name>/ by default when "
            "--include-transformers is set — useful for quick sanity-check "
            "runs where you don't want to overwrite the real saved model)."
        ),
    )
    parser.set_defaults(save_models=True)
    return parser.parse_args()


def _print_device_status(requested_device: str | None) -> None:
    """Print which device the transformer models will actually run on.

    Mirrors the GPU-check cell in
    notebooks/train_and_save_roberta_label_dependency.ipynb, so this is
    visible in the terminal too instead of only happening silently inside
    ``_roberta_base.RobertaMultiLabelBase.fit``.
    """

    import torch

    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "--device cuda was requested but torch.cuda.is_available() is "
            "False — no GPU visible in this environment. Fix the GPU setup "
            "(or drop --device to fall back to CPU) before running."
        )

    device = requested_device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        print(f"✅ Dùng GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠️  KHÔNG CÓ GPU — đang chạy CPU, sẽ rất chậm cho model transformer.")


def main() -> None:
    args = parse_args()
    if args.include_transformers:
        _print_device_status(args.device)

    repository = CsvFileDatasetRepository(args.data)
    fold_results, summary = run_experiment(
        repository=repository,
        output_dir=args.output,
        n_splits=args.folds,
        max_features=args.max_features,
        include_transformer_models=args.include_transformers,
        device=args.device,
        save_transformer_models=args.save_models,
    )
    print(f"Saved fold metrics: {args.output / 'cross_validation_results.csv'}")
    print(f"Saved summary metrics: {args.output / 'summary_results.csv'}")
    print(summary)

