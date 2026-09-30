"""Command-line entry point for classical and transformer workflows."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _transformer_parser() -> argparse.ArgumentParser:
    from toxic_comments.config import PROCESSED_DATA_DIR, RESULTS_DIR

    parser = argparse.ArgumentParser(description="Toxic comment classification experiment")
    parser.add_argument("--data", type=Path, default=PROCESSED_DATA_DIR / "train_clean.csv")
    parser.add_argument("--output", type=Path, default=RESULTS_DIR)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--include-transformers", action="store_true")
    parser.add_argument("--device", choices=["cuda", "cpu"], default=None)
    parser.add_argument("--no-save-models", dest="save_models", action="store_false")
    parser.set_defaults(save_models=True)
    return parser


def _run_transformer_workflow() -> None:
    import torch
    from toxic_comments.experiment import run_experiment
    from toxic_comments.repositories import CsvFileDatasetRepository

    args = _transformer_parser().parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but no CUDA device is available")
    if args.include_transformers:
        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
    fold_results, summary = run_experiment(
        repository=CsvFileDatasetRepository(args.data),
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


def main() -> None:
    # Preserve the established train.py CLI unless the new experiment flag is used.
    if "--include-transformers" in sys.argv:
        _run_transformer_workflow()
        return
    from toxic_comments.train import main as train_main
    train_main()
