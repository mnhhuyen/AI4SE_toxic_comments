"""Business workflow for loading data, training models, and saving results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from toxic_comments.config import HEAVY_TEXT_COLUMN, MODELS_DIR
from toxic_comments.evaluation import cross_validate_model, summarize_results
from toxic_comments.folds import make_kfold_splits
from toxic_comments.cleaning import process_cleaning
from toxic_comments.repositories import DatasetRepository, validate_training_data
from toxic_comments.models.registry import build_models

# Columns process_cleaning() adds — if the loaded file already has them
# (e.g. a pre-cleaned file like data/processed/train_clean.csv), cleaning is
# skipped instead of redone, since it's already exactly that output.
_CLEANED_MARKER_COLUMNS = {HEAVY_TEXT_COLUMN, "is_empty_heavy"}


def _load_and_prepare_data(repository: DatasetRepository) -> pd.DataFrame:
    data = validate_training_data(repository.load())
    if _CLEANED_MARKER_COLUMNS.issubset(data.columns):
        print(
            f"Data đã có sẵn cột {HEAVY_TEXT_COLUMN!r} — bỏ qua process_cleaning(), "
            "dùng thẳng file đã clean."
        )
    else:
        data = process_cleaning(data, is_train=True, verbose=False)
    return data[data["is_empty_heavy"] == 0].reset_index(drop=True)


def run_experiment(
    repository: DatasetRepository,
    output_dir: Path,
    n_splits: int = 5,
    max_features: int = 50_000,
    include_transformer_models: bool = False,
    device: str | None = None,
    save_transformer_models: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run baseline and ML classifier evaluation with a repository abstraction.

    Fold results are appended to ``cross_validation_results.csv`` and the
    summary is refreshed after every fold — not just once at the very end.
    This matters most for the slow transformer models (see
    ``models/roberta_label_dependency.py``): a long-running fine-tune can
    now be inspected mid-run (e.g. with the ``view_experiment_results``
    notebook) and, if it crashes or is interrupted, the folds/models that
    already finished are not lost.

    ``cross_validate_model`` clones + fits + discards a model per fold, by
    design, since the point of CV is comparing methods, not producing a
    deployable artifact. But when ``save_transformer_models`` is True
    (default), the LAST fold's fitted transformer model is kept and saved to
    ``models/<model_name>/`` via its ``.save()`` — so a normal CLI/CV run
    also leaves behind a model usable for inference (see ``predict.py``),
    without needing a second, separate training pass just for that.
    """

    data = _load_and_prepare_data(repository)
    output_dir.mkdir(parents=True, exist_ok=True)

    models = build_models(
        max_features=max_features,
        include_transformer_models=include_transformer_models,
        device=device,
    )
    splits = make_kfold_splits(data, n_splits=n_splits, text_column=HEAVY_TEXT_COLUMN)

    fold_results_path = output_dir / "cross_validation_results.csv"
    summary_path = output_dir / "summary_results.csv"
    # Start this run from a clean file rather than appending to a stale one
    # left over from a previous run.
    if fold_results_path.exists():
        fold_results_path.unlink()

    all_results: list[pd.DataFrame] = []

    def _persist_fold(result) -> None:
        row = pd.DataFrame([result.__dict__])
        row.to_csv(
            fold_results_path,
            mode="a",
            header=not fold_results_path.exists(),
            index=False,
        )
        # Cheap to recompute after every fold, and means summary_results.csv
        # always reflects everything finished so far, not just the final state.
        so_far = pd.concat(all_results + [row], ignore_index=True)
        summarize_results(so_far).to_csv(summary_path)

    for name, model in models.items():
        last_fitted = {}

        def _on_fold_complete(result, fold_estimator, _last_fitted=last_fitted) -> None:
            _persist_fold(result)
            _last_fitted["estimator"] = fold_estimator

        model_fold_results = cross_validate_model(
            model,
            data,
            model_name=name,
            n_splits=n_splits,
            text_column=HEAVY_TEXT_COLUMN,
            splits=splits,
            on_fold_complete=_on_fold_complete,
        )
        all_results.append(model_fold_results)

        fitted = last_fitted.get("estimator")
        if save_transformer_models and fitted is not None and hasattr(fitted, "save"):
            save_path = MODELS_DIR / name
            fitted.save(save_path)
            print(
                f"Đã lưu model '{name}' (fold cuối) tại {save_path} — "
                f"dùng `python -m toxic_comments.predict --model {save_path}` để infer."
            )

    fold_results = pd.concat(all_results, ignore_index=True)
    summary = summarize_results(fold_results)
    summary.to_csv(summary_path)
    return fold_results, summary

