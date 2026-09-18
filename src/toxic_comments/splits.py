"""Create reusable dataset splits for training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split

from toxic_comments.config import (
    HEAVY_TEXT_COLUMN,
    LABEL_COLUMNS,
    K_FOLD_DATA_DIR,
    PROCESSED_DATA_DIR,
)

FoldSplit = tuple[np.ndarray, np.ndarray]


def make_multilabel_strata(
    data: pd.DataFrame,
    n_splits: int = 5,
    label_columns: list[str] | None = None,
) -> pd.Series:
    """Build stratification labels from multi-label target combinations."""

    if label_columns is None:
        label_columns = LABEL_COLUMNS

    missing = [column for column in label_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing label columns: {', '.join(missing)}")

    label_combinations = data[label_columns].astype(int).astype(str).agg("".join, axis=1)
    combination_counts = label_combinations.value_counts()
    rare_combinations = combination_counts[combination_counts < n_splits].index
    return label_combinations.mask(
        label_combinations.isin(rare_combinations),
        "rare_label_combo",
    )


def make_kfold_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    text_column: str = HEAVY_TEXT_COLUMN,
) -> list[FoldSplit]:
    """Create reusable k-fold train/test index splits for fair model comparison."""

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(data))


def make_stratified_kfold_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    label_columns: list[str] | None = None,
) -> list[FoldSplit]:
    """Create stratified splits using multi-label combinations as strata."""

    strata = make_multilabel_strata(
        data,
        n_splits=n_splits,
        label_columns=label_columns,
    )
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(splitter.split(data, strata))


def train_validation_test_split(
    data: pd.DataFrame,
    validation_size: float = 0.1,
    test_size: float = 0.1,
    random_state: int = 42,
    stratified: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split one dataframe into train, validation, and test dataframes."""

    if validation_size <= 0 or test_size <= 0 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be > 0 and sum to < 1")

    strata = None
    if stratified:
        strata = make_multilabel_strata(data, n_splits=2)

    train_data, temp_data = train_test_split(
        data,
        test_size=validation_size + test_size,
        random_state=random_state,
        shuffle=True,
        stratify=strata,
    )

    temp_strata = None
    if stratified:
        temp_strata = make_multilabel_strata(temp_data, n_splits=2)

    validation_fraction = validation_size / (validation_size + test_size)
    validation_data, test_data = train_test_split(
        temp_data,
        test_size=1 - validation_fraction,
        random_state=random_state,
        shuffle=True,
        stratify=temp_strata,
    )

    return (
        train_data.reset_index(drop=True),
        validation_data.reset_index(drop=True),
        test_data.reset_index(drop=True),
    )


def make_fold_splits(
    data: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    text_column: str = HEAVY_TEXT_COLUMN,
    strategy: str = "stratified",
) -> list[FoldSplit]:
    """Create reusable train/test index splits for fair model comparison."""

    if strategy == "kfold":
        return make_kfold_splits(
            data,
            n_splits=n_splits,
            random_state=random_state,
            text_column=text_column,
        )
    if strategy == "stratified":
        return make_stratified_kfold_splits(
            data,
            n_splits=n_splits,
            random_state=random_state,
        )
    raise ValueError("strategy must be either 'kfold' or 'stratified'")


def save_kfold_datasets(
    data: pd.DataFrame,
    output_dir: Path = K_FOLD_DATA_DIR,
    n_splits: int = 5,
    random_state: int = 42,
    text_column: str = HEAVY_TEXT_COLUMN,
    strategy: str = "stratified",
) -> list[Path]:
    """Save fold_N/train.csv and fold_N/test.csv files from one dataframe."""

    splits = make_fold_splits(
        data,
        n_splits=n_splits,
        random_state=random_state,
        text_column=text_column,
        strategy=strategy,
    )
    written_files: list[Path] = []

    for fold_index, (train_index, test_index) in enumerate(splits, start=1):
        fold_dir = output_dir / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_path = fold_dir / "train.csv"
        test_path = fold_dir / "test.csv"

        data.iloc[train_index].to_csv(train_path, index=False)
        data.iloc[test_index].to_csv(test_path, index=False)

        written_files.extend([train_path, test_path])

    return written_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create split CSV files.")
    parser.add_argument(
        "--data",
        type=Path,
        default=PROCESSED_DATA_DIR / "train_clean.csv",
        help="Path to cleaned training CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=K_FOLD_DATA_DIR,
        help="Directory where fold_N/train.csv and fold_N/test.csv are written.",
    )
    parser.add_argument("--folds", type=int, default=5, help="Number of folds.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--strategy",
        choices=["stratified", "kfold"],
        default="stratified",
        help="Split strategy.",
    )
    parser.add_argument(
        "--text-column",
        default=HEAVY_TEXT_COLUMN,
        help="Column used only to determine the number of rows to split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.data)
    written_files = save_kfold_datasets(
        data=data,
        output_dir=args.output,
        n_splits=args.folds,
        random_state=args.random_state,
        text_column=args.text_column,
        strategy=args.strategy,
    )

    print(f"Saved {len(written_files)} files to {args.output}")
    for path in written_files:
        print(path)


if __name__ == "__main__":
    main()
