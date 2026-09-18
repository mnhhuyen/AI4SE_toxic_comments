import pandas as pd

from toxic_comments.config import HEAVY_TEXT_COLUMN, ID_COLUMN, LABEL_COLUMNS, TEXT_COLUMN
from toxic_comments.splits import (
    make_kfold_splits,
    make_stratified_kfold_splits,
    save_kfold_datasets,
)


def test_make_kfold_splits_can_be_reused_across_models():
    data = pd.DataFrame(
        {
            TEXT_COLUMN: [f"comment {index}" for index in range(10)],
            **{label: [index % 2 for index in range(10)] for label in LABEL_COLUMNS},
        }
    )

    first = make_kfold_splits(data, n_splits=5, random_state=42)
    second = make_kfold_splits(data, n_splits=5, random_state=42)

    assert len(first) == 5
    assert [
        (train_index.tolist(), test_index.tolist())
        for train_index, test_index in first
    ] == [
        (train_index.tolist(), test_index.tolist())
        for train_index, test_index in second
    ]


def test_save_kfold_datasets_writes_train_and_test_files(tmp_path):
    data = pd.DataFrame(
        {
            ID_COLUMN: [f"id_{index}" for index in range(10)],
            TEXT_COLUMN: [f"raw comment {index}" for index in range(10)],
            HEAVY_TEXT_COLUMN: [f"comment {index}" for index in range(10)],
            **{label: [index % 2 for index in range(10)] for label in LABEL_COLUMNS},
        }
    )

    written_files = save_kfold_datasets(data, output_dir=tmp_path, n_splits=5)

    assert len(written_files) == 10
    test_ids = []
    for fold_index in range(1, 6):
        train = pd.read_csv(tmp_path / f"fold_{fold_index}" / "train.csv")
        test = pd.read_csv(tmp_path / f"fold_{fold_index}" / "test.csv")

        assert len(train) + len(test) == len(data)
        test_ids.extend(test[ID_COLUMN].tolist())

    assert sorted(test_ids) == sorted(data[ID_COLUMN].tolist())


def test_make_stratified_kfold_splits_supports_multilabel_targets():
    data = pd.DataFrame(
        {
            TEXT_COLUMN: [f"comment {index}" for index in range(20)],
            **{
                label: [1 if index % (label_index + 2) == 0 else 0 for index in range(20)]
                for label_index, label in enumerate(LABEL_COLUMNS)
            },
        }
    )

    splits = make_stratified_kfold_splits(data, n_splits=5, random_state=42)

    assert len(splits) == 5
    assert sum(len(test_index) for _, test_index in splits) == len(data)
