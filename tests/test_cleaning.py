import pandas as pd

from toxic_comments.config import HEAVY_TEXT_COLUMN, LABEL_COLUMNS, LIGHT_TEXT_COLUMN, TEXT_COLUMN
from toxic_comments.cleaning import clean_heavy, clean_light, process_cleaning, validate_training_data


def test_clean_light_removes_markup_and_web_noise():
    text = "[[User:Example]] '''Hello!!!!''' visit https://example.com 12:30, 1 January 2020 (UTC)"

    cleaned = clean_light(text)

    assert "User:Example" not in cleaned
    assert "https://example.com" not in cleaned
    assert "UTC" not in cleaned
    assert "Hello!!!" in cleaned


def test_clean_heavy_deobfuscates_and_expands_contractions():
    cleaned = clean_heavy("You're f.u.c.k.i.n.g annoying!!!!")

    assert "you" in cleaned
    assert "are" not in cleaned
    assert "fucking" in cleaned
    assert "annoying" in cleaned


def test_process_cleaning_adds_clean_text_and_feature_columns():
    data = pd.DataFrame(
        {
            TEXT_COLUMN: ["YOU are awful!!!", ""],
            **{label: [1, 0] for label in LABEL_COLUMNS},
        }
    )

    cleaned = process_cleaning(data, is_train=True, verbose=False)

    assert len(cleaned) == 1
    assert LIGHT_TEXT_COLUMN in cleaned.columns
    assert HEAVY_TEXT_COLUMN in cleaned.columns
    assert "caps_ratio" in cleaned.columns
    assert cleaned.loc[0, HEAVY_TEXT_COLUMN] == "you awful"


def test_validate_training_data_fills_missing_text_and_labels():
    data = pd.DataFrame(
        {
            TEXT_COLUMN: [None],
            **{label: [None] for label in LABEL_COLUMNS},
        }
    )

    validated = validate_training_data(data)

    assert validated.loc[0, TEXT_COLUMN] == ""
    assert validated[LABEL_COLUMNS].iloc[0].tolist() == [0, 0, 0, 0, 0, 0]
