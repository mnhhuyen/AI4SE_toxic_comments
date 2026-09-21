import numpy as np
import pandas as pd
import pytest

from toxic_comments.config import HEAVY_TEXT_COLUMN, LABEL_COLUMNS, TEXT_COLUMN
from toxic_comments.models.handcrafted import HandCraftedFeatures
from toxic_comments.models.nbsvm import NbFeaturer, build_nbsvm
from toxic_comments.models.registry import build_models
from toxic_comments.models.linear import (
    build_raw_text_logistic_regression,
    build_tfidf_complement_nb,
    build_tfidf_linear_svc,
    build_tfidf_sgd_logistic,
    build_word_char_logistic_regression,
)
from scipy import sparse

TOXIC = "You are a stupid idiot and you should shut up now"
CLEAN = "Thank you for the helpful edit to this article page"


def _training_data(n_rows: int = 20) -> pd.DataFrame:
    rows = []
    for index in range(n_rows):
        is_toxic = index % 2 == 0
        rows.append(
            {
                TEXT_COLUMN: f"{TOXIC if is_toxic else CLEAN} {index}",
                HEAVY_TEXT_COLUMN: f"{TOXIC if is_toxic else CLEAN} {index}".lower(),
                "toxic": int(is_toxic),
                "severe_toxic": int(index % 4 == 0),
                "obscene": int(index % 3 == 0),
                "threat": int(index % 5 == 0),
                "insult": int(index % 3 == 1),
                "identity_hate": int(index % 7 == 0),
            }
        )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "builder",
    [
        build_word_char_logistic_regression,
        build_tfidf_linear_svc,
        build_tfidf_sgd_logistic,
        build_tfidf_complement_nb,
        build_nbsvm,
    ],
)
def test_models_fit_and_predict(builder):
    data = _training_data()
    x = data[HEAVY_TEXT_COLUMN]
    y = data[LABEL_COLUMNS].to_numpy()

    predictions = builder(max_features=200).fit(x, y).predict(x)

    assert predictions.shape == (len(data), len(LABEL_COLUMNS))
    assert set(np.unique(predictions)) <= {0, 1}


def test_raw_text_model_fits_on_uncleaned_text():
    data = _training_data()
    x = data[TEXT_COLUMN]
    y = data[LABEL_COLUMNS].to_numpy()

    model = build_raw_text_logistic_regression(max_features=200).fit(x, y)

    assert model.predict(x).shape == (len(data), len(LABEL_COLUMNS))


def test_registry_exposes_the_new_models():
    models = build_models(max_features=20)

    for name in [
        "word_char_logistic_regression",
        "raw_text_logistic_regression",
        "nbsvm",
        "tfidf_linear_svc",
        "tfidf_sgd_logistic",
        "tfidf_complement_nb",
    ]:
        assert name in models


def test_hand_crafted_features_stay_non_negative():
    transformer = HandCraftedFeatures().fit(_training_data()[TEXT_COLUMN])

    matrix = transformer.transform(pd.Series(["tiny", "WORD " * 500 + "!" * 100]))

    assert matrix.min() >= 0.0
    assert matrix.max() <= 1.0


def test_nb_featurer_ranks_positive_features_higher():
    counts = sparse.csr_matrix(np.array([[1, 1, 0], [1, 1, 0], [0, 1, 1], [0, 1, 1]]))
    y = np.array([1, 1, 0, 0])

    ratios = NbFeaturer().fit(counts, y).log_ratio_.toarray().ravel()

    assert ratios[0] > ratios[1] > ratios[2]


def test_nb_featurer_handles_a_label_with_no_positives():
    counts = sparse.csr_matrix(np.array([[1, 0], [0, 1], [1, 1]]))

    ratios = NbFeaturer().fit(counts, np.zeros(3, dtype=int)).log_ratio_.toarray()

    assert np.all(np.isfinite(ratios))
