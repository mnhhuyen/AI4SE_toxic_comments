"""Model factories for the toxic comment classifiers."""

from __future__ import annotations

from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline


def build_dummy_baseline() -> MultiOutputClassifier:
    """Return a simple baseline that predicts the most frequent class."""

    return MultiOutputClassifier(DummyClassifier(strategy="most_frequent"))