"""Hand-crafted numeric text features as a scikit-learn transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from toxic_comments.cleaning import build_features


class HandCraftedFeatures(BaseEstimator, TransformerMixin):
    """Scale the features of build_features() into a sparse matrix.

    Feed this the RAW text column: casing and punctuation are the signal,
    and heavy cleaning deletes exactly those characters.
    """

    def _frame(self, x) -> pd.DataFrame:
        series = pd.Series(x).reset_index(drop=True).fillna("").astype(str)
        features = build_features(series)
        return features.replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def fit(self, x, y=None):
        features = self._frame(x)
        self.feature_names_ = list(features.columns)
        # clip=True keeps the output non-negative for Naive Bayes even when a
        # test comment is longer than anything seen during fit.
        self.scaler_ = MinMaxScaler(clip=True).fit(features.to_numpy(dtype=float))
        return self

    def transform(self, x) -> sparse.csr_matrix:
        features = self._frame(x)[self.feature_names_]
        return sparse.csr_matrix(self.scaler_.transform(features.to_numpy(dtype=float)))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)
