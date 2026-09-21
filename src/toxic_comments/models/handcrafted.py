"""Hand-crafted numeric text features as a scikit-learn transformer."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler

from toxic_comments.cleaning import build_features


class HandCraftedFeatures(BaseEstimator, TransformerMixin):
    """Scale build_features() output into a sparse matrix. Expects raw text."""

    def _frame(self, x) -> pd.DataFrame:
        series = pd.Series(x).reset_index(drop=True).fillna("").astype(str)
        return build_features(series).replace([np.inf, -np.inf], 0.0).fillna(0.0)

    def fit(self, x, y=None):
        features = self._frame(x)
        self.feature_names_ = list(features.columns)
        # clip keeps the output non-negative for Naive Bayes on unseen extremes
        self.scaler_ = MinMaxScaler(clip=True).fit(features.to_numpy(dtype=float))
        return self

    def transform(self, x) -> sparse.csr_matrix:
        features = self._frame(x)[self.feature_names_]
        return sparse.csr_matrix(self.scaler_.transform(features.to_numpy(dtype=float)))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.feature_names_, dtype=object)
