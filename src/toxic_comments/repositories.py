"""Dataset persistence abstractions.

The training and evaluation code depends on DatasetRepository, not on a
specific storage detail. Switching between in-memory data and CSV files should
therefore require only a different repository object.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pandas as pd

from toxic_comments.config import LABEL_COLUMNS, TEXT_COLUMN


class DatasetRepository(Protocol):
    """Common interface for dataset persistence layers."""

    def load(self) -> pd.DataFrame:
        """Return a dataframe containing comment text and label columns."""

    def save(self, data: pd.DataFrame) -> None:
        """Persist a dataframe containing comment text and label columns."""


@dataclass
class InMemoryDatasetRepository:
    """Repository useful for notebooks, tests, and small experiments."""

    data: pd.DataFrame

    def load(self) -> pd.DataFrame:
        return self.data.copy()

    def save(self, data: pd.DataFrame) -> None:
        self.data = data.copy()


@dataclass
class CsvFileDatasetRepository:
    """Repository backed by CSV files on the local filesystem."""

    path: Path

    def load(self) -> pd.DataFrame:
        return pd.read_csv(self.path)

    def save(self, data: pd.DataFrame) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(self.path, index=False)


def validate_training_data(data: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize the columns needed by the classifiers."""

    required_columns = [TEXT_COLUMN, *LABEL_COLUMNS]
    missing = [column for column in required_columns if column not in data.columns]
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    normalized = data.copy()
    normalized[TEXT_COLUMN] = normalized[TEXT_COLUMN].fillna("").astype(str)
    for label in LABEL_COLUMNS:
        normalized[label] = pd.to_numeric(normalized[label], errors="coerce").fillna(0).astype(int)
    return normalized
