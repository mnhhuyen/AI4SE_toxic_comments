"""Helpers for optional model dependencies."""

from __future__ import annotations


def missing_dependency(package_name: str, model_name: str) -> ImportError:
    return ImportError(
        f"{model_name} requires optional dependency '{package_name}'. "
        f"Install it before training this model."
    )


class MissingDependencyEstimator:
    """Estimator placeholder that fails clearly when fit is called."""

    def __init__(self, package_name: str, model_name: str) -> None:
        self.package_name = package_name
        self.model_name = model_name

    def fit(self, x, y=None):
        raise missing_dependency(self.package_name, self.model_name)

    def predict(self, x):
        raise missing_dependency(self.package_name, self.model_name)

    def predict_proba(self, x):
        raise missing_dependency(self.package_name, self.model_name)
