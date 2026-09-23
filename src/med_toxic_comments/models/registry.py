from toxic_comments.models.baseline import build_dummy_baseline
from toxic_comments.models.tfidf_logreg import build_tfidf_logistic_regression


def build_models(
    max_features: int = 50_000,
    include_transformer_models: bool = False,
    device: str | None = None,
):
    """Return the registry of models to evaluate.

    ``include_transformer_models`` is off by default: fine-tuning a RoBERTa
    model inside a 5-fold CV loop is far more expensive than the sklearn
    baselines above, so leaving it off keeps existing quick runs
    (``python -m toxic_comments``) and the test suite fast. Pass
    ``--include-transformers`` on the CLI, or the flag directly here, to add
    Method 3 (and, once implemented, Methods 1/2/4) to the comparison.

    ``device`` is forwarded to the transformer model(s) — ``None`` (default)
    auto-detects GPU vs CPU; pass ``"cuda"``/``"cpu"`` to force one.
    """

    models = {
        "dummy_most_frequent": build_dummy_baseline(),
        "tfidf_logistic_regression": build_tfidf_logistic_regression(
            max_features=max_features
        ),
    }

    if include_transformer_models:
        from toxic_comments.models.roberta_label_dependency import (
            build_roberta_label_dependency,
        )

        models["roberta_label_dependency"] = build_roberta_label_dependency(device=device)

    return models

