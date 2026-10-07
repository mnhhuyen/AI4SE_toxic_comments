# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Multi-label classifier for the Kaggle Jigsaw Toxic Comment Classification Challenge (AI for Software Engineering course project). Six labels, defined once in `src/toxic_comments/config.py` (`LABEL_COLUMNS`): `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`. Some user-facing messages and comments are written in Vietnamese, and tests match on them (e.g. `"Không thấy model"`), so don't translate them.

## Setup

```bash
python -m venv .venv
pip install -e ".[dev]"            # package must be installed (src/ layout) for tests/imports
pip install -e ".[embeddings]"     # gensim + bpemb, for *_logistic_regression embedding models
pip install -e ".[transformers]"   # torch + transformers, for roberta_label_dependency
pip install -e ".[advanced]"       # lightgbm (+ tensorflow)
```

Raw Kaggle CSVs are tracked in `data/raw/`. Generated outputs (`data/processed`, `data/k_fold`, `data/embeddings`, `models/`, `results/`) are gitignored except `.gitkeep`/`README.md`. Pretrained embedding download instructions and expected filenames are in `data/embeddings/README.md`.

## Commands

```bash
pytest                                            # all tests
pytest tests/test_train.py::test_name             # single test
python -m toxic_comments.splits                   # write data/k_fold/fold_N/{train,test}.csv from data/processed/train_clean.csv
python -m toxic_comments.train --mode holdout --model tfidf_logistic_regression
python -m toxic_comments.train --mode folds --model nbsvm --tune-thresholds
python -m toxic_comments.train --mode full --model ...          # train on everything, no eval
python -m toxic_comments --include-transformers --folds 5       # CV experiment incl. RoBERTa
python -m toxic_comments.predict --model models/<...>.joblib --text "..."
```

Useful `train` flags: `--text-column` (default `comment_heavy`), `--embedding-pooling {mean,max,attention,max_attention}`, `--max-embedding-vectors N` (cap embedding load for smoke tests). Tests that need torch use `pytest.importorskip`.

## Architecture

**Data pipeline.** `cleaning.process_cleaning()` adds two cleaned text columns from `comment_text`: `comment_light` (case/punctuation preserved) and `comment_heavy` (derived from light; lowercased, stopwords dropped — the default model input), plus handcrafted features and `is_empty_light`/`is_empty_heavy`/`is_non_latin` flags. It dedupes and drops empty rows only when `is_train=True`. The cleaned file `data/processed/train_clean.csv` is produced by the notebooks (`notebooks/01_cleaning.ipynb`); `experiment.py` skips re-cleaning if those marker columns already exist. `predict.py` applies `clean_light` → `clean_heavy` to raw input unless `--raw`.

**Model registry.** `models/registry.build_models()` returns a dict `name → unfitted estimator`; this is the single place to register a model, and `--model` names come from it. Every model follows the sklearn contract: `fit(X: Series[str], y: ndarray[n, 6])`, `predict`, and `predict_proba`/`decision_function`. Most are sklearn `Pipeline`s (vectorizer + `OneVsRestClassifier`). `evaluation.predict_scores()` normalizes whichever score method exists into an `(n, 6)` array.

**Optional dependencies.** Models needing uninstalled packages must still be importable/registrable: use `models/_optional.MissingDependencyEstimator` (fails on `fit`) or raise `missing_dependency()` inside the builder. `roberta_label_dependency` imports torch at module level, so the registry wraps the *module import* in try/except. `dpcnn` is a stub not wired into the registry.

**Recurrent models.** `birnn_max_attention`, `bilstm_max_attention` and `bigru_max_attention` (factories in `models/rnn.py`, `bilstm.py` and `bigru.py`) all wrap `models/_rnn_base.RecurrentMultiLabelClassifier` (PyTorch). They differ only in `cell_type`. Each is bidirectional with max and attention pooling. The vocabulary is built per training fold, and embeddings are initialised from frozen fastText vectors (`embedding_file=None` trains them from scratch). Pickling follows the same CPU-safe pattern as RoBERTa. `notebooks/03_rnn_models.ipynb` runs all three on Colab.

**Embeddings.** `embeddings/` holds one vectorizer factory per source (fasttext, glove twitter, word2vec, lexvec, bpemb) sharing `pooling.py` (sklearn transformers with mean/max/attention/max_attention pooling). Text vector files load lazily on `fit`; BPEmb uses its own SentencePiece tokenizer.

**RoBERTa.** `models/_roberta_base.RobertaMultiLabelBase` is a sklearn-compatible estimator owning tokenization/training/inference; subclasses override `_build_model` and optionally `_compute_loss`. Constructor params must be stored verbatim (no derived state) so `sklearn.base.clone` works — `ThresholdedClassifier` and CV rely on it. Custom `__getstate__`/`__setstate__` move weights to CPU so joblib pickles load on CPU-only machines. `roberta_label_dependency.py` adds a label-dependency graph layer on top.

**Training/eval workflows.** Two entry points:
- `train.py` — one model at a time; holdout (stratified train/val/test), pre-built folds from `data/k_fold/`, or full. Saves `.joblib` to `models/` (`models/fold_N/` for folds) and metric CSVs (`*_holdout_metrics.csv`, `*_per_label.csv`, …) to `results/`.
- `experiment.py` (via `cli.py`/`__main__`, only when `--include-transformers` is passed; otherwise `cli` delegates to `train.main`) — cross-validates a small model set through a `DatasetRepository` (`repositories.py`), appending `cross_validation_results.csv` and refreshing `summary_results.csv` after every fold so long runs are inspectable/resumable.

Splits use multilabel stratification (`splits.make_multilabel_strata`), seed 42. `thresholds.ThresholdedClassifier` wraps any estimator to tune a per-label decision threshold instead of 0.5. `submission.py` builds Kaggle submission frames.
