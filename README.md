# Toxic Comment Classification

Project for **Artificial Intelligence for Software Engineering**.

The application solves the Kaggle Jigsaw Toxic Comment Classification Challenge as a
multi-label classification problem with six labels (defined in
`src/toxic_comments/config.py` as `LABEL_COLUMNS`):

- `toxic`
- `severe_toxic`
- `obscene`
- `threat`
- `insult`
- `identity_hate`

## Project Structure

```text
data/
  raw/                         # Kaggle CSVs: train, test, test_labels, sample_submission
  processed/                   # train_clean.csv produced by notebooks/01_cleaning.ipynb
  k_fold/fold_{1..5}/          # train.csv / test.csv written by toxic_comments.splits
  embeddings/                  # pretrained vectors (download manually, see data/embeddings/README.md)
    fasttext/  glove_twitter/  word2vec/  lexvec/  bpemb/en/
models/                        # saved models (.joblib, transformer checkpoints)
notebooks/
  00_toxic_comment_eda1.ipynb              # exploratory data analysis
  01_cleaning.ipynb                        # cleaning -> data/processed/train_clean.csv
  02_trainning.ipynb                       # classical baselines (TF-IDF, linear models)
  03_rnn_models.ipynb                      # BiRNN / BiLSTM / BiGRU comparison
  04_compare_embeddings.ipynb              # embedding sources x pooling strategies
  04_dpcnn.ipynb                           # DPCNN training
  04_logreg_gbm_dpcnn_comparison_v2.ipynb  # LogReg vs LightGBM vs DPCNN
  05_pooled_fasttext_lr_lightgbm.ipynb     # pooled FastText features with LR / LightGBM
  train_bigru.ipynb                        # BiGRU max+attention training
  train_roberta_standard_folds.ipynb       # RoBERTa baseline, k-fold
  train_roberta_label_dependency_folds.ipynb  # RoBERTa + label-dependency head, k-fold
  toxic_comment_seminar_2.ipynb            # seminar notebook
report/
  Report-Seminar2.pdf          # seminar 2 report
  figures/
    software-architecture.svg, dataset-backends.tex, implementation-workflows.tex
    rq1-representation-branches.tex, rq1_drawio.{png,svg}   # RQ1 diagrams
    rq2-three-branches.tex                                  # RQ2 diagram
    rq3-label-dependency.tex                                # RQ3 diagram
    bigru_architecture.png, roberta_standard_architecture.png, roberta_ld_architecture.png
results/                       # CV metrics, per-label metrics, plots, run folders (e.g. rq2_pooled_<timestamp>/)
src/toxic_comments/
  __main__.py, cli.py          # `python -m toxic_comments` entry point
  config.py                    # paths and label columns
  cleaning.py                  # comment_light / comment_heavy + handcrafted features
  splits.py                    # k-fold split generation (CLI)
  train.py                     # holdout / folds / full training (CLI)
  predict.py                   # predict from a saved model (CLI)
  experiment.py                # cross-validation experiment runner
  evaluation.py                # per-label + aggregate metrics, CV loop
  thresholds.py                # per-label decision threshold tuning
  submission.py                # Kaggle submission builder
  repositories.py              # dataset repositories (in-memory / CSV)
  embeddings/                  # FastText, GloVe Twitter, Word2Vec, LexVec, BPEmb + pooling.py
  models/
    registry.py                # name -> estimator factory (build_models)
    _optional.py               # placeholder for missing optional dependencies
    baseline.py, linear.py, nbsvm.py, tfidf_logreg.py, vectorizers.py, handcrafted.py
    lightgbm.py                # TF-IDF / FastText + LightGBM
    embedding_logreg.py        # pooled embeddings + logistic regression
    rnn.py, bilstm.py, bigru.py, bigru_shared.py, _rnn_base.py, _bigru_torch.py
    dpcnn.py, _dpcnn_torch.py
    roberta_standard.py, roberta_label_dependency.py, _roberta_base.py
tests/                         # pytest unit tests
pyproject.toml, requirements.txt
```

## Available Models

Names accepted by `--model` (see `src/toxic_comments/models/registry.py`):

| Family | Models |
| --- | --- |
| Baseline | `dummy_most_frequent` |
| TF-IDF / linear | `tfidf_logistic_regression`, `word_char_logistic_regression`, `raw_text_logistic_regression`, `nbsvm`, `tfidf_linear_svc`, `tfidf_sgd_logistic`, `tfidf_complement_nb` |
| Gradient boosting | `tfidf_lightgbm`, `fasttext_lightgbm` |
| Pooled embeddings + LR | `fasttext_logistic_regression`, `glove_twitter_logistic_regression`, `bpemb_logistic_regression`, `lexvec_logistic_regression` |
| Recurrent / CNN | `birnn_max_attention`, `bilstm_max_attention`, `bigru_max_attention`, `bigru_max_attention_glove`, `bigru_max_attention_random`, `bigru_shared_max_attention`, `dpcnn` |
| Transformers | `roberta_standard`, `roberta_label_dependency` |

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional extras (install only what you need):

```bash
pip install -e ".[advanced]"      # lightgbm, tensorflow
pip install -e ".[embeddings]"    # gensim, bpemb
pip install -e ".[transformers]"  # torch, transformers, tqdm
```

The Kaggle CSVs are already in `data/raw/`. Pretrained embeddings must be downloaded
manually into `data/embeddings/` — see `data/embeddings/README.md` for file names.

## Usage

1. **Clean the data** — run `notebooks/01_cleaning.ipynb` to produce
   `data/processed/train_clean.csv`.
2. **Create folds**

   ```bash
   python -m toxic_comments.splits --folds 5 --strategy stratified
   ```

3. **Train / evaluate**

   ```bash
   python -m toxic_comments.train --mode holdout --model tfidf_logistic_regression
   python -m toxic_comments.train --mode folds --model bigru_max_attention --tune-thresholds
   python -m toxic_comments --include-transformers --device cuda   # CV incl. RoBERTa
   ```

4. **Predict**

   ```bash
   python -m toxic_comments.predict --model models/<model>.joblib --text "your comment"
   ```

## Tests

```bash
pytest
pytest tests/test_bigru.py -k pickle
```

Tests that require torch are skipped automatically when it is not installed.
