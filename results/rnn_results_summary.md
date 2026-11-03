# RNN results: Bi-RNN vs Bi-LSTM vs Bi-GRU

Same architecture for all three (frozen fastText 300d → bidirectional cell, 128 units/direction → max + attention pooling → 6 sigmoid outputs); only the recurrent cell differs.
5 stratified folds (seed 42), input `comment_heavy`, 4 epochs, decision threshold 0.5. Trained on Apple M4 Pro GPU (MPS).

## Overall (mean ± std over 5 folds)

| Model | micro F1 | macro F1 | micro ROC-AUC | macro ROC-AUC | Train time (5 folds) |
|---|---|---|---|---|---|
| Bi-GRU | 0.762 ± 0.006 | 0.594 ± 0.021 | 0.991 ± 0.000 | 0.988 ± 0.001 | 45 min |
| Bi-LSTM | 0.758 ± 0.007 | 0.538 ± 0.024 | 0.990 ± 0.000 | 0.987 ± 0.001 | 46 min |
| Bi-RNN | 0.742 ± 0.023 | 0.518 ± 0.038 | 0.990 ± 0.001 | 0.985 ± 0.001 | 36 min |

## F1 per label (mean over 5 folds)

| label         |   Bi-RNN |   Bi-LSTM |   Bi-GRU |
|:--------------|---------:|----------:|---------:|
| toxic         |    0.787 |     0.802 |    0.8   |
| obscene       |    0.81  |     0.825 |    0.825 |
| insult        |    0.713 |     0.729 |    0.74  |
| identity_hate |    0.37  |     0.44  |    0.51  |
| severe_toxic  |    0.389 |     0.357 |    0.404 |
| threat        |    0.04  |     0.077 |    0.285 |

## ROC-AUC per label (mean over 5 folds)

| label         |   Bi-RNN |   Bi-LSTM |   Bi-GRU |
|:--------------|---------:|----------:|---------:|
| toxic         |    0.981 |     0.982 |    0.982 |
| obscene       |    0.992 |     0.992 |    0.992 |
| insult        |    0.986 |     0.986 |    0.987 |
| identity_hate |    0.986 |     0.988 |    0.989 |
| severe_toxic  |    0.989 |     0.99  |    0.99  |
| threat        |    0.977 |     0.982 |    0.986 |

Figures: `rnn_comparison.png`, `rnn_loss_curves.png`. Raw per-fold numbers: `*_fold_metrics.csv`, `*_fold_per_label.csv`.
