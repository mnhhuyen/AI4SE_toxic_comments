# DPCNN results

DPCNN (frozen fastText 300d -> region conv, 250 filters -> 8 downsampling blocks -> global max pool -> 6 sigmoid outputs), trained with the same setup as the RNN models: 5 stratified folds (seed 42), input `comment_heavy`, 4 epochs, decision threshold 0.5. Trained on a Colab T4 GPU: 26.9 min for 5 folds (about 70 s per epoch). fastText covers 61.8% of the fold 1 vocabulary.

## Overall (mean ± std over 5 folds)

| Model | micro F1 | macro F1 | micro ROC-AUC | macro ROC-AUC |
|---|---|---|---|---|
| DPCNN | 0.769 ± 0.005 | 0.626 ± 0.012 | 0.991 ± 0.000 | 0.988 ± 0.001 |
| Bi-GRU | 0.762 ± 0.006 | 0.594 ± 0.021 | 0.991 ± 0.000 | 0.988 ± 0.001 |
| Bi-LSTM | 0.758 ± 0.007 | 0.538 ± 0.024 | 0.990 ± 0.000 | 0.987 ± 0.001 |
| Bi-RNN | 0.742 ± 0.023 | 0.518 ± 0.038 | 0.990 ± 0.001 | 0.985 ± 0.001 |

## F1 per label (mean over 5 folds)

| label | DPCNN | Bi-GRU | Bi-LSTM | Bi-RNN |
|:--|--:|--:|--:|--:|
| toxic | 0.807 | 0.800 | 0.802 | 0.787 |
| obscene | 0.827 | 0.825 | 0.825 | 0.810 |
| insult | 0.743 | 0.740 | 0.729 | 0.713 |
| identity_hate | 0.547 | 0.510 | 0.440 | 0.370 |
| severe_toxic | 0.428 | 0.404 | 0.357 | 0.389 |
| threat | 0.404 | 0.285 | 0.077 | 0.040 |

## ROC-AUC per label (mean over 5 folds)

| label | DPCNN | Bi-GRU | Bi-LSTM | Bi-RNN |
|:--|--:|--:|--:|--:|
| toxic | 0.983 | 0.982 | 0.982 | 0.981 |
| obscene | 0.993 | 0.993 | 0.992 | 0.992 |
| insult | 0.987 | 0.987 | 0.986 | 0.986 |
| identity_hate | 0.987 | 0.989 | 0.988 | 0.986 |
| severe_toxic | 0.990 | 0.990 | 0.990 | 0.989 |
| threat | 0.987 | 0.986 | 0.982 | 0.977 |

Figures: `dpcnn_vs_rnn.png`, `dpcnn_loss_curves.png`. Raw per-fold numbers: `dpcnn_fold_metrics.csv`, `dpcnn_fold_per_label.csv`.
