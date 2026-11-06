"""Evaluate fold-local supervised FastText pooling with LR and LightGBM."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import json
import time
import hashlib
import platform
import numpy as np
import pandas as pd
import torch
import sklearn
import lightgbm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from toxic_comments.embeddings.pooling import MaxSupervisedAttentionEmbeddingVectorizer

LABELS = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
OUT = ROOT / 'results' / ('rq2_pooled_' + time.strftime('%Y%m%d_%H%M%S'))
OUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(4)
config = dict(text_column='comment_heavy', labels=LABELS, seed=42,
    embedding=str(ROOT / 'data/embeddings/fasttext/wiki-news-300d-1M.vec'),
    pooling='max_supervised_attention', max_length=200, attention_epochs=3,
    attention_batch_size=256, attention_learning_rate=0.001, attention_dropout=0.1,
    device='cpu', lr_C=1.0, lr_class_weight='balanced', lr_max_iter=1000,
    lightgbm_n_estimators=300, lightgbm_learning_rate=0.05, lightgbm_num_leaves=31,
    threshold=0.5, std_ddof=0, python=platform.python_version(),
    torch=torch.__version__, sklearn=sklearn.__version__, lightgbm=lightgbm.__version__)
(OUT / 'config.json').write_text(json.dumps(config, indent=2))
print('Output:', OUT, flush=True)
rows, per_label, membership = [], [], []
seen_test = set()
for fold in range(1, 6):
    start = time.perf_counter()
    folder = ROOT / 'data/k_fold' / f'fold_{fold}'
    train = pd.read_csv(folder / 'train.csv', usecols=['id', 'comment_heavy'] + LABELS)
    test = pd.read_csv(folder / 'test.csv', usecols=['id', 'comment_heavy'] + LABELS)
    train_ids, test_ids = set(train.id), set(test.id)
    assert len(train_ids) == len(train) and len(test_ids) == len(test)
    assert not train_ids & test_ids, 'Train/test ID overlap'
    assert not seen_test & test_ids, 'Repeated test IDs across folds'
    seen_test.update(test_ids)
    membership.append(dict(fold=fold, train_size=len(train), test_size=len(test),
        train_ids_sha256=hashlib.sha256('\n'.join(sorted(map(str, train_ids))).encode()).hexdigest(),
        test_ids_sha256=hashlib.sha256('\n'.join(sorted(map(str, test_ids))).encode()).hexdigest()))
    (OUT / 'fold_membership.json').write_text(json.dumps(membership, indent=2))
    ytr, yte = train[LABELS].to_numpy(), test[LABELS].to_numpy()
    pool = MaxSupervisedAttentionEmbeddingVectorizer(config['embedding'],
        max_length=200, num_epochs=3, batch_size=256, random_state=42, device='cpu')
    print(f'Fold {fold}: fitting attention on {len(train)} training rows', flush=True)
    pool.fit(train.comment_heavy.fillna(''), ytr)
    print(f'Fold {fold}: attention loss {pool.loss_history_}; extracting features', flush=True)
    xtr = pool.transform(train.comment_heavy.fillna(''))
    xte = pool.transform(test.comment_heavy.fillna(''))
    pool_seconds = time.perf_counter() - start
    np.savez_compressed(OUT / f'fold_{fold}_attention.npz', W=pool.attention_W_, b=pool.attention_b_, v=pool.attention_v_, loss=pool.loss_history_)
    del pool
    for name in ['LR', 'LightGBM']:
        tick = time.perf_counter()
        if name == 'LR':
            scaler = StandardScaler()
            a, b = scaler.fit_transform(xtr), scaler.transform(xte)
            model = OneVsRestClassifier(LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, random_state=42))
        else:
            a, b = xtr, xte
            model = OneVsRestClassifier(lightgbm.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42, n_jobs=4, verbosity=-1))
        print(f'Fold {fold}: training {name}', flush=True)
        model.fit(a, ytr)
        scores = model.predict_proba(b)
        pred = scores >= 0.5
        record = dict(fold=fold, model=name, macro_roc_auc=roc_auc_score(yte, scores, average='macro'),
            micro_roc_auc=roc_auc_score(yte, scores, average='micro'),
            macro_f1=f1_score(yte, pred, average='macro', zero_division=0),
            micro_f1=f1_score(yte, pred, average='micro', zero_division=0),
            micro_precision=precision_score(yte, pred, average='micro', zero_division=0),
            micro_recall=recall_score(yte, pred, average='micro', zero_division=0),
            pooling_seconds=pool_seconds, classifier_seconds=time.perf_counter()-tick)
        rows.append(record)
        np.savez_compressed(OUT / f'fold_{fold}_{name}_predictions.npz', ids=test.id.to_numpy(dtype=str), targets=yte, scores=scores)
        for j, label in enumerate(LABELS):
            per_label.append(dict(fold=fold, model=name, label=label, roc_auc=roc_auc_score(yte[:,j], scores[:,j]), f1=f1_score(yte[:,j], pred[:,j], zero_division=0)))
        pd.DataFrame(rows).to_csv(OUT / 'fold_metrics.csv', index=False)
        pd.DataFrame(per_label).to_csv(OUT / 'per_label_metrics.csv', index=False)
        print(record, flush=True)
        del model, a, b
    del train, test, xtr, xte
frame = pd.DataFrame(rows)
metrics = ['macro_roc_auc', 'micro_roc_auc', 'macro_f1', 'micro_f1', 'micro_precision', 'micro_recall']
summary = []
for name, group in frame.groupby('model'):
    for metric in metrics:
        summary.append(dict(model=name, metric=metric, mean=group[metric].mean(), std=group[metric].std(ddof=0), folds=len(group)))
pd.DataFrame(summary).to_csv(OUT / 'summary.csv', index=False)
print('COMPLETE', OUT, flush=True)
print(pd.DataFrame(summary).to_string(index=False), flush=True)
