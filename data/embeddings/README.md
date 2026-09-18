# Pretrained Embeddings

Place pretrained embedding files here. Large vector files are ignored by git.

Expected paths:

- `fasttext/wiki-news-300d-1M.vec`
- `glove_twitter/glove.twitter.27B.200d.txt`
- `bpemb/en/en.wiki.bpe.vs10000.model`
- `bpemb/en/en.wiki.bpe.vs10000.d300.w2v.bin`
- `word2vec/GoogleNews-vectors-negative300.txt`
- `lexvec/lexvec.commoncrawl.300d.W.pos.neg3.vectors`

The word embedding loader supports text vector files in this format:

```text
token 0.123 -0.456 ...
```

Word2Vec binary files such as `GoogleNews-vectors-negative300.bin` must be
converted to text format before training with this project.

BPEmb uses its own SentencePiece tokenizer and binary subword vectors. Install
`pip install -e '.[embeddings]'` in the notebook's environment. Its files are
downloaded to the paths above on the first fit if missing, then reused locally.
Importing the package does not load or download embeddings.

All five vectorizers support `mean`, `max`, `attention`, and `max_attention`.
Attention here uses a global query computed from training token vectors; it is
not a neural attention layer trained with classification labels.
`max_attention` outputs twice the embedding dimension, including for empty text.
`max_vectors` limits text-file loading for smoke tests; BPEmb retains its full
subword vocabulary so tokenizer IDs remain aligned with the embedding matrix.
Google News lookup preserves case. Prefer `comment_light` when retaining case
and punctuation is important; lowercased input cannot recover proper-name case.
FastText `.vec` files provide fixed word vectors, not subword inference for OOVs.
