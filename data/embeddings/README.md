# Pretrained Embeddings

Place pretrained embedding files here. 

## Setup

```bash
python -m pip install -e .
python -m pip install gensim bpemb
mkdir -p data/embeddings/{fasttext,glove_twitter,word2vec,bpemb,lexvec}
```

Run the Python blocks below in a notebook using the same environment,
or in a Python interpreter.

## FastText (300 dimensions)

Source: [FastText English vectors](https://fasttext.cc/docs/en/english-vectors.html).

```bash
wget -c https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip \
  -P data/embeddings/fasttext/

unzip -n data/embeddings/fasttext/wiki-news-300d-1M.vec.zip \
  -d data/embeddings/fasttext/
```

## GloVe Twitter (200 dimensions)

Source: [Stanford GloVe](https://nlp.stanford.edu/projects/glove/).
The archive contains several dimensions; extract the 200-dimensional version
used by this project:

```bash
wget -c https://nlp.stanford.edu/data/glove.twitter.27B.zip \
  -P data/embeddings/glove_twitter/

unzip -n data/embeddings/glove_twitter/glove.twitter.27B.zip \
  glove.twitter.27B.200d.txt -d data/embeddings/glove_twitter/
```

`wget -c` resumes an incomplete download; `unzip -n` keeps existing extracted
files. If an extracted file is incomplete, these commands will not replace it.

## Word2Vec Google News (300 dimensions)

Source: [Gensim data repository](https://github.com/piskvorky/gensim-data).
Download through Gensim and export to text for the project's loader:

```python
import gensim.downloader as api
from toxic_comments.config import EMBEDDINGS_DIR

path = EMBEDDINGS_DIR / "word2vec" / "GoogleNews-vectors-negative300.txt"
path.parent.mkdir(parents=True, exist_ok=True)

if not path.exists():
    vectors = api.load("word2vec-google-news-300")
    vectors.save_word2vec_format(str(path), binary=False)
    del vectors
else:
    print(f"Already exists: {path}")
```

Gensim keeps its own download cache in addition to the exported text file.
The existence check avoids exporting again; it does not validate an interrupted
export. A binary `.bin` file cannot be converted by simply renaming it `.txt`.

## BPEmb (English, 10,000 subwords, 300 dimensions)

Source: [BPEmb](https://github.com/bheinzerling/bpemb).
Installing `bpemb` only installs the library. Run this Python block to download
the SentencePiece tokenizer and matching pretrained vectors into this project:

```python
from bpemb import BPEmb
from toxic_comments.config import EMBEDDINGS_DIR

bpemb = BPEmb(
    lang="en",
    vs=10000,
    dim=300,
    cache_dir=EMBEDDINGS_DIR / "bpemb",
    vs_fallback=False,
)
print(bpemb.model_file)
print(bpemb.emb_file)
```

Existing cached files are reused. Keep both the `.model` tokenizer and `.bin`
vectors; no `bpemb_en.vec` export is needed. The project uses BPEmb's subword
tokenizer rather than looking up whole words in these vectors.

## LexVec (300 dimensions)

Open the [LexVec pretrained vectors page](https://github.com/alexandres/lexvec#pre-trained-vectors).
Under **LexVec**, choose **Common Crawl, lowercased, Word Vectors (2.2GB)**,
then extract the downloaded archive and place the vector file at:

```text
data/embeddings/lexvec/lexvec.commoncrawl.300d.W.pos.neg3.vectors
```

This is the filename configured in `src/toxic_comments/embeddings/lexvec.py`.
The older name `lexvec.commoncrawl.300d.W.pos.vectors` does not match that
configuration. Select word vectors, rather than the binary subword model or
the Word + Context variant.

## Expected Files

Paths relative to `data/embeddings/`:

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
