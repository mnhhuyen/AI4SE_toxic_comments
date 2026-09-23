"""Text cleaning and feature engineering for the toxic comments dataset."""

from __future__ import annotations

import html
import re
import unicodedata

import numpy as np
import pandas as pd

from toxic_comments.config import HEAVY_TEXT_COLUMN, ID_COLUMN, LABEL_COLUMNS, LIGHT_TEXT_COLUMN, TEXT_COLUMN

CONTRACTIONS = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "couldn't": "could not",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'll": "he will",
    "he's": "he is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it's": "it is",
    "let's": "let us",
    "mustn't": "must not",
    "shan't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "shouldn't": "should not",
    "that's": "that is",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "we'd": "we would",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what's": "what is",
    "where's": "where is",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "you've": "you have",
    "y'all": "you all",
    "wanna": "want to",
    "gonna": "going to",
    "gotta": "got to",
}
CONTRACTION_RE = re.compile(
    r"\b(" + "|".join(re.escape(key) for key in sorted(CONTRACTIONS, key=len, reverse=True)) + r")\b",
    flags=re.IGNORECASE,
)

KEEP_WORDS = {
    "you",
    "your",
    "yours",
    "yourself",
    "u",
    "ur",
    "youre",
    "no",
    "not",
    "nor",
    "never",
    "none",
    "nothing",
    "nobody",
    "cant",
    "cannot",
    "dont",
    "doesnt",
    "didnt",
    "wont",
    "shouldnt",
    "wouldnt",
    "couldnt",
    "isnt",
    "arent",
    "wasnt",
    "werent",
    "aint",
    "why",
    "who",
    "very",
    "too",
    "so",
    "just",
    "only",
    "own",
    "same",
}
BASE_STOPWORDS = {
    "i",
    "me",
    "my",
    "myself",
    "we",
    "our",
    "ours",
    "ourselves",
    "he",
    "him",
    "his",
    "himself",
    "she",
    "her",
    "hers",
    "herself",
    "it",
    "its",
    "itself",
    "they",
    "them",
    "their",
    "theirs",
    "themselves",
    "what",
    "which",
    "whom",
    "this",
    "that",
    "these",
    "those",
    "am",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "have",
    "has",
    "had",
    "having",
    "do",
    "does",
    "did",
    "doing",
    "a",
    "an",
    "the",
    "and",
    "but",
    "if",
    "or",
    "because",
    "as",
    "until",
    "while",
    "of",
    "at",
    "by",
    "for",
    "with",
    "about",
    "against",
    "between",
    "into",
    "through",
    "during",
    "before",
    "after",
    "above",
    "below",
    "to",
    "from",
    "up",
    "down",
    "in",
    "out",
    "on",
    "off",
    "over",
    "under",
    "again",
    "further",
    "then",
    "once",
    "here",
    "there",
    "when",
    "where",
    "how",
    "all",
    "any",
    "both",
    "each",
    "few",
    "more",
    "most",
    "other",
    "some",
    "such",
    "than",
    "s",
    "t",
    "can",
    "will",
    "should",
    "now",
    "d",
    "ll",
    "m",
    "o",
    "re",
    "ve",
    "y",
}
STOPWORDS = BASE_STOPWORDS - KEEP_WORDS 
# not any word in both BASE_STOPWORDS and KEEP_WORDS, so actually STOPWORDS = BASE_STOPWORDS ?
LEET_MAP = str.maketrans(
    {
        "@": "a",
        "$": "s",
        "0": "o",
        "1": "i",
        "3": "e",
        "4": "a",
        "5": "s",
        "7": "t",
        "|": "i",
        "+": "t",
    }
)

RE_WIKI_USER = re.compile(
    r"\[\[\s*(?:user|user talk|special:contributions)\s*:[^\]]*\]\]", re.IGNORECASE
)
RE_WIKI_IMG = re.compile(r"\[\[\s*(?:image|file|category)\s*:[^\]]*\]\]", re.IGNORECASE)
RE_WIKI_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]]*)\]\]")
RE_EXT_LINK = re.compile(r"\[(?:https?://\S+)\s*([^\]]*)\]")
RE_TEMPLATE = re.compile(r"\{\{[^{}]*\}\}")
RE_HEADING = re.compile(r"={2,}\s*([^=]+?)\s*={2,}")
RE_WIKI_EMPH = re.compile(r"'{2,5}")
RE_TALKMARK = re.compile(r"^[:*#]+", re.MULTILINE)
RE_TIMESTAMP = re.compile(r"\d{1,2}:\d{2},\s*\d{1,2}\s+\w+\s+\d{4}\s*\(?UTC\)?", re.IGNORECASE)
RE_UTC = re.compile(r"\(\s*UTC\s*\)", re.IGNORECASE)
RE_HTML_TAG = re.compile(r"<[^>]{1,200}>")
RE_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
RE_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
RE_IP = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
RE_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
RE_NEWLINE = re.compile(r"[\r\n\t\u000b\u000c\u0085\u2028\u2029]+")
RE_REPEAT_CHAR = re.compile(r"([A-Za-z])\1{2,}")
RE_REPEAT_PUNC = re.compile(r"([!?.,])\1{3,}")
RE_WS = re.compile(r"\s+")
RE_NON_ALPHA = re.compile(r"[^a-z\s]")
RE_SPACED_WORD = re.compile(r"\b(?:[a-zA-Z]\s){2,}[a-zA-Z]\b")
RE_DOTTED_WORD = re.compile(r"\b(?:[a-zA-Z][.\-_*]){2,}[a-zA-Z]\b")
RE_MASKED = re.compile(r"\b[a-zA-Z]+[*@$#!]{1,4}[a-zA-Z]*\b")

FEATURE_COLUMNS = [
    "n_chars",
    "n_words",
    "n_unique_words",
    "unique_word_ratio",
    "mean_word_len",
    "caps_ratio",
    "n_exclaim",
    "n_question",
    "punct_ratio",
    "n_newlines",
    "n_urls",
    "n_ips",
    "n_you",
    "n_masked_words",
    "has_shouting",
]


def normalize_unicode(text: str) -> str:
    """NFKC compat-decompose (Ｆuck -> Fuck) + drop control characters."""
    text = unicodedata.normalize("NFKC", text)
    return RE_CONTROL.sub(" ", text)


def strip_accents(text: str) -> str:
    """café -> cafe (used for classical only)."""
    return "".join(
        c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c)
    )


def strip_wiki_markup(text: str) -> str:
    """Remove Wikipedia Talk-page artifacts specific to this dataset."""
    text = RE_WIKI_USER.sub(" ", text)     # drop [[User:...]] entirely
    text = RE_WIKI_IMG.sub(" ", text)      # drop [[Image/File/Category:...]]
    text = RE_TEMPLATE.sub(" ", text)      # drop {{template}}
    text = RE_HEADING.sub(r" \1 ", text)   # keep heading text
    text = RE_WIKI_LINK.sub(r" \1 ", text) # keep display text of wiki links
    text = RE_EXT_LINK.sub(r" \1 ", text)  # keep display text of [url text]
    text = RE_WIKI_EMPH.sub(" ", text)     # drop '', ''', ''''', ...
    text = RE_TALKMARK.sub(" ", text)      # drop ^:, ^*, ^# talk indents
    return text


def strip_web_noise(text: str) -> str:
    """Remove URLs, emails, and IPv4 addresses."""
    text = RE_URL.sub(" ", text)
    text = RE_EMAIL.sub(" ", text)
    text = RE_IP.sub(" ", text)
    return text


def deobfuscate(text: str) -> str:
    """
    Undo common toxicity-obfuscation tricks:
      f u c k       -> fuck        (spaced letters)
      f.u.c.k       -> fuck        (dotted)
      f-u-c-k       -> fuck        (hyphenated)
      f**k / sh1t   -> fuck / shit (leetspeak + mask chars)
    """
    text = RE_SPACED_WORD.sub(lambda m: m.group(0).replace(" ", ""), text)
    text = RE_DOTTED_WORD.sub(lambda m: re.sub(r"[.\-_*]", "", m.group(0)), text)
    text = RE_MASKED.sub(
        lambda m: re.sub(r"[*#!]", "", m.group(0).translate(LEET_MAP)), text
    )
    return text


def expand_contractions(text: str) -> str:
    """can't -> cannot, you're -> you are (case-insensitive dict lookup)."""
    return CONTRACTION_RE.sub(lambda m: CONTRACTIONS[m.group(0).lower()], text)


def clean_light(text: str) -> str:
    """
    LIGHT cleaning for pretrained transformers (BERT / DistilBERT / RoBERTa).
    Keeps casing, punctuation, and contractions — the tokenizer needs them.
    """
    if not isinstance(text, str):
        return ""
    text = html.unescape(html.unescape(text))     # double-encoded entities exist
    text = normalize_unicode(text)
    text = RE_HTML_TAG.sub(" ", text)
    text = strip_wiki_markup(text)
    text = RE_TIMESTAMP.sub(" ", text)
    text = RE_UTC.sub(" ", text)
    text = strip_web_noise(text)
    text = RE_NEWLINE.sub(" ", text)
    text = RE_REPEAT_CHAR.sub(r"\1\1", text)       # letters: keep 2
    text = RE_REPEAT_PUNC.sub(r"\1\1\1", text)     # punct:   keep 3
    return RE_WS.sub(" ", text).strip()


def clean_heavy(text: str, drop_stopwords: bool = True) -> str:
    """
    HEAVY cleaning for bag-of-words models (TF-IDF + LogReg / SVM / NB).
    Runs on top of LIGHT output. Aggressive: lowercase, accent-strip,
    deobfuscate, expand contractions, letters only, optional stopword drop.
    """
    if not text:
        return ""
    text = strip_accents(text.lower())
    text = deobfuscate(text)
    text = expand_contractions(text)
    text = RE_NON_ALPHA.sub(" ", text)
    tokens = [t for t in text.split() if len(t) > 1]
    if drop_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return " ".join(tokens)


def build_features(raw: pd.Series) -> pd.DataFrame:
    """
    Hand-crafted numeric features from RAW text. These are strong signals for
    classical models — you feed them alongside TF-IDF (hstack).
    """
    s = raw.fillna("")
    words = s.str.split()
    n_words = words.str.len().replace(0, np.nan)
    n_chars = s.str.len().replace(0, np.nan)

    f = pd.DataFrame(index=s.index)
    f["n_chars"] = s.str.len()
    f["n_words"] = words.str.len()
    f["n_unique_words"] = words.apply(lambda w: len(set(w)) if isinstance(w, list) else 0)
    f["unique_word_ratio"] = (f["n_unique_words"] / n_words).fillna(0)
    f["mean_word_len"] = (s.str.replace(r"\s", "", regex=True).str.len() / n_words).fillna(0)
    f["caps_ratio"] = (s.str.count(r"[A-Z]") / n_chars).fillna(0)
    f["n_exclaim"] = s.str.count(r"!")
    f["n_question"] = s.str.count(r"\?")
    f["punct_ratio"] = (s.str.count(r"[^\w\s]") / n_chars).fillna(0)
    f["n_newlines"] = s.str.count("\n")
    f["n_urls"] = s.str.count(RE_URL)
    f["n_ips"] = s.str.count(RE_IP)
    f["n_you"] = s.str.count(r"(?i)\b(you|your|u|ur)\b")   # 2nd-person address
    f["n_masked_words"] = s.str.count(r"\b[a-zA-Z]+[*@$#]{1,4}[a-zA-Z]*\b")
    f["has_shouting"] = (f["caps_ratio"] > 0.30).astype(int)
    return f

def process_cleaning(
    df: pd.DataFrame,
    name: str = "dataset",
    is_train: bool = True,
    drop_duplicate_rows: bool = True,
    drop_empty_light_train_rows: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Clean one dataframe and add text/features columns.

    This function mirrors the original notebook's ``process`` step, but it uses
    plain pandas ``apply`` so it also works in scripts, tests, and CLI runs
    without requiring tqdm setup.
    """

    if verbose:
        print(f"\n>>> {name} ({len(df):,} rows)")
    df = df.copy()
    df[TEXT_COLUMN] = df[TEXT_COLUMN].fillna("").astype(str)

    # Dedupe TRAIN only (never touch test integrity)
    if is_train and drop_duplicate_rows:
        before = len(df)
        subset = [TEXT_COLUMN] + [c for c in LABEL_COLUMNS if c in df.columns]
        df = df.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
        if verbose and before != len(df):
            print(f"    dropped {before - len(df):,} exact duplicate rows")

    # Feature engineering from RAW text
    feats = build_features(df[TEXT_COLUMN])

    # Apply BOTH cleaning levels
    if verbose:
        print("    cleaning level 1 (light) ...")
    df[LIGHT_TEXT_COLUMN] = df[TEXT_COLUMN].apply(clean_light)
    if verbose:
        print("    cleaning level 2 (heavy) ...")
    df[HEAVY_TEXT_COLUMN] = df[LIGHT_TEXT_COLUMN].apply(clean_heavy)

    # Attach features + emptiness flags
    df = pd.concat([df, feats], axis=1)
    df["is_empty_light"] = (df[LIGHT_TEXT_COLUMN].str.strip() == "").astype(int)
    df["is_empty_heavy"] = (df[HEAVY_TEXT_COLUMN].str.strip() == "").astype(int)
    # If light survives but heavy is empty -> likely non-Latin script / emoji only
    df["is_non_latin"] = (
        (df["is_empty_heavy"] == 1) & (df["is_empty_light"] == 0)
    ).astype(int)

    n_heavy = int(df["is_empty_heavy"].sum())
    n_light = int(df["is_empty_light"].sum())
    n_nonlatin = int(df["is_non_latin"].sum())
    if verbose and n_nonlatin:
        print(f"    {n_nonlatin:,} rows empty in HEAVY but fine in LIGHT "
              f"(non-Latin / emoji) -> KEPT, flagged as is_non_latin")
    if is_train and drop_empty_light_train_rows and n_light:
        df = df[df["is_empty_light"] == 0].reset_index(drop=True)
        if verbose:
            print(f"    dropped {n_light:,} rows with no usable text at all")
    if verbose:
        print(f"    {n_heavy:,} rows unusable for bag-of-words -> "
              f"filter on is_empty_heavy for TF-IDF")
    return df


def audit_dataset(data: pd.DataFrame) -> dict[str, int | dict[str, int]]:
    """Return basic data quality statistics without printing from library code."""

    label_counts = {
        label: int(data[label].sum())
        for label in LABEL_COLUMNS
        if label in data.columns
    }
    return {
        "rows": len(data),
        "null_comment_text": int(data[TEXT_COLUMN].isna().sum()),
        "empty_comment_text": int((data[TEXT_COLUMN].fillna("").str.strip() == "").sum()),
        "duplicate_ids": int(data[ID_COLUMN].duplicated().sum()) if ID_COLUMN in data.columns else 0,
        "duplicate_texts": int(data[TEXT_COLUMN].duplicated().sum()),
        "label_counts": label_counts,
    }
