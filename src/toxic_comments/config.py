"""Shared project configuration."""

from pathlib import Path

LABEL_COLUMNS = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
]

TEXT_COLUMN = "comment_text"
LIGHT_TEXT_COLUMN = "comment_light"
HEAVY_TEXT_COLUMN = "comment_heavy"
ID_COLUMN = "id"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
K_FOLD_DATA_DIR = DATA_DIR / "k_fold"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
