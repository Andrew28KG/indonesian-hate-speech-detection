"""
Configuration file for the Indonesian Hate Speech Detection App
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "outputs" / "models"
ABUSIVE_WORDS_DIR = PROJECT_ROOT / "IndonesianAbusiveWords"

# Model file paths
BINARY_MODEL_PATH = MODELS_DIR / "binary_classifier.pkl"
MULTILABEL_MODEL_PATH = MODELS_DIR / "multilabel_classifier.pkl"
VECTORIZER_PATH = MODELS_DIR / "tfidf_vectorizer.pkl"

# Data file paths
RAW_DATA_PATH = DATA_DIR / "raw" / "data.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "tweets_cleaned.csv"
ABUSIVE_WORDS_PATH = ABUSIVE_WORDS_DIR / "data.csv"

# App configuration
APP_CONFIG = {
    "page_title": "Indonesian Hate Speech Detection",
    "page_icon": "🇮🇩",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Model categories for multi-label classification
HATE_SPEECH_CATEGORIES = [
    'HS_Individual',
    'HS_Group', 
    'HS_Religion',
    'HS_Race',
    'HS_Physical',
    'HS_Gender',
    'HS_Other'
]

# Fallback abusive words (if dictionary not found)
FALLBACK_ABUSIVE_WORDS = {
    'anjing', 'bangsat', 'bego', 'bodoh', 'goblok', 'tolol', 'idiot', 'dungu',
    'kampret', 'bajingan', 'kontol', 'memek', 'lonte', 'pelacur',
    'banci', 'homo', 'lesbi', 'kafir', 'murtad', 'munafik',
    'cebong', 'kampret', 'bajingan', 'bangsat'
}

# Text preprocessing options
TEXT_PREPROCESSING_OPTIONS = {
    "remove_urls": True,
    "remove_mentions": True,
    "remove_hashtags": True,
    "normalize_slang": True,
    "remove_stopwords": True,
    "apply_stemming": True
} 