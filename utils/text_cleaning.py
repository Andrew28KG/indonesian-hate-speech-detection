"""
Indonesian Hate Speech Detection - Text Cleaning Utilities

This module contains reusable text preprocessing functions for Indonesian text,
specifically designed for hate speech detection tasks.

Author: Data Science Team
Date: 2025-06-16
"""

import pandas as pd
import numpy as np
import re
import string
import warnings
warnings.filterwarnings('ignore')

try:
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
    SASTRAWI_AVAILABLE = True
except ImportError:
    print("Warning: Sastrawi not installed. Some Indonesian language features will be unavailable.")
    print("Install with: pip install sastrawi")
    SASTRAWI_AVAILABLE = False


class IndonesianTextCleaner:
    """
    A comprehensive text cleaning class for Indonesian text preprocessing.
    """
    
    def __init__(self, use_sastrawi=True):
        """
        Initialize the text cleaner.
        
        Args:
            use_sastrawi (bool): Whether to use Sastrawi for Indonesian language processing
        """
        self.use_sastrawi = use_sastrawi and SASTRAWI_AVAILABLE
        
        if self.use_sastrawi:
            # Initialize Sastrawi components
            factory = StemmerFactory()
            self.stemmer = factory.create_stemmer()
            
            stopword_factory = StopWordRemoverFactory()
            self.stopword_remover = stopword_factory.create_stop_word_remover()
        else:
            self.stemmer = None
            self.stopword_remover = None
    
    def clean_text_basic(self, text):
        """
        Basic text cleaning for Indonesian tweets.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text is None:
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove USER mentions (anonymized users)
        text = re.sub(r'\buser\b', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove RT (retweet) markers
        text = re.sub(r'^rt\s+', '', text)
        
        # Remove mentions and hashtags
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def remove_punctuation_and_numbers(self, text):
        """
        Remove punctuation and numbers, keeping only alphabetic characters.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text with only alphabetic characters
        """
        if not text:
            return ""
        
        # Remove punctuation and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def remove_indonesian_stopwords(self, text):
        """
        Remove Indonesian stopwords using Sastrawi.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Text without stopwords
        """
        if not text or not self.use_sastrawi:
            return text
        
        return self.stopword_remover.remove(text)
    
    def stem_indonesian_text(self, text):
        """
        Apply Indonesian stemming using Sastrawi.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Stemmed text
        """
        if not text or not self.use_sastrawi:
            return text
        
        return self.stemmer.stem(text)
    
    def remove_repeated_characters(self, text, max_repeats=2):
        """
        Remove excessive character repetitions (e.g., 'hahahahaha' -> 'haha').
        
        Args:
            text (str): Input text
            max_repeats (int): Maximum allowed character repetitions
            
        Returns:
            str: Text with reduced character repetitions
        """
        if not text:
            return ""
        
        # Replace repeated characters
        pattern = r'(.)\1{' + str(max_repeats) + ',}'
        return re.sub(pattern, r'\1' * max_repeats, text)
    
    def normalize_slang_and_abbreviations(self, text):
        """
        Normalize common Indonesian slang and abbreviations.
        
        Args:
            text (str): Input text
            
        Returns:
            str: Normalized text
        """
        if not text:
            return ""
        
        # Common Indonesian slang normalization
        slang_dict = {
            'gw': 'saya',
            'gue': 'saya',
            'lu': 'kamu',
            'loe': 'kamu',
            'elo': 'kamu',
            'lo': 'kamu',
            'emang': 'memang',
            'udah': 'sudah',
            'udh': 'sudah',
            'blm': 'belum',
            'blom': 'belum',
            'gimana': 'bagaimana',
            'gmn': 'bagaimana',
            'knp': 'kenapa',
            'knapa': 'kenapa',
            'jgn': 'jangan',
            'jangan': 'jangan',
            'bgt': 'banget',
            'bgt': 'banget',
            'yg': 'yang',
            'krn': 'karena',
            'karna': 'karena',
            'dgn': 'dengan',
            'utk': 'untuk',
            'tdk': 'tidak',
            'ga': 'tidak',
            'gak': 'tidak',
            'ngga': 'tidak',
            'nggak': 'tidak'
        }
        
        # Replace slang words
        words = text.split()
        normalized_words = [slang_dict.get(word, word) for word in words]
        
        return ' '.join(normalized_words)
    
    def preprocess_text_pipeline(self, text, 
                                remove_stopwords=True, 
                                apply_stemming=True,
                                normalize_slang=True,
                                remove_repeats=True):
        """
        Complete text preprocessing pipeline.
        
        Args:
            text (str): Input text
            remove_stopwords (bool): Whether to remove stopwords
            apply_stemming (bool): Whether to apply stemming
            normalize_slang (bool): Whether to normalize slang
            remove_repeats (bool): Whether to remove repeated characters
            
        Returns:
            str: Fully preprocessed text
        """
        # Basic cleaning
        text = self.clean_text_basic(text)
        
        # Remove repeated characters
        if remove_repeats:
            text = self.remove_repeated_characters(text)
        
        # Normalize slang
        if normalize_slang:
            text = self.normalize_slang_and_abbreviations(text)
        
        # Remove punctuation and numbers
        text = self.remove_punctuation_and_numbers(text)
        
        # Remove stopwords
        if remove_stopwords:
            text = self.remove_indonesian_stopwords(text)
        
        # Apply stemming
        if apply_stemming:
            text = self.stem_indonesian_text(text)
        
        return text

    def clean_text(self, text):
        """
        Simple method to clean text using the full preprocessing pipeline.
        
        Args:
            text (str): Input text to clean
            
        Returns:
            str: Cleaned text
        """
        return self.preprocess_text_pipeline(text)


def clean_dataset(df, text_column='Tweet', 
                 target_columns=None,
                 save_intermediate=False):
    """
    Clean an entire dataset using the Indonesian text cleaner.
    
    Args:
        df (pd.DataFrame): Input dataframe
        text_column (str): Name of the text column to clean
        target_columns (list): List of target column names to preserve
        save_intermediate (bool): Whether to save intermediate cleaning steps
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    if target_columns is None:
        target_columns = ['HS', 'Abusive', 'HS_Individual', 'HS_Group', 
                         'HS_Religion', 'HS_Race', 'HS_Physical', 'HS_Gender', 
                         'HS_Other', 'HS_Weak', 'HS_Moderate', 'HS_Strong']
    
    # Initialize cleaner
    cleaner = IndonesianTextCleaner()
    
    # Create a copy of the dataframe
    df_clean = df.copy()
    
    print(f"Starting dataset cleaning...")
    print(f"Original dataset shape: {df_clean.shape}")
    
    # Apply basic cleaning
    print("Applying basic text cleaning...")
    df_clean[f'{text_column}_basic_clean'] = df_clean[text_column].apply(
        cleaner.clean_text_basic
    )
    
    # Remove punctuation and numbers
    print("Removing punctuation and numbers...")
    df_clean[f'{text_column}_no_punct'] = df_clean[f'{text_column}_basic_clean'].apply(
        cleaner.remove_punctuation_and_numbers
    )
    
    # Apply full preprocessing pipeline
    print("Applying full preprocessing pipeline...")
    df_clean[f'{text_column}_cleaned'] = df_clean[text_column].apply(
        cleaner.preprocess_text_pipeline
    )
    
    # Remove empty texts
    empty_mask = df_clean[f'{text_column}_cleaned'].str.len() == 0
    if empty_mask.sum() > 0:
        print(f"Removing {empty_mask.sum()} empty texts after cleaning...")
        df_clean = df_clean[~empty_mask].reset_index(drop=True)
    
    # Select final columns
    final_columns = [text_column, f'{text_column}_cleaned'] + [col for col in target_columns if col in df_clean.columns]
    df_final = df_clean[final_columns].copy()
    
    print(f"Final dataset shape: {df_final.shape}")
    print(f"Text length reduction: {((df_clean[text_column].str.len().mean() - df_clean[f'{text_column}_cleaned'].str.len().mean()) / df_clean[text_column].str.len().mean() * 100):.1f}%")
    
    return df_final


def get_text_statistics(text_series):
    """
    Get comprehensive statistics about a text series.
    
    Args:
        text_series (pd.Series): Series containing text data
        
    Returns:
        dict: Dictionary containing text statistics
    """
    stats = {
        'total_texts': len(text_series),
        'empty_texts': text_series.str.len().eq(0).sum(),
        'avg_length': text_series.str.len().mean(),
        'median_length': text_series.str.len().median(),
        'min_length': text_series.str.len().min(),
        'max_length': text_series.str.len().max(),
        'avg_words': text_series.str.split().str.len().mean(),
        'unique_texts': text_series.nunique(),
        'duplicate_texts': len(text_series) - text_series.nunique()
    }
    
    return stats


def extract_features(text_series, feature_type='tfidf', **kwargs):
    """
    Extract features from text data.
    
    Args:
        text_series (pd.Series): Series containing text data
        feature_type (str): Type of features to extract ('tfidf', 'count')
        **kwargs: Additional arguments for the vectorizer
        
    Returns:
        tuple: (feature_matrix, vectorizer)
    """
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    
    # Default parameters
    default_params = {
        'max_features': 10000,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.95
    }
    
    # Update with provided kwargs
    params = {**default_params, **kwargs}
    
    # Initialize vectorizer
    if feature_type == 'tfidf':
        vectorizer = TfidfVectorizer(**params)
    elif feature_type == 'count':
        vectorizer = CountVectorizer(**params)
    else:
        raise ValueError("feature_type must be 'tfidf' or 'count'")
    
    # Fit and transform
    feature_matrix = vectorizer.fit_transform(text_series)
    
    return feature_matrix, vectorizer


def evaluate_text_cleaning(original_texts, cleaned_texts, sample_size=10):
    """
    Evaluate the effectiveness of text cleaning by comparing samples.
    
    Args:
        original_texts (pd.Series): Original text data
        cleaned_texts (pd.Series): Cleaned text data
        sample_size (int): Number of samples to display
        
    Returns:
        dict: Evaluation metrics
    """
    # Calculate statistics
    original_stats = get_text_statistics(original_texts)
    cleaned_stats = get_text_statistics(cleaned_texts)
    
    # Calculate reduction percentages
    length_reduction = ((original_stats['avg_length'] - cleaned_stats['avg_length']) / 
                       original_stats['avg_length'] * 100)
    
    word_reduction = ((original_stats['avg_words'] - cleaned_stats['avg_words']) / 
                     original_stats['avg_words'] * 100)
    
    print(f"Text Cleaning Evaluation:")
    print(f"{'='*50}")
    print(f"Average length reduction: {length_reduction:.1f}%")
    print(f"Average word count reduction: {word_reduction:.1f}%")
    print(f"Original empty texts: {original_stats['empty_texts']}")
    print(f"Cleaned empty texts: {cleaned_stats['empty_texts']}")
    
    # Show sample comparisons
    print(f"\nSample Comparisons:")
    print(f"{'='*50}")
    
    sample_indices = np.random.choice(len(original_texts), min(sample_size, len(original_texts)), replace=False)
    
    for i, idx in enumerate(sample_indices):
        print(f"\nSample {i+1}:")
        print(f"Original: {original_texts.iloc[idx][:100]}...")
        print(f"Cleaned:  {cleaned_texts.iloc[idx][:100]}...")
    
    return {
        'original_stats': original_stats,
        'cleaned_stats': cleaned_stats,
        'length_reduction_pct': length_reduction,
        'word_reduction_pct': word_reduction
    }


# Convenience functions for quick usage
def quick_clean_text(text, level='full'):
    """
    Quickly clean a single text with predefined settings.
    
    Args:
        text (str): Input text
        level (str): Cleaning level ('basic', 'medium', 'full')
        
    Returns:
        str: Cleaned text
    """
    cleaner = IndonesianTextCleaner()
    
    if level == 'basic':
        return cleaner.clean_text_basic(text)
    elif level == 'medium':
        text = cleaner.clean_text_basic(text)
        text = cleaner.remove_punctuation_and_numbers(text)
        return text
    elif level == 'full':
        return cleaner.preprocess_text_pipeline(text)
    else:
        raise ValueError("level must be 'basic', 'medium', or 'full'")


def batch_clean_texts(texts, level='full', show_progress=True):
    """
    Clean multiple texts efficiently.
    
    Args:
        texts (list or pd.Series): List or series of texts to clean
        level (str): Cleaning level ('basic', 'medium', 'full')
        show_progress (bool): Whether to show progress
        
    Returns:
        list: List of cleaned texts
    """
    if isinstance(texts, pd.Series):
        texts = texts.tolist()
    
    cleaner = IndonesianTextCleaner()
    cleaned_texts = []
    
    for i, text in enumerate(texts):
        if show_progress and i % 1000 == 0:
            print(f"Processed {i}/{len(texts)} texts...")
        
        cleaned_text = quick_clean_text(text, level)
        cleaned_texts.append(cleaned_text)
    
    if show_progress:
        print(f"Completed cleaning {len(texts)} texts!")
    
    return cleaned_texts


if __name__ == "__main__":
    # Example usage
    sample_text = "RT @user: halo semua!!! gimana kabarnya??? gw lagi belajar NLP nih... 😊 http://example.com"
    
    print("Indonesian Text Cleaning Utilities - Example Usage")
    print("="*60)
    print(f"Original text: {sample_text}")
    
    # Initialize cleaner
    cleaner = IndonesianTextCleaner()
    
    # Apply different levels of cleaning
    basic_clean = cleaner.clean_text_basic(sample_text)
    print(f"Basic clean: {basic_clean}")
    
    full_clean = cleaner.preprocess_text_pipeline(sample_text)
    print(f"Full clean: {full_clean}")
    
    # Quick clean
    quick_full = quick_clean_text(sample_text, 'full')
    print(f"Quick full: {quick_full}")
