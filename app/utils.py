"""
Utility functions for the Indonesian Hate Speech Detection App
"""

import pandas as pd
import numpy as np
import re
from collections import Counter
import streamlit as st
from pathlib import Path

def load_abusive_words_dictionary(file_path):
    """
    Load Indonesian abusive words dictionary from file
    
    Args:
        file_path (str): Path to the abusive words file
        
    Returns:
        set: Set of abusive words
    """
    try:
        if Path(file_path).exists():
            df = pd.read_csv(file_path)
            
            # Try different column names
            word_columns = ['word', 'kata', 'text', 'abusive_word']
            for col in word_columns:
                if col in df.columns:
                    words = set(df[col].str.lower().dropna())
                    return words
            
            # If no specific column found, use first column
            if len(df.columns) > 0:
                words = set(df.iloc[:, 0].str.lower().dropna())
                return words
                
    except Exception as e:
        st.warning(f"Could not load abusive words from {file_path}: {e}")
    
    return set()

def count_abusive_words(text, abusive_words_dict):
    """
    Count abusive words in the given text
    
    Args:
        text (str): Input text to analyze
        abusive_words_dict (set): Set of abusive words
        
    Returns:
        tuple: (count, list_of_found_words)
    """
    if not text or not abusive_words_dict:
        return 0, []
    
    # Convert to lowercase and split into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Find abusive words
    found_abusive = [word for word in words if word in abusive_words_dict]
    
    return len(found_abusive), found_abusive

def analyze_text_statistics(text):
    """
    Analyze basic text statistics
    
    Args:
        text (str): Input text
        
    Returns:
        dict: Dictionary with text statistics
    """
    if not text:
        return {
            'char_count': 0,
            'word_count': 0,
            'sentence_count': 0,
            'avg_word_length': 0
        }
    
    # Basic statistics
    char_count = len(text)
    words = text.split()
    word_count = len(words)
    
    # Sentence count (simple heuristic)
    sentences = re.split(r'[.!?]+', text)
    sentence_count = len([s for s in sentences if s.strip()])
    
    # Average word length
    avg_word_length = np.mean([len(word) for word in words]) if words else 0
    
    return {
        'char_count': char_count,
        'word_count': word_count,
        'sentence_count': sentence_count,
        'avg_word_length': round(avg_word_length, 2)
    }

def format_prediction_results(result):
    """
    Format prediction results for display
    
    Args:
        result (dict): Prediction results
        
    Returns:
        dict: Formatted results
    """
    if not result:
        return None
    
    # Format binary prediction
    binary_status = "HATE SPEECH DETECTED" if result.get('binary_prediction') == 1 else "NO HATE SPEECH DETECTED"
    binary_confidence = result.get('binary_probability', 0)
    
    # Format multi-label predictions
    multilabel_results = []
    for category, pred in result.get('multilabel_predictions', {}).items():
        prob = result.get('multilabel_probabilities', {}).get(category, 0)
        category_name = category.replace('HS_', '').replace('_', ' ').title()
        status = "DETECTED" if pred == 1 else "NOT DETECTED"
        multilabel_results.append({
            'category': category_name,
            'status': status,
            'confidence': prob,
            'prediction': pred
        })
    
    return {
        'binary_status': binary_status,
        'binary_confidence': binary_confidence,
        'multilabel_results': multilabel_results,
        'cleaned_text': result.get('cleaned_text', ''),
        'model_used': result.get('model_used', 'Unknown')
    }

def create_word_frequency_chart(text, top_n=10):
    """
    Create word frequency analysis for text
    
    Args:
        text (str): Input text
        top_n (int): Number of top words to show
        
    Returns:
        dict: Word frequency data
    """
    if not text:
        return {}
    
    # Clean and tokenize
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove common stop words
    stop_words = {'dan', 'di', 'ke', 'dari', 'yang', 'untuk', 'dengan', 'ini', 'itu', 'atau', 'tapi', 'jika', 'karena', 'seperti', 'akan', 'sudah', 'masih', 'juga', 'hanya', 'bisa', 'harus', 'akan', 'dapat', 'mau', 'ingin', 'perlu', 'boleh', 'tidak', 'bukan', 'jangan', 'saya', 'kamu', 'dia', 'mereka', 'kita', 'kami', 'anda', 'beliau'}
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count frequencies
    word_freq = Counter(words)
    
    # Get top words
    top_words = word_freq.most_common(top_n)
    
    return {
        'words': [word for word, _ in top_words],
        'frequencies': [freq for _, freq in top_words],
        'total_words': len(words)
    }

def validate_text_input(text):
    """
    Validate text input
    
    Args:
        text (str): Input text
        
    Returns:
        tuple: (is_valid, error_message)
    """
    if not text or not text.strip():
        return False, "Please enter some text to analyze."
    
    if len(text.strip()) < 3:
        return False, "Text must be at least 3 characters long."
    
    if len(text) > 10000:
        return False, "Text is too long. Please enter text with less than 10,000 characters."
    
    return True, ""

def get_model_status():
    """
    Check the status of available models
    
    Returns:
        dict: Model status information
    """
    from app.config import BINARY_MODEL_PATH, MULTILABEL_MODEL_PATH, VECTORIZER_PATH
    
    status = {
        'binary_model': BINARY_MODEL_PATH.exists(),
        'multilabel_model': MULTILABEL_MODEL_PATH.exists(),
        'vectorizer': VECTORIZER_PATH.exists(),
        'all_models': all([BINARY_MODEL_PATH.exists(), VECTORIZER_PATH.exists()])
    }
    
    return status 