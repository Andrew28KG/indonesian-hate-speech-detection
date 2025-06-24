"""
Indonesian Hate Speech Detection - Utilities Package

This package contains utility functions and classes for:
- Text preprocessing and cleaning
- Model training and evaluation
- Bias analysis and performance assessment
"""

from .text_cleaning import IndonesianTextCleaner
from .model_utils import ModelEvaluator, ModelSaver, BiasAnalyzer, create_model_pipeline

__version__ = "1.0.0"
__author__ = "Indonesian Hate Speech Detection Project"

__all__ = [
    'IndonesianTextCleaner',
    'ModelEvaluator', 
    'ModelSaver',
    'BiasAnalyzer',
    'create_model_pipeline'
]
