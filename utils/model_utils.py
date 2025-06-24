"""
Model utilities for Indonesian Hate Speech Detection
This module contains utility functions for model training, evaluation, and deployment.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    """Class for comprehensive model evaluation"""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_binary_classification(self, y_true, y_pred, y_pred_proba=None, model_name="Model"):
        """
        Evaluate binary classification model
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities (optional)
            model_name: Name of the model for reporting
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='weighted'),
            'recall': recall_score(y_true, y_pred, average='weighted'),
            'f1_score': f1_score(y_true, y_pred, average='weighted')
        }
        
        if y_pred_proba is not None:
            # For binary classification, take probabilities of positive class
            if y_pred_proba.ndim == 2:
                y_pred_proba = y_pred_proba[:, 1]
            metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba)
        
        self.results[model_name] = metrics
        return metrics
    
    def evaluate_multilabel_classification(self, y_true, y_pred, model_name="MultiLabel_Model"):
        """
        Evaluate multi-label classification model
        
        Args:
            y_true: True labels (multi-label)
            y_pred: Predicted labels (multi-label)
            model_name: Name of the model for reporting
        
        Returns:
            dict: Dictionary containing evaluation metrics
        """
        from sklearn.metrics import hamming_loss, jaccard_score
        
        metrics = {
            'model_name': model_name,
            'hamming_loss': hamming_loss(y_true, y_pred),
            'jaccard_score': jaccard_score(y_true, y_pred, average='samples'),
            'f1_micro': f1_score(y_true, y_pred, average='micro'),
            'f1_macro': f1_score(y_true, y_pred, average='macro')
        }
        
        # Calculate subset accuracy (exact match)
        subset_accuracy = np.mean(np.all(y_true == y_pred, axis=1))
        metrics['subset_accuracy'] = subset_accuracy
        
        self.results[model_name] = metrics
        return metrics
    
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, title="Confusion Matrix"):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.show()
    
    def compare_models(self):
        """Compare performance of multiple models"""
        if not self.results:
            print("No model results to compare!")
            return
        
        df_results = pd.DataFrame(self.results).T
        
        # Sort by F1 score for binary classification or F1 micro for multi-label
        sort_column = 'f1_score' if 'f1_score' in df_results.columns else 'f1_micro'
        df_results = df_results.sort_values(sort_column, ascending=False)
        
        print("Model Comparison Results:")
        print("=" * 50)
        print(df_results.round(4))
        
        return df_results
    
    def cross_validate_model(self, model, X, y, cv=5, scoring='f1_weighted'):
        """
        Perform cross-validation on a model
        
        Args:
            model: Sklearn model to evaluate
            X: Features
            y: Target labels
            cv: Number of cross-validation folds
            scoring: Scoring metric
        
        Returns:
            dict: Cross-validation results
        """
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
        
        results = {
            'cv_scores': cv_scores,
            'mean_score': cv_scores.mean(),
            'std_score': cv_scores.std(),
            'min_score': cv_scores.min(),
            'max_score': cv_scores.max()
        }
        
        print(f"Cross-Validation Results ({scoring}):")
        print(f"Mean: {results['mean_score']:.4f} (+/- {results['std_score']*2:.4f})")
        print(f"Scores: {cv_scores}")
        
        return results

class ModelSaver:
    """Class for saving and loading trained models"""
    
    @staticmethod
    def save_model(model, filepath, use_joblib=True):
        """
        Save a trained model to disk
        
        Args:
            model: Trained model object
            filepath: Path to save the model
            use_joblib: Whether to use joblib (True) or pickle (False)
        """
        try:
            if use_joblib:
                joblib.dump(model, filepath)
            else:
                with open(filepath, 'wb') as f:
                    pickle.dump(model, f)
            print(f"✅ Model saved successfully to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving model: {e}")
    
    @staticmethod
    def load_model(filepath, use_joblib=True):
        """
        Load a trained model from disk
        
        Args:
            filepath: Path to the saved model
            use_joblib: Whether to use joblib (True) or pickle (False)
        
        Returns:
            Loaded model object
        """
        try:
            if use_joblib:
                model = joblib.load(filepath)
            else:
                with open(filepath, 'rb') as f:
                    model = pickle.load(f)
            print(f"✅ Model loaded successfully from: {filepath}")
            return model
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    @staticmethod
    def save_training_history(history, filepath):
        """Save training history (for deep learning models)"""
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(history, f)
            print(f"✅ Training history saved to: {filepath}")
        except Exception as e:
            print(f"❌ Error saving training history: {e}")

class BiasAnalyzer:
    """Class for analyzing model bias across different groups"""
    
    def __init__(self):
        self.bias_results = {}
    
    def analyze_performance_by_group(self, y_true, y_pred, group_labels, group_column_name):
        """
        Analyze model performance across different groups
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            group_labels: Group identifiers for each sample
            group_column_name: Name of the group column
        
        Returns:
            dict: Performance metrics by group
        """
        unique_groups = np.unique(group_labels)
        group_results = {}
        
        for group in unique_groups:
            mask = group_labels == group
            group_y_true = y_true[mask]
            group_y_pred = y_pred[mask]
            
            if len(group_y_true) > 0:
                group_results[group] = {
                    'count': len(group_y_true),
                    'accuracy': accuracy_score(group_y_true, group_y_pred),
                    'f1_score': f1_score(group_y_true, group_y_pred, average='weighted'),
                    'precision': precision_score(group_y_true, group_y_pred, average='weighted'),
                    'recall': recall_score(group_y_true, group_y_pred, average='weighted')
                }
        
        self.bias_results[group_column_name] = group_results
        
        # Print results
        print(f"Performance Analysis by {group_column_name}:")
        print("=" * 50)
        for group, metrics in group_results.items():
            print(f"{group}:")
            print(f"  Count: {metrics['count']}")
            print(f"  Accuracy: {metrics['accuracy']:.4f}")
            print(f"  F1-Score: {metrics['f1_score']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall: {metrics['recall']:.4f}")
            print()
        
        return group_results
    
    def plot_bias_analysis(self, group_column_name, metric='f1_score'):
        """Plot bias analysis results"""
        if group_column_name not in self.bias_results:
            print(f"No bias analysis results found for {group_column_name}")
            return
        
        results = self.bias_results[group_column_name]
        groups = list(results.keys())
        values = [results[group][metric] for group in groups]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(groups, values)
        plt.title(f'{metric.replace("_", " ").title()} by {group_column_name}')
        plt.xlabel(group_column_name)
        plt.ylabel(metric.replace("_", " ").title())
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()

def create_model_pipeline(model, vectorizer, text_cleaner):
    """
    Create a complete prediction pipeline
    
    Args:
        model: Trained ML model
        vectorizer: Fitted text vectorizer
        text_cleaner: Text cleaning utility
    
    Returns:
        Pipeline function
    """
    def predict_pipeline(text):
        """Complete prediction pipeline"""
        # Clean text
        cleaned_text = text_cleaner.clean_text(text)
        
        # Vectorize text
        vectorized_text = vectorizer.transform([cleaned_text])
        
        # Make prediction
        prediction = model.predict(vectorized_text)[0]
        
        # Get prediction probability if available
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(vectorized_text)[0]
        else:
            probabilities = None
        
        return {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'prediction': prediction,
            'probabilities': probabilities
        }
    
    return predict_pipeline

# Example usage and utility functions
def print_model_summary(model_name, metrics):
    """Print a formatted summary of model performance"""
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    
    for metric_name, value in metrics.items():
        if metric_name != 'model_name':
            print(f"{metric_name.replace('_', ' ').title()}: {value:.4f}")
    
    print(f"{'='*50}")

# Constants for common model configurations
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Common model parameters
LOGISTIC_REGRESSION_PARAMS = {
    'C': 1.0,
    'max_iter': 1000,
    'random_state': RANDOM_STATE
}

SVM_PARAMS = {
    'C': 1.0,
    'kernel': 'linear',
    'random_state': RANDOM_STATE
}

RANDOM_FOREST_PARAMS = {
    'n_estimators': 100,
    'max_depth': None,
    'random_state': RANDOM_STATE
}

LSTM_PARAMS = {
    'embedding_dim': 128,
    'lstm_units': 64,
    'dropout': 0.5,
    'batch_size': 32,
    'epochs': 10
}
