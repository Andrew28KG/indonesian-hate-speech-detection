#!/usr/bin/env python3
"""
Indonesian Hate Speech Detection - GUI Application

A simple Tkinter-based GUI application for detecting hate speech in Indonesian text.
Uses the best trained model from our machine learning pipeline.

Usage:
    python gui.py

Requirements:
    - Trained models in ../models/ directory
    - Required packages: tkinter, joblib, pandas, numpy, scikit-learn

Author: Data Science Team
Date: 2025-06-21
"""

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import joblib
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
import threading
import re

class HateSpeechDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🇮🇩 Indonesian Hate Speech Detector")
        self.root.geometry("800x600")
        self.root.configure(bg='#f0f0f0')
        
        # Initialize variables
        self.model = None
        self.vectorizer = None
        self.scaler = None
        self.prediction_history = []
        
        # Load models
        self.load_models()
        
        # Create GUI
        self.create_widgets()
        
    def load_models(self):
        """Load the trained model and vectorizer"""
        try:
            # Get the directory where this script is located
            script_dir = os.path.dirname(os.path.abspath(__file__))
            # Go up one level to project root, then into models
            models_dir = os.path.join(script_dir, "..", "models")
            
            model_path = os.path.join(models_dir, "best_hate_speech_model.pkl")
            vectorizer_path = os.path.join(models_dir, "tfidf_vectorizer.pkl")
            scaler_path = os.path.join(models_dir, "feature_scaler.pkl")
            
            print(f"Script directory: {script_dir}")
            print(f"Looking for models in: {os.path.abspath(models_dir)}")
            print(f"Model file exists: {os.path.exists(model_path)}")
            print(f"Vectorizer file exists: {os.path.exists(vectorizer_path)}")
            print(f"Scaler file exists: {os.path.exists(scaler_path)}")
            
            if os.path.exists(model_path) and os.path.exists(vectorizer_path):
                print("Loading models...")
                self.model = joblib.load(model_path)
                self.vectorizer = joblib.load(vectorizer_path)
                
                # Load scaler if available
                if os.path.exists(scaler_path):
                    self.scaler = joblib.load(scaler_path)
                    print("SUCCESS: All models loaded successfully (including scaler)")
                else:
                    print("WARNING: Scaler not found, using features without scaling")
                    
                # Load abusive words dictionary
                self.load_abusive_words()
                
                print("SUCCESS: Models loaded successfully")
            else:
                print("ERROR: Model files not found")
                print(f"Expected paths:")
                print(f"  Model: {os.path.abspath(model_path)}")
                print(f"  Vectorizer: {os.path.abspath(vectorizer_path)}")
                messagebox.showerror("Error", 
                    f"Model files not found. Please ensure models are trained and saved in:\n{os.path.abspath(models_dir)}")
                
        except Exception as e:
            print(f"ERROR: Failed to load models: {e}")
            messagebox.showerror("Error", f"Failed to load models: {e}")
    
    def load_abusive_words(self):
        """Load abusive words dictionary"""
        try:
            script_dir = os.path.dirname(os.path.abspath(__file__))
            abusive_path = os.path.join(script_dir, "..", "IndonesianAbusiveWords", "abusive.csv")
            
            if os.path.exists(abusive_path):
                abusive_df = pd.read_csv(abusive_path)
                self.abusive_words = set(abusive_df['ABUSIVE'].str.lower())
                print(f"SUCCESS: Loaded {len(self.abusive_words)} abusive words")
            else:
                print("WARNING: Abusive words dictionary not found")
                self.abusive_words = set()
        except Exception as e:
            print(f"WARNING: Could not load abusive words: {e}")
            self.abusive_words = set()
    
    def basic_text_clean(self, text):
        """Basic text cleaning for Indonesian text"""
        if pd.isna(text) or text is None:
            return ""
        
        text = str(text).lower()
        text = re.sub(r'\buser\b', '', text)
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^rt\s+', '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def create_widgets(self):
        """Create and arrange GUI widgets"""
        # Main title
        title_frame = tk.Frame(self.root, bg='#f0f0f0')
        title_frame.pack(pady=10, fill='x')
        
        title_label = tk.Label(
            title_frame, 
            text="🇮🇩 Indonesian Hate Speech Detector",
            font=('Arial', 18, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack()
        
        subtitle_label = tk.Label(
            title_frame,
            text="Deteksi Ujaran Kebencian dalam Bahasa Indonesia",
            font=('Arial', 12),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        subtitle_label.pack()
        
        # Input section
        input_frame = tk.LabelFrame(
            self.root, 
            text="📝 Input Text", 
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        input_frame.pack(pady=10, padx=20, fill='both', expand=True)
        
        # Text input area
        self.text_input = scrolledtext.ScrolledText(
            input_frame,
            height=8,
            width=70,
            font=('Arial', 11),
            wrap=tk.WORD,
            bg='white',
            fg='black'
        )
        self.text_input.pack(pady=5, fill='both', expand=True)
        
        # Placeholder text
        placeholder_text = "Masukkan teks dalam bahasa Indonesia di sini untuk dianalisis...\n\nContoh:\n- Saya sangat menghargai pendapat Anda\n- Terima kasih atas bantuan Anda"
        self.text_input.insert('1.0', placeholder_text)
        self.text_input.configure(fg='gray')
        
        # Bind events for placeholder behavior
        self.text_input.bind('<FocusIn>', self.on_text_focus_in)
        self.text_input.bind('<FocusOut>', self.on_text_focus_out)
        
        # Button frame
        button_frame = tk.Frame(input_frame, bg='#f0f0f0')
        button_frame.pack(pady=10, fill='x')
        
        # Analyze button
        self.analyze_button = tk.Button(
            button_frame,
            text="🔍 Analyze Text",
            font=('Arial', 12, 'bold'),
            bg='#3498db',
            fg='white',
            command=self.analyze_text,
            padx=20,
            pady=5,
            cursor='hand2'
        )
        self.analyze_button.pack(side='left', padx=5)
        
        # Clear button
        clear_button = tk.Button(
            button_frame,
            text="🗑️ Clear",
            font=('Arial', 12),
            bg='#95a5a6',
            fg='white',
            command=self.clear_text,
            padx=20,
            pady=5,
            cursor='hand2'
        )
        clear_button.pack(side='left', padx=5)
        
        # Example button
        example_button = tk.Button(
            button_frame,
            text="📄 Example",
            font=('Arial', 12),
            bg='#f39c12',
            fg='white',
            command=self.load_example,
            padx=20,
            pady=5,
            cursor='hand2'
        )
        example_button.pack(side='left', padx=5)
        
        # Dictionary button
        dict_button = tk.Button(
            button_frame,
            text="📚 Dictionary",
            font=('Arial', 12),
            bg='#9b59b6',
            fg='white',
            command=self.show_dictionary,
            padx=20,
            pady=5,
            cursor='hand2'
        )
        dict_button.pack(side='left', padx=5)
        
        # Results section
        results_frame = tk.LabelFrame(
            self.root,
            text="📊 Analysis Results",
            font=('Arial', 12, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50',
            padx=10,
            pady=10
        )
        results_frame.pack(pady=10, padx=20, fill='x')
        
        # Result display
        self.result_frame = tk.Frame(results_frame, bg='#f0f0f0')
        self.result_frame.pack(fill='x', pady=5)
        
        # Statistics frame
        stats_frame = tk.Frame(results_frame, bg='#f0f0f0')
        stats_frame.pack(fill='x', pady=5)
        
        # Text statistics
        self.char_count_label = tk.Label(
            stats_frame,
            text="Characters: 0",
            font=('Arial', 10),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        self.char_count_label.pack(side='left', padx=10)
        
        self.word_count_label = tk.Label(
            stats_frame,
            text="Words: 0",
            font=('Arial', 10),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        self.word_count_label.pack(side='left', padx=10)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready" if self.model is not None else "Models not loaded")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            relief=tk.SUNKEN,
            anchor='w',
            bg='#ecf0f1',
            fg='#2c3e50'
        )
        status_bar.pack(side='bottom', fill='x')
        
        # Initial result display
        self.show_initial_result()
    
    def on_text_focus_in(self, event):
        """Handle text input focus in"""
        if self.text_input.get('1.0', tk.END).strip().startswith("Masukkan teks"):
            self.text_input.delete('1.0', tk.END)
            self.text_input.configure(fg='black')
    
    def on_text_focus_out(self, event):
        """Handle text input focus out"""
        if not self.text_input.get('1.0', tk.END).strip():
            placeholder_text = "Masukkan teks dalam bahasa Indonesia di sini untuk dianalisis...\n\nContoh:\n- Saya sangat menghargai pendapat Anda\n- Terima kasih atas bantuan Anda"
            self.text_input.insert('1.0', placeholder_text)
            self.text_input.configure(fg='gray')
    
    def load_example(self):
        """Load example text for testing"""
        examples = [
            "Saya sangat menghargai pendapat Anda tentang topik ini.",
            "Terima kasih atas bantuan yang sangat berharga.",
            "Mari kita berdiskusi dengan sopan dan saling menghormati."
        ]
        
        # Show example selection dialog
        example_window = tk.Toplevel(self.root)
        example_window.title("Select Example")
        example_window.geometry("400x200")
        example_window.configure(bg='#f0f0f0')
        
        tk.Label(example_window, text="Choose an example:", 
                font=('Arial', 12), bg='#f0f0f0').pack(pady=10)
        
        for i, example in enumerate(examples):
            btn = tk.Button(
                example_window,
                text=f"Example {i+1}: {example[:30]}...",
                command=lambda ex=example: self.set_example_text(ex, example_window),
                bg='#3498db',
                fg='white',
                pady=5
            )
            btn.pack(pady=2, padx=20, fill='x')
    
    def set_example_text(self, text, window):
        """Set example text and close window"""
        self.clear_text()
        self.text_input.delete('1.0', tk.END)
        self.text_input.insert('1.0', text)
        self.text_input.configure(fg='black')
        self.update_text_statistics(text)
        window.destroy()
    
    def show_dictionary(self):
        """Show the abusive words dictionary"""
        if not hasattr(self, 'abusive_words') or not self.abusive_words:
            messagebox.showwarning("Warning", "Abusive words dictionary not loaded.")
            return
        
        # Create dictionary window
        dict_window = tk.Toplevel(self.root)
        dict_window.title("Abusive Words Dictionary")
        dict_window.geometry("500x400")
        dict_window.configure(bg='#f0f0f0')
        
        # Title
        title_label = tk.Label(
            dict_window, 
            text=f"📚 Abusive Words Dictionary ({len(self.abusive_words)} words)",
            font=('Arial', 14, 'bold'),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        title_label.pack(pady=10)
        
        # Search frame
        search_frame = tk.Frame(dict_window, bg='#f0f0f0')
        search_frame.pack(fill='x', padx=20, pady=5)
        
        tk.Label(search_frame, text="Search:", font=('Arial', 10), bg='#f0f0f0').pack(side='left')
        search_entry = tk.Entry(search_frame, font=('Arial', 10))
        search_entry.pack(side='left', fill='x', expand=True, padx=5)
        
        # Words display
        words_frame = tk.Frame(dict_window, bg='#f0f0f0')
        words_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Scrollable text widget
        words_text = scrolledtext.ScrolledText(
            words_frame,
            height=15,
            width=60,
            font=('Arial', 10),
            wrap=tk.WORD,
            bg='white',
            fg='black'
        )
        words_text.pack(fill='both', expand=True)
        
        # Display words in columns
        words_list = sorted(list(self.abusive_words))
        columns = 3
        words_per_column = len(words_list) // columns + 1
        
        display_text = ""
        for i in range(words_per_column):
            row_words = []
            for col in range(columns):
                idx = col * words_per_column + i
                if idx < len(words_list):
                    row_words.append(f"{words_list[idx]:<15}")
            display_text += "  ".join(row_words) + "\n"
        
        words_text.insert('1.0', display_text)
        words_text.config(state='disabled')
        
        # Search functionality
        def search_words():
            search_term = search_entry.get().lower()
            if not search_term:
                words_text.config(state='normal')
                words_text.delete('1.0', tk.END)
                words_text.insert('1.0', display_text)
                words_text.config(state='disabled')
                return
            
            filtered_words = [word for word in words_list if search_term in word]
            if filtered_words:
                filtered_display = "\n".join(filtered_words)
                words_text.config(state='normal')
                words_text.delete('1.0', tk.END)
                words_text.insert('1.0', f"Found {len(filtered_words)} word(s) matching '{search_term}':\n\n{filtered_display}")
                words_text.config(state='disabled')
            else:
                words_text.config(state='normal')
                words_text.delete('1.0', tk.END)
                words_text.insert('1.0', f"No words found matching '{search_term}'")
                words_text.config(state='disabled')
        
        search_entry.bind('<KeyRelease>', lambda e: search_words())
        
        # Close button
        close_button = tk.Button(
            dict_window,
            text="Close",
            font=('Arial', 12),
            bg='#95a5a6',
            fg='white',
            command=dict_window.destroy,
            padx=20,
            pady=5
        )
        close_button.pack(pady=10)
    
    def clear_text(self):
        """Clear the text input"""
        self.text_input.delete('1.0', tk.END)
        self.on_text_focus_out(None)
        self.update_text_statistics("")
        self.show_initial_result()
    
    def show_initial_result(self):
        """Display initial result state"""
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        initial_label = tk.Label(
            self.result_frame,
            text="Enter Indonesian text above and click 'Analyze Text' to begin",
            font=('Arial', 11),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        initial_label.pack(pady=10)
    
    def update_text_statistics(self, text):
        """Update character and word count"""
        char_count = len(text)
        word_count = len(text.split()) if text.strip() else 0
        
        self.char_count_label.config(text=f"Characters: {char_count}")
        self.word_count_label.config(text=f"Words: {word_count}")
    
    def analyze_text(self):
        """Analyze the input text for hate speech"""
        input_text = self.text_input.get('1.0', tk.END).strip()
        
        # Check if placeholder text
        if input_text.startswith("Masukkan teks") or not input_text:
            messagebox.showwarning("Warning", "Please enter some text to analyze.")
            return
        
        if self.model is None or self.vectorizer is None:
            messagebox.showerror("Error", "Models not loaded. Cannot perform analysis.")
            return
        
        # Update statistics
        self.update_text_statistics(input_text)
        
        # Disable analyze button during processing
        self.analyze_button.config(state='disabled', text="🔄 Analyzing...")
        self.status_var.set("Analyzing text...")
        
        # Perform analysis in separate thread
        analysis_thread = threading.Thread(target=self.perform_analysis, args=(input_text,))
        analysis_thread.daemon = True
        analysis_thread.start()
    
    def perform_analysis(self, input_text):
        """Perform the actual hate speech analysis"""
        try:
            # Clean the text
            cleaned_text = self.basic_text_clean(input_text)
            
            # Create TF-IDF features
            vectorized_text = self.vectorizer.transform([cleaned_text])
            
            # Create numeric features
            def count_abusive_words(text):
                if not text or not hasattr(self, 'abusive_words'):
                    return 0, []
                words = text.lower().split()
                found_abusive = [word for word in words if word in self.abusive_words]
                return len(found_abusive), found_abusive
            
            char_length = len(input_text)
            word_count = len(input_text.split())
            abusive_word_count, found_abusive_words = count_abusive_words(input_text)
            
            numeric_features = np.array([[char_length, word_count, abusive_word_count]])
            
            # Scale numeric features if scaler is available
            if self.scaler is not None:
                numeric_features = self.scaler.transform(numeric_features)
            
            # Combine TF-IDF and numeric features
            from scipy.sparse import hstack
            combined_features = hstack([vectorized_text, numeric_features])
            
            # Make prediction
            prediction = self.model.predict(combined_features)[0]
            
            # Get prediction probabilities
            if hasattr(self.model, 'predict_proba'):
                probabilities = self.model.predict_proba(combined_features)[0]
                confidence = max(probabilities)
                hate_speech_prob = probabilities[1] if len(probabilities) > 1 else 0
            else:
                confidence = 1.0
                hate_speech_prob = float(prediction)
            
            # Store in history
            self.prediction_history.append({
                'timestamp': datetime.now(),
                'text': input_text[:100] + "..." if len(input_text) > 100 else input_text,
                'prediction': prediction,
                'confidence': confidence
            })
            
            # Schedule UI update on main thread
            self.root.after(0, self.display_results, prediction, confidence, hate_speech_prob, cleaned_text, abusive_word_count, found_abusive_words)
            
        except Exception as e:
            self.root.after(0, self.handle_analysis_error, str(e))
    
    def display_results(self, prediction, confidence, hate_speech_prob, cleaned_text, abusive_word_count=0, found_abusive_words=None):
        """Display analysis results"""
        if found_abusive_words is None:
            found_abusive_words = []
            
        # Clear previous results
        for widget in self.result_frame.winfo_children():
            widget.destroy()
        
        # Determine result styling
        if prediction == 1:  # Hate speech detected
            result_text = "⚠️ HATE SPEECH DETECTED"
            result_color = '#e74c3c'
            bg_color = '#fadbd8'
        else:  # Normal text
            result_text = "✅ NORMAL TEXT"
            result_color = '#27ae60'
            bg_color = '#d5f4e6'
        
        # Main result display
        result_main_frame = tk.Frame(self.result_frame, bg=bg_color, relief='ridge', bd=2)
        result_main_frame.pack(fill='x', pady=5, padx=10)
        
        # Result label
        result_label = tk.Label(
            result_main_frame,
            text=result_text,
            font=('Arial', 14, 'bold'),
            fg=result_color,
            bg=bg_color
        )
        result_label.pack(pady=10)
        
        # Confidence information
        confidence_frame = tk.Frame(self.result_frame, bg='#f0f0f0')
        confidence_frame.pack(fill='x', pady=5)
        
        confidence_label = tk.Label(
            confidence_frame,
            text=f"Confidence: {confidence:.1%}",
            font=('Arial', 11),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        confidence_label.pack(side='left', padx=10)
        
        prob_label = tk.Label(
            confidence_frame,
            text=f"Hate Speech Probability: {hate_speech_prob:.1%}",
            font=('Arial', 11),
            bg='#f0f0f0',
            fg='#2c3e50'
        )
        prob_label.pack(side='left', padx=10)
        
        # Abusive Words Detection Section
        if abusive_word_count > 0:
            abusive_frame = tk.LabelFrame(
                self.result_frame,
                text="🚨 Abusive Words Detection",
                font=('Arial', 10, 'bold'),
                bg='#f0f0f0',
                fg='#e74c3c',
                padx=10,
                pady=5
            )
            abusive_frame.pack(fill='x', pady=5, padx=10)
            
            # Count info
            count_info = tk.Label(
                abusive_frame,
                text=f"Found {abusive_word_count} abusive word(s):",
                font=('Arial', 10, 'bold'),
                bg='#f0f0f0',
                fg='#e74c3c'
            )
            count_info.pack(anchor='w', pady=2)
            
                        # List of found words
            words_text = ", ".join(found_abusive_words)
            if len(words_text) > 50:  # Truncate if too long
                words_text = words_text[:47] + "..."
                
            words_label = tk.Label(
                abusive_frame,
                text=f"• {words_text}",
                font=('Arial', 10),
                bg='#f0f0f0',
                fg='#c0392b',
                wraplength=400
            )
            words_label.pack(anchor='w', padx=10, pady=2)
            
            # Highlighted text preview
            if len(found_abusive_words) > 0:
                preview_label = tk.Label(
                    abusive_frame,
                    text="Text with highlighted words:",
                    font=('Arial', 9, 'bold'),
                    bg='#f0f0f0',
                    fg='#2c3e50'
                )
                preview_label.pack(anchor='w', pady=(5, 2))
                
                # Create highlighted text
                original_text = self.text_input.get('1.0', tk.END).strip()
                highlighted_text = self.highlight_abusive_words(original_text, found_abusive_words)
                
                text_preview = tk.Text(
                    abusive_frame,
                    height=3,
                    font=('Arial', 9),
                    bg='#fdfefe',
                    fg='#2c3e50',
                    wrap=tk.WORD,
                    relief='sunken',
                    bd=1
                )
                text_preview.pack(fill='x', padx=10, pady=2)
                
                # Insert text with highlighting
                text_preview.insert('1.0', original_text)
                
                # Highlight abusive words
                self.highlight_text_in_widget(text_preview, original_text, found_abusive_words)
                
                text_preview.config(state='disabled')
            
            # Warning message
            if abusive_word_count >= 2:
                warning_label = tk.Label(
                    abusive_frame,
                    text="⚠️ Multiple abusive words detected - High risk of hate speech",
                    font=('Arial', 9, 'italic'),
                    bg='#f0f0f0',
                    fg='#e74c3c'
                )
                warning_label.pack(anchor='w', pady=2)
            elif abusive_word_count == 1:
                info_label = tk.Label(
                    abusive_frame,
                    text="ℹ️ Single abusive word - Context considered for classification",
                    font=('Arial', 9, 'italic'),
                    bg='#f0f0f0',
                    fg='#f39c12'
                )
                info_label.pack(anchor='w', pady=2)
        else:
            # Clean text indicator
            clean_frame = tk.Frame(self.result_frame, bg='#f0f0f0')
            clean_frame.pack(fill='x', pady=5)
            
            clean_label = tk.Label(
                clean_frame,
                text="✅ No abusive words detected",
                font=('Arial', 11),
                bg='#f0f0f0',
                fg='#27ae60'
            )
            clean_label.pack(side='left', padx=10)
        
        # Additional analysis info (text statistics)
        stats_frame = tk.Frame(self.result_frame, bg='#f0f0f0')
        stats_frame.pack(fill='x', pady=5)
        
        # Analysis method info
        method_label = tk.Label(
            stats_frame,
            text="Analysis: TF-IDF + Numeric Features + Abusive Words Count",
            font=('Arial', 9, 'italic'),
            bg='#f0f0f0',
            fg='#7f8c8d'
        )
        method_label.pack(side='left', padx=10)
        
        # Re-enable analyze button
        self.analyze_button.config(state='normal', text="🔍 Analyze Text")
        self.status_var.set("Analysis complete")
    
    def handle_analysis_error(self, error_msg):
        """Handle analysis errors"""
        messagebox.showerror("Analysis Error", f"An error occurred during analysis: {error_msg}")
        self.analyze_button.config(state='normal', text="🔍 Analyze Text")
        self.status_var.set("Analysis failed")
    
    def highlight_abusive_words(self, text, abusive_words):
        """Create text with highlighted abusive words"""
        if not abusive_words:
            return text
        
        # Simple highlighting by surrounding with markers
        highlighted = text
        for word in abusive_words:
            # Case insensitive replacement
            import re
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            highlighted = pattern.sub(f"[{word}]", highlighted)
        
        return highlighted
    
    def highlight_text_in_widget(self, text_widget, original_text, abusive_words):
        """Highlight abusive words in a text widget"""
        if not abusive_words:
            return
        
        # Configure highlight tag
        text_widget.tag_configure("highlight", background="#ffcccc", foreground="#8b0000", font=('Arial', 9, 'bold'))
        
        # Find and highlight each abusive word
        for word in abusive_words:
            # Search for the word (case insensitive)
            start_pos = '1.0'
            while True:
                pos = text_widget.search(word, start_pos, tk.END, nocase=True)
                if not pos:
                    break
                
                # Calculate end position
                end_pos = f"{pos}+{len(word)}c"
                
                # Apply highlight tag
                text_widget.tag_add("highlight", pos, end_pos)
                
                # Move to next occurrence
                start_pos = end_pos

def main():
    """Main application entry point"""
    root = tk.Tk()
    app = HateSpeechDetectorGUI(root)
    
    # Center the window
    root.update_idletasks()
    x = (root.winfo_screenwidth() - root.winfo_width()) // 2
    y = (root.winfo_screenheight() - root.winfo_height()) // 2
    root.geometry(f"+{x}+{y}")
    
    root.mainloop()

if __name__ == "__main__":
    main() 