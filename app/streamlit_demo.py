import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
import sys
import os
import re
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path
project_root = os.path.join(os.path.dirname(__file__), '..')
sys.path.append(project_root)

# Import text cleaner
try:
    from utils.text_cleaning import IndonesianTextCleaner
    TEXT_CLEANER_AVAILABLE = True
except ImportError:
    # Silently fall back to built-in cleaner
    TEXT_CLEANER_AVAILABLE = False
    
    # Create a simple fallback text cleaner
    class IndonesianTextCleaner:
        def clean_text(self, text):
            """Enhanced fallback text cleaning for Indonesian text"""
            if not text:
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
            
            # Normalize common Indonesian slang
            slang_dict = {
                'gw': 'saya', 'gue': 'saya', 'lu': 'kamu', 'loe': 'kamu', 'elo': 'kamu',
                'emang': 'memang', 'udah': 'sudah', 'udh': 'sudah', 'blm': 'belum',
                'gimana': 'bagaimana', 'gmn': 'bagaimana', 'knp': 'kenapa', 'jgn': 'jangan',
                'bgt': 'banget', 'yg': 'yang', 'krn': 'karena', 'karna': 'karena',
                'dgn': 'dengan', 'utk': 'untuk', 'tdk': 'tidak', 'ga': 'tidak',
                'gak': 'tidak', 'ngga': 'tidak'
            }
            
            # Apply slang normalization
            words = text.split()
            words = [slang_dict.get(word, word) for word in words]
            text = ' '.join(words)
            
            # Remove excessive character repetitions
            text = re.sub(r'(.)\1{2,}', r'\1\1', text)
            
            # Remove punctuation and numbers (keep only letters and spaces)
            text = re.sub(r'[^a-zA-Z\s]', ' ', text)
            
            # Remove extra whitespace
            text = re.sub(r'\s+', ' ', text)
            
            return text.strip()

import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Indonesian Hate Speech Detection",
    page_icon="🇮🇩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .result-box {
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 2px solid;
    }
    .hate-speech {
        background-color: #ffebee;
        border-color: #f44336;
        color: #d32f2f;
    }
    .non-hate-speech {
        background-color: #e8f5e8;
        border-color: #4caf50;
        color: #388e3c;
    }
    .warning-box {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    .abusive-word {
        background-color: #ffcdd2;
        border: 1px solid #f44336;
        border-radius: 4px;
        padding: 0.25rem 0.5rem;
        margin: 0.2rem;
        display: inline-block;
        font-size: 0.9rem;
        font-weight: bold;
        color: #d32f2f;
    }
    .clean-word {
        background-color: #c8e6c9;
        border: 1px solid #4caf50;
        border-radius: 4px;
        padding: 0.25rem 0.5rem;
        margin: 0.1rem;
        display: inline-block;
        font-size: 0.9rem;
        color: #388e3c;
    }
    .analysis-section {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        border-left: 4px solid #007bff;
    }
</style>
""", unsafe_allow_html=True)

# Initialize text cleaner
@st.cache_resource
def load_text_cleaner():
    if TEXT_CLEANER_AVAILABLE:
        st.success("✅ Advanced Indonesian text cleaner loaded!")
        return IndonesianTextCleaner()
    else:
        st.info("ℹ️ Using optimized built-in text cleaner")
        return IndonesianTextCleaner()

# Load abusive words dictionary
@st.cache_resource
def load_abusive_words():
    """Load Indonesian abusive words dictionary"""
    try:
        # Load from the updated abusive words file
        abusive_words_path = os.path.join(os.path.dirname(__file__), '..', 'IndonesianAbusiveWords', 'abusive.csv')
        if os.path.exists(abusive_words_path):
            df_abusive = pd.read_csv(abusive_words_path)
            if 'ABUSIVE' in df_abusive.columns:
                words = set(df_abusive['ABUSIVE'].str.lower().dropna())
                st.success(f"✅ Loaded {len(words)} abusive words from dictionary")
                return words
            elif len(df_abusive.columns) > 0:
                words = set(df_abusive.iloc[:, 0].str.lower().dropna())
                st.success(f"✅ Loaded {len(words)} abusive words from dictionary")
                return words
        
        # Fallback: common Indonesian abusive words
        fallback_words = {
            'anjing', 'bangsat', 'bego', 'bodoh', 'goblok', 'tolol', 'idiot', 'dungu',
            'kampret', 'bajingan', 'kontol', 'memek', 'lonte', 'pelacur',
            'banci', 'homo', 'lesbi', 'kafir', 'murtad', 'munafik', 'tai',
            'fuck', 'shit', 'damn', 'kntl', 'mmk'
        }
        st.warning("⚠️ Using fallback abusive words dictionary")
        return fallback_words
    except Exception as e:
        st.error(f"❌ Error loading abusive words dictionary: {e}")
        return set()

# Load models with metadata
@st.cache_resource
def load_model():
    """Load the hate speech detection model with metadata"""
    try:
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'best_hate_speech_model.pkl')
        metadata_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_metadata.json')
        
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            
            # Load metadata if available
            metadata = {}
            if os.path.exists(metadata_path):
                import json
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            
            model_type = metadata.get('model_type', 'Logistic Regression')
            st.success(f"✅ {model_type} loaded successfully!")
            return model, metadata
        else:
            st.warning("⚠️ Model file not found. Analysis will be based on abusive words only.")
            return None, {}
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None, {}

@st.cache_resource
def load_vectorizer():
    """Load the TF-IDF vectorizer with details"""
    try:
        vectorizer_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'tfidf_vectorizer.pkl')
        if os.path.exists(vectorizer_path):
            vectorizer = joblib.load(vectorizer_path)
            vocab_size = len(vectorizer.vocabulary_) if hasattr(vectorizer, 'vocabulary_') else 'Unknown'
            st.success(f"✅ TF-IDF Vectorizer loaded (Vocab: {vocab_size} words)!")
            return vectorizer, vocab_size
        else:
            st.warning("⚠️ Vectorizer file not found.")
            return None, 0
    except Exception as e:
        st.error(f"❌ Error loading vectorizer: {e}")
        return None, 0

@st.cache_resource
def load_scaler():
    """Load the feature scaler for numeric features"""
    try:
        scaler_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_scaler.pkl')
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            st.success("✅ Feature scaler loaded successfully!")
            return scaler
        else:
            st.warning("⚠️ Feature scaler not found. Using features without scaling.")
            return None
    except Exception as e:
        st.error(f"❌ Error loading scaler: {e}")
        return None

def analyze_abusive_words(text, abusive_words_dict):
    """
    Analyze abusive words in the given text
    """
    if not text or not abusive_words_dict:
        return {
            'count': 0,
            'words': [],
            'percentage': 0,
            'word_positions': {}
        }
    
    # Convert to lowercase and split into words
    words = re.findall(r'\b\w+\b', text.lower())
    total_words = len(words)
    
    # Find abusive words with positions
    found_abusive = []
    word_positions = {}
    
    for i, word in enumerate(words):
        if word in abusive_words_dict:
            found_abusive.append(word)
            if word in word_positions:
                word_positions[word].append(i)
            else:
                word_positions[word] = [i]
    
    # Calculate percentage
    percentage = (len(found_abusive) / total_words * 100) if total_words > 0 else 0
    
    return {
        'count': len(found_abusive),
        'words': list(set(found_abusive)),  # Unique words
        'all_instances': found_abusive,  # All instances including duplicates
        'percentage': round(percentage, 2),
        'word_positions': word_positions,
        'total_words': total_words
    }

def predict_hate_speech(text, model_data, vectorizer_data, scaler_data, abusive_words, text_cleaner):
    """
    Predict hate speech for given text using Logistic Regression with combined features
    """
    if not text:
        return None
    
    model, metadata = model_data if model_data else (None, {})
    vectorizer, vocab_size = vectorizer_data if vectorizer_data else (None, 0)
    
    # Clean the text
    cleaned_text = text_cleaner.clean_text(text)
    
    # Check if we have actual models
    if model is not None and vectorizer is not None:
        try:
            # 1. Create TF-IDF features
            vectorized_text = vectorizer.transform([cleaned_text])
            
            # 2. Create numeric features
            def count_abusive_words(text):
                if not text or not abusive_words:
                    return 0
                words = text.lower().split()
                return sum(1 for word in words if word in abusive_words)
            
            char_length = len(text)
            word_count = len(text.split())
            abusive_word_count = count_abusive_words(text)
            
            numeric_features = np.array([[char_length, word_count, abusive_word_count]])
            
            # 3. Scale numeric features if scaler is available
            if scaler_data is not None:
                numeric_features = scaler_data.transform(numeric_features)
            
            # 4. Combine TF-IDF and numeric features
            from scipy.sparse import hstack
            combined_features = hstack([vectorized_text, numeric_features])
            
            # 5. Make prediction
            prediction = model.predict(combined_features)[0]
            probability = model.predict_proba(combined_features)[0]
            
            return {
                'prediction': int(prediction),
                'confidence': round(max(probability) * 100, 2),
                'probabilities': {
                    'not_hate_speech': round(probability[0] * 100, 2),
                    'hate_speech': round(probability[1] * 100, 2)
                },
                'cleaned_text': cleaned_text,
                'features': {
                    'char_length': char_length,
                    'word_count': word_count,
                    'abusive_word_count': abusive_word_count,
                    'tfidf_features': vectorized_text.shape[1],
                    'total_features': combined_features.shape[1]
                },
                'metadata': metadata
            }
        except Exception as e:
            st.error(f"❌ Error in prediction: {e}")
            return None
    else:
        return None

def get_text_statistics(text):
    """Get basic text statistics"""
    if not text:
        return {}
    
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    return {
        'characters': len(text),
        'characters_no_spaces': len(text.replace(' ', '')),
        'words': len(words),
        'sentences': len(sentences),
        'avg_word_length': round(np.mean([len(word) for word in words]), 2) if words else 0,
        'avg_sentence_length': round(np.mean([len(sentence.split()) for sentence in sentences]), 2) if sentences else 0
    }

def create_word_frequency_chart(text, top_n=10):
    """Create word frequency visualization"""
    if not text:
        return None
    
    # Clean and tokenize
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove common stop words
    stop_words = {'dan', 'di', 'ke', 'dari', 'yang', 'untuk', 'dengan', 'ini', 'itu', 
                  'atau', 'tapi', 'jika', 'karena', 'seperti', 'akan', 'sudah', 'masih', 
                  'juga', 'hanya', 'bisa', 'harus', 'dapat', 'mau', 'ingin', 'perlu', 
                  'boleh', 'tidak', 'bukan', 'jangan', 'saya', 'kamu', 'dia', 'mereka', 
                  'kita', 'kami', 'anda', 'beliau', 'ya', 'iya', 'oh', 'ah', 'eh'}
    
    words = [word for word in words if word not in stop_words and len(word) > 2]
    
    if not words:
        return None
    
    # Count frequencies
    word_freq = Counter(words)
    top_words = word_freq.most_common(top_n)
    
    if not top_words:
        return None
    
    # Create chart
    words_list = [word for word, _ in top_words]
    frequencies = [freq for _, freq in top_words]
    
    fig = px.bar(
        x=frequencies,
        y=words_list,
        orientation='h',
        title=f'Top {len(top_words)} Most Frequent Words',
        labels={'x': 'Frequency', 'y': 'Words'}
    )
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    
    return fig

def highlight_abusive_words(text, abusive_words):
    """Highlight abusive words in text"""
    if not text or not abusive_words:
        return text
    
    # Create a copy of text for highlighting
    highlighted_text = text
    
    # Sort abusive words by length (longer first to avoid partial replacements)
    sorted_abusive = sorted(abusive_words, key=len, reverse=True)
    
    for word in sorted_abusive:
        # Use word boundaries to match whole words only
        pattern = r'\b' + re.escape(word) + r'\b'
        highlighted_text = re.sub(
            pattern, 
            f'<span class="abusive-word">{word}</span>',
            highlighted_text,
            flags=re.IGNORECASE
        )
    
    return highlighted_text

def main():
    # Initialize session state for text input if not exists
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    
    # Header
    st.markdown('<h1 class="main-header">🇮🇩 Indonesian Hate Speech Detection</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Analyze Indonesian text for hate speech and abusive language</p>', unsafe_allow_html=True)
    
    # Load resources
    text_cleaner = load_text_cleaner()
    abusive_words = load_abusive_words()
    model, metadata = load_model()
    vectorizer, vocab_size = load_vectorizer()
    scaler = load_scaler()
    
    # Sidebar
    st.sidebar.header("📊 Analysis Options")
    
    # Analysis options
    show_text_stats = st.sidebar.checkbox("Show Text Statistics", value=True)
    show_word_freq = st.sidebar.checkbox("Show Word Frequency", value=True)
    show_abusive_analysis = st.sidebar.checkbox("Show Abusive Words Analysis", value=True)
    show_model_prediction = st.sidebar.checkbox("Show Model Prediction", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 Instructions")
    st.sidebar.markdown("""
    1. Enter Indonesian text in the text area
    2. Click 'Analyze Text' to see results
    3. View detailed analysis including:
       - Abusive words detection
       - Text statistics
       - Hate speech prediction
       - Word frequency analysis
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📚 Dictionary")
    if st.sidebar.button("📖 View Abusive Words Dictionary"):
        if abusive_words:
            # Create a dictionary display dialog
            with st.sidebar:
                st.markdown("#### 📚 Abusive Words Dictionary")
                st.markdown(f"**Total words:** {len(abusive_words)}")
                
                # Search functionality
                search_term = st.text_input("🔍 Search words:", placeholder="Type to search...")
                
                if search_term:
                    filtered_words = [word for word in sorted(abusive_words) if search_term.lower() in word.lower()]
                    if filtered_words:
                        st.markdown(f"**Found {len(filtered_words)} words:**")
                        # Display in columns
                        words_text = ", ".join(filtered_words[:50])  # Limit to 50 words
                        if len(filtered_words) > 50:
                            words_text += f"... and {len(filtered_words) - 50} more"
                        st.text_area("Results:", value=words_text, height=100, disabled=True)
                    else:
                        st.info("No words found")
                else:
                    # Show sample words
                    sample_words = list(sorted(abusive_words))[:20]
                    st.markdown("**Sample words:**")
                    st.text_area("Sample:", value=", ".join(sample_words), height=100, disabled=True)
                    st.info(f"Showing 20 of {len(abusive_words)} words. Use search to find specific words.")
        else:
            st.sidebar.error("Dictionary not loaded")
    
    # Main content
    st.markdown("### 📝 Enter Text for Analysis")
    
    # Text input
    text_input = st.text_area(
        "Enter Indonesian text to analyze:",
        value=st.session_state.text_input,
        height=150,
        placeholder="Masukkan teks bahasa Indonesia di sini untuk dianalisis...",
        help="Enter any Indonesian text to check for hate speech and abusive language",
        key="main_text_input"
    )
    
    # Example texts
    st.markdown("#### 💡 Example Texts")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("Normal Text Example"):
            st.session_state.text_input = "Selamat pagi semua! Hari ini cuaca sangat cerah dan indah. Semoga hari kalian menyenangkan."
    
    with col2:
        if st.button("Mild Negative Example"):
            st.session_state.text_input = "Kamu itu bodoh banget sih, tidak bisa mengerjakan tugas sederhana seperti ini."
    
    with col3:
        if st.button("Strong Negative Example"):
            st.session_state.text_input = "Dasar anjing lu! Bangsat emang kelakuan lu, kontol!"
    
    # Analysis button
    if st.button("🔍 Analyze Text", type="primary"):
        # Update session state with current text input
        st.session_state.text_input = text_input
        
        if not text_input.strip():
            st.warning("⚠️ Please enter some text to analyze.")
            return
        
        if len(text_input.strip()) < 3:
            st.warning("⚠️ Please enter at least 3 characters.")
            return
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Main Results", "📊 Text Statistics", "📈 Word Analysis", "🔧 Technical Details"])
        
        with tab1:
            st.markdown("### 🎯 Analysis Results")
            
            # Combined Analysis Section
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🚨 Abusive Words Detection")
                # Abusive words analysis
                if show_abusive_analysis and abusive_words:
                    abusive_analysis = analyze_abusive_words(text_input, abusive_words)
                    
                    # Display result box
                    if abusive_analysis['count'] > 0:
                        st.markdown(f"""
                        <div class="result-box hate-speech">
                            <h4>⚠️ ABUSIVE WORDS FOUND</h4>
                            <p><strong>Total instances:</strong> {abusive_analysis['count']}</p>
                            <p><strong>Unique words:</strong> {len(abusive_analysis['words'])}</p>
                            <p><strong>Percentage:</strong> {abusive_analysis['percentage']}%</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show abusive words
                        st.markdown("**Detected words:**")
                        abusive_words_html = ""
                        for word in abusive_analysis['words']:
                            count = abusive_analysis['all_instances'].count(word)
                            abusive_words_html += f'<span class="abusive-word">{word} ({count}x)</span> '
                        st.markdown(abusive_words_html, unsafe_allow_html=True)
                        
                        # Warning based on count
                        if len(abusive_analysis['words']) >= 2:
                            st.warning("⚠️ Multiple abusive words detected - High risk of hate speech")
                        elif len(abusive_analysis['words']) == 1:
                            st.info("ℹ️ Single abusive word - Context considered for classification")
                        
                    else:
                        st.markdown(f"""
                        <div class="result-box non-hate-speech">
                            <h4>✅ NO ABUSIVE WORDS</h4>
                            <p>No abusive words found in the text.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("Abusive words analysis disabled")
            
            with col2:
                st.markdown("#### 🤖 AI Hate Speech Detection")
                # Model prediction
                if show_model_prediction:
                    prediction_result = predict_hate_speech(text_input, (model, metadata), (vectorizer, vocab_size), scaler, abusive_words, text_cleaner)
                
                    if prediction_result:
                        if prediction_result['prediction'] == 1:
                            st.markdown(f"""
                            <div class="result-box hate-speech">
                                <h4>🚨 HATE SPEECH DETECTED</h4>
                                <p><strong>Confidence:</strong> {prediction_result['confidence']}%</p>
                                <p><strong>Probability:</strong> {prediction_result['probabilities']['hate_speech']}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="result-box non-hate-speech">
                                <h4>✅ NORMAL TEXT</h4>
                                <p><strong>Confidence:</strong> {prediction_result['confidence']}%</p>
                                <p><strong>Probability:</strong> {prediction_result['probabilities']['not_hate_speech']}%</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Show feature information
                        st.markdown("**Analysis features:**")
                        features = prediction_result['features']
                        st.write(f"• Character length: {features['char_length']}")
                        st.write(f"• Word count: {features['word_count']}")
                        st.write(f"• Abusive words: {features['abusive_word_count']}")
                        st.write(f"• Total features: {features['total_features']}")
                        
                    else:
                        st.markdown(f"""
                        <div class="result-box">
                            <h4>❌ MODEL UNAVAILABLE</h4>
                            <p>Machine learning model not loaded.</p>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("AI hate speech detection disabled")
            
            # Combined Summary
            st.markdown("---")
            st.markdown("### 📊 Combined Analysis Summary")
            
            # Get both results for summary
            if show_abusive_analysis and abusive_words:
                abusive_analysis = analyze_abusive_words(text_input, abusive_words)
            else:
                abusive_analysis = {'count': 0, 'words': []}
                
            if show_model_prediction:
                prediction_result = predict_hate_speech(text_input, (model, metadata), (vectorizer, vocab_size), scaler, abusive_words, text_cleaner)
            else:
                prediction_result = None
            
            # Create summary
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                st.metric(
                    "Abusive Words",
                    abusive_analysis['count'],
                    delta=f"{len(abusive_analysis['words'])} unique" if abusive_analysis['count'] > 0 else "Clean"
                )
            
            with summary_col2:
                if prediction_result:
                    ai_result = "Hate Speech" if prediction_result['prediction'] == 1 else "Normal Text"
                    confidence = prediction_result['confidence']
                    st.metric(
                        "AI Classification",
                        ai_result,
                        delta=f"{confidence}% confidence"
                    )
                else:
                    st.metric("AI Classification", "Unavailable")
            
            with summary_col3:
                # Overall risk assessment
                risk_level = "Low"
                if abusive_analysis['count'] >= 2:
                    risk_level = "High"
                elif abusive_analysis['count'] == 1:
                    risk_level = "Medium"
                elif prediction_result and prediction_result['prediction'] == 1:
                    risk_level = "Medium"
                
                risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                st.metric(
                    "Risk Level",
                    f"{risk_color[risk_level]} {risk_level}",
                    delta="Based on combined analysis"
                )
            
            # Highlight abusive words in original text
            if abusive_analysis['count'] > 0:
                st.markdown("#### 📝 Text with Highlighted Abusive Words:")
                highlighted_text = highlight_abusive_words(text_input, abusive_analysis['words'])
                st.markdown(f'<div style="background-color: #f9f9f9; padding: 1rem; border-radius: 5px; border-left: 4px solid #007bff;">{highlighted_text}</div>', unsafe_allow_html=True)
            
            # Model details in expander
            if prediction_result and metadata:
                with st.expander("🤖 Model Details", expanded=False):
                    model_col1, model_col2 = st.columns(2)
                    
                    with model_col1:
                        st.write(f"**Model Type:** {metadata.get('model_type', 'Unknown')}")
                        if 'performance_metrics' in metadata:
                            perf = metadata['performance_metrics']
                            st.write(f"**Accuracy:** {perf.get('accuracy', 0):.2%}")
                            st.write(f"**Precision:** {perf.get('precision', 0):.2%}")
                            st.write(f"**Recall:** {perf.get('recall', 0):.2%}")
                        else:
                            st.write("**Accuracy:** N/A")
                            st.write("**Precision:** N/A")
                            st.write("**Recall:** N/A")
                    
                    with model_col2:
                        if 'performance_metrics' in metadata:
                            perf = metadata['performance_metrics']
                            st.write(f"**F1-Score:** {perf.get('f1_score', 0):.2%}")
                        else:
                            st.write("**F1-Score:** N/A")
                        st.write(f"**Vocabulary Size:** {vocab_size:,} words")
                        st.write(f"**Training Date:** {metadata.get('training_date', 'Unknown')}")
                    
                    # Show probability chart
                    if prediction_result:
                        fig = go.Figure(data=[
                            go.Bar(
                                x=['Not Hate Speech', 'Hate Speech'],
                                y=[prediction_result['probabilities']['not_hate_speech'], 
                                   prediction_result['probabilities']['hate_speech']],
                                marker_color=['green', 'red']
                            )
                        ])
                        fig.update_layout(
                            title="Model Prediction Probabilities",
                            yaxis_title="Probability (%)",
                            height=300
                        )
                        st.plotly_chart(fig, use_container_width=True)
        
        with tab2:
            if show_text_stats:
                st.markdown("### 📊 Text Statistics")
                
                stats = get_text_statistics(text_input)
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Characters", stats['characters'])
                    st.metric("Words", stats['words'])
                
                with col2:
                    st.metric("Characters (no spaces)", stats['characters_no_spaces'])
                    st.metric("Sentences", stats['sentences'])
                
                with col3:
                    st.metric("Avg. Word Length", f"{stats['avg_word_length']:.1f}")
                    st.metric("Avg. Sentence Length", f"{stats['avg_sentence_length']:.1f}")
        
        with tab3:
            if show_word_freq:
                st.markdown("### 📈 Word Frequency Analysis")
                
                freq_chart = create_word_frequency_chart(text_input)
                if freq_chart:
                    st.plotly_chart(freq_chart, use_container_width=True)
                else:
                    st.info("ℹ️ Not enough words for frequency analysis.")
        
        with tab4:
            st.markdown("### 🔧 Technical Details")
            
            # Show cleaned text
            if text_cleaner:
                cleaned = text_cleaner.clean_text(text_input)
                st.markdown("#### 🧹 Cleaned Text:")
                st.code(cleaned)
            
            # Show comprehensive model info
            st.markdown("#### 🤖 Model Information:")
            
            # Main model status
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**📊 Model Status:**")
                if model:
                    st.success(f"✅ **{metadata.get('model_type', 'Unknown')}** - Loaded")
                    if metadata and 'performance_metrics' in metadata:
                        perf = metadata['performance_metrics']
                        st.info(f"🎯 **Accuracy:** {perf.get('accuracy', 0):.2%}")
                        st.info(f"🏆 **F1-Score:** {perf.get('f1_score', 0):.2%}")
                        st.info(f"📈 **Precision:** {perf.get('precision', 0):.2%}")
                else:
                    st.error("❌ Machine Learning Model - Not Available")
                
                if vectorizer:
                    st.success(f"✅ **TF-IDF Vectorizer** - Loaded")
                    st.info(f"📚 **Vocabulary:** {vocab_size:,} words")
                else:
                    st.error("❌ TF-IDF Vectorizer - Not Available")
            
            with col2:
                st.markdown("**📚 Dictionary Status:**")
                if abusive_words:
                    st.success(f"✅ **Abusive Words Dictionary** - Loaded")
                    st.info(f"🚫 **Total Words:** {len(abusive_words):,}")
                    st.info(f"🔤 **Languages:** Indonesian + English")
                else:
                    st.error("❌ Abusive Words Dictionary - Not Available")
                
                if TEXT_CLEANER_AVAILABLE:
                    st.success("✅ **Advanced Text Cleaner** - Full Features")
                else:
                    st.success("✅ **Built-in Text Cleaner** - Optimized")
            

            
            # System information
            with st.expander("💻 System Information"):
                import platform
                st.write(f"**Python Version:** {platform.python_version()}")
                st.write(f"**Streamlit Version:** {st.__version__}")
                st.write(f"**Platform:** {platform.system()} {platform.release()}")
                
                # Abusive words sample
                if abusive_words:
                    sample_words = list(abusive_words)[:10]
                    st.write(f"**Sample Abusive Words:** {', '.join(sample_words)}")
                    
            # Processing pipeline info
            with st.expander("🔄 Text Processing Pipeline"):
                st.markdown("""
                **1. Text Preprocessing:**
                - Convert to lowercase
                - Remove URLs, mentions, hashtags
                - Normalize Indonesian slang
                - Remove punctuation and numbers
                - Handle character repetitions
                
                **2. Abusive Words Detection:**
                - Dictionary-based matching
                - Word boundary detection
                - Count and position tracking
                
                **3. Machine Learning Analysis:**
                - TF-IDF feature extraction
                - Logistic Regression classification
                - Probability estimation
                - Confidence scoring
                """)
            

    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <h4>📖 About This Tool</h4>
        <p>This tool analyzes Indonesian text for hate speech and abusive language using:</p>
        <ul style="text-align: left; max-width: 600px; margin: 0 auto;">
            <li><strong>Abusive Words Dictionary:</strong> A comprehensive list of Indonesian offensive words</li>
            <li><strong>Machine Learning Model:</strong> Trained on Indonesian text data for context-aware detection</li>
            <li><strong>Text Analysis:</strong> Statistical analysis of text characteristics</li>
        </ul>
        <p style="margin-top: 1rem;"><em>Use responsibly for content moderation and research purposes.</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
