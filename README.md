# Indonesian Hate Speech Detection

A machine learning project that detects hate speech and abusive content in Indonesian social media text. The system combines AI-powered classification with dictionary-based abusive word detection to provide comprehensive text analysis.

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/release/python-380/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/)

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Applications](#applications)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Dataset Information](#dataset-information)
- [Support](#support)
- [License](#license)

## Overview

This project helps identify potentially harmful content in Indonesian text using two complementary approaches:

1. **AI Classification**: A machine learning model trained on Indonesian social media data that understands context and meaning
2. **Abusive Words Detection**: A dictionary of 260 Indonesian abusive words with real-time detection and counting
3. **Smart Analysis**: The system considers context - for example, "anjing saya lucu" (my dog is cute) vs "kamu anjing" (you dog) are handled differently

### Key Intelligence
The AI model is context-aware and doesn't just look for bad words. It understands:
- Single abusive words in normal context are usually fine
- Multiple abusive words together often indicate hate speech
- The overall meaning and intent of the text matters

## Features

### Dual Detection System
- **AI Hate Speech Detection**: Provides confidence scores and probability percentages
- **Abusive Words Detection**: Counts and highlights problematic words in text
- **Risk Assessment**: Combines both analyses to give Low/Medium/High risk levels

### Text Analysis
- Character and word count
- Text cleaning preview
- Abusive words highlighting
- Statistical analysis

### User-Friendly Interface
- Modern desktop application (GUI)
- Web-based application (browser)
- Real-time analysis as you type
- Searchable dictionary of abusive words
- Easy-to-understand results

### Indonesian Language Support
- Specialized for Indonesian text and slang
- Understands cultural context
- Handles informal social media language

## Applications

### Desktop Application
**File**: `notebooks/gui.py`

A professional desktop interface with:
- Clean, easy-to-use design
- Real-time text analysis
- Highlighted abusive words (red background)
- Detailed results with confidence scores
- Built-in dictionary browser

### Web Application
**File**: `app/streamlit_demo.py`

A modern web interface featuring:
- Works in any web browser
- Side-by-side comparison of detection methods
- Interactive dashboard
- Risk assessment display
- Text statistics and charts

## Installation

### Requirements
- Python 3.8 or newer
- Internet connection for initial setup

### Quick Setup
```bash
# 1. Download the project
git clone https://github.com/yourusername/indonesian-hate-speech-detection.git
cd indonesian-hate-speech-detection

# 2. Install required packages
pip install -r requirements.txt

# 3. Download language data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"

# 4. Test installation
python start_app.py
```

## Usage

### Starting the Applications

**Option 1: Desktop Application**
```bash
cd notebooks
python gui.py
```
Best for: Detailed analysis, desktop use, offline work

**Option 2: Web Application** 
```bash
streamlit run app/streamlit_demo.py
```
Best for: Browser access, sharing, modern interface
Opens at: http://localhost:8501

**Option 3: Choose Interactively**
```bash
python start_app.py
```
Shows a menu to pick which application to run

### Example Analysis

**Input**: "kamu anjing bangsat tolol"
**Results**:
- AI Detection: 72.6% Hate Speech
- Abusive Words: 3 words found (anjing, bangsat, tolol)
- Risk Level: HIGH
- Recommendation: Content likely contains hate speech

**Input**: "anjing saya sangat lucu dan menggemaskan"
**Results**:
- AI Detection: 15.2% Normal (context: talking about a pet)
- Abusive Words: 1 word found (anjing)
- Risk Level: LOW
- Recommendation: Safe content, referring to pet

### Research and Development
Explore the machine learning pipeline using Jupyter notebooks:
```bash
jupyter notebook
```
Then open the notebooks in order:
1. `01_data_retrieval.ipynb` - Data loading
2. `02_data_preparation.ipynb` - Data preprocessing
3. `03_data_exploring.ipynb` - Data analysis
4. `04_data_modelling.ipynb` - Model training
5. `05_presentation_and_automation.ipynb` - Results

## Project Structure

```
indonesian-hate-speech-detection/
├── Applications
│   ├── app/streamlit_demo.py         # Web application
│   └── notebooks/gui.py              # Desktop application
├── Models and AI
│   ├── models/                       # Trained AI models
│   └── utils/                        # Text processing tools
├── Data and Analysis
│   ├── data/                         # Training datasets
│   └── notebooks/                    # Research notebooks
├── Dictionary and Resources
│   ├── IndonesianAbusiveWords/       # Abusive words database
│   └── outputs/                      # Charts and visualizations
└── Project Files
    ├── start_app.py                  # Application launcher
    ├── requirements.txt              # Required packages
    └── README.md                     # This file
```

## How It Works

### AI Model
- **Type**: Logistic Regression trained on Indonesian Twitter data
- **Features**: Analyzes 5003 different text characteristics
- **Training**: ~13,000 Indonesian social media posts
- **Accuracy**: ~85% on test data

### Abusive Words Dictionary
- **Size**: 260 carefully selected Indonesian abusive words
- **Source**: Community-curated list from `IndonesianAbusiveWords/abusive.csv`
- **Method**: Direct word matching with frequency counting

### Text Processing
1. **Cleaning**: Removes URLs, mentions, extra spaces
2. **Normalization**: Converts to lowercase, handles Indonesian text
3. **Analysis**: Extracts features for AI model
4. **Detection**: Scans for abusive words simultaneously

### Smart Decision Making
The system combines both methods intelligently:
- If no abusive words found → LOW risk
- Single abusive word + low AI confidence → MEDIUM risk
- Single abusive word + high AI confidence → HIGH risk
- Multiple abusive words → HIGH risk (regardless of AI)

## Dataset Information

### Training Data
- **Size**: ~13,169 Indonesian tweets
- **Language**: Indonesian (Bahasa Indonesia) including slang
- **Labels**: Hate Speech vs Normal content
- **Balance**: Processed to handle class imbalance

### Sample Examples
```
Normal: "selamat pagi semua, semoga hari ini menyenangkan"
        (good morning everyone, hope today is pleasant)

Hate Speech: "dasar bangsa tolol, tidak berguna semua"
             (stupid nation, all useless)

Context-Aware: "anjing tetangga menggonggong terus malam ini"
               (neighbor's dog keeps barking tonight)
```

### Performance
- **Accuracy**: 85.2% on diverse test data
- **Speed**: Real-time analysis (less than 0.1 seconds)
- **Cultural Sensitivity**: Trained specifically on Indonesian social media
- **False Positives**: Less than 15% due to context-aware design

## Support

### Troubleshooting
- **Model not loading**: Check that all files in `models/` folder are present
- **GUI won't start**: Install tkinter with `pip install tk`
- **Web app issues**: Update streamlit with `pip install --upgrade streamlit`
- **Indonesian text problems**: Install Sastrawi with `pip install Sastrawi`

### Additional Help
- **Detailed Documentation**: Check `app/README.md` for complete guides
- **Technical Details**: See `models/model_metadata.json` for model specifications
- **Issues**: Report problems via GitHub Issues

### Cleanup
Keep your project organized:
```bash
python cleanup_project.py
```
This removes development files while keeping essential components.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Indonesian NLP community for linguistic resources
- Sastrawi library developers for Indonesian text processing
- Streamlit team for the web framework
- Open source community for machine learning tools

---

**Important**: This project is for educational, research, and content moderation purposes. The AI provides suggestions, not final judgments. Human review is recommended for important decisions.

**Research Use**: Suitable for academic research in natural language processing and hate speech detection in Indonesian language.
