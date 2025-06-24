# Indonesian Hate Speech Detection - Streamlit App

This Streamlit application provides an interactive interface for analyzing Indonesian text for hate speech and abusive language.

## Features

### 🎯 Main Analysis
- **Abusive Words Detection**: Identifies offensive words using a comprehensive Indonesian abusive words dictionary (260+ words)
- **Machine Learning Prediction**: Uses trained models to detect hate speech with confidence scores
- **Real-time Analysis**: Instant feedback on text input

### 📊 Text Analysis
- **Text Statistics**: Character count, word count, sentence count, average lengths
- **Word Frequency Analysis**: Interactive charts showing most common words
- **Text Highlighting**: Visual highlighting of detected abusive words in the original text

### 🔧 Technical Features
- **Text Preprocessing**: Shows cleaned text after processing
- **Model Information**: Displays status of loaded models and dictionaries
- **Multiple Analysis Options**: Toggle different analysis features on/off

## Getting Started

### 1. Run the Application

```bash
# Option 1: Using the demo runner
python run_demo.py

# Option 2: Direct Streamlit command
streamlit run app/streamlit_demo.py
```

### 2. Using the Interface

1. **Enter Text**: Type or paste Indonesian text in the text area
2. **Use Examples**: Click on example buttons to test with sample texts
3. **Configure Analysis**: Use sidebar options to enable/disable features
4. **Analyze**: Click "Analyze Text" to see results

### 3. Understanding Results

#### Abusive Words Analysis
- **Count**: Number of abusive word instances found
- **Unique Words**: List of different abusive words detected
- **Percentage**: Proportion of abusive words in the text
- **Highlighted Text**: Original text with abusive words highlighted

#### Machine Learning Prediction
- **Binary Classification**: Hate speech or not (with confidence)
- **Probability Distribution**: Visual chart showing model confidence
- **Cleaned Text**: Preprocessed text used by the model

## Example Texts

The app includes three types of example texts:

1. **Normal Text**: Clean, non-offensive Indonesian text
2. **Mild Negative**: Slightly negative but not hate speech
3. **Strong Negative**: Contains clear abusive language and hate speech

## Technical Details

### Models Used
- **Hate Speech Model**: `best_hate_speech_model.pkl` - Binary classifier for hate speech detection
- **TF-IDF Vectorizer**: `tfidf_vectorizer.pkl` - Text feature extraction
- **Abusive Words Dictionary**: `IndonesianAbusiveWords/abusive.csv` - 260+ Indonesian offensive words

### Text Preprocessing
- URL removal
- Mention (@user) removal  
- Hashtag (#tag) removal
- Slang normalization
- Text cleaning and standardization

### Analysis Capabilities
- Real-time abusive word detection
- Context-aware hate speech classification
- Statistical text analysis
- Word frequency visualization
- Performance metrics and confidence scores

## Dependencies

Key libraries used:
- `streamlit` - Web application framework
- `plotly` - Interactive visualizations
- `scikit-learn` - Machine learning models
- `pandas` - Data manipulation
- `numpy` - Numerical computations

## File Structure

```
app/
├── streamlit_demo.py    # Main Streamlit application
├── utils.py            # Utility functions
├── config.py           # Configuration settings
├── models/             # Model metadata
└── README.md          # This file
```

## Usage Examples

### Basic Text Analysis
```python
# Enter this text in the app:
"Selamat pagi semua! Hari ini cuaca cerah."
# Result: No abusive content detected
```

### Abusive Content Detection
```python
# Enter this text in the app:
"Dasar anjing lu! Bangsat emang!"
# Result: Multiple abusive words detected with highlighting
```

## Troubleshooting

### Common Issues

1. **Models not loading**
   - Ensure model files exist in the `models/` directory
   - Check file permissions and paths

2. **Abusive words dictionary not found**
   - Verify `IndonesianAbusiveWords/abusive.csv` exists
   - Check CSV format (should have 'ABUSIVE' column)

3. **Import errors**
   - Install all required dependencies: `pip install -r requirements.txt`
   - Check Python path configuration

### Performance Tips

- For best results, use Indonesian text
- Keep text length reasonable (< 10,000 characters)
- Use the sidebar options to customize analysis features

## Contributing

When adding new features:
1. Update the abusive words dictionary as needed
2. Maintain the modular structure
3. Add appropriate error handling
4. Update this README

## Support

For issues or questions:
1. Check the troubleshooting section
2. Verify all dependencies are installed
3. Ensure model files are properly placed
4. Test with the provided example texts

---

**Note**: This tool is designed for research and content moderation purposes. Use responsibly and consider the context of the analyzed text. 