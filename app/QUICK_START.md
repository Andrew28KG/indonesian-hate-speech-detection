# Quick Start Guide - Indonesian Hate Speech Detection App

## 🚀 Running the App

### Option 1: Using the Start Script (Recommended)
```bash
python start_app.py
```

### Option 2: Using Streamlit Directly
```bash
streamlit run app/streamlit_demo.py --server.port 8501
```

### Option 3: Using the Demo Runner
```bash
python run_demo.py
```

## 📱 Accessing the App

Once started, the app will be available at:
- **Local URL**: http://localhost:8501
- The app should automatically open in your default browser

## ✨ Features

### 🎯 Text Analysis
- **Abusive Words Detection**: Detects 260+ Indonesian offensive words
- **Hate Speech Prediction**: ML model classification with confidence scores
- **Text Statistics**: Character, word, and sentence analysis
- **Word Frequency**: Interactive charts of most common words

### 🔧 How to Use
1. Enter Indonesian text in the text area
2. Or click example buttons to test with sample texts
3. Click "Analyze Text" to see results
4. Explore different tabs for detailed analysis

### 📊 Example Results

**Normal Text**: "Selamat pagi semua! Hari ini cuaca cerah."
- ✅ No abusive content detected
- Shows text statistics and word frequency

**Offensive Text**: "Dasar anjing lu! Bangsat emang!"
- ⚠️ Abusive content detected with highlighted words
- ML model prediction with confidence scores

## 🛠️ Troubleshooting

### Common Issues

**Port already in use**
```bash
# Stop any running Streamlit instances
pkill -f streamlit
# Or use a different port
streamlit run app/streamlit_demo.py --server.port 8502
```

**Import warnings**
- The app has fallback text cleaning if the main module isn't available
- All core features still work with the built-in cleaner

**Model not found**
- App will use abusive words detection only
- Core functionality remains available

## 🎮 Testing the App

Try these example texts:

1. **Normal**: `"Selamat pagi! Semoga hari kalian menyenangkan."`
2. **Mild Negative**: `"Kamu bodoh banget sih."`
3. **Strong Offensive**: `"Anjing lu! Kontol bangsat!"`

## 🔍 Features Breakdown

- **Main Results Tab**: Abusive words detection + ML predictions
- **Text Statistics Tab**: Character/word counts, averages
- **Word Analysis Tab**: Interactive frequency charts
- **Technical Details Tab**: Cleaned text + model status

## 📝 Notes

- Optimized for Indonesian text
- Uses comprehensive abusive words dictionary
- Real-time analysis with visual feedback
- Professional UI with color-coded results 