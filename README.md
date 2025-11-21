# 🚂 Railway Crack Detection System

Real-time railway crack detection using acoustic emissions, advanced signal processing, and GenAI-augmented machine learning.

![Railway Detection](https://img.shields.io/badge/Railway-Detection-blue)
![Python](https://img.shields.io/badge/Python-3.10+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28.0-red)
![License](https://img.shields.io/badge/License-Educational-yellow)

## 🎯 Features

- **🎵 Acoustic Analysis**: MFCC, spectral features, fractal dimensions
- **🤖 GenAI Augmentation**: Synthetic defect sample generation
- **📊 Ensemble Learning**: Random Forest + XGBoost + SVM
- **⚡ Real-time Detection**: Upload and analyze in seconds
- **💻 Interactive UI**: Clean Streamlit interface
- **📈 Analytics Dashboard**: Performance metrics and visualizations

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- Virtual environment support
- 2GB+ free disk space

### Installation

```bash
# 1. Clone the repository
cd /media/brittytino/Data/Work\'s/Github/railway_crack_detection

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Generate sample data (optional but recommended)
python3 generate_sample_data.py
```

### 🎯 Running the Application

**Option 1: Use the convenience script (Recommended)**
```bash
./run.sh
```

**Option 2: Run commands manually**

```bash
# Activate virtual environment
source venv/bin/activate

# Launch Streamlit app
streamlit run app.py

# OR Launch Jupyter notebooks
jupyter notebook notebooks/

# OR Generate sample data
python3 generate_sample_data.py
```

The application will open in your browser at `http://localhost:8501`

## 📁 Project Structure

```
railway_crack_detection/
├── app.py                      # Main Streamlit application
├── generate_sample_data.py     # Sample audio data generator
├── run.sh                      # Convenience startup script
├── requirements.txt            # Python dependencies
│
├── config/                     # Configuration
│   ├── __init__.py
│   └── settings.py            # System settings
│
├── src/                       # Core modules
│   ├── preprocessing/         # Audio preprocessing
│   │   ├── audio_loader.py
│   │   └── noise_filter.py
│   ├── feature_extraction/    # Feature extraction
│   │   ├── mfcc_extractor.py
│   │   ├── spectral_features.py
│   │   └── fractal_analysis.py
│   ├── models/                # ML models
│   │   ├── classifier.py
│   │   └── ensemble.py
│   ├── augmentation/          # GenAI augmentation
│   │   └── diffusion_generator.py
│   ├── evaluation/            # Model evaluation
│   │   └── metrics.py
│   └── utils/                 # Utilities
│       ├── audio_utils.py
│       └── visualization.py
│
├── ui/                        # User interface
│   ├── pages/
│   │   ├── home.py           # Main detection page
│   │   ├── training.py       # Model training page
│   │   ├── analytics.py      # Analytics dashboard
│   │   └── settings.py       # Configuration page
│   └── components/
│       ├── audio_player.py
│       ├── feature_display.py
│       └── result_panel.py
│
├── data/                      # Data storage
│   ├── raw/                  # Raw audio files
│   │   ├── healthy/         # Healthy rail samples
│   │   └── defective/       # Defective rail samples
│   ├── processed/           # Processed features
│   └── augmented/           # GenAI augmented data
│
├── models/                   # Trained models
│   ├── trained/             # Saved models
│   └── checkpoints/         # Training checkpoints
│
├── notebooks/               # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_training.ipynb
│
└── tests/                   # Unit tests
    ├── test_preprocessing.py
    ├── test_feature_extraction.py
    └── test_classifier.py
```

## 📊 Dataset

### Using Sample Data
The system comes with a sample data generator:

```bash
python3 generate_sample_data.py
```

This creates:
- 10 healthy rail audio samples
- 15 defective samples (cracks, corrugations, weld failures)

### Using Your Own Data
Place audio files (WAV format preferred) in:
- `data/raw/healthy/` - Healthy rail recordings
- `data/raw/defective/` - Defective rail recordings

**Audio Requirements:**
- Format: WAV, MP3, FLAC, OGG
- Sample Rate: 22050 Hz (will be resampled automatically)
- Duration: 3-10 seconds recommended
- Mono or Stereo (converted to mono)

## 🎓 How It Works

### 1. **Audio Acquisition**
   - Trackside or onboard microphone recordings
   - Pass-by acoustic emissions from wheel-rail contact

### 2. **Preprocessing**
   - Noise filtering (bandpass 300-8000 Hz)
   - Normalization
   - Silence removal

### 3. **Feature Extraction**
   - **MFCCs**: 20 Mel-frequency cepstral coefficients
   - **Spectral Features**: Centroid, bandwidth, rolloff, flux
   - **Fractal Dimension**: Higuchi algorithm for complexity

### 4. **Classification**
   - Random Forest
   - XGBoost
   - Support Vector Machine
   - Ensemble voting

### 5. **GenAI Augmentation** (Optional)
   - Diffusion-based synthetic defect generation
   - Addresses class imbalance

## 🔧 Configuration

Edit `config/settings.py` or use the Settings page in the UI:

```python
# Audio Processing
SAMPLE_RATE = 22050        # Hz
DURATION = 3.0             # seconds
N_MFCC = 20               # MFCC coefficients

# Detection Thresholds
CONFIDENCE_THRESHOLD = 0.7
ANOMALY_THRESHOLD = 0.85

# Model Configuration
CLASSIFIER_TYPE = 'ensemble'
USE_AUGMENTATION = True
USE_FRACTAL_ANALYSIS = True
```

## 📈 Performance

Expected metrics on sample data:
- **Accuracy**: 90-95%
- **Precision**: 88-93%
- **Recall**: 89-94%
- **F1-Score**: 88-93%

*Note: Actual performance depends on dataset quality and size*

## 🛠️ Development

### Running Tests
```bash
source venv/bin/activate
pytest tests/
```

### Training Custom Models
1. Prepare dataset in `data/raw/`
2. Navigate to Training page in UI
3. Configure parameters
4. Click "Train Model"

### Jupyter Notebooks
```bash
jupyter notebook notebooks/
```

Explore:
- Data exploration and visualization
- Feature analysis
- Model training and evaluation

## 🐛 Troubleshooting

### Common Issues

**1. Import Errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

**2. No Audio Files Found**
```bash
# Generate sample data
python3 generate_sample_data.py
```

**3. Streamlit Won't Start**
```bash
# Check Python version
python3 --version  # Should be 3.10+

# Verify streamlit installation
streamlit --version
```

**4. Permission Denied on run.sh**
```bash
chmod +x run.sh
```

## 📝 License

Educational/Research Use

This project is intended for educational and research purposes. For commercial use, please contact the author.

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📧 Contact

For questions or support, please open an issue on GitHub.

## 🙏 Acknowledgments

- Railway acoustic emission dataset sources
- Librosa for audio processing
- Streamlit for UI framework
- Scikit-learn, XGBoost for ML models

---

## 🔬 Technical Background

Railway safety critically depends on timely detection of rail defects such as cracks, corrugations, and weld failures. Traditional inspection methods (manual patrols, ultrasonic testing cars, and visual inspections) are periodic and costly, potentially missing fast-emerging defects between inspection cycles. 

This project proposes a low-cost, scalable, and near-real-time crack detection system using pass-by acoustic emissions captured via trackside or onboard microphones, advanced signal processing, and Generative AI (GenAI) models augmented with classical machine learning. 

The system extracts robust acoustic features (MFCCs, spectral flux, kurtosis, fractal measures), synthesizes rare defect signatures via diffusion-based audio augmentation, and classifies rail conditions using ensemble methods for high accuracy and reliability.

---

**Built with ❤️ for Railway Safety**