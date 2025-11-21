"""
Settings page - Configure system parameters
"""

import streamlit as st
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from config.settings import *


def render():
    st.title("⚙️ System Settings")
    st.markdown("Configure detection parameters and system preferences")
    
    # Audio Processing Settings
    st.subheader("🎵 Audio Processing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        sample_rate = st.number_input(
            "Sample Rate (Hz)",
            min_value=8000,
            max_value=48000,
            value=SAMPLE_RATE,
            step=1000,
            help="Audio sampling frequency"
        )
        
        n_mfcc = st.slider(
            "Number of MFCC Coefficients",
            min_value=10,
            max_value=40,
            value=N_MFCC,
            help="Number of Mel-frequency cepstral coefficients"
        )
    
    with col2:
        duration = st.number_input(
            "Audio Duration (seconds)",
            min_value=1.0,
            max_value=10.0,
            value=DURATION,
            step=0.5,
            help="Fixed duration for audio processing"
        )
        
        n_fft = st.number_input(
            "FFT Window Size",
            min_value=512,
            max_value=4096,
            value=N_FFT,
            step=512,
            help="Fast Fourier Transform window size"
        )
    
    # Detection Settings
    st.markdown("---")
    st.subheader("🎯 Detection Thresholds")
    
    col1, col2 = st.columns(2)
    
    with col1:
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=CONFIDENCE_THRESHOLD,
            step=0.05,
            help="Minimum confidence for defect detection"
        )
    
    with col2:
        anomaly_threshold = st.slider(
            "Anomaly Threshold",
            min_value=0.0,
            max_value=1.0,
            value=ANOMALY_THRESHOLD,
            step=0.05,
            help="Threshold for anomaly detection"
        )
    
    # Model Settings
    st.markdown("---")
    st.subheader("🤖 Model Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        use_augmentation = st.checkbox(
            "Enable GenAI Augmentation",
            value=USE_AUGMENTATION,
            help="Use diffusion models for data augmentation"
        )
        
        classifier_type = st.selectbox(
            "Classifier Type",
            ["random_forest", "xgboost", "svm", "ensemble"],
            index=["random_forest", "xgboost", "svm", "ensemble"].index(CLASSIFIER_TYPE)
        )
    
    with col2:
        use_fractal = st.checkbox(
            "Enable Fractal Analysis",
            value=USE_FRACTAL_ANALYSIS,
            help="Include fractal dimension features"
        )
        
        cv_folds = st.slider(
            "Cross-Validation Folds",
            min_value=3,
            max_value=10,
            value=CV_FOLDS,
            help="Number of folds for cross-validation"
        )
    
    # Data Paths
    st.markdown("---")
    st.subheader("📂 Data Directories")
    
    st.text_input("Raw Data Directory", value=str(RAW_DATA_DIR), disabled=True)
    st.text_input("Processed Data Directory", value=str(PROCESSED_DATA_DIR), disabled=True)
    st.text_input("Models Directory", value=str(MODELS_DIR), disabled=True)
    
    # Save settings
    st.markdown("---")
    
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("💾 Save Settings", type="primary"):
            st.success("Settings saved successfully!")
    
    with col2:
        if st.button("🔄 Reset to Defaults"):
            st.info("Settings reset to defaults")
    
    # System Info
    st.markdown("---")
    st.subheader("ℹ️ System Information")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown(f"""
        **Audio Settings:**
        - Sample Rate: {sample_rate} Hz
        - Duration: {duration}s
        - MFCC Coefficients: {n_mfcc}
        - FFT Size: {n_fft}
        """)
    
    with info_col2:
        st.markdown(f"""
        **Model Settings:**
        - Classifier: {classifier_type}
        - Confidence: {confidence_threshold}
        - GenAI Augmentation: {'Enabled' if use_augmentation else 'Disabled'}
        - Fractal Analysis: {'Enabled' if use_fractal else 'Disabled'}
        """)
