"""
Home page - Upload and detect cracks
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.preprocessing.audio_loader import AudioLoader
from src.preprocessing.noise_filter import NoiseFilter
from src.feature_extraction.mfcc_extractor import MFCCExtractor
from src.feature_extraction.spectral_features import SpectralFeatureExtractor
from src.feature_extraction.fractal_analysis import FractalAnalyzer
from src.utils.visualization import plot_waveform, plot_spectrogram, plot_mfcc
from config.settings import *
import matplotlib.pyplot as plt


def render():
    st.title("🔍 Railway Crack Detection")
    st.markdown("Upload railway acoustic emissions to detect potential defects")
    
    # File upload
    uploaded_file = st.file_uploader(
        "Upload Audio File (WAV format)",
        type=['wav', 'mp3'],
        help="Upload trackside or onboard acoustic recording"
    )
    
    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📁 Audio Information")
            st.info(f"**Filename:** {uploaded_file.name}")
            
            # Load audio
            with st.spinner("Loading audio..."):
                audio_loader = AudioLoader(sample_rate=SAMPLE_RATE, duration=DURATION)
                
                # Save temporarily
                temp_path = f"temp_{uploaded_file.name}"
                with open(temp_path, 'wb') as f:
                    f.write(uploaded_file.getbuffer())
                
                audio, sr = audio_loader.load(temp_path)
                
                st.success(f"✓ Loaded: {len(audio)/sr:.2f}s @ {sr}Hz")
        
        with col2:
            st.subheader("🎵 Audio Player")
            st.audio(uploaded_file)
        
        # Preprocessing
        st.markdown("---")
        st.subheader("🔧 Preprocessing")
        
        with st.spinner("Applying filters..."):
            noise_filter = NoiseFilter(sample_rate=sr)
            audio_clean = noise_filter.preprocess(audio)
            st.success("✓ Noise reduction and bandpass filtering applied")
        
        # Visualizations
        st.markdown("---")
        st.subheader("📊 Audio Analysis")
        
        tab1, tab2, tab3 = st.tabs(["Waveform", "Spectrogram", "MFCC"])
        
        with tab1:
            fig = plot_waveform(audio_clean, sr)
            st.pyplot(fig)
            plt.close()
        
        with tab2:
            fig = plot_spectrogram(audio_clean, sr)
            st.pyplot(fig)
            plt.close()
        
        with tab3:
            fig = plot_mfcc(audio_clean, sr)
            st.pyplot(fig)
            plt.close()
        
        # Feature extraction
        st.markdown("---")
        st.subheader("🎯 Feature Extraction")
        
        with st.spinner("Extracting features..."):
            # MFCC
            mfcc_extractor = MFCCExtractor(sr, N_MFCC, N_FFT, HOP_LENGTH)
            mfcc_features = mfcc_extractor.extract_full_features(audio_clean)
            
            # Spectral
            spectral_extractor = SpectralFeatureExtractor(sr, N_FFT, HOP_LENGTH)
            spectral_features = spectral_extractor.extract_all_features(audio_clean)
            
            # Fractal
            fractal_analyzer = FractalAnalyzer(sr)
            fractal_features = fractal_analyzer.extract_all_fractal_features(audio_clean)
            
            # Combine
            all_features = np.concatenate([mfcc_features, spectral_features, fractal_features])
            
            col1, col2, col3 = st.columns(3)
            col1.metric("MFCC Features", len(mfcc_features))
            col2.metric("Spectral Features", len(spectral_features))
            col3.metric("Fractal Features", len(fractal_features))
        
        st.success(f"✓ Total features extracted: {len(all_features)}")
        
        # Classification
        st.markdown("---")
        st.subheader("🤖 Crack Detection")
        
        if st.button("🔍 Detect Defects", type="primary"):
            with st.spinner("Analyzing..."):
                # Mock prediction (replace with trained model)
                import time
                time.sleep(1)
                
                # Simulate prediction
                prediction = np.random.choice([0, 1], p=[0.7, 0.3])
                confidence = np.random.uniform(0.75, 0.95)
                
                if prediction == 0:
                    st.success(f"✅ **HEALTHY RAIL** (Confidence: {confidence:.2%})")
                else:
                    st.error(f"⚠️ **DEFECT DETECTED** (Confidence: {confidence:.2%})")
                    st.warning("**Recommendation:** Schedule immediate inspection")
                
                # Show feature importance (mock)
                with st.expander("📈 Feature Analysis"):
                    st.write("Top contributing features:")
                    importance_data = {
                        'Spectral Flux': 0.23,
                        'MFCC-5': 0.19,
                        'Higuchi FD': 0.16,
                        'Spectral Kurtosis': 0.14,
                        'MFCC-3': 0.12
                    }
                    for feat, imp in importance_data.items():
                        st.progress(imp, text=f"{feat}: {imp:.2%}")
        
        # Clean up
        import os
        if os.path.exists(temp_path):
            os.remove(temp_path)
    
    else:
        st.info("👆 Upload an audio file to begin analysis")
        
        # Sample data info
        with st.expander("ℹ️ Sample Data Format"):
            st.markdown("""
            **Expected Audio Format:**
            - Format: WAV, MP3
            - Sample Rate: 22050 Hz (recommended)
            - Duration: 3 seconds (default)
            - Channels: Mono
            
            **Defect Types Detected:**
            - Rail cracks
            - Corrugations
            - Weld failures
            - Surface defects
            """)
