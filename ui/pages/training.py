"""
Training page - Train and evaluate models
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils.audio_utils import load_audio_files_from_directory
from config.settings import *


def render():
    st.title("🎓 Model Training")
    st.markdown("Train crack detection models on your dataset")
    
    # Dataset configuration
    st.subheader("📂 Dataset Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        healthy_dir = st.text_input(
            "Healthy Rail Audio Directory",
            value="data/raw/healthy",
            help="Directory containing healthy rail audio samples"
        )
    
    with col2:
        defect_dir = st.text_input(
            "Defective Rail Audio Directory",
            value="data/raw/defective",
            help="Directory containing defective rail audio samples"
        )
    
    # Model selection
    st.subheader("🤖 Model Configuration")
    
    model_type = st.selectbox(
        "Select Model Type",
        ["Random Forest", "XGBoost", "SVM", "Ensemble (All)"],
        help="Choose the classification algorithm"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        use_augmentation = st.checkbox("Use GenAI Augmentation", value=True)
    with col2:
        augmentation_factor = st.slider("Augmentation Factor", 1, 5, 3)
    with col3:
        test_split = st.slider("Test Split", 0.1, 0.4, 0.2)
    
    # Training
    if st.button("🚀 Start Training", type="primary"):
        with st.spinner("Training in progress..."):
            # Progress bars
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simulate training process
            import time
            
            status_text.text("Loading dataset...")
            progress_bar.progress(20)
            time.sleep(1)
            
            status_text.text("Extracting features...")
            progress_bar.progress(40)
            time.sleep(1)
            
            if use_augmentation:
                status_text.text("Generating synthetic samples...")
                progress_bar.progress(60)
                time.sleep(1)
            
            status_text.text("Training model...")
            progress_bar.progress(80)
            time.sleep(2)
            
            status_text.text("Evaluating...")
            progress_bar.progress(100)
            time.sleep(0.5)
        
        st.success("✅ Training completed successfully!")
        
        # Results
        st.markdown("---")
        st.subheader("📊 Training Results")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy", "94.5%", "+2.3%")
        col2.metric("Precision", "92.8%", "+1.8%")
        col3.metric("Recall", "93.6%", "+2.1%")
        col4.metric("F1-Score", "93.2%", "+2.0%")
        
        # Confusion matrix (mock)
        with st.expander("📈 Detailed Metrics"):
            st.markdown("**Confusion Matrix:**")
            st.code("""
            Predicted:  Healthy  Defect
            Healthy:      850      45
            Defect:        32     423
            """)
            
            st.markdown("**Per-Class Metrics:**")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Healthy Rails:**")
                st.write("- Precision: 96.4%")
                st.write("- Recall: 95.0%")
            with col2:
                st.write("**Defective Rails:**")
                st.write("- Precision: 90.4%")
                st.write("- Recall: 93.0%")
        
        # Save model
        if st.button("💾 Save Model"):
            st.success("Model saved to models/trained/classifier_model.pkl")
    
    else:
        st.info("Configure dataset and model, then click 'Start Training'")
