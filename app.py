"""
Railway Crack Detection System - Main Application
Streamlit-based UI for acoustic crack detection
"""

import streamlit as st
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Configure page
st.set_page_config(
    page_title="Railway Crack Detection",
    page_icon="🚂",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .uploadedFile {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🚂 Railway Crack Detection")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "🎓 Training", "📊 Analytics", "⚙️ Settings"]
)

# Route to pages
if page == "🏠 Home":
    from ui.pages import home
    home.render()
elif page == "🎓 Training":
    from ui.pages import training
    training.render()
elif page == "📊 Analytics":
    from ui.pages import analytics
    analytics.render()
elif page == "⚙️ Settings":
    from ui.pages import settings
    settings.render()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
    ### About
    Real-time railway crack detection using:
    - **MFCC** features
    - **Spectral** analysis
    - **Fractal** dimensions
    - **GenAI** augmentation
    - **Ensemble** ML models
""")
