"""
Configuration settings for Railway Crack Detection System
"""

import os
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
AUGMENTED_DATA_DIR = DATA_DIR / "augmented"
MODELS_DIR = BASE_DIR / "models" / "trained"
LOGS_DIR = BASE_DIR / "logs"

# Audio Processing Parameters
SAMPLE_RATE = 22050  # Hz
DURATION = 3.0  # seconds
N_MFCC = 20  # Number of MFCC coefficients
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

# Feature Extraction
SPECTRAL_FEATURES = ['spectral_centroid', 'spectral_flux', 'spectral_rolloff', 'spectral_bandwidth']
USE_FRACTAL_ANALYSIS = True

# Classification Parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5

# Model Configuration
CLASSIFIER_TYPE = 'random_forest'  # Options: 'random_forest', 'svm', 'xgboost', 'ensemble'
RF_N_ESTIMATORS = 100
RF_MAX_DEPTH = 20
SVM_KERNEL = 'rbf'
XGB_N_ESTIMATORS = 100
XGB_LEARNING_RATE = 0.1

# GenAI Augmentation
USE_AUGMENTATION = True
AUGMENTATION_FACTOR = 3  # Generate 3x synthetic samples for minority class
DIFFUSION_STEPS = 50

# Detection Thresholds
CONFIDENCE_THRESHOLD = 0.7  # Minimum confidence for defect detection
ANOMALY_THRESHOLD = 0.85  # Threshold for anomaly detection

# Class Labels
CLASS_LABELS = {
    0: 'Healthy',
    1: 'Crack',
    2: 'Corrugation',
    3: 'Weld Failure'
}

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
