#!/bin/bash

# Railway Crack Detection Project Setup Script
# Run this script to create the complete project structure

echo "Creating Railway Crack Detection Project Structure..."

# Create main project directory
mkdir -p railway-crack-detection
cd railway-crack-detection

# Create config directory
mkdir -p config
touch config/__init__.py
touch config/settings.py

# Create data directories
mkdir -p data/raw/healthy
mkdir -p data/raw/defective
mkdir -p data/processed
mkdir -p data/augmented

# Create models directories
mkdir -p models/trained
mkdir -p models/checkpoints

# Create src directories and files
mkdir -p src/preprocessing
touch src/__init__.py
touch src/preprocessing/__init__.py
touch src/preprocessing/audio_loader.py
touch src/preprocessing/noise_filter.py

mkdir -p src/feature_extraction
touch src/feature_extraction/__init__.py
touch src/feature_extraction/mfcc_extractor.py
touch src/feature_extraction/spectral_features.py
touch src/feature_extraction/fractal_analysis.py

mkdir -p src/augmentation
touch src/augmentation/__init__.py
touch src/augmentation/diffusion_generator.py

mkdir -p src/models
touch src/models/__init__.py
touch src/models/classifier.py
touch src/models/ensemble.py

mkdir -p src/evaluation
touch src/evaluation/__init__.py
touch src/evaluation/metrics.py

mkdir -p src/utils
touch src/utils/__init__.py
touch src/utils/audio_utils.py
touch src/utils/visualization.py

# Create UI directories and files
mkdir -p ui/components
mkdir -p ui/pages
touch ui/__init__.py
touch ui/components/__init__.py
touch ui/components/audio_player.py
touch ui/components/feature_display.py
touch ui/components/result_panel.py
touch ui/pages/home.py
touch ui/pages/training.py
touch ui/pages/analytics.py
touch ui/pages/settings.py

# Create notebooks directory
mkdir -p notebooks
touch notebooks/01_data_exploration.ipynb
touch notebooks/02_feature_analysis.ipynb
touch notebooks/03_model_training.ipynb

# Create tests directory
mkdir -p tests
touch tests/__init__.py
touch tests/test_preprocessing.py
touch tests/test_feature_extraction.py
touch tests/test_classifier.py

# Create logs directory
mkdir -p logs
touch logs/app.log

# Create assets directories
mkdir -p assets/images
mkdir -p assets/styles
touch assets/styles/custom.css

# Create main files
touch app.py
touch requirements.txt
touch README.md
touch .gitignore

echo "Project structure created successfully!"
echo ""
echo "Next steps:"
echo "1. Navigate to the project: cd railway-crack-detection"
echo "2. Create a virtual environment: python3 -m venv venv"
echo "3. Activate virtual environment: source venv/bin/activate"
echo "4. Install dependencies: pip install -r requirements.txt"
echo ""
echo "Project structure ready for development!"
