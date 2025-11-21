"""
Spectral feature extraction module
Extracts spectral characteristics from audio signals
"""

import numpy as np
import librosa
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class SpectralFeatureExtractor:
    """Extract spectral features from audio signals"""
    
    def __init__(self, sample_rate: int = 22050, n_fft: int = 2048, 
                 hop_length: int = 512):
        """
        Initialize SpectralFeatureExtractor
        
        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
    
    def extract(self, audio: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract all spectral features
        
        Args:
            audio: Audio waveform
            
        Returns:
            Dictionary of spectral features
        """
        features = {}
        
        try:
            # Spectral centroid
            centroid = librosa.feature.spectral_centroid(
                y=audio, 
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            features['spectral_centroid'] = centroid
            features['spectral_centroid_mean'] = np.mean(centroid)
            features['spectral_centroid_std'] = np.std(centroid)
            
            # Spectral bandwidth
            bandwidth = librosa.feature.spectral_bandwidth(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            features['spectral_bandwidth'] = bandwidth
            features['spectral_bandwidth_mean'] = np.mean(bandwidth)
            features['spectral_bandwidth_std'] = np.std(bandwidth)
            
            # Spectral rolloff
            rolloff = librosa.feature.spectral_rolloff(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )[0]
            features['spectral_rolloff'] = rolloff
            features['spectral_rolloff_mean'] = np.mean(rolloff)
            features['spectral_rolloff_std'] = np.std(rolloff)
            
            # Spectral contrast
            contrast = librosa.feature.spectral_contrast(
                y=audio,
                sr=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            features['spectral_contrast'] = contrast
            features['spectral_contrast_mean'] = np.mean(contrast, axis=1)
            features['spectral_contrast_std'] = np.std(contrast, axis=1)
            
            # Zero crossing rate
            zcr = librosa.feature.zero_crossing_rate(
                audio,
                hop_length=self.hop_length
            )[0]
            features['zero_crossing_rate'] = zcr
            features['zero_crossing_rate_mean'] = np.mean(zcr)
            features['zero_crossing_rate_std'] = np.std(zcr)
            
            # Spectral flux (difference between consecutive spectra)
            flux = self._compute_spectral_flux(audio)
            features['spectral_flux'] = flux
            features['spectral_flux_mean'] = np.mean(flux)
            features['spectral_flux_std'] = np.std(flux)
            
            logger.debug(f"Extracted {len(features)} spectral features")
            
        except Exception as e:
            logger.error(f"Error extracting spectral features: {str(e)}")
            raise
        
        return features
    
    def _compute_spectral_flux(self, audio: np.ndarray) -> np.ndarray:
        """
        Compute spectral flux
        
        Args:
            audio: Audio waveform
            
        Returns:
            Spectral flux values
        """
        # Compute STFT
        stft = librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length)
        magnitude = np.abs(stft)
        
        # Compute flux as difference between consecutive frames
        flux = np.sqrt(np.sum(np.diff(magnitude, axis=1)**2, axis=0))
        
        return flux
    
    def extract_summary_stats(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract summary statistics of spectral features as a feature vector
        
        Args:
            audio: Audio waveform
            
        Returns:
            1D feature vector
        """
        features = self.extract(audio)
        
        # Collect scalar features
        feature_vector = []
        
        # Add mean and std for each feature
        for key in ['spectral_centroid', 'spectral_bandwidth', 
                   'spectral_rolloff', 'zero_crossing_rate', 'spectral_flux']:
            if f'{key}_mean' in features:
                feature_vector.append(features[f'{key}_mean'])
            if f'{key}_std' in features:
                feature_vector.append(features[f'{key}_std'])
        
        # Add spectral contrast statistics
        if 'spectral_contrast_mean' in features:
            feature_vector.extend(features['spectral_contrast_mean'])
            feature_vector.extend(features['spectral_contrast_std'])
        
        return np.array(feature_vector)
    
    def get_feature_names(self) -> List[str]:
        """
        Get names of features in the feature vector
        
        Returns:
            List of feature names
        """
        names = []
        
        for key in ['spectral_centroid', 'spectral_bandwidth', 
                   'spectral_rolloff', 'zero_crossing_rate', 'spectral_flux']:
            names.append(f'{key}_mean')
            names.append(f'{key}_std')
        
        # Spectral contrast has 7 bands by default
        for i in range(7):
            names.append(f'spectral_contrast_band{i}_mean')
        for i in range(7):
            names.append(f'spectral_contrast_band{i}_std')
        
        return names


def extract_spectral_features(audio: np.ndarray, sample_rate: int = 22050,
                              n_fft: int = 2048, hop_length: int = 512) -> Dict[str, np.ndarray]:
    """
    Convenience function to extract spectral features
    
    Args:
        audio: Audio waveform
        sample_rate: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        
    Returns:
        Dictionary of spectral features
    """
    extractor = SpectralFeatureExtractor(sample_rate, n_fft, hop_length)
    return extractor.extract(audio)
