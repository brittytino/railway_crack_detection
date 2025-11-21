"""
Spectral feature extraction module
"""

import numpy as np
import librosa
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class SpectralFeatureExtractor:
    def __init__(self, sample_rate: int = 22050, n_fft: int = 2048, 
                 hop_length: int = 512):
        """
        Initialize Spectral Feature Extractor

        Args:
            sample_rate: Audio sample rate
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length

    def spectral_centroid(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral centroid"""
        centroid = librosa.feature.spectral_centroid(
            y=audio, sr=self.sample_rate, 
            n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        return centroid

    def spectral_rolloff(self, audio: np.ndarray, roll_percent: float = 0.85) -> np.ndarray:
        """Extract spectral rolloff"""
        rolloff = librosa.feature.spectral_rolloff(
            y=audio, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length,
            roll_percent=roll_percent
        )[0]
        return rolloff

    def spectral_bandwidth(self, audio: np.ndarray) -> np.ndarray:
        """Extract spectral bandwidth"""
        bandwidth = librosa.feature.spectral_bandwidth(
            y=audio, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length
        )[0]
        return bandwidth

    def spectral_contrast(self, audio: np.ndarray, n_bands: int = 6) -> np.ndarray:
        """Extract spectral contrast"""
        contrast = librosa.feature.spectral_contrast(
            y=audio, sr=self.sample_rate,
            n_fft=self.n_fft, hop_length=self.hop_length,
            n_bands=n_bands
        )
        return contrast

    def spectral_flux(self, audio: np.ndarray) -> np.ndarray:
        """
        Calculate spectral flux (rate of change in spectrum)
        """
        # Compute spectrogram
        S = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))

        # Compute flux as difference between consecutive frames
        flux = np.sqrt(np.sum(np.diff(S, axis=1)**2, axis=0))

        return flux

    def zero_crossing_rate(self, audio: np.ndarray) -> np.ndarray:
        """Extract zero crossing rate"""
        zcr = librosa.feature.zero_crossing_rate(
            audio, frame_length=self.n_fft, hop_length=self.hop_length
        )[0]
        return zcr

    def spectral_kurtosis(self, audio: np.ndarray) -> float:
        """
        Calculate spectral kurtosis (measure of peakedness)
        """
        # Compute power spectrum
        S = np.abs(librosa.stft(audio, n_fft=self.n_fft, hop_length=self.hop_length))

        # Calculate kurtosis across frequency bins
        kurtosis_values = stats.kurtosis(S, axis=0)

        return np.mean(kurtosis_values)

    def extract_all_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract all spectral features and return statistics

        Args:
            audio: Audio waveform

        Returns:
            Feature vector with statistics of all spectral features
        """
        features = []

        # Spectral centroid
        centroid = self.spectral_centroid(audio)
        features.extend([np.mean(centroid), np.std(centroid)])

        # Spectral rolloff
        rolloff = self.spectral_rolloff(audio)
        features.extend([np.mean(rolloff), np.std(rolloff)])

        # Spectral bandwidth
        bandwidth = self.spectral_bandwidth(audio)
        features.extend([np.mean(bandwidth), np.std(bandwidth)])

        # Spectral flux
        flux = self.spectral_flux(audio)
        features.extend([np.mean(flux), np.std(flux)])

        # Zero crossing rate
        zcr = self.zero_crossing_rate(audio)
        features.extend([np.mean(zcr), np.std(zcr)])

        # Spectral kurtosis
        kurtosis = self.spectral_kurtosis(audio)
        features.append(kurtosis)

        # Spectral contrast
        contrast = self.spectral_contrast(audio)
        features.extend([np.mean(contrast), np.std(contrast)])

        feature_vector = np.array(features)
        logger.debug(f"Extracted spectral features: {feature_vector.shape}")

        return feature_vector
