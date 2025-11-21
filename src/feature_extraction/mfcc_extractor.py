"""
MFCC feature extraction module
"""

import numpy as np
import librosa
import logging

logger = logging.getLogger(__name__)


class MFCCExtractor:
    def __init__(self, sample_rate: int = 22050, n_mfcc: int = 20, 
                 n_fft: int = 2048, hop_length: int = 512):
        """
        Initialize MFCC Extractor

        Args:
            sample_rate: Audio sample rate
            n_mfcc: Number of MFCC coefficients
            n_fft: FFT window size
            hop_length: Hop length for STFT
        """
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length

    def extract(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract MFCC features from audio

        Args:
            audio: Audio waveform

        Returns:
            MFCC feature matrix (n_mfcc x time_frames)
        """
        try:
            mfcc = librosa.feature.mfcc(
                y=audio,
                sr=self.sample_rate,
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )

            logger.debug(f"Extracted MFCC features: {mfcc.shape}")
            return mfcc

        except Exception as e:
            logger.error(f"Error extracting MFCC: {str(e)}")
            raise

    def extract_statistics(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract statistical features from MFCC

        Args:
            audio: Audio waveform

        Returns:
            Feature vector with mean, std, min, max of each MFCC coefficient
        """
        mfcc = self.extract(audio)

        # Compute statistics across time dimension
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.min(mfcc, axis=1),
            np.max(mfcc, axis=1)
        ])

        logger.debug(f"Extracted MFCC statistics: {features.shape}")
        return features

    def extract_delta(self, audio: np.ndarray, order: int = 1) -> np.ndarray:
        """
        Extract MFCC delta features (temporal derivatives)

        Args:
            audio: Audio waveform
            order: Delta order (1 for delta, 2 for delta-delta)

        Returns:
            Delta MFCC features
        """
        mfcc = self.extract(audio)
        delta = librosa.feature.delta(mfcc, order=order)

        logger.debug(f"Extracted MFCC delta (order={order}): {delta.shape}")
        return delta

    def extract_full_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract comprehensive MFCC-based features

        Args:
            audio: Audio waveform

        Returns:
            Combined feature vector (MFCC + delta + delta-delta statistics)
        """
        # Extract MFCC
        mfcc = self.extract(audio)
        delta1 = self.extract_delta(audio, order=1)
        delta2 = self.extract_delta(audio, order=2)

        # Combine all features with statistics
        features = np.concatenate([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(delta1, axis=1),
            np.std(delta1, axis=1),
            np.mean(delta2, axis=1),
            np.std(delta2, axis=1)
        ])

        logger.debug(f"Extracted full MFCC features: {features.shape}")
        return features
