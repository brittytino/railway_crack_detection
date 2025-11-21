"""
Audio noise filtering and preprocessing module
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt
import logging

logger = logging.getLogger(__name__)


class NoiseFilter:
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize NoiseFilter

        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate

    def bandpass_filter(self, audio: np.ndarray, lowcut: float = 300, 
                       highcut: float = 8000, order: int = 5) -> np.ndarray:
        """
        Apply bandpass filter to isolate relevant frequencies

        Args:
            audio: Audio waveform
            lowcut: Low frequency cutoff (Hz)
            highcut: High frequency cutoff (Hz)
            order: Filter order

        Returns:
            Filtered audio
        """
        nyquist = 0.5 * self.sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist

        b, a = butter(order, [low, high], btype='band')
        filtered_audio = filtfilt(b, a, audio)

        logger.debug(f"Applied bandpass filter: {lowcut}-{highcut} Hz")
        return filtered_audio

    def remove_silence(self, audio: np.ndarray, threshold: float = 0.01) -> np.ndarray:
        """
        Remove silent portions from audio

        Args:
            audio: Audio waveform
            threshold: Amplitude threshold for silence

        Returns:
            Audio with silence removed
        """
        non_silent = np.abs(audio) > threshold
        if np.any(non_silent):
            audio = audio[non_silent]

        logger.debug(f"Removed silence (threshold: {threshold})")
        return audio

    def reduce_noise_spectral(self, audio: np.ndarray, 
                             noise_profile_duration: float = 0.5) -> np.ndarray:
        """
        Spectral noise reduction using noise profile

        Args:
            audio: Audio waveform
            noise_profile_duration: Duration of noise profile in seconds

        Returns:
            Noise-reduced audio
        """
        # Simple spectral subtraction
        profile_samples = int(noise_profile_duration * self.sample_rate)
        noise_profile = audio[:profile_samples]

        # Compute noise spectrum
        noise_fft = np.fft.rfft(noise_profile)
        noise_power = np.abs(noise_fft) ** 2

        # Process full audio
        audio_fft = np.fft.rfft(audio)
        audio_power = np.abs(audio_fft) ** 2

        # Subtract noise
        clean_power = np.maximum(audio_power - noise_power.mean(), 0)
        clean_fft = np.sqrt(clean_power) * np.exp(1j * np.angle(audio_fft))

        clean_audio = np.fft.irfft(clean_fft, n=len(audio))

        logger.debug("Applied spectral noise reduction")
        return clean_audio

    def preprocess(self, audio: np.ndarray, apply_bandpass: bool = True,
                  apply_noise_reduction: bool = True) -> np.ndarray:
        """
        Complete preprocessing pipeline

        Args:
            audio: Audio waveform
            apply_bandpass: Whether to apply bandpass filter
            apply_noise_reduction: Whether to apply noise reduction

        Returns:
            Preprocessed audio
        """
        if apply_bandpass:
            audio = self.bandpass_filter(audio)

        if apply_noise_reduction:
            audio = self.reduce_noise_spectral(audio)

        # Normalize
        audio = audio / (np.abs(audio).max() + 1e-8)

        return audio
