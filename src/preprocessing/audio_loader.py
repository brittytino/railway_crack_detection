"""
Audio file loading and validation module
"""

import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class AudioLoader:
    def __init__(self, sample_rate: int = 22050, duration: Optional[float] = None):
        """
        Initialize AudioLoader

        Args:
            sample_rate: Target sample rate for audio
            duration: Fixed duration to load (seconds). None for full file
        """
        self.sample_rate = sample_rate
        self.duration = duration

    def load(self, file_path: str) -> Tuple[np.ndarray, int]:
        """
        Load audio file and return waveform

        Args:
            file_path: Path to audio file

        Returns:
            Tuple of (audio_data, sample_rate)
        """
        try:
            # Load audio file
            audio, sr = librosa.load(
                file_path, 
                sr=self.sample_rate, 
                duration=self.duration,
                mono=True
            )

            logger.info(f"Loaded audio: {file_path} | Duration: {len(audio)/sr:.2f}s")
            return audio, sr

        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
            raise

    def load_batch(self, file_paths: list) -> list:
        """
        Load multiple audio files

        Args:
            file_paths: List of audio file paths

        Returns:
            List of (audio_data, sample_rate) tuples
        """
        audio_data = []
        for path in file_paths:
            try:
                audio, sr = self.load(path)
                audio_data.append((audio, sr))
            except Exception as e:
                logger.warning(f"Skipping {path}: {str(e)}")
                continue

        return audio_data

    def validate_audio(self, audio: np.ndarray, sr: int) -> bool:
        """
        Validate audio data

        Args:
            audio: Audio waveform
            sr: Sample rate

        Returns:
            True if valid, False otherwise
        """
        if len(audio) == 0:
            logger.error("Audio is empty")
            return False

        if np.all(audio == 0):
            logger.error("Audio contains only silence")
            return False

        if np.any(np.isnan(audio)) or np.any(np.isinf(audio)):
            logger.error("Audio contains NaN or Inf values")
            return False

        return True

    def normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize audio to [-1, 1] range

        Args:
            audio: Audio waveform

        Returns:
            Normalized audio
        """
        max_val = np.abs(audio).max()
        if max_val > 0:
            audio = audio / max_val
        return audio
