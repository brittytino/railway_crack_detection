
import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


def load_audio_files_from_directory(directory: str, 
                                    sample_rate: int = 22050,
                                    duration: float = None) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load all audio files from a directory

    Args:
        directory: Directory path
        sample_rate: Target sample rate
        duration: Duration to load (seconds)

    Returns:
        Tuple of (audio_list, filename_list)
    """
    audio_files = []
    filenames = []

    directory_path = Path(directory)

    # Supported audio formats
    extensions = ['*.wav', '*.mp3', '*.flac', '*.ogg']

    for ext in extensions:
        for file_path in directory_path.glob(ext):
            try:
                audio, sr = librosa.load(str(file_path), sr=sample_rate, duration=duration)
                audio_files.append(audio)
                filenames.append(file_path.name)
                logger.debug(f"Loaded: {file_path.name}")
            except Exception as e:
                logger.warning(f"Error loading {file_path.name}: {str(e)}")

    logger.info(f"Loaded {len(audio_files)} audio files from {directory}")
    return audio_files, filenames


def save_audio(audio: np.ndarray, filepath: str, sample_rate: int = 22050):
    """
    Save audio to file

    Args:
        audio: Audio waveform
        filepath: Output file path
        sample_rate: Sample rate
    """
    try:
        sf.write(filepath, audio, sample_rate)
        logger.info(f"Audio saved to {filepath}")
    except Exception as e:
        logger.error(f"Error saving audio: {str(e)}")
        raise


def normalize_length(audio: np.ndarray, target_length: int, 
                     pad_mode: str = 'constant') -> np.ndarray:
    """
    Normalize audio length by padding or truncating

    Args:
        audio: Audio waveform
        target_length: Target length in samples
        pad_mode: Padding mode ('constant', 'edge', 'wrap')

    Returns:
        Audio with target length
    """
    current_length = len(audio)

    if current_length < target_length:
        # Pad
        pad_width = target_length - current_length
        audio = np.pad(audio, (0, pad_width), mode=pad_mode)
    elif current_length > target_length:
        # Truncate
        audio = audio[:target_length]

    return audio


def split_audio(audio: np.ndarray, segment_length: int, 
                overlap: float = 0.5) -> List[np.ndarray]:
    """
    Split audio into segments

    Args:
        audio: Audio waveform
        segment_length: Length of each segment
        overlap: Overlap ratio (0-1)

    Returns:
        List of audio segments
    """
    hop_length = int(segment_length * (1 - overlap))
    segments = []

    for start in range(0, len(audio) - segment_length + 1, hop_length):
        segment = audio[start:start + segment_length]
        segments.append(segment)

    logger.debug(f"Split audio into {len(segments)} segments")
    return segments


def calculate_snr(audio: np.ndarray, noise_duration: float = 0.5,
                 sample_rate: int = 22050) -> float:
    """
    Calculate Signal-to-Noise Ratio

    Args:
        audio: Audio waveform
        noise_duration: Duration of noise sample (seconds)
        sample_rate: Sample rate

    Returns:
        SNR in dB
    """
    noise_samples = int(noise_duration * sample_rate)

    if len(audio) < noise_samples * 2:
        logger.warning("Audio too short for SNR calculation")
        return 0.0

    # Estimate noise from beginning
    noise = audio[:noise_samples]
    signal = audio[noise_samples:]

    signal_power = np.mean(signal ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float('inf')

    snr = 10 * np.log10(signal_power / noise_power)
    return snr


def apply_gain(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """
    Apply gain to audio

    Args:
        audio: Audio waveform
        gain_db: Gain in dB

    Returns:
        Audio with gain applied
    """
    gain_linear = 10 ** (gain_db / 20)
    audio_gained = audio * gain_linear

    # Clip to prevent overflow
    audio_gained = np.clip(audio_gained, -1.0, 1.0)

    return audio_gained


def detect_silence(audio: np.ndarray, threshold: float = 0.01,
                  min_silence_duration: float = 0.1,
                  sample_rate: int = 22050) -> List[Tuple[int, int]]:
    """
    Detect silent regions in audio

    Args:
        audio: Audio waveform
        threshold: Amplitude threshold for silence
        min_silence_duration: Minimum silence duration (seconds)
        sample_rate: Sample rate

    Returns:
        List of (start, end) indices of silent regions
    """
    min_samples = int(min_silence_duration * sample_rate)

    # Binary mask: 1 for silence, 0 for sound
    is_silent = np.abs(audio) < threshold

    # Find contiguous silent regions
    silent_regions = []
    in_silence = False
    start_idx = 0

    for i, silent in enumerate(is_silent):
        if silent and not in_silence:
            start_idx = i
            in_silence = True
        elif not silent and in_silence:
            if i - start_idx >= min_samples:
                silent_regions.append((start_idx, i))
            in_silence = False

    # Check last region
    if in_silence and len(audio) - start_idx >= min_samples:
        silent_regions.append((start_idx, len(audio)))

    return silent_regions


def get_audio_stats(audio: np.ndarray, sample_rate: int = 22050) -> dict:
    """
    Get audio statistics

    Args:
        audio: Audio waveform
        sample_rate: Sample rate

    Returns:
        Dictionary of audio statistics
    """
    stats = {
        'duration': len(audio) / sample_rate,
        'sample_rate': sample_rate,
        'num_samples': len(audio),
        'max_amplitude': np.abs(audio).max(),
        'mean_amplitude': np.abs(audio).mean(),
        'rms': np.sqrt(np.mean(audio ** 2)),
        'dynamic_range': 20 * np.log10(np.abs(audio).max() / (np.abs(audio).mean() + 1e-10))
    }

    return stats