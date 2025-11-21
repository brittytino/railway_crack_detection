
import numpy as np
import torch
import torch.nn as nn
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class SimpleDiffusionModel(nn.Module):
    """
    Simplified diffusion model for audio augmentation
    """
    def __init__(self, audio_length: int = 66150):
        super().__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(audio_length, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
        )

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 512),
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, audio_length),
        )

    def forward(self, x, t):
        # Encode audio
        z = self.encoder(x)

        # Add time embedding
        t_emb = self.time_embed(t)
        z = z + t_emb

        # Decode
        out = self.decoder(z)
        return out


class DiffusionAudioGenerator:
    def __init__(self, sample_rate: int = 22050, duration: float = 3.0,
                 device: str = 'cpu'):
        """
        Initialize Diffusion Audio Generator

        Args:
            sample_rate: Audio sample rate
            duration: Audio duration in seconds
            device: Computing device ('cpu' or 'cuda')
        """
        self.sample_rate = sample_rate
        self.duration = duration
        self.audio_length = int(sample_rate * duration)
        self.device = device

        # Initialize model
        self.model = SimpleDiffusionModel(self.audio_length).to(device)
        self.num_steps = 50

        logger.info(f"Initialized DiffusionAudioGenerator on {device}")

    def add_noise(self, audio: np.ndarray, timestep: int) -> np.ndarray:
        """
        Add noise to audio based on timestep

        Args:
            audio: Clean audio waveform
            timestep: Diffusion timestep (0 to num_steps)

        Returns:
            Noisy audio
        """
        # Linear noise schedule
        alpha = 1 - (timestep / self.num_steps)
        noise = np.random.randn(*audio.shape)
        noisy_audio = np.sqrt(alpha) * audio + np.sqrt(1 - alpha) * noise

        return noisy_audio

    def generate_variations(self, seed_audio: np.ndarray, 
                           num_variations: int = 3) -> List[np.ndarray]:
        """
        Generate variations of seed audio using simple augmentation
        (Simplified version without full diffusion training)

        Args:
            seed_audio: Original audio sample
            num_variations: Number of variations to generate

        Returns:
            List of augmented audio samples
        """
        variations = []

        for i in range(num_variations):
            # Time stretching
            stretch_factor = np.random.uniform(0.9, 1.1)
            stretched = self._time_stretch(seed_audio, stretch_factor)

            # Pitch shifting
            pitch_shift = np.random.uniform(-2, 2)
            pitched = self._pitch_shift(stretched, pitch_shift)

            # Add controlled noise
            noise_level = np.random.uniform(0.01, 0.05)
            noisy = pitched + noise_level * np.random.randn(*pitched.shape)

            # Normalize
            if np.abs(noisy).max() > 0:
                noisy = noisy / np.abs(noisy).max()

            variations.append(noisy)

        logger.info(f"Generated {num_variations} audio variations")
        return variations

    def _time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """Simple time stretching"""
        indices = np.arange(0, len(audio), rate)
        indices = indices[indices < len(audio)].astype(int)
        stretched = audio[indices]

        # Pad or trim to original length
        if len(stretched) < len(audio):
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
        else:
            stretched = stretched[:len(audio)]

        return stretched

    def _pitch_shift(self, audio: np.ndarray, n_steps: float) -> np.ndarray:
        """Simple pitch shifting using resampling"""
        rate = 2 ** (n_steps / 12)
        indices = np.arange(0, len(audio), rate)
        indices = indices[indices < len(audio)].astype(int)
        shifted = audio[indices]

        # Pad or trim to original length
        if len(shifted) < len(audio):
            shifted = np.pad(shifted, (0, len(audio) - len(shifted)))
        else:
            shifted = shifted[:len(audio)]

        return shifted

    def augment_minority_class(self, audio_samples: List[np.ndarray],
                               augmentation_factor: int = 3) -> List[np.ndarray]:
        """
        Augment minority class samples

        Args:
            audio_samples: List of audio samples from minority class
            augmentation_factor: How many variations per sample

        Returns:
            List of original + augmented samples
        """
        augmented_data = list(audio_samples)  # Keep originals

        for audio in audio_samples:
            # Ensure correct length
            if len(audio) < self.audio_length:
                audio = np.pad(audio, (0, self.audio_length - len(audio)))
            elif len(audio) > self.audio_length:
                audio = audio[:self.audio_length]

            # Generate variations
            variations = self.generate_variations(audio, augmentation_factor)
            augmented_data.extend(variations)

        logger.info(f"Augmented {len(audio_samples)} samples to {len(augmented_data)} samples")
        return augmented_data