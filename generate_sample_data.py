"""
Generate sample audio data for testing Railway Crack Detection System
Creates synthetic audio files for healthy and defective rail samples
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_healthy_rail_audio(duration=3.0, sample_rate=22050):
    """
    Generate synthetic healthy rail audio
    Low frequency rumble + minimal high frequency content
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        
    Returns:
        numpy array of audio samples
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Base rumble (50-200 Hz)
    rumble = 0.3 * np.sin(2 * np.pi * 80 * t)
    rumble += 0.2 * np.sin(2 * np.pi * 120 * t)
    rumble += 0.15 * np.sin(2 * np.pi * 150 * t)
    
    # Periodic wheel impacts (smooth)
    wheel_freq = 2.5  # Hz (wheel passing frequency)
    wheel_impact = 0.2 * np.sin(2 * np.pi * wheel_freq * t) ** 2
    
    # Low noise
    noise = 0.05 * np.random.randn(len(t))
    
    audio = rumble + wheel_impact + noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32)


def generate_defective_rail_audio(duration=3.0, sample_rate=22050, defect_type='crack'):
    """
    Generate synthetic defective rail audio
    Includes high frequency transients and irregularities
    
    Args:
        duration: Duration in seconds
        sample_rate: Sample rate in Hz
        defect_type: 'crack', 'corrugation', or 'weld_failure'
        
    Returns:
        numpy array of audio samples
    """
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Base rumble (similar to healthy)
    rumble = 0.25 * np.sin(2 * np.pi * 80 * t)
    rumble += 0.15 * np.sin(2 * np.pi * 120 * t)
    
    if defect_type == 'crack':
        # Sharp high-frequency impacts at irregular intervals
        num_impacts = np.random.randint(8, 15)
        impact_positions = np.random.choice(len(t), num_impacts, replace=False)
        
        impacts = np.zeros(len(t))
        for pos in impact_positions:
            # Create sharp transient
            impact_duration = int(0.01 * sample_rate)  # 10ms
            if pos + impact_duration < len(t):
                transient = np.exp(-np.linspace(0, 10, impact_duration))
                transient *= np.sin(2 * np.pi * 3000 * np.linspace(0, 0.01, impact_duration))
                impacts[pos:pos+impact_duration] += 0.6 * transient
        
        audio = rumble + impacts
        
    elif defect_type == 'corrugation':
        # Regular high-frequency modulation
        corrugation_freq = 35  # Hz (corrugation wavelength)
        corrugation = 0.4 * np.sin(2 * np.pi * corrugation_freq * t)
        corrugation *= np.sin(2 * np.pi * 1500 * t)  # Modulated at 1.5 kHz
        
        audio = rumble + corrugation
        
    elif defect_type == 'weld_failure':
        # Periodic large impacts
        weld_freq = 1.5  # Hz
        weld_impacts = 0.5 * np.abs(np.sin(2 * np.pi * weld_freq * t)) ** 8
        weld_impacts *= np.sin(2 * np.pi * 500 * t)
        
        audio = rumble + weld_impacts
    
    else:
        audio = rumble
    
    # Add moderate noise
    noise = 0.1 * np.random.randn(len(t))
    audio += noise
    
    # Normalize
    audio = audio / np.max(np.abs(audio)) * 0.8
    
    return audio.astype(np.float32)


def main():
    """Generate sample dataset"""
    
    # Configuration
    SAMPLE_RATE = 22050
    DURATION = 3.0
    NUM_HEALTHY = 10
    NUM_DEFECTIVE_PER_TYPE = 5
    
    # Create directories
    base_dir = Path(__file__).parent
    healthy_dir = base_dir / "data" / "raw" / "healthy"
    defective_dir = base_dir / "data" / "raw" / "defective"
    
    healthy_dir.mkdir(parents=True, exist_ok=True)
    defective_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Creating directories:")
    logger.info(f"  Healthy: {healthy_dir}")
    logger.info(f"  Defective: {defective_dir}")
    
    # Generate healthy samples
    logger.info(f"\nGenerating {NUM_HEALTHY} healthy rail samples...")
    for i in range(NUM_HEALTHY):
        audio = generate_healthy_rail_audio(duration=DURATION, sample_rate=SAMPLE_RATE)
        filename = healthy_dir / f"healthy_rail_{i+1:03d}.wav"
        sf.write(filename, audio, SAMPLE_RATE)
        logger.info(f"  Created: {filename.name}")
    
    # Generate defective samples
    defect_types = ['crack', 'corrugation', 'weld_failure']
    
    for defect_type in defect_types:
        logger.info(f"\nGenerating {NUM_DEFECTIVE_PER_TYPE} {defect_type} samples...")
        for i in range(NUM_DEFECTIVE_PER_TYPE):
            audio = generate_defective_rail_audio(
                duration=DURATION,
                sample_rate=SAMPLE_RATE,
                defect_type=defect_type
            )
            filename = defective_dir / f"{defect_type}_{i+1:03d}.wav"
            sf.write(filename, audio, SAMPLE_RATE)
            logger.info(f"  Created: {filename.name}")
    
    # Summary
    total_healthy = len(list(healthy_dir.glob("*.wav")))
    total_defective = len(list(defective_dir.glob("*.wav")))
    
    logger.info("\n" + "="*50)
    logger.info("Sample Data Generation Complete!")
    logger.info("="*50)
    logger.info(f"Healthy samples: {total_healthy}")
    logger.info(f"Defective samples: {total_defective}")
    logger.info(f"Total samples: {total_healthy + total_defective}")
    logger.info(f"\nSample rate: {SAMPLE_RATE} Hz")
    logger.info(f"Duration: {DURATION} seconds")
    logger.info(f"\nData location:")
    logger.info(f"  {healthy_dir}")
    logger.info(f"  {defective_dir}")
    logger.info("\nYou can now run: streamlit run app.py")


if __name__ == "__main__":
    main()
