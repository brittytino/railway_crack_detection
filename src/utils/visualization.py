
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import seaborn as sns
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def plot_waveform(audio: np.ndarray, sample_rate: int = 22050, 
                 title: str = "Audio Waveform", figsize: tuple = (12, 4)):
    """
    Plot audio waveform

    Args:
        audio: Audio waveform
        sample_rate: Sample rate
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    time = np.arange(len(audio)) / sample_rate
    ax.plot(time, audio, linewidth=0.5)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_spectrogram(audio: np.ndarray, sample_rate: int = 22050,
                    n_fft: int = 2048, hop_length: int = 512,
                    title: str = "Spectrogram", figsize: tuple = (12, 6)):
    """
    Plot spectrogram

    Args:
        audio: Audio waveform
        sample_rate: Sample rate
        n_fft: FFT window size
        hop_length: Hop length
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Compute spectrogram
    D = librosa.amplitude_to_db(
        np.abs(librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)),
        ref=np.max
    )

    img = librosa.display.specshow(D, sr=sample_rate, hop_length=hop_length,
                                   x_axis='time', y_axis='hz', ax=ax, cmap='viridis')
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    plt.tight_layout()
    return fig


def plot_mel_spectrogram(audio: np.ndarray, sample_rate: int = 22050,
                        n_mels: int = 128, n_fft: int = 2048, 
                        hop_length: int = 512,
                        title: str = "Mel Spectrogram", figsize: tuple = (12, 6)):
    """
    Plot mel spectrogram

    Args:
        audio: Audio waveform
        sample_rate: Sample rate
        n_mels: Number of mel bands
        n_fft: FFT window size
        hop_length: Hop length
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio, sr=sample_rate, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length
    )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    img = librosa.display.specshow(mel_spec_db, sr=sample_rate, hop_length=hop_length,
                                   x_axis='time', y_axis='mel', ax=ax, cmap='magma')
    ax.set_title(title)
    fig.colorbar(img, ax=ax, format='%+2.0f dB')

    plt.tight_layout()
    return fig


def plot_mfcc(audio: np.ndarray, sample_rate: int = 22050, n_mfcc: int = 20,
             n_fft: int = 2048, hop_length: int = 512,
             title: str = "MFCC Features", figsize: tuple = (12, 6)):
    """
    Plot MFCC features

    Args:
        audio: Audio waveform
        sample_rate: Sample rate
        n_mfcc: Number of MFCC coefficients
        n_fft: FFT window size
        hop_length: Hop length
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Compute MFCC
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=n_mfcc,
                               n_fft=n_fft, hop_length=hop_length)

    img = librosa.display.specshow(mfcc, sr=sample_rate, hop_length=hop_length,
                                   x_axis='time', ax=ax, cmap='coolwarm')
    ax.set_ylabel('MFCC Coefficient')
    ax.set_title(title)
    fig.colorbar(img, ax=ax)

    plt.tight_layout()
    return fig


def plot_confusion_matrix(cm: np.ndarray, class_labels: list,
                         title: str = "Confusion Matrix", 
                         figsize: tuple = (8, 6), normalize: bool = False):
    """
    Plot confusion matrix

    Args:
        cm: Confusion matrix
        class_labels: List of class labels
        title: Plot title
        figsize: Figure size
        normalize: Whether to normalize values

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels,
                ax=ax, cbar_kws={'label': 'Count' if not normalize else 'Proportion'})

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(title)

    plt.tight_layout()
    return fig


def plot_feature_importance(importance: np.ndarray, feature_names: list = None,
                           top_n: int = 20, title: str = "Feature Importance",
                           figsize: tuple = (10, 8)):
    """
    Plot feature importance

    Args:
        importance: Feature importance values
        feature_names: List of feature names
        top_n: Number of top features to show
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    if feature_names is None:
        feature_names = [f"Feature {i}" for i in range(len(importance))]

    # Sort by importance
    indices = np.argsort(importance)[::-1][:top_n]

    # Plot
    ax.barh(range(top_n), importance[indices])
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Importance')
    ax.set_title(title)
    ax.invert_yaxis()

    plt.tight_layout()
    return fig


def plot_training_history(history: dict, metrics: list = ['accuracy', 'loss'],
                         title: str = "Training History", figsize: tuple = (12, 5)):
    """
    Plot training history

    Args:
        history: Training history dictionary
        metrics: List of metrics to plot
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(1, len(metrics), figsize=figsize)

    if len(metrics) == 1:
        axes = [axes]

    for i, metric in enumerate(metrics):
        if metric in history:
            axes[i].plot(history[metric], label=f'Train {metric}')
            if f'val_{metric}' in history:
                axes[i].plot(history[f'val_{metric}'], label=f'Val {metric}')
            axes[i].set_xlabel('Epoch')
            axes[i].set_ylabel(metric.capitalize())
            axes[i].set_title(f'{metric.capitalize()} over Epochs')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

    plt.suptitle(title)
    plt.tight_layout()
    return fig


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, roc_auc: float,
                  title: str = "ROC Curve", figsize: tuple = (8, 6)):
    """
    Plot ROC curve

    Args:
        fpr: False positive rate
        tpr: True positive rate
        roc_auc: ROC-AUC score
        title: Plot title
        figsize: Figure size

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(fpr, tpr, color='darkorange', lw=2, 
            label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(title)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig