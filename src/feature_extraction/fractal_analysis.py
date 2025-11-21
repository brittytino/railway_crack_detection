
import numpy as np
from scipy import signal
import logging

logger = logging.getLogger(__name__)


class FractalAnalyzer:
    def __init__(self, sample_rate: int = 22050):
        """
        Initialize Fractal Analyzer

        Args:
            sample_rate: Audio sample rate
        """
        self.sample_rate = sample_rate

    def higuchi_fractal_dimension(self, audio: np.ndarray, kmax: int = 10) -> float:
        """
        Calculate Higuchi Fractal Dimension

        Args:
            audio: Audio waveform
            kmax: Maximum k value

        Returns:
            Fractal dimension value
        """
        N = len(audio)
        L = []
        x = []

        for k in range(1, kmax + 1):
            Lk = []
            for m in range(k):
                Lmk = 0
                for i in range(1, int((N - m) / k)):
                    Lmk += abs(audio[m + i * k] - audio[m + (i - 1) * k])
                Lmk = Lmk * (N - 1) / (((N - m) / k) * k)
                Lk.append(Lmk)
            L.append(np.log(np.mean(Lk)))
            x.append(np.log(1.0 / k))

        # Linear regression to find slope
        coeffs = np.polyfit(x, L, 1)
        fd = coeffs[0]

        logger.debug(f"Higuchi FD: {fd:.4f}")
        return fd

    def petrosian_fractal_dimension(self, audio: np.ndarray) -> float:
        """
        Calculate Petrosian Fractal Dimension

        Args:
            audio: Audio waveform

        Returns:
            Fractal dimension value
        """
        N = len(audio)

        # Count zero crossings
        diff = np.diff(audio)
        N_delta = np.sum(diff[:-1] * diff[1:] < 0)

        # Calculate PFD
        if N_delta == 0:
            return 1.0

        pfd = np.log10(N) / (np.log10(N) + np.log10(N / (N + 0.4 * N_delta)))

        logger.debug(f"Petrosian FD: {pfd:.4f}")
        return pfd

    def katz_fractal_dimension(self, audio: np.ndarray) -> float:
        """
        Calculate Katz Fractal Dimension

        Args:
            audio: Audio waveform

        Returns:
            Fractal dimension value
        """
        n = len(audio)

        # Calculate distances
        dists = np.abs(np.diff(audio))
        L = np.sum(dists)  # Total length

        # Maximum distance from first point
        if n > 1:
            d = np.max(np.abs(audio - audio[0]))
        else:
            return 1.0

        # Avoid division by zero
        if d == 0 or L == 0:
            return 1.0

        kfd = np.log10(n) / (np.log10(n) + np.log10(d / L))

        logger.debug(f"Katz FD: {kfd:.4f}")
        return kfd

    def hurst_exponent(self, audio: np.ndarray, max_lag: int = 100) -> float:
        """
        Calculate Hurst Exponent (related to fractal dimension)

        Args:
            audio: Audio waveform
            max_lag: Maximum lag for R/S analysis

        Returns:
            Hurst exponent value
        """
        lags = range(2, min(max_lag, len(audio) // 2))
        tau = []

        for lag in lags:
            # Divide series into chunks
            n_chunks = len(audio) // lag
            chunks = audio[:n_chunks * lag].reshape(n_chunks, lag)

            # Calculate R/S for each chunk
            rs_values = []
            for chunk in chunks:
                mean_chunk = np.mean(chunk)
                deviations = chunk - mean_chunk
                cumulative_dev = np.cumsum(deviations)

                R = np.max(cumulative_dev) - np.min(cumulative_dev)
                S = np.std(chunk)

                if S > 0:
                    rs_values.append(R / S)

            if rs_values:
                tau.append(np.mean(rs_values))
            else:
                tau.append(0)

        # Linear regression on log-log plot
        if len(tau) > 0 and np.all(np.array(tau) > 0):
            log_lags = np.log(list(lags))
            log_tau = np.log(tau)
            coeffs = np.polyfit(log_lags, log_tau, 1)
            hurst = coeffs[0]
        else:
            hurst = 0.5

        logger.debug(f"Hurst Exponent: {hurst:.4f}")
        return hurst

    def extract_all_fractal_features(self, audio: np.ndarray) -> np.ndarray:
        """
        Extract all fractal-based features

        Args:
            audio: Audio waveform

        Returns:
            Array of fractal features
        """
        features = []

        try:
            # Higuchi FD
            hfd = self.higuchi_fractal_dimension(audio)
            features.append(hfd)

            # Petrosian FD
            pfd = self.petrosian_fractal_dimension(audio)
            features.append(pfd)

            # Katz FD
            kfd = self.katz_fractal_dimension(audio)
            features.append(kfd)

            # Hurst Exponent
            hurst = self.hurst_exponent(audio)
            features.append(hurst)

        except Exception as e:
            logger.error(f"Error extracting fractal features: {str(e)}")
            features = [0.0, 0.0, 0.0, 0.5]  # Default values

        feature_vector = np.array(features)
        logger.debug(f"Extracted fractal features: {feature_vector.shape}")

        return feature_vector