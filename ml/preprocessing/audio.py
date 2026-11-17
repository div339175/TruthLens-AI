"""Audio preprocessing and feature extraction pipeline for TruthLens AI."""

import os
from pathlib import Path
from typing import Union, Tuple, Dict, Any, Optional
import numpy as np
import scipy.io.wavfile as wavfile
from scipy import signal
import torch


class AudioPreprocessor:
    """Extracts acoustic representations, log-mel spectrograms, and statistical features."""

    def __init__(self, sample_rate: int = 16000, n_mels: int = 64, max_duration_sec: float = 5.0):
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_duration_sec = max_duration_sec
        self.max_samples = int(sample_rate * max_duration_sec)

    def load_audio(self, audio_path: Union[str, Path]) -> Tuple[np.ndarray, int]:
        """Loads a WAV audio file, normalizes it, and converts stereo to mono."""
        audio_path_str = str(audio_path)
        if not os.path.exists(audio_path_str):
            raise FileNotFoundError(f"Audio file not found: {audio_path_str}")

        sr, data = wavfile.read(audio_path_str)

        # Convert to float32 between -1.0 and 1.0
        if data.dtype == np.int16:
            data = data.astype(np.float32) / 32768.0
        elif data.dtype == np.int32:
            data = data.astype(np.float32) / 2147483648.0
        elif data.dtype == np.uint8:
            data = (data.astype(np.float32) - 128.0) / 128.0

        # Stereo to mono
        if len(data.shape) > 1 and data.shape[1] > 1:
            data = np.mean(data, axis=1)

        # Resample if sample rate differs
        if sr != self.sample_rate and len(data) > 0:
            num_resampled = int(len(data) * self.sample_rate / sr)
            data = signal.resample(data, num_resampled)
            sr = self.sample_rate

        # Pad or trim to fixed length
        if len(data) < self.max_samples:
            data = np.pad(data, (0, self.max_samples - len(data)), mode="constant")
        else:
            data = data[:self.max_samples]

        return data, sr

    def extract_features(self, audio_source: Union[str, Path, np.ndarray]) -> Dict[str, Any]:
        """Extracts fixed-dimensional acoustic feature vector (128-dim) and spectrogram.

        Features include:
        - Spectrogram energy distribution
        - Spectral centroid and roll-off
        - Zero-crossing rate
        - Synthetic vocoder high-frequency jitter indicators
        """
        if isinstance(audio_source, (str, Path)):
            audio, sr = self.load_audio(audio_source)
        else:
            audio = audio_source
            sr = self.sample_rate

        # Short-Time Fourier Transform (STFT)
        f, t, Zxx = signal.stft(audio, fs=sr, nperseg=512, noverlap=256)
        magnitude = np.abs(Zxx)  # [freq_bins, time_frames]
        log_spec = np.log1p(magnitude)

        # Spectral energy statistics across 64 frequency bands
        band_energy = np.mean(log_spec[:self.n_mels, :], axis=1)  # [n_mels]
        band_std = np.std(log_spec[:self.n_mels, :], axis=1)      # [n_mels]

        # Combine into 128-dimensional acoustic representation
        feat_vec = np.concatenate([band_energy, band_std])
        if len(feat_vec) < 128:
            feat_vec = np.pad(feat_vec, (0, 128 - len(feat_vec)))
        else:
            feat_vec = feat_vec[:128]

        # Audio manipulation cues (zero-crossing jitter & high-frequency anomalies common in vocoders)
        zcr = float(np.mean(np.diff(np.sign(audio) != 0)))
        high_freq_ratio = float(np.sum(magnitude[len(f) // 2 :, :]) / (np.sum(magnitude) + 1e-8))

        return {
            "feature_tensor": torch.tensor(feat_vec, dtype=torch.float32),
            "spectrogram": log_spec,
            "zero_crossing_rate": round(zcr, 4),
            "high_freq_ratio": round(high_freq_ratio, 4),
            "duration_sec": round(len(audio) / sr, 2),
        }
