"""MFCC and spectral-centroid features for one in-memory audio signal."""

import numpy as np


def extract_acoustic_features(
    waveform: np.ndarray, sample_rate: int, *, n_mfcc: int = 13
) -> np.ndarray:
    """Return a finite `(frames, n_mfcc + 1)` matrix; does not read/download audio."""
    signal = np.asarray(waveform, dtype=np.float32)
    if signal.ndim != 1 or signal.size == 0 or sample_rate <= 0 or n_mfcc < 1:
        raise ValueError("provide a non-empty mono signal, positive rate, and n_mfcc")
    try:
        import librosa
    except ImportError as exc:
        raise RuntimeError("install librosa in the project uv environment") from exc
    mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc)
    centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate)
    features = np.concatenate([mfcc, centroid], axis=0).T
    if not np.isfinite(features).all():
        raise ValueError("audio features contain non-finite values")
    return features
