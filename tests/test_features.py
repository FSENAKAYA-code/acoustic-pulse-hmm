import numpy as np
import pytest

from audio.features import extract_acoustic_features


def test_feature_extraction_rejects_invalid_signal_before_optional_import():
    with pytest.raises(ValueError, match="mono signal"):
        extract_acoustic_features(np.zeros((10, 2)), 16000)
