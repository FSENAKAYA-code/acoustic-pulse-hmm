import pytest

hmmlearn = pytest.importorskip("hmmlearn")
import numpy as np

from audio.acoustic_models import ClassConditionalHMM


def test_hmm_rejects_mismatched_sequences_and_labels():
    with pytest.raises(ValueError, match="equally many"):
        ClassConditionalHMM().fit([np.ones((5, 2))], [])
