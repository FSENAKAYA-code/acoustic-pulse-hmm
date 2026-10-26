import numpy as np
import pytest
from sklearn.datasets import make_classification

from audio.splits import split_data, split_sequence_indices


def test_split_has_three_disjoint_deterministic_partitions():
    X, y = make_classification(n_samples=200, n_features=8, random_state=5)
    first = split_data(X, y)
    second = split_data(X, y)
    assert len(first.X_train) == 120
    assert len(first.X_validation) == 40
    assert len(first.X_test) == 40
    np.testing.assert_array_equal(first.X_test, second.X_test)


def test_split_rejects_too_small_classes():
    with pytest.raises(ValueError, match="five or more"):
        split_data(np.arange(8).reshape(4, 2), np.array([0, 0, 1, 1]))


def test_sequence_split_keeps_utterances_disjoint():
    labels = np.array(["a"] * 20 + ["b"] * 20)
    train, validation, test = split_sequence_indices(labels)
    assert not set(train) & set(validation)
    assert not set(train) & set(test)
    assert not set(validation) & set(test)
    assert len(train) + len(validation) + len(test) == len(labels)
