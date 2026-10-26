"""One Gaussian HMM per label for fixed-length feature sequences."""

import numpy as np
from hmmlearn.hmm import GMMHMM


class ClassConditionalHMM:
    def __init__(self, n_states: int = 3, random_state: int = 42) -> None:
        if n_states < 1:
            raise ValueError("n_states must be positive")
        self.n_states = n_states
        self.random_state = random_state
        self.models: dict[str, GMMHMM] = {}

    def fit(self, sequences: list[np.ndarray], labels: list[str]) -> "ClassConditionalHMM":
        if not sequences or len(sequences) != len(labels):
            raise ValueError("provide equally many non-empty sequences and labels")
        for label in sorted(set(labels)):
            selected = [np.asarray(x, dtype=float) for x, y in zip(sequences, labels, strict=True) if y == label]
            if any(x.ndim != 2 or len(x) < self.n_states for x in selected):
                raise ValueError("each sequence must be 2D and at least n_states long")
            lengths = [len(x) for x in selected]
            model = GMMHMM(
                n_components=self.n_states,
                n_mix=2,
                covariance_type="diag",
                n_iter=100,
                random_state=self.random_state,
            )
            model.fit(np.concatenate(selected), lengths)
            self.models[label] = model
        return self

    def predict(self, sequences: list[np.ndarray]) -> list[str]:
        if not self.models:
            raise RuntimeError("fit must be called before predict")
        return [max(self.models, key=lambda label: self.models[label].score(x)) for x in sequences]
