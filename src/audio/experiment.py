"""Leakage-aware baseline training and final held-out evaluation."""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from audio.acoustic_models import ClassConditionalHMM
from audio.splits import split_sequence_indices


def run_experiment(
    sequences: np.ndarray, labels: np.ndarray, *, random_state: int = 42
) -> dict[str, float]:
    """Fit class-conditional GMM-HMMs on train, select on validation, test last."""
    if sequences.ndim != 3:
        raise ValueError("features must have shape (utterances, frames, coefficients)")
    labels = np.asarray(labels)
    if len(sequences) != len(labels):
        raise ValueError("features and labels must have matching utterance counts")
    train_idx, validation_idx, test_idx = split_sequence_indices(
        labels, random_state=random_state
    )
    train_sequences = [sequences[index] for index in train_idx]
    train_labels = labels[train_idx].astype(str).tolist()
    validation_sequences = [sequences[index] for index in validation_idx]
    validation_labels = labels[validation_idx].astype(str)
    model = ClassConditionalHMM(random_state=random_state).fit(
        train_sequences, train_labels
    )
    validation_predictions = model.predict(validation_sequences)
    train_loss = _mean_negative_log_likelihood(model, train_sequences, train_labels)
    validation_loss = _mean_negative_log_likelihood(
        model, validation_sequences, validation_labels.tolist()
    )
    validation_score = f1_score(
        validation_labels, validation_predictions, average="weighted", zero_division=0
    )
    final_idx = np.concatenate([train_idx, validation_idx])
    final_model = ClassConditionalHMM(random_state=random_state).fit(
        [sequences[index] for index in final_idx], labels[final_idx].astype(str).tolist()
    )
    test_predictions = final_model.predict([sequences[index] for index in test_idx])
    test_labels = labels[test_idx].astype(str)
    return {
        "train_loss_nll": float(train_loss),
        "validation_loss_nll": float(validation_loss),
        "validation_f1_weighted": float(validation_score),
        "test_accuracy": float(accuracy_score(test_labels, test_predictions)),
        "test_precision_weighted": float(
            precision_score(test_labels, test_predictions, average="weighted", zero_division=0)
        ),
        "test_recall_weighted": float(
            recall_score(test_labels, test_predictions, average="weighted", zero_division=0)
        ),
        "test_f1_weighted": float(
            f1_score(test_labels, test_predictions, average="weighted", zero_division=0)
        ),
    }


def _mean_negative_log_likelihood(
    model: ClassConditionalHMM, sequences: list[np.ndarray], labels: list[str]
) -> float:
    return float(
        np.mean(
            [
                -model.models[label].score(sequence) / max(len(sequence), 1)
                for sequence, label in zip(sequences, labels, strict=True)
            ]
        )
    )
