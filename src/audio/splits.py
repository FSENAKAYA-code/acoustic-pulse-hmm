"""Deterministic, stratified three-way splits with basic input checks."""

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DataSplits:
    X_train: Any
    X_validation: Any
    X_test: Any
    y_train: Any
    y_validation: Any
    y_test: Any


def split_sequence_indices(labels: Any, *, random_state: int = 42) -> tuple[Any, Any, Any]:
    """Return stratified utterance indices; keep every sequence intact."""
    labels = np.asarray(labels)
    if labels.ndim != 1:
        raise ValueError("labels must be one-dimensional")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or counts.min() < 5:
        raise ValueError("each of at least two labels needs five or more utterances")
    indices = np.arange(len(labels))
    train_validation, test = train_test_split(
        indices, test_size=0.2, random_state=random_state, stratify=labels
    )
    train, validation = train_test_split(
        train_validation,
        test_size=0.25,
        random_state=random_state,
        stratify=labels[train_validation],
    )
    return train, validation, test


def split_data(X: Any, y: Any, *, random_state: int = 42) -> DataSplits:
    """Split aligned labeled examples 60/20/20, stratifying by class."""
    labels = np.asarray(y)
    if labels.ndim != 1:
        raise ValueError("y must be one-dimensional")
    if len(X) != len(labels):
        raise ValueError("X and y must have the same number of rows")
    classes, counts = np.unique(labels, return_counts=True)
    if len(classes) < 2 or counts.min() < 5:
        raise ValueError("each of at least two classes needs five or more examples")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    X_train, X_validation, y_train, y_validation = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_val,
    )
    return DataSplits(X_train, X_validation, X_test, y_train, y_validation, y_test)
