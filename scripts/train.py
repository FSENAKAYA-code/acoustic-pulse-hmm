"""Train an HMM from local feature arrays; no audio is downloaded."""

import argparse
from pathlib import Path

import numpy as np

from audio.experiment import run_experiment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True, help="NPZ with features and labels arrays")
    args = parser.parse_args()
    with np.load(args.data, allow_pickle=False) as archive:
        if not {"features", "labels"}.issubset(archive.files):
            raise ValueError("NPZ must contain features and labels arrays")
        features = archive["features"].astype(float)
        labels = archive["labels"].astype(str)
    print(run_experiment(features, labels))


if __name__ == "__main__":
    main()
