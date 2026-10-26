# acoustic-pulse-hmm

> Educational portfolio scaffold by Fatma Sena Kaya · **Audio and speech**

Audio feature experiments with Gaussian mixtures and hidden Markov models

## Scope and status

This repository is a reproducible learning project scaffold, not a production service. It contains an executable baseline, clear data boundaries, and extension points. Results are meaningful only when run on documented, appropriate data. No dataset, model weight, credential, or private record is included.

## Quick start

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run python scripts/train.py --data data/processed/features.npz
uv run pytest
uv run ruff check .
```

The audio training command expects a local NPZ with `features` shaped `(utterances, frames, coefficients)` and one string `labels` entry per utterance. Extract features from recordings you are allowed to use, record their source and license, and keep recordings and feature arrays out of Git. The command does not download data.

## Data and evaluation

- Expected input: Feature sequences extracted from licensed audio.
- The baseline separates training, validation, and test partitions with a fixed seed. The test partition is held out until final evaluation.
- Fit preprocessing steps on training data only. Use validation results for model selection; report the test result once for the chosen configuration.
- Check the project's `src/audio/` code and tests before adapting it to real data.

## Repository layout

```text
data/       empty raw/ and processed/ locations; contents are ignored by Git
models/     empty local model-output location; model weights are ignored
scripts/    runnable demo entry points
src/        importable project code
tests/      small deterministic unit tests
```

## Responsible use

GMM-HMM experiments

## Roadmap

1. Add a documented, licensed dataset adapter.
2. Add domain-specific preprocessing and baselines.
3. Tune against validation data and report held-out test metrics with limitations.
4. Add deployment only after privacy, security, and monitoring review.

## Author

Fatma Sena Kaya · [GitHub](https://github.com/FSENAKAYA-code) · [LinkedIn](https://www.linkedin.com/in/sena-kaya-8634ba383/)
