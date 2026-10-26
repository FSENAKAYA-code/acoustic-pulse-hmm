# Domain implementation notes

The feature helper extracts MFCCs and spectral centroid with librosa. Class-conditional GMM-HMM models consume utterance-level features; stratified utterance splits keep sequences intact, and evaluation reports train/validation negative log likelihood and held-out classification metrics. No recordings or extracted arrays are included.
