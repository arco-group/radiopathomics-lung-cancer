# Radiopathomics Lung Cancer

Companion repository for the study "RadioPathomics: Multimodal Learning in
Non-Small Cell Lung Cancer for Adaptive Radiotherapy".

## Repository structure

- `Codice/`: Python scripts for handcrafted feature experiments, multimodal
  fusion, bootstrap evaluation, and deep-feature training.
- `Codice/lib/`: shared utilities for feature aggregation, fusion helpers, and
  metric computation.
- `Codice/DeepFeatures/`: PyTorch training utilities for image-based deep
  features.
- `data/`: input data directory.
- `docs/DATA.md`: expected data layout.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The scripts assume CSV feature matrices with the subject or sample identifier in
the first column and the binary target in the last column. Place the input files
under `data/` using the filenames described in `docs/DATA.md`.
