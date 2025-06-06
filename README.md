# CS182-Final-Project

This repository contains the implementation for our CS182 (Deep Learning) final project on protein analysis and prediction.

## Project Overview

This project focuses on developing machine learning models for protein analysis, including protein function prediction, protein-protein interaction prediction, and other protein-related tasks using state-of-the-art embeddings and neural network architectures.

## Project Structure

```
CS182-Final-Project/
├── src/
│   ├── v5/                   # V5 PPI Classifier (Latest)
│   │   ├── v5.py            # Main V5 architecture
│   │   ├── v5_2.py          # Hybrid V5.2 (V2 MAE + V5 downstream)
│   │   ├── v5_2_train.py    # V5.2 training script
│   │   └── evaluate_v5_model.py # V5 evaluation script
│   ├── v4/                   # V4 Implementation
│   │   └── v4_1.py          # Data loading and V4 models
│   ├── mask_autoencoder/     # MAE implementations
│   │   ├── v2.py            # V2 MAE (pretrained available)
│   │   ├── v3.py            # V3 MAE
│   │   └── models/          # Trained model checkpoints
│   ├── data_process/         # Data processing utilities
│   └── tcn_encoder/          # TCN encoder implementation
├── data/
│   ├── full_dataset/         # Main protein dataset
│   │   └── embeddings/       # ESM protein embeddings (960-dim)
│   ├── medium_set/           # Medium-sized dataset
│   └── small_set/            # Small dataset for testing
├── experiments/              # Experimental results and notebooks
├── results/                  # Model outputs and evaluations
├── logs/                     # Training logs and plots
├── models/                   # Trained model checkpoints
├── requirements.txt          # Python dependencies
└── README.md                # This file
```

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.9+
- CUDA-capable GPU (recommended)
- Conda environment with ESM dependencies

### Installation

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Activate the ESM environment: `conda activate esm`

## Contributors

CS182 Final Project Team

## License

This project is developed for educational purposes as part of CS182 coursework.
