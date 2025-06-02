# CS182-Final-Project

This repository contains the implementation for our CS182 (Deep Learning) final project on protein analysis and prediction.

## Project Overview

This project focuses on developing machine learning models for protein analysis, including protein function prediction, protein-protein interaction prediction, and other protein-related tasks using state-of-the-art embeddings and neural network architectures.

## Project Structure

```
CS182-Final-Project/
├── src/
│   └── tcn_encoder/          # TCN encoder implementation
├── data/
│   ├── full_dataset/         # Main dataset
│   │   └── embeddings/       # Protein embeddings (original & transformed)
│   ├── medium_set/           # Medium-sized dataset
│   └── small_set/            # Small dataset for testing
├── experiments/              # Experimental results and notebooks
├── results/                  # Model outputs and evaluations
├── docs/                     # Documentation
├── transform_full_dataset.py # Main transformation script
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