# CS182-Final-Project

This repository contains the implementation for our CS182 (Deep Learning) final project on protein analysis and prediction.

## Project Overview

This project focuses on developing machine learning models for protein analysis, including protein function prediction, protein-protein interaction prediction, and other protein-related tasks using state-of-the-art embeddings and neural network architectures.

## TCN Encoder Component

### Overview
One of the key components of this project is a **Temporal Convolutional Network (TCN) encoder** designed to transform variable-length ESM-C protein embeddings into fixed-length representations suitable for downstream tasks.

### Purpose
- **Input**: Variable-length ESM-C embeddings `(L × 960)` where L varies by protein
- **Output**: Fixed-length protein vectors `(512 dimensions)`
- **Goal**: Enable efficient batch processing and consistent input sizes for downstream models

### Architecture
The TCN encoder uses:
- **5 dilated convolutional blocks** with dilations [1, 2, 4, 8, 16] for capturing long-range dependencies
- **Residual connections** with LayerNorm for stable training
- **Attention pooling** to convert variable-length sequences to fixed-length vectors
- **Configurable output dimensions** (default: 512)

### Implementation
- **Location**: `src/tcn_encoder/`
- **Main Files**:
  - `encoder.py`: Core TCN implementation
  - `model.py`: Task-specific model wrappers
  - `dataset.py`: Data handling utilities
  - `train.py`: Training infrastructure

### Data Processing Status
✅ **Transformation Complete**: All 12,026 proteins in the dataset have been successfully transformed from variable-length embeddings to fixed 512-dimensional vectors.

- **Input Data**: `data/full_dataset/embeddings/embeddings_merged.pkl` (~11GB)
- **Output Data**: `data/full_dataset/embeddings/embeddings_tcn_512.pkl` (~27MB)
- **Transformation Script**: `transform_full_dataset.py`

### Usage
The TCN encoder can be used for various protein-related tasks:
- Single protein classification/regression
- Protein-protein interaction prediction
- Multi-task learning scenarios

For detailed usage instructions and API documentation, see `src/tcn_encoder/README.md`.

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

### Quick Start
```python
# Load transformed embeddings
import pickle
with open('data/full_dataset/embeddings/embeddings_tcn_512.pkl', 'rb') as f:
    embeddings = pickle.load(f)

# Use TCN encoder for new data
from src.tcn_encoder import ProteinTCNEncoder
encoder = ProteinTCNEncoder(d_out=512)
# ... (see src/tcn_encoder/README.md for details)
```

## Contributors
CS182 Final Project Team

## License
This project is developed for educational purposes as part of CS182 coursework.