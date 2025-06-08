# Improving PPI Prediction with ESMC-derived Features

## Overview

This repository contains code and resources for a machine learning project focused on **protein feature extraction** and **supervised learning** models. The project is divided into several key components, such as data preprocessing, embedding extraction, model training, and evaluation. The primary goal is to use different supervised learning algorithms to process and predict protein-related data.

## Directory Structure

```
code/
├── README.md                     # This file - project documentation
├── requirements.txt              # Python package dependencies for the entire project
├── data/                         # Contains the data used for training and validation
│   └── benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.pkl
├── data_process/                 # Scripts for data preprocessing
│   ├── separate_dataset.ipynb   # Dataset separation and processing
│   └── medium_dataset.ipynb     # Medium dataset processing
├── extract_embeddings/           # Scripts for extracting embeddings
│   ├── embed.ipynb              # Main embedding extraction
│   └── embeddings_standardize.ipynb # Embedding standardization
├── MAE/                          # Masked AutoEncoder models and training scripts
│   ├── logs/                     # Training logs and loss curves
│   │   ├── v2_mae_train_20250528-174157.json
│   │   ├── v2_loss_curve_20250528-174157.png
│   │   ├── v3_mae_pairs_train_20250530-151013.json
│   │   └── v3_loss_curve_pairs_20250530-151013.png
│   ├── mae_v1.ipynb              # MAE model version 1 (Jupyter Notebook)
│   ├── mae_v2.py                 # MAE model version 2
│   ├── mae_v3.py                 # MAE model version 3
│   ├── generate_compressed_fearture2.py # Feature compression script v2 (note: typo in filename)
│   └── generate_compressed_feature3.py  # Feature compression script v3
├── logs/                         # Top-level logs and visualization files of DNN_v4
│   ├── fixed_model_comparison.png
│   └── fixed_model_metrics_comparison.png
├── models/                       # Top-level saved model files of DNN_v4
│   └── DNN_v4.pth               # Trained DNN model
└── supervised_learning/          # Supervised learning model scripts
    ├── v4.py                     # Data preparation for DNN_v4.py (moved from MAE/)
    ├── DNN_v4.py                 # Deep neural network model definition, training, and test (moved from MAE/)
    ├── L2_cos.ipynb              # L2 norm & cosine similarity model
    ├── logistic_regression_medium.ipynb # Logistic regression on medium dataset
    ├── XGB_cls.ipynb             # XGBoost classifier using CLS token
    ├── Xgboost_average_pooling.ipynb # XGBoost with average pooling
    ├── Xgboost_MAE_v1.ipynb      # XGBoost with MAE version 1
    └── Xgboost_MAE_v2.ipynb      # XGBoost with MAE version 2
```

## Installation

1. Install the required dependencies:

   ```bash
   cd code
   pip install -r requirements.txt
   ```

   Note: This project requires Python 3.10+ and CUDA support for GPU acceleration. Some packages (like `flash-attn` and NVIDIA CUDA libraries) are optional and will be skipped if not compatible with your system.

## Data

### Original Data Source

The initial dataset `data/benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.pkl` is obtained from the official GitHub repository of the paper:

**"Pitfalls of machine learning models for protein–protein interaction networks"** by Loïc Lannelongue and Michael Inouye  
- Paper: [Bioinformatics, 2024](https://doi.org/10.1093/bioinformatics/btae012)  
- GitHub Repository: [https://github.com/Llannelongue/B4PPI](https://github.com/Llannelongue/B4PPI)

This dataset is part of the B4PPI (Benchmarking Pipeline for the Prediction of Protein-Protein Interactions) framework, which provides curated benchmarking datasets for human and yeast protein-protein interactions.

The data processing pipeline creates several datasets and embeddings through a sequential workflow:

### 1. Data Processing Pipeline

**Run jupyter notebooks in the `data_process/`,`extract_embeddings/` directory to process the data.**

**Step 1: Dataset Separation** (`data_process/separate_dataset.ipynb`)
- Processes the raw dataset: `data/benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.pkl`
- Creates train/validation/test splits and saves to `data/full_dataset/`:
  * `train_data.pkl`: The training dataset for the model
  * `validation_data.pkl`: The validation dataset for model evaluation
  * `test1_data.pkl`, `test2_data.pkl`: Test datasets for model performance evaluation

**Step 2: Medium Dataset Creation** (`data_process/medium_dataset.ipynb`)
- Creates smaller, more manageable datasets for experimentation
- Saves to `data/medium_set/` with balanced sampling

**Step 3: Embedding Extraction** (`extract_embeddings/embed.ipynb`)
- Extracts ESM-C embeddings for all unique proteins
- Saves embeddings to `data/full_dataset/embeddings/embeddings_merged.pkl`

**Step 4: Embedding Standardization** (`extract_embeddings/embeddings_standardize.ipynb`)
- Merges batch embeddings and standardizes them (mean=0, std=1)
- Creates final standardized embeddings:
  * `embeddings_standardized.pkl`: Standardized embeddings for protein data (dictionary format)

### 2. Compressed Features Generation

The MAE models generate compressed protein features:

**Generated by `MAE/generate_compressed_fearture2.py`:**
- `data/full_dataset/compressed_protein_features2.pkl`: Compressed features using MAE v2 model

**Generated by `MAE/generate_compressed_feature3.py`:**
- `data/full_dataset/train_data_with_mae_embeddings_*.pkl`: Training data with MAE embeddings
- `data/full_dataset/val_data_with_mae_embeddings_*.pkl`: Validation data with MAE embeddings  
- `data/full_dataset/mae_embeddings_*.npz`: Numpy arrays of compressed embeddings

### 3. Expected Data Structure

After running the complete pipeline, your data directory should contain:
```
data/
├── benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.pkl  # Raw input data
├── full_dataset/                    # Full-scale processed datasets
│   ├── train_data.pkl
│   ├── validation_data.pkl
│   ├── test1_data.pkl
│   ├── test2_data.pkl
│   ├── unique_proteins.pkl
│   ├── compressed_protein_features2.pkl # MAE v2 compressed features
│   └── embeddings/
│       ├── embeddings_merged.pkl
│       ├── embeddings_standardized.pkl
├── medium_set/                      # Smaller datasets for quick experimentation
│   ├── train_data.pkl
│   ├── validation_data.pkl
│   ├── test1_data.pkl
│   └── test2_data.pkl

```

## MAE (masked autoencoder)

The final adopted version is  `mae_v2.py` for feature extraction and `DNN_v4.py` for supervised learning.

* `mae_v1.py`: the initial model version.
* `mae_v2.py`: Adding "padding_start", only mask meaningful positions. Using data of mean=0, std=1. 50% mask ratio.
* `mae_v3.py` : First splicing data, then compressing. The result is not good, you don't need to use it.

**Note:** `DNN_v4.py` and `v4.py` have been moved to `supervised_learning/` folder as they're primarily used for supervised learning rather than unsupervised MAE training. The trained model `DNN_v4.pth` is stored in the top-level `models/` directory.

Run `mae_v2.py` and `generate_compressed_feature2.py` to get `compressed_protein_features2.pkl`.

```bash
   python MAE/mae_v2.py
   python MAE/generate_compressed_feature2.py
```


## Model Description

This project includes multiple machine learning models to process protein data:

### Supervised Learning Models

Located in the `supervised_learning/` folder:

* **`DNN_v4.py`**: Deep neural network model for protein-protein interaction prediction (moved from MAE folder)
* **`v4.py`**: Data preparation script for DNN_v4.py (moved from MAE folder)
* **Jupyter Notebooks**: Various supervised learning approaches including:
  - XGBoost variants (with different feature types)
  - Logistic regression
  - L2 norm & cosine similarity models

**Model Storage**: Trained models are saved in the top-level `models/` directory, and training logs/plots are stored in the `logs/` directory.

Run these jupyter notebooks and the DNN script in the `supervised_learning/` folder to see results.
