# Protein TCN Encoder

A PyTorch implementation of a **dilated 1D Temporal Convolutional Network (TCN)** for encoding protein sequences from ESM-C embeddings into fixed-length vectors.

## Overview

This module converts variable-length ESM-C residue embeddings `(L × 960)` into fixed-length protein vectors `(512)` through:

1. **Dilated 1D TCN**: 5 residual blocks with dilations [1, 2, 4, 8, 16]
2. **Attention Pooling**: Learnable query vector for sequence-to-vector conversion
3. **Task Heads**: Flexible heads for classification, regression, and protein-pair tasks

## File Structure and Descriptions

### Core Implementation Files

#### `encoder.py`
- **Purpose**: Core TCN encoder implementation
- **Main Class**: `ProteinTCNEncoder`
- **Function**: Transforms variable-length embeddings `(B, L, 960)` → fixed-length vectors `(B, 512)`
- **Key Features**:
  - 5 dilated convolutional blocks with residual connections
  - Attention pooling mechanism
  - Configurable output dimensions

#### `model.py`
- **Purpose**: Task-specific model wrappers and heads
- **Main Classes**:
  - `SingleProteinModel`: For protein-level tasks (classification/regression)
  - `ProteinPairModel`: For protein-protein interaction tasks
  - `MultiTaskModel`: For multi-task learning scenarios
- **Function**: Combines TCN encoder with task-specific prediction heads

#### `dataset.py`
- **Purpose**: Data handling and preprocessing utilities
- **Main Classes**:
  - `ProteinDataset`: PyTorch dataset for protein embeddings
  - `ProteinPairDataset`: Dataset for protein pair tasks
- **Main Functions**:
  - `collate_single_protein`: Batch collation with padding and masking
  - `collate_protein_pairs`: Collation for protein pair data
  - `create_dataloader`: Convenient dataloader creation
- **Features**: Automatic padding, masking, and data augmentation

#### `train.py`
- **Purpose**: Training utilities and scripts
- **Main Classes**:
  - `VanillaTrainer`: Basic PyTorch training loop
  - `LightningTrainer`: PyTorch Lightning wrapper
- **Function**: Provides training infrastructure with logging, checkpointing, and evaluation

#### `__init__.py`
- **Purpose**: Package initialization and exports
- **Function**: Exposes main classes and functions for easy importing
- **Exports**: All major classes and utility functions

## Data Storage and Locations

### Input Data
- **Original ESM-C Embeddings**: `data/full_dataset/embeddings/embeddings_merged.pkl`
  - Format: Dictionary `{protein_id: torch.Tensor(L, 960)}`
  - Size: ~11GB
  - Contains: 12,026 proteins with variable-length embeddings

### Transformed Data
- **Fixed-Length Embeddings**: `data/full_dataset/embeddings/embeddings_tcn_512.pkl`
  - Format: Dictionary `{protein_id: torch.Tensor(512)}`
  - Size: ~27MB
  - Contains: 12,026 proteins with fixed 512-dimensional vectors
  - **Status**: ✅ Transformation complete

### Other Data Files
- **Embeddings Manifest**: `data/full_dataset/embeddings/embeddings_manifest.pkl`
  - Contains metadata about the embeddings
- **Training Data**: Various `train_data_*.pkl` and `validation_data_*.pkl` files
  - Preprocessed datasets for specific tasks
- **Benchmark Data**: `data/benchmarkingGS_v1-0_similarityMeasure_sequence_v3-1.pkl`
  - Benchmark dataset for evaluation

## Quick Usage

### Basic Embedding Transformation

```python
import torch
from src.tcn_encoder import ProteinTCNEncoder

# Load the encoder
encoder = ProteinTCNEncoder(d_out=512)
encoder.eval()

# Transform embeddings
embeddings = torch.randn(batch_size, seq_len, 960)  # ESM-C embeddings
mask = torch.ones(batch_size, seq_len, dtype=torch.bool)  # Sequence mask

with torch.no_grad():
    protein_vectors = encoder(embeddings, mask)  # (batch_size, 512)
```

### Loading Transformed Data

```python
import pickle

# Load pre-transformed embeddings
with open('data/full_dataset/embeddings/embeddings_tcn_512.pkl', 'rb') as f:
    transformed_embeddings = pickle.load(f)

# Access specific protein
protein_id = list(transformed_embeddings.keys())[0]
protein_vector = transformed_embeddings[protein_id]  # torch.Tensor(512)
```

### Single Protein Classification

```python
from src.tcn_encoder import SingleProteinModel

# Create model for binary classification
model = SingleProteinModel(
    n_classes=2,
    task_type='binary_classification'
)

# Forward pass
logits = model(embeddings, mask)  # (batch_size, 2)
```

### Protein Pair Interaction

```python
from src.tcn_encoder import ProteinPairModel

# Create PPI model
ppi_model = ProteinPairModel(
    n_classes=1,
    task_type='binary_classification'
)

# Predict interaction between two proteins
logits = ppi_model(embA, maskA, embB, maskB)  # (batch_size, 1)
```

## Data Transformation Script

The main transformation script is located in the project root:

- **Script**: `transform_full_dataset.py`
- **Purpose**: Transforms all embeddings from variable-length to fixed 512-dimensional vectors
- **Status**: ✅ Completed successfully
- **Usage**: `python transform_full_dataset.py` (requires `conda activate esm`)

## Architecture Details

```
ESM-C embeddings (B, L, 960)
           │  (permute to B, 960, L)
           ▼
TCN Block 1 (dilation=1)    ─┐
TCN Block 2 (dilation=2)     │ 5 residual blocks
TCN Block 3 (dilation=4)     │ with LayerNorm
TCN Block 4 (dilation=8)     │ and dropout
TCN Block 5 (dilation=16)   ─┘
           │  (B, 512, L)
           ▼
Attention Pooling           ─ learnable query vector
           │  (B, 512)
           ▼
Task-specific heads         ─ MLP/bilinear for downstream tasks
```

## Performance Characteristics

- **Transformation Speed**: ~1,000 proteins/second (GPU)
- **Memory Usage**: ~16GB GPU memory for batch processing
- **Model Size**: ~2.5M parameters
- **Output Quality**: Fixed 512-dimensional representations suitable for downstream tasks

## Dependencies

See `requirements.txt` in the project root for full dependencies. Key requirements:
- PyTorch >= 1.9.0
- NumPy
- tqdm (for progress bars)
- pickle (for data serialization)

## Notes

- All embeddings are processed in float32 precision
- Batch processing is used for memory efficiency
- GPU acceleration is automatically detected and used when available
- The transformation preserves protein identifiers for easy mapping
