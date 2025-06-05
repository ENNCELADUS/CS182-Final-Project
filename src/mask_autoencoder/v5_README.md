# V5 PPI Classifier - Cross-Attention Architecture

This directory contains the implementation of the V5 Protein-Protein Interaction (PPI) classifier, which uses a cross-attention architecture built on top of a pre-trained Masked Autoencoder (MAE).

## 🏗️ Architecture Overview

The V5 model implements the cross-attention classifier design as specified in the build manual:

### Core Components

1. **MAEEncoder** (12 layers, 768-dim)

   - Adapted from v2.py TransformerMAE
   - Supports loading pre-trained weights
   - Optional weight freezing for transfer learning
2. **InteractionCrossAttention** (2 layers, 8 heads)

   - Learnable `[CLS_int]` token as query
   - Keys/values from concatenated protein tokens
   - 768-dimensional throughout
3. **InteractionMLPHead** (768 → 512 → 128 → 1)

   - LayerNorm after each linear layer
   - GELU activation and dropout
   - Xavier uniform initialization
4. **PPIClassifier** (Main glue module)

   - Combines all components
   - Handles patchification (47 tokens + CLS)
   - Supports LoRA fine-tuning

## 📁 Files

- **`v5.py`** - Main model architecture
- **`v5_train_eval.py`** - Training and evaluation script
- **`test_v5_architecture.py`** - Architecture validation tests
- **`v5_README.md`** - This documentation

## 🚀 Quick Start

### 1. Test the Architecture

```bash
python test_v5_architecture.py
```

This will run comprehensive tests to verify:

- Model creation and parameter counting
- Forward pass with dummy data
- Real data loading and processing
- Gradient flow and training steps
- Frozen vs unfrozen encoder behavior
- Various sequence lengths

### 2. Train the Model

```bash
python v5_train_eval.py
```

This will:

- Load data using v4_1 data loading functions
- Create the V5 PPI classifier
- Train for 50 epochs with early stopping
- Evaluate on test sets
- Save models and generate plots

## ⚙️ Configuration

### Model Configuration

```python
# Create classifier with default settings
model = create_ppi_classifier(
    mae_checkpoint_path=None,  # Path to pre-trained MAE weights
    freeze_encoder=True,       # Freeze encoder for transfer learning
    use_lora=False            # Use LoRA for efficient fine-tuning
)
```

### Training Configuration

Edit the `config` dictionary in `v5_train_eval.py`:

```python
config = {
    'batch_size': 16,          # Batch size for training
    'learning_rate': 1e-4,     # Learning rate
    'num_epochs': 50,          # Maximum epochs
    'weight_decay': 0.01,      # Weight decay
    'use_scheduler': True,     # Use OneCycleLR scheduler
    'early_stopping': True,    # Enable early stopping
    'patience': 10,            # Early stopping patience
  
    # Model specific
    'mae_checkpoint_path': None,  # Pre-trained MAE path
    'freeze_encoder': True,       # Freeze encoder weights
    'use_lora': False,           # Use LoRA adaptation
}
```

## 📊 Model Statistics

From the architecture test:

- **Total Parameters**: 101,587,201 (~102M)
- **Trainable Parameters** (frozen encoder): 14,638,849 (~15M)
- **Frozen Parameters**: 86,948,352 (~87M)

## 🔄 Data Flow

1. **Input**: Protein embeddings (B, L, 960)
2. **Patchification**: Convert to patches + CLS token (B, 47+1, 768)
3. **MAE Encoding**: Transform through 12-layer encoder (B, 48, 768)
4. **Cross-Attention**: Interaction modeling with CLS_int token (B, 768)
5. **Classification**: MLP head produces logits (B, 1)

## 🎯 Key Features

### Memory Efficient

- Processes sequences up to ~1500 residues
- Patchifies to fixed 47 tokens + CLS
- Gradient checkpointing available

### Transfer Learning Ready

- Load pre-trained MAE weights
- Optional encoder freezing
- LoRA support for efficient fine-tuning

### Flexible Training

- Comprehensive evaluation metrics
- Early stopping and checkpointing
- Visualization and logging

### Robust Architecture

- Handles variable sequence lengths
- Gradient clipping for stability
- Proper weight initialization

## 📈 Expected Performance

Based on the architecture design:

- **Memory Usage**: ~40MB for cross-attention layers
- **Training Speed**: Efficient with frozen encoder
- **Convergence**: Expected within 20-30 epochs

## 🔧 Advanced Usage

### Loading Pre-trained MAE Weights

```python
model = create_ppi_classifier(
    mae_checkpoint_path="path/to/mae_pretrained.pth",
    freeze_encoder=True
)
```

### Using LoRA for Fine-tuning

```python
model = create_ppi_classifier(
    mae_checkpoint_path="path/to/mae_pretrained.pth",
    freeze_encoder=False,
    use_lora=True
)
```

### Extract Interaction Embeddings

```python
# Get 768-dimensional interaction vectors
interaction_vectors = model.get_interaction_embeddings(emb_a, emb_b)
```

### Custom Training Loop

```python
from v5_train_eval import train_epoch, validate_epoch, collate_fn_v5

# Create data loaders with custom collate function
train_loader = DataLoader(dataset, collate_fn=collate_fn_v5, ...)

# Training loop
for epoch in range(num_epochs):
    train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
    val_metrics = validate_epoch(model, val_loader, criterion, device)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce batch size or sequence length
2. **Import Errors**: Ensure v4_1.py is in the same directory
3. **Data Not Found**: Check data paths in v4_1.py load_data()

### Performance Tips

1. **Use frozen encoder** for faster training
2. **Enable mixed precision** for memory efficiency
3. **Use gradient accumulation** for larger effective batch sizes

## 📚 References

- Built following the "from-repo to new model" build manual
- Uses data loading from v4_1.py
- Implements cross-attention design for PPI prediction
- Compatible with ESM protein embeddings (960-dim)

## 🧪 Testing

The test suite covers:

- ✅ Model creation and parameter counting
- ✅ Forward pass with various input sizes
- ✅ Real data loading and processing
- ✅ Gradient flow and optimization
- ✅ Frozen vs unfrozen behavior
- ✅ Different sequence lengths (10-1200 residues)

Run tests regularly to ensure architecture integrity:

```bash
python test_v5_architecture.py
```
