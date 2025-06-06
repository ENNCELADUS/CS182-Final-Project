# V5 MAE Pre-training Script

This script (`v5_mae_train.py`) trains **only the MAE (Masked Autoencoder) component** of the v5 architecture for self-supervised pre-training on protein sequences.

## 🎯 Purpose

The MAE pre-training provides:

- **Self-supervised representation learning** on protein embeddings
- **Pre-trained encoder weights** for the v5 PPI classifier
- **Better initialization** than random weights for downstream tasks

## 🏗️ Architecture

The V5MAE model includes:

### Encoder (from v5.py)

- **12 transformer layers** (768-dim, 12 heads)
- **Patchification** to 47 patches + CLS token
- **Positional embeddings**
- **Same architecture** as the encoder in v5 PPI classifier

### Decoder (MAE-specific)

- **8 transformer layers** (768-dim, 16 heads)
- **Separate positional embeddings**
- **Mask tokens** for masked positions
- **Reconstruction head** (768 → 960 dim)

### Key Features

- **75% masking ratio** (following MAE paper)
- **Reconstruction loss** only on masked patches
- **Handles variable sequence lengths**
- **Compatible with v5 encoder architecture**

## 🚀 Usage

### 1. Run MAE Pre-training

```bash
cd src/v5
python v5_mae_train.py
```

### 2. Key Configuration Parameters

Edit the `config` dictionary in `main()`:

```python
config = {
    # Data parameters
    'batch_size': 4,        # Small due to large model size
    'max_len': 1500,        # Maximum sequence length
  
    # Model architecture (matches v5)
    'embed_dim': 768,       # Embedding dimension
    'encoder_layers': 12,   # Encoder layers (matches v5)
    'encoder_heads': 12,    # Encoder attention heads
    'decoder_layers': 8,    # Decoder layers
    'decoder_heads': 16,    # Decoder attention heads
    'ff_dim': 3072,        # Feed-forward dimension
  
    # MAE specific
    'mask_ratio': 0.75,     # 75% masking ratio
  
    # Training
    'num_epochs': 100,
    'learning_rate': 1.5e-4,
    'weight_decay': 0.05,
    'use_scheduler': True,
}
```

### 3. Use Pre-trained MAE in v5_train.py

After training, use the best checkpoint in your PPI classifier:

```python
# In v5_train.py config
config = {
    'mae_checkpoint_path': 'models/v5_mae_pretrain_20250101_120000/mae_checkpoint_epoch_45_best.pth',
    'freeze_encoder': True,  # Freeze the pre-trained encoder
    # ... other config
}
```

## 📊 Expected Output

### Training Progress

```
🧬 V5 MAE PRE-TRAINING
==================================================
Using device: cuda
CUDA device name: NVIDIA RTX 4090

📊 Loading protein embeddings...
Available proteins: 25,000
Training proteins: 25,000
Validation proteins: 2,500

🔧 Creating V5 MAE model...
Total parameters: 180,500,000
Trainable parameters: 180,500,000

🚀 Starting MAE pre-training for 100 epochs...
Mask ratio: 0.75

Epoch 1/100
------------------------------
Training MAE: 100%|██████████| 6250/6250 [32:15<00:00, 3.23it/s, loss=0.8234]
Validation: 100%|██████████| 625/625 [02:45<00:00, 3.78it/s, loss=0.7891]
Train Loss: 0.823400
Val Loss: 0.789100
🎉 New best validation loss: 0.789100
```

### Files Generated

```
models/v5_mae_pretrain_20250101_120000/
├── mae_checkpoint_epoch_1.pth
├── mae_checkpoint_epoch_2.pth
├── ...
└── mae_checkpoint_epoch_45_best.pth  # Use this for v5_train.py

logs/v5_mae_pretrain_20250101_120000/
├── mae_training_results.json
├── mae_training_curves.png
├── reconstruction_epoch_1.png
├── reconstruction_epoch_5.png
└── ...
```

## 📈 Monitoring Training

### 1. Training Curves

- **Loss curves**: Monitor reconstruction loss decrease
- **Learning rate**: Cosine annealing schedule
- **Saved to**: `logs/v5_mae_pretrain_*/mae_training_curves.png`

### 2. Reconstruction Samples

- **Visual examples** of original vs reconstructed sequences
- **Masked positions** highlighted in red
- **Generated every 5 epochs**
- **Saved to**: `logs/v5_mae_pretrain_*/reconstruction_epoch_*.png`

### 3. Metrics JSON

Complete training history saved to:

```json
{
  "config": {...},
  "history": {
    "train_loss": [0.823, 0.756, ...],
    "val_loss": [0.789, 0.723, ...],
    "learning_rate": [1.5e-4, 1.49e-4, ...]
  },
  "best_epoch": 45,
  "best_val_loss": 0.634
}
```

## 🔧 Advanced Configuration

### Memory Optimization

```python
# For limited GPU memory
config = {
    'batch_size': 2,        # Reduce batch size
    'encoder_layers': 8,    # Smaller encoder
    'decoder_layers': 4,    # Smaller decoder
    'ff_dim': 2048,        # Smaller FF dimension
}
```

### Fast Prototyping

```python
# For quick testing
config = {
    'num_epochs': 10,
    'mask_ratio': 0.5,      # Less aggressive masking
    'max_len': 500,         # Shorter sequences
}
```

## 📋 Model Statistics

### Default Configuration (768-dim, 12 layers)

- **Total Parameters**: ~180M
- **Encoder Parameters**: ~87M (matches v5)
- **Decoder Parameters**: ~93M (MAE-specific)
- **Memory Usage**: ~40GB (batch_size=4)
- **Training Time**: ~8 hours/epoch (RTX 4090)

### Parameter Breakdown

```
Encoder Components:
├── PatchEmbedding: 737,280 parameters
├── Position Embeddings: 1,155,072 parameters  
├── CLS Token: 768 parameters
├── 12 Transformer Layers: 85,055,232 parameters
└── Layer Norm: 1,536 parameters

Decoder Components:
├── Decoder Embedding: 589,824 parameters
├── Mask Token: 768 parameters
├── Decoder Pos Embeddings: 36,864 parameters
├── 8 Transformer Layers: 92,536,832 parameters
├── Decoder Norm: 1,536 parameters
└── Reconstruction Head: 737,280 parameters
```

## 🎯 Expected Performance

### Training Progression

- **Epoch 1**: Loss ~0.85 (random initialization)
- **Epoch 10**: Loss ~0.75 (basic patterns learned)
- **Epoch 30**: Loss ~0.65 (good reconstruction)
- **Epoch 50+**: Loss ~0.60 (convergence)

### Quality Indicators

- **Reconstruction quality**: Visual similarity in plots
- **Loss convergence**: Smooth decrease over epochs
- **Validation gap**: Train/val loss difference < 0.05

## 🔄 Integration with v5_train.py

### 1. Update Configuration

```python
# In v5_train.py
config = {
    'mae_checkpoint_path': 'path/to/mae_checkpoint_epoch_X_best.pth',
    'freeze_encoder': True,  # Recommended for transfer learning
}
```

### 2. Expected Improvements

- **Faster convergence** of PPI classifier
- **Better validation performance**
- **More stable training**
- **Higher final accuracy**

### 3. Parameter Counts (with pre-trained MAE)

```
v5 PPI Classifier with Pre-trained MAE:
├── Frozen Encoder: 86,948,352 parameters (pre-trained)
├── Cross-attention: 14,338,560 parameters (trainable)
└── MLP Head: 300,289 parameters (trainable)

Total: 101,587,201 parameters
Trainable: 14,638,849 parameters (14.4%)
```

## 🐛 Troubleshooting

### Common Issues

1. **CUDA OOM Error**

   ```python
   config['batch_size'] = 2  # Reduce batch size
   config['max_len'] = 1000  # Shorter sequences
   ```
2. **Loss Not Decreasing**

   - Check learning rate (try 1e-4 to 5e-4)
   - Verify data loading (check protein_embeddings dict)
   - Reduce mask_ratio to 0.5 initially
3. **Import Errors**

   ```bash
   # Ensure you're in the right directory
   cd src/v5
   # Check v5.py and v4/v4_1.py exist
   ls v5.py ../v4/v4_1.py
   ```

### Performance Tips

1. **Use mixed precision** (add to training loop):

   ```python
   from torch.cuda.amp import GradScaler, autocast
   scaler = GradScaler()

   with autocast():
       loss, pred, mask = model(embeddings, lengths=lengths)
   ```
2. **Gradient accumulation** for larger effective batch size:

   ```python
   accumulation_steps = 4
   if (batch_idx + 1) % accumulation_steps == 0:
       optimizer.step()
       optimizer.zero_grad()
   ```

## 🔗 Next Steps

1. **Train MAE**: Run `python v5_mae_train.py`
2. **Monitor progress**: Check logs and plots
3. **Select best model**: Use lowest validation loss epoch
4. **Transfer to PPI**: Update `v5_train.py` with checkpoint path
5. **Train PPI classifier**: Run `python v5_train.py`

The pre-trained MAE should significantly improve your PPI classification performance compared to training from scratch!
