#!/usr/bin/env python3
"""
Training script for V5 MAE (Masked Autoencoder) pre-training
This script trains only the MAE component with reconstruction loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
import os
import json
import pickle
from datetime import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error

# Import base components from v5
from v5 import MAEEncoder, PatchEmbedding

# Import data loading from v4.1
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'v4'))
from v4_1 import load_data, ProteinPairDataset, detect_column_names

class V5MAE(nn.Module):
    """
    Complete MAE for v5 architecture with encoder and decoder
    """
    def __init__(self,
                 input_dim=960,
                 embed_dim=768,
                 num_layers=12,
                 nhead=12,
                 ff_dim=3072,
                 max_len=1502,
                 dropout=0.1,
                 mask_ratio=0.75,
                 decoder_layers=8,
                 decoder_heads=16):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len
        self.mask_ratio = mask_ratio
        self.target_length = 47  # Fixed patch length as in v5
        
        # Encoder (from v5)
        self.encoder = MAEEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            nhead=nhead,
            ff_dim=ff_dim,
            max_len=max_len,
            dropout=dropout
        )
        
        # Decoder specific components
        self.decoder_embed = nn.Linear(embed_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Decoder positional embedding (separate from encoder)
        self.decoder_pos_embed = nn.Parameter(torch.randn(1, self.target_length + 1, embed_dim) * 0.02)
        
        # Decoder transformer
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=decoder_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_layers)
        
        # Decoder norm and head
        self.decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder_head = nn.Linear(embed_dim, input_dim)  # Reconstruct to original 960-dim
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)
        
        # Initialize decoder components
        nn.init.xavier_uniform_(self.decoder_embed.weight)
        nn.init.xavier_uniform_(self.decoder_head.weight)
        if self.decoder_embed.bias is not None:
            nn.init.zeros_(self.decoder_embed.bias)
        if self.decoder_head.bias is not None:
            nn.init.zeros_(self.decoder_head.bias)
    
    def random_masking(self, x, mask_ratio, lengths=None):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        
        Args:
            x: (B, L, D) - input sequence (already patchified)
            mask_ratio: ratio of patches to mask
            lengths: (B,) - actual sequence lengths (if provided)
        
        Returns:
            x_masked: (B, L_keep, D) - visible patches
            mask: (B, L) - 0 is keep, 1 is remove
            ids_restore: (B, L) - indices to restore original order
        """
        B, L, D = x.shape  # L includes CLS token (should be target_length + 1)
        
        # Exclude CLS token from masking (first token)
        len_keep = int((L - 1) * (1 - mask_ratio)) + 1  # +1 for CLS token
        
        # Generate random noise for each sample
        noise = torch.rand(B, L - 1, device=x.device)  # Exclude CLS from noise
        
        # If lengths provided, mask out padding positions
        if lengths is not None:
            # Create mask for valid positions (exclude CLS token)
            valid_mask = torch.arange(L - 1, device=x.device).unsqueeze(0) < (lengths.unsqueeze(1) - 1)
            noise = noise.masked_fill(~valid_mask, float('inf'))  # Don't mask padding
        
        # Sort noise to identify masked and unmasked
        ids_shuffle = torch.argsort(noise, dim=1)  # Ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        
        # Generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([B, L - 1], device=x.device)
        mask[:, :len_keep - 1] = 0  # -1 because we exclude CLS
        
        # Unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        
        # Add CLS token back to mask (CLS is never masked)
        cls_mask = torch.zeros(B, 1, device=x.device)
        mask = torch.cat([cls_mask, mask], dim=1)
        
        # Keep the unmasked tokens + CLS
        ids_keep_with_cls = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=x.device), 
                                      ids_shuffle[:, :len_keep - 1] + 1], dim=1)
        x_masked = torch.gather(x, dim=1, index=ids_keep_with_cls.unsqueeze(-1).repeat(1, 1, D))
        
        # Restore ids should include CLS token
        ids_restore_with_cls = torch.cat([torch.zeros(B, 1, dtype=torch.long, device=x.device),
                                         ids_restore + 1], dim=1)
        
        return x_masked, mask, ids_restore_with_cls
    
    def forward_encoder(self, x, mask_ratio=None, lengths=None):
        """
        Forward pass through encoder with masking
        
        Args:
            x: (B, L, 960) - protein embeddings
            mask_ratio: masking ratio (default uses self.mask_ratio)
            lengths: (B,) - actual sequence lengths
            
        Returns:
            x_encoded: (B, L_visible, D) - encoded visible patches
            mask: (B, L) - mask used
            ids_restore: (B, L) - ids to restore patches
        """
        if mask_ratio is None:
            mask_ratio = self.mask_ratio
        
        # Patchify input
        x_patches = self.encoder.patchify(x, target_length=self.target_length)  # (B, T+1, 768)
        
        # Random masking
        x_masked, mask, ids_restore = self.random_masking(x_patches, mask_ratio, lengths)
        
        # Apply encoder (only on visible patches)
        x_encoded = self.encoder(x_masked)
        
        return x_encoded, mask, ids_restore
    
    def forward_decoder(self, x_encoded, ids_restore):
        """
        Forward pass through decoder
        
        Args:
            x_encoded: (B, L_visible, D) - encoded visible patches
            ids_restore: (B, L) - indices to restore original order
        
        Returns:
            x_decoded: (B, L, input_dim) - reconstructed sequence
        """
        # Embed tokens
        x = self.decoder_embed(x_encoded)
        
        # Append mask tokens to sequence
        B, L_visible, D = x.shape
        L_full = self.target_length + 1  # Including CLS token
        
        # Create mask tokens
        mask_tokens = self.mask_token.repeat(B, L_full - L_visible, 1)
        
        # Concatenate visible tokens and mask tokens
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, L_full, D)
        
        # Unshuffle to restore original order
        x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
        
        # Add positional embedding
        x_full = x_full + self.decoder_pos_embed
        
        # Apply decoder
        x_decoded = self.decoder(x_full)
        x_decoded = self.decoder_norm(x_decoded)
        
        # Remove CLS token and predict pixel values
        x_decoded = x_decoded[:, 1:]  # Remove CLS token
        x_decoded = self.decoder_head(x_decoded)  # (B, T, 960)
        
        return x_decoded
    
    def forward_loss(self, imgs, pred, mask):
        """
        Compute reconstruction loss
        
        Args:
            imgs: (B, L, 960) - original images (patchified without CLS)
            pred: (B, L, 960) - reconstructed images  
            mask: (B, L+1) - mask (includes CLS position)
        
        Returns:
            loss: reconstruction loss
        """
        # Remove CLS token from mask
        mask = mask[:, 1:]  # (B, L)
        
        # Truncate/pad imgs to match prediction length
        B, L_pred, D = pred.shape
        B_img, L_img, D_img = imgs.shape
        
        if L_img > L_pred:
            imgs = imgs[:, :L_pred]
        elif L_img < L_pred:
            padding = torch.zeros(B_img, L_pred - L_img, D_img, device=imgs.device, dtype=imgs.dtype)
            imgs = torch.cat([imgs, padding], dim=1)
        
        # Compute loss only on masked patches
        loss = F.mse_loss(pred, imgs, reduction='none')  # (B, L, 960)
        loss = loss.mean(dim=-1)  # (B, L) - average over features
        
        # Apply mask (only compute loss on masked patches)
        loss = loss * mask
        loss_sum = loss.sum()
        mask_sum = mask.sum()
        
        return loss_sum / (mask_sum + 1e-8)  # Avoid division by zero
    
    def forward(self, imgs, mask_ratio=None, lengths=None):
        """
        Full forward pass for training
        
        Args:
            imgs: (B, L, 960) - input protein embeddings
            mask_ratio: masking ratio
            lengths: (B,) - sequence lengths
            
        Returns:
            loss: reconstruction loss
            pred: predictions
            mask: mask used
        """
        # Get encoder output and projection to original space
        imgs_patches = self.encoder.patchify(imgs, target_length=self.target_length)
        imgs_for_loss = imgs_patches[:, 1:]  # Remove CLS token for loss computation
        imgs_for_loss = self.encoder.patch_embed.projection.weight.T @ imgs_for_loss.transpose(-1, -2)
        imgs_for_loss = imgs_for_loss.transpose(-1, -2)  # Back to (B, L, 960)
        
        # Forward encoder
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio, lengths)
        
        # Forward decoder
        pred = self.forward_decoder(latent, ids_restore)
        
        # Compute loss
        loss = self.forward_loss(imgs_for_loss, pred, mask)
        
        return loss, pred, mask

class ProteinMAEDataset(Dataset):
    """Dataset for MAE pre-training on individual proteins"""
    def __init__(self, protein_embeddings, max_len=1502):
        """
        Args:
            protein_embeddings: Dict mapping protein IDs to embeddings
            max_len: Maximum sequence length
        """
        self.protein_ids = list(protein_embeddings.keys())
        self.protein_embeddings = protein_embeddings
        self.max_len = max_len
    
    def __len__(self):
        return len(self.protein_ids)
    
    def __getitem__(self, idx):
        protein_id = self.protein_ids[idx]
        embedding = self.protein_embeddings[protein_id]  # (L, 960)
        
        # Convert to tensor if needed
        if not isinstance(embedding, torch.Tensor):
            embedding = torch.tensor(embedding, dtype=torch.float32)
        
        seq_len = embedding.shape[0]
        
        # Truncate if too long
        if seq_len > self.max_len:
            embedding = embedding[:self.max_len]
            seq_len = self.max_len
        
        return {
            'embedding': embedding,  # (L, 960)
            'length': seq_len,
            'protein_id': protein_id
        }

def collate_fn_mae(batch):
    """Collate function for MAE training"""
    embeddings = [item['embedding'] for item in batch]
    lengths = [item['length'] for item in batch]
    protein_ids = [item['protein_id'] for item in batch]
    
    # Pad to same length
    max_len = max(lengths)
    batch_size = len(batch)
    
    padded_embeddings = torch.zeros(batch_size, max_len, 960)
    
    for i, emb in enumerate(embeddings):
        padded_embeddings[i, :lengths[i]] = emb
    
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)
    
    return padded_embeddings, lengths_tensor, protein_ids

def count_parameters(model):
    """Count total and trainable parameters"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def train_epoch(model, train_loader, optimizer, device, scheduler=None):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc="Training MAE")
    for batch_idx, (embeddings, lengths, protein_ids) in enumerate(pbar):
        embeddings = embeddings.to(device).float()
        lengths = lengths.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        loss, pred, mask = model(embeddings, lengths=lengths)
        
        # Check for NaN/Inf
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"Warning: Invalid loss at batch {batch_idx}: {loss.item()}")
            continue
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        # Update progress bar
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        # Clear CUDA cache periodically
        if batch_idx % 50 == 0 and device.type == 'cuda':
            torch.cuda.empty_cache()
    
    return {'loss': total_loss / max(num_batches, 1)}

def validate_epoch(model, val_loader, device):
    """Validate for one epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for embeddings, lengths, protein_ids in pbar:
            embeddings = embeddings.to(device).float()
            lengths = lengths.to(device)
            
            # Forward pass
            loss, pred, mask = model(embeddings, lengths=lengths)
            
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return {'loss': total_loss / max(num_batches, 1)}

def save_checkpoint(model, optimizer, scheduler, epoch, train_metrics, val_metrics, 
                   config, save_path, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'train_metrics': train_metrics,
        'val_metrics': val_metrics,
        'config': config,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    torch.save(checkpoint, save_path)
    if is_best:
        best_path = save_path.replace('.pth', '_best.pth')
        torch.save(checkpoint, best_path)
        print(f"💾 Saved best model: {best_path}")

def plot_training_curves(history, save_path):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle('MAE Training Progress', fontsize=16, fontweight='bold')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_title('Reconstruction Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Learning rate
    if 'learning_rate' in history:
        axes[1].plot(epochs, history['learning_rate'], 'g-', linewidth=2)
        axes[1].set_title('Learning Rate Schedule')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Learning Rate')
        axes[1].set_yscale('log')
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, 'No LR Schedule', ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title('Learning Rate Schedule')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📊 Saved training curves: {save_path}")

def plot_reconstruction_sample(model, sample_loader, device, save_path, epoch):
    """Plot reconstruction examples"""
    model.eval()
    
    with torch.no_grad():
        # Get one batch
        embeddings, lengths, protein_ids = next(iter(sample_loader))
        embeddings = embeddings.to(device).float()
        lengths = lengths.to(device)
        
        # Forward pass with no masking to see full reconstruction
        loss, pred, mask = model(embeddings[:1], mask_ratio=0.75, lengths=lengths[:1])
        
        # Convert to numpy
        original = embeddings[0].cpu().numpy()  # (L, 960)
        reconstructed = pred[0].cpu().numpy()   # (T, 960)
        mask_np = mask[0, 1:].cpu().numpy()     # (T,) - remove CLS token
        
        # Plot first few features
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'MAE Reconstruction Examples - Epoch {epoch}', fontsize=16)
        
        for i, ax in enumerate(axes.flat):
            if i >= 4:
                break
                
            feature_idx = i * 240  # Show different features
            if feature_idx >= 960:
                break
                
            seq_len = min(original.shape[0], reconstructed.shape[0])
            
            ax.plot(original[:seq_len, feature_idx], label='Original', alpha=0.7, linewidth=2)
            ax.plot(reconstructed[:seq_len, feature_idx], '--', label='Reconstructed', linewidth=1.5)
            
            # Highlight masked positions
            if len(mask_np) >= seq_len:
                masked_positions = np.where(mask_np[:seq_len])[0]
                ax.scatter(masked_positions, original[masked_positions, feature_idx], 
                          color='red', marker='x', s=50, label='Masked', alpha=0.8)
            
            ax.set_title(f'Feature {feature_idx}')
            ax.set_xlabel('Sequence Position')
            ax.set_ylabel('Feature Value')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"📊 Saved reconstruction sample: {save_path}")

def train_mae(config):
    """Main MAE training function"""
    print("🧬 V5 MAE PRE-TRAINING")
    print("=" * 50)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if device.type == 'cuda':
        print(f"CUDA device name: {torch.cuda.get_device_name()}")
        torch.cuda.empty_cache()
    
    # Create output directories
    os.makedirs(config['save_dir'], exist_ok=True)
    os.makedirs(config['log_dir'], exist_ok=True)
    
    # Load data
    print("\n📊 Loading protein embeddings...")
    train_data, val_data, test1_data, test2_data, protein_embeddings = load_data()
    
    print(f"Available proteins: {len(protein_embeddings)}")
    
    # Create datasets (use all available proteins for pre-training)
    train_dataset = ProteinMAEDataset(protein_embeddings, max_len=config['max_len'])
    
    # Split for validation (use a subset)
    val_proteins = list(protein_embeddings.keys())[:len(protein_embeddings) // 10]  # 10% for validation
    val_protein_dict = {pid: protein_embeddings[pid] for pid in val_proteins}
    val_dataset = ProteinMAEDataset(val_protein_dict, max_len=config['max_len'])
    
    print(f"Training proteins: {len(train_dataset)}")
    print(f"Validation proteins: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        collate_fn=collate_fn_mae,
        num_workers=0,
        pin_memory=False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        collate_fn=collate_fn_mae,
        num_workers=0,
        pin_memory=False
    )
    
    # Sample loader for visualization
    sample_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn_mae
    )
    
    # Create model
    print(f"\n🔧 Creating V5 MAE model...")
    model = V5MAE(
        input_dim=960,
        embed_dim=config['embed_dim'],
        num_layers=config['encoder_layers'],
        nhead=config['encoder_heads'],
        ff_dim=config['ff_dim'],
        max_len=config['max_len'],
        dropout=config['dropout'],
        mask_ratio=config['mask_ratio'],
        decoder_layers=config['decoder_layers'],
        decoder_heads=config['decoder_heads']
    )
    model = model.to(device)
    
    # Count parameters
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Setup optimizer and scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.95)  # Following MAE paper
    )
    
    scheduler = None
    if config['use_scheduler']:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=config['num_epochs'] // 4,
            T_mult=2,
            eta_min=config['learning_rate'] * 0.01
        )
    
    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    best_val_loss = float('inf')
    best_epoch = 0
    
    # Training loop
    print(f"\n🚀 Starting MAE pre-training for {config['num_epochs']} epochs...")
    print(f"Mask ratio: {config['mask_ratio']}")
    
    for epoch in range(1, config['num_epochs'] + 1):
        print(f"\nEpoch {epoch}/{config['num_epochs']}")
        print("-" * 30)
        
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, scheduler)
        
        # Validate
        val_metrics = validate_epoch(model, val_loader, device)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['val_loss'].append(val_metrics['loss'])
        
        if scheduler:
            history['learning_rate'].append(scheduler.get_last_lr()[0])
        else:
            history['learning_rate'].append(config['learning_rate'])
        
        # Print metrics
        print(f"Train Loss: {train_metrics['loss']:.6f}")
        print(f"Val Loss: {val_metrics['loss']:.6f}")
        
        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            best_epoch = epoch
            print(f"🎉 New best validation loss: {best_val_loss:.6f}")
        
        checkpoint_path = os.path.join(config['save_dir'], f"mae_checkpoint_epoch_{epoch}.pth")
        save_checkpoint(model, optimizer, scheduler, epoch, train_metrics, val_metrics,
                       config, checkpoint_path, is_best)
        
        # Plot reconstruction samples
        if epoch % 5 == 0 or epoch == 1:
            recon_path = os.path.join(config['log_dir'], f'reconstruction_epoch_{epoch}.png')
            plot_reconstruction_sample(model, sample_loader, device, recon_path, epoch)
        
        # Early stopping
        if config.get('early_stopping', False) and epoch - best_epoch >= config.get('patience', 15):
            print(f"Early stopping at epoch {epoch}")
            break
    
    # Save final results
    results_dict = {
        'config': config,
        'history': history,
        'best_epoch': best_epoch,
        'best_val_loss': best_val_loss,
        'model_parameters': {
            'total': total_params,
            'trainable': trainable_params
        }
    }
    
    results_path = os.path.join(config['log_dir'], 'mae_training_results.json')
    with open(results_path, 'w') as f:
        json.dump(results_dict, f, indent=2, default=str)
    print(f"💾 Saved results: {results_path}")
    
    # Generate plots
    plot_path = os.path.join(config['log_dir'], 'mae_training_curves.png')
    plot_training_curves(history, plot_path)
    
    return results_dict

def main():
    """Main function"""
    # MAE training configuration
    config = {
        # Data parameters
        'batch_size': 4,  # Small batch size due to large model
        'max_len': 1500,  # Maximum sequence length
        
        # Model architecture (matching v5 specifications)
        'embed_dim': 768,        # Embedding dimension
        'encoder_layers': 12,    # Number of encoder layers
        'encoder_heads': 12,     # Number of attention heads in encoder
        'decoder_layers': 8,     # Number of decoder layers
        'decoder_heads': 16,     # Number of attention heads in decoder
        'ff_dim': 3072,         # Feed-forward dimension
        'dropout': 0.1,         # Dropout rate
        
        # MAE specific
        'mask_ratio': 0.75,     # Masking ratio (75% as in MAE paper)
        
        # Training parameters
        'num_epochs': 100,
        'learning_rate': 1.5e-4,  # Base learning rate (scaled for batch size)
        'weight_decay': 0.05,     # Weight decay
        'use_scheduler': True,
        'early_stopping': True,
        'patience': 15,
        
        # Paths
        'save_dir': 'models/v5_mae_pretrain',
        'log_dir': 'logs/v5_mae_pretrain'
    }
    
    # Add timestamp to directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    config['save_dir'] = f"{config['save_dir']}_{timestamp}"
    config['log_dir'] = f"{config['log_dir']}_{timestamp}"
    
    print("V5 MAE Pre-training Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Train MAE
    results = train_mae(config)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("🎉 MAE PRE-TRAINING COMPLETED!")
    print("=" * 60)
    print(f"Best validation loss: {results['best_val_loss']:.6f} (epoch {results['best_epoch']})")
    print(f"Total parameters: {results['model_parameters']['total']:,}")
    print(f"Trainable parameters: {results['model_parameters']['trainable']:,}")
    
    print(f"\nMAE model saved to: {config['save_dir']}")
    print(f"Logs saved to: {config['log_dir']}")
    print("\nYou can now use the pre-trained MAE encoder in v5_train.py by setting:")
    print(f"'mae_checkpoint_path': '{config['save_dir']}/mae_checkpoint_epoch_{results['best_epoch']}_best.pth'")

if __name__ == "__main__":
    main()