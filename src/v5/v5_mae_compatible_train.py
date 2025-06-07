#!/usr/bin/env python3
"""
V5 MAE Training Script - Compatible with v5_memory_friendly.py
This script trains a MAE that exactly matches the MAEEncoder architecture 
in v5_memory_friendly.py for zero missing keys.
"""

import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from datetime import datetime
import json
import torch.nn.functional as F  
import matplotlib
matplotlib.use('Agg') 
import gc

class ProteinDataset(Dataset):
    def __init__(self, protein_dict, max_len=1502):
        self.keys = list(protein_dict.keys())
        self.protein_dict = protein_dict
        self.max_len = max_len
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        seq = torch.from_numpy(self.protein_dict[key]) # shape: (seq_len, 960)
        seq_len = seq.shape[0]

        need_cleanup = False
        # padding to max_len
        if seq_len < self.max_len:
            pad_size = (self.max_len - seq_len, 960)
            padding = torch.zeros(pad_size, dtype=seq.dtype)
            seq = torch.cat([seq, padding], dim=0)
            need_cleanup = True
        else:
            seq = seq[:self.max_len]  # truncate if needed
            seq_len = self.max_len
            
        if need_cleanup:
            del padding
            
        return {
            "seq": seq.clone(),                 # (max_len, 960)
            "padding_start": seq_len            # int
        }

def collate_fn(batch):
    seqs = torch.stack([item["seq"] for item in batch], dim=0)              # (B, L, 960)
    lengths = torch.tensor([item["padding_start"] for item in batch])        # (B,)
    return seqs, lengths

class PatchEmbedding(nn.Module):
    """Converts 960-d residue embeddings into 512-d tokens"""
    def __init__(self, input_dim=960, embed_dim=512):
        super().__init__()
        self.embed_dim = embed_dim
        self.projection = nn.Linear(input_dim, embed_dim)
        
    def forward(self, x):
        # x: (B, L, 960) -> (B, L, 512)
        return self.projection(x)

class CompatibleMAEEncoder(nn.Module):
    """MAE Encoder that exactly matches v5_memory_friendly.py MAEEncoder"""
    def __init__(self,
                 input_dim=960,
                 embed_dim=512,
                 num_layers=8,
                 nhead=8,
                 ff_dim=2048,
                 max_len=1502,
                 dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

        # ---- EXACT SAME ARCHITECTURE as v5_memory_friendly.py ----
        self.patch_embed = PatchEmbedding(input_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))  # CLS token for v5

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)  # Named 'norm' not 'encoder_norm'

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def patchify(self, x, target_length=47):
        """Same patchify logic as v5_memory_friendly.py"""
        B, L, D = x.shape
        x_emb = self.patch_embed(x)  # (B, L, 512)
        
        # Add positional encoding
        x_emb = x_emb + self.pos_embed[:, :L]
        
        # Dynamic pooling to target length
        if L > target_length:
            pool_size = L // target_length
            remainder = L % target_length
            
            pooled_parts = []
            start_idx = 0
            
            for i in range(target_length):
                if i < remainder:
                    end_idx = start_idx + pool_size + 1
                else:
                    end_idx = start_idx + pool_size
                
                if start_idx < L:
                    chunk = x_emb[:, start_idx:min(end_idx, L)]
                    pooled = chunk.mean(dim=1, keepdim=True)
                    pooled_parts.append(pooled)
                
                start_idx = end_idx
            
            if pooled_parts:
                x_emb = torch.cat(pooled_parts, dim=1)
            else:
                x_emb = x_emb[:, :target_length]
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x_emb = torch.cat([cls_tokens, x_emb], dim=1)
        
        return x_emb

    def forward(self, x):
        """Forward pass through encoder"""
        # Create attention mask for padding
        B, L, _ = x.shape
        
        # Pass through encoder layers
        x = self.encoder(x)
        x = self.norm(x)
        
        return x

class CompatibleMAEForTraining(nn.Module):
    """Complete MAE model for training with decoder"""
    def __init__(self,
                 input_dim=960,
                 embed_dim=512,
                 mask_ratio=0.5,
                 num_layers=8,
                 nhead=8,
                 ff_dim=2048,
                 max_len=1502,
                 dropout=0.1):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.max_len = max_len
        self.embed_dim = embed_dim

        # ---- Use the compatible encoder ----
        self.mae_encoder = CompatibleMAEEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            num_layers=num_layers,
            nhead=nhead,
            ff_dim=ff_dim,
            max_len=max_len,
            dropout=dropout
        )

        # ---- Mask token for training ----
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # ---- Decoder head ----
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim)
        )

        # ---- Compression head ----
        self.compress_head = nn.Linear(embed_dim, input_dim)
        
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x: Tensor (B, L, 960)
        lengths: Tensor (B,)
        return:
          - recon: Tensor (B, L, 960)
          - compressed: Tensor (B, 960)
          - mask_bool: Tensor (B, L)
        """
        device = x.device
        B, L, _ = x.shape

        # 1) Create padding mask
        arange = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        mask_pad = arange >= lengths.unsqueeze(1)

        # 2) Calculate mask positions for each sample
        len_per_sample = lengths
        num_mask_per_sample = (len_per_sample.float() * self.mask_ratio).long()

        # 3) Generate random mask
        noise = torch.rand(B, L, device=device)
        noise = noise.masked_fill(mask_pad, float("inf"))
        sorted_indices = torch.argsort(noise, dim=1)

        # 4) Create boolean mask
        mask_bool = torch.zeros(B, L, dtype=torch.bool, device=device)
        for i in range(B):
            k = num_mask_per_sample[i].item()
            mask_bool[i, sorted_indices[i, :k]] = True

        # 5) Project to embed_dim and apply masking
        x_emb = self.mae_encoder.patch_embed(x)  # (B, L, 512)
        
        # Add positional encoding
        x_emb = x_emb + self.mae_encoder.pos_embed[:, :L]
        
        # Replace masked positions with mask token
        mask_tokens = self.mask_token.expand(B, L, -1)
        x_emb = torch.where(mask_bool.unsqueeze(-1), mask_tokens, x_emb)

        # 6) Pass through encoder (without CLS token for training)
        src_key_padding_mask = mask_pad
        enc_out = self.mae_encoder.encoder(x_emb, src_key_padding_mask=src_key_padding_mask)
        enc_out = self.mae_encoder.norm(enc_out)

        # 7) Decoder reconstruction
        recon = self.decoder(enc_out)  # (B, L, 960)

        # 8) Compression vector (mean pooling over non-padded positions)
        enc_sum = torch.zeros(B, self.embed_dim, device=device)
        for i in range(B):
            enc_sum[i] = enc_out[i, :lengths[i]].mean(dim=0)
        compressed = self.compress_head(enc_sum)  # (B, 960)

        return recon, compressed, mask_bool

    def get_encoder_state_dict(self):
        """Extract only the encoder state dict for downstream use"""
        encoder_state_dict = {}
        for key, value in self.mae_encoder.state_dict().items():
            encoder_state_dict[key] = value
        return encoder_state_dict

# Loss function (same as before)
def mae_loss(recon: torch.Tensor, orig: torch.Tensor, mask_bool: torch.Tensor, delta: float = 0.5):
    scale_factor = 5
    loss = F.huber_loss(
        recon[mask_bool] * scale_factor,
        orig[mask_bool] * scale_factor,
        delta=delta,
    )
    return loss

def plot_reconstruction(orig, recon, mask_bool, epoch, batch_idx, ts):
    plt.figure(figsize=(12, 4))
    orig_np = orig[0, :, 0].cpu().numpy().copy()
    recon_np = recon[0, :, 0].cpu().numpy().copy()
    plt.plot(orig_np, label="Original", alpha=0.7, linewidth=2)
    plt.plot(recon_np, "--", label="Reconstructed", linewidth=1.5)

    mask_pos = torch.where(mask_bool[0])[0].cpu().numpy()
    plt.scatter(mask_pos, orig[0, mask_pos, 0].cpu().numpy(), color="red", marker="x", s=50, label="Masked")

    plt.legend()
    plt.title(f"Epoch {epoch} Batch {batch_idx} - Compatible V5 MAE Reconstruction")
    plt.xlabel("Sequence Position")
    plt.ylabel("Feature Value")
    plt.grid(True, alpha=0.3)

    os.makedirs(f"logs/v5_compatible_recon_{ts}", exist_ok=True)
    plt.savefig(f"logs/v5_compatible_recon_{ts}/epoch{epoch}_batch{batch_idx}.png")
    plt.close()

def train(protein_dict, epochs=50):
    """Train the compatible MAE"""
    # Prepare data
    dataset = ProteinDataset(protein_dict)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1,
                            persistent_workers=False,
                            pin_memory=False,
                            collate_fn=collate_fn)

    # Model & optimizer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CompatibleMAEForTraining().to(device)
    
    # Optimizer settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                T_0=50,
                                                T_mult=1,
                                                eta_min=1e-6)
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # Logging & saving
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_path = f'logs/v5_compatible_mae_{ts}.json'
    best_path = f'models/v5_compatible_mae_best_{ts}.pth'

    best_loss = float('inf')
    history = []
    all_epoch_losses = []

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Compatible V5 MAE Model - Total parameters: {total_params:,}")

    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        for batch_idx, (batch, lengths) in enumerate(dataloader):
            batch = batch.to(device, non_blocking=True).float()
            lengths = lengths.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Training with mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    recon, compressed, mask_bool = model(batch, lengths)
                    loss = mae_loss(recon, batch, mask_bool)
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                recon, compressed, mask_bool = model(batch, lengths)
                loss = mae_loss(recon, batch, mask_bool)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            # Cleanup
            loss_item = loss.item()
            del recon, compressed, mask_bool, loss
            torch.cuda.empty_cache()
            
            losses.append(loss_item)

            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch}/{epochs}  Batch {batch_idx}  Loss {loss_item:.4f}  LR {current_lr:.2e}')
                
            if batch_idx == 1:  # Visualization
                with torch.no_grad():
                    sample = batch[:1]
                    sample_len = lengths[:1]
                    recon, _, mask_bool_vis = model(sample, sample_len)
                    plot_reconstruction(sample.cpu(), recon.cpu(), mask_bool_vis.cpu(), epoch, batch_idx, ts)

        epoch_loss = np.mean(losses)
        history.append(epoch_loss)

        # Save best model with ONLY encoder weights
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            
            # Save only the encoder part for downstream use
            encoder_state_dict = model.get_encoder_state_dict()
            torch.save({
                'epoch': epoch,
                'model_state_dict': encoder_state_dict,  # Only encoder weights
                'full_model_state_dict': model.state_dict(),  # Full model for resuming training
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'config': {
                    'embed_dim': 512,
                    'num_layers': 8,
                    'nhead': 8,
                    'ff_dim': 2048,
                    'mask_ratio': 0.5,
                    'compatible_with': 'v5_memory_friendly.py'
                }
            }, best_path)
            print(f'>>> New best Compatible V5 MAE model, Epoch={epoch}  Loss={best_loss:.4f}')

        # Write logs
        with open(log_path, 'a') as f:
            log = {
                'epoch': epoch,
                'loss': epoch_loss,
                'best_loss': best_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            f.write(json.dumps(log) + '\n')

        print(f'--- Compatible V5 MAE Epoch {epoch} completed, Avg Loss={epoch_loss:.4f}, Best Loss={best_loss:.4f} ---')
        all_epoch_losses.append(epoch_loss)
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # Loss curve visualization
    plt.figure(figsize=(10, 6))
    plt.plot(all_epoch_losses, label='Compatible V5 MAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Compatible V5 MAE Training Loss Curve (Final: {epoch_loss:.4f})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'logs/v5_compatible_mae_loss_{ts}.png')
    plt.close()

    # Save final model
    final_path = f'models/v5_compatible_mae_final_{ts}.pth'
    encoder_state_dict = model.get_encoder_state_dict()
    torch.save({
        'epoch': epochs,
        'model_state_dict': encoder_state_dict,
        'full_model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': history[-1],
        'config': {
            'embed_dim': 512,
            'num_layers': 8,
            'nhead': 8,
            'ff_dim': 2048,
            'mask_ratio': 0.5,
            'compatible_with': 'v5_memory_friendly.py'
        }
    }, final_path)
    print(f'Compatible V5 MAE training completed, final model saved to {final_path}')

    return history

if __name__=='__main__':
    protein_dict = pd.read_pickle('data/full_dataset/embeddings/embeddings_standardized.pkl')
    print("Starting Compatible V5 MAE training...")
    print("This MAE will be 100% compatible with v5_memory_friendly.py")
    history = train(protein_dict, epochs=50)