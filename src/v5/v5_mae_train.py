import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
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

class V5TransformerMAE(nn.Module):
    """V5 MAE with decoder for training, based on v5 MAEEncoder architecture"""
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

        # ---- Patch embedding & mask token & pos embed ----
        self.patch_embed = PatchEmbedding(input_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim) * 0.02)

        # ---- Transformer encoder (v5 style) ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.encoder_norm = nn.LayerNorm(embed_dim)

        # ---- Decoder head (MLP) ----
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, input_dim)
        )

        # ---- Compression head: (embed_dim -> input_dim) ----
        self.compress_head = nn.Linear(embed_dim, input_dim)
        
        self._init_weights()

    def _init_weights(self):
        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor):
        """
        x: Tensor (B, L, 960)
        lengths: Tensor (B,)  # 每个样本的有效长度（非 padding 部分）
        return:
          - recon: Tensor (B, L, 960)   # 重建整个序列
          - compressed: Tensor (B, 960) # 池化后的压缩向量
          - mask_bool: Tensor (B, L)  # True 表示该位置被掩码
        """
        device = x.device
        B, L, _ = x.shape

        # 1) 依据 lengths 构造 padding mask（True 表示该位置为 padding）
        arange = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B, L)
        mask_pad = arange >= lengths.unsqueeze(1)                          # (B, L)

        # 2) 计算每个样本需要掩码的数量（非 padding 区域的 mask_ratio）
        len_per_sample = lengths
        num_mask_per_sample = (len_per_sample.float() * self.mask_ratio).long()  # (B,)

        # 3) 为所有样本生成随机噪声，并屏蔽 padding 为 +inf，排序后取前 num_mask_i
        noise = torch.rand(B, L, device=device)
        noise = noise.masked_fill(mask_pad, float("inf"))  # padding 位置永远不会被选中
        sorted_indices = torch.argsort(noise, dim=1)        # (B, L)

        # 4) 生成 boolean 掩码矩阵 mask_bool
        mask_bool = torch.zeros(B, L, dtype=torch.bool, device=device)  # False = 不 mask
        for i in range(B):
            k = num_mask_per_sample[i].item()
            mask_bool[i, sorted_indices[i, :k]] = True

        # 5) Project to embed_dim and replace masked positions with mask_token
        x_emb = self.patch_embed(x)  # (B, L, 512)
        
        # Replace masked positions with mask token
        mask_tokens = self.mask_token.expand(B, L, -1)
        x_emb = torch.where(mask_bool.unsqueeze(-1), mask_tokens, x_emb)

        # 6) 位置编码 & Transformer 编码器
        x_emb = x_emb + self.pos_embed[:, :L]
        
        # Create attention mask for padding
        src_key_padding_mask = mask_pad
        
        # Pass through encoder
        enc_out = self.encoder(x_emb, src_key_padding_mask=src_key_padding_mask)
        enc_out = self.encoder_norm(enc_out)

        # 7) 解码器重建
        recon = self.decoder(enc_out)  # (B, L, 960)

        # 8) 生成压缩向量 (mean pooling over non-padded positions)
        # Use lengths to compute proper mean
        enc_sum = torch.zeros(B, self.embed_dim, device=device)
        for i in range(B):
            enc_sum[i] = enc_out[i, :lengths[i]].mean(dim=0)
        compressed = self.compress_head(enc_sum)  # (B, 960)

        return recon, compressed, mask_bool


# MSE 只在 mask 位置上计算 (same as v2)
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
    plt.title(f"Epoch {epoch} Batch {batch_idx} - V5 MAE Reconstruction")
    plt.xlabel("Sequence Position")
    plt.ylabel("Feature Value")
    plt.grid(True, alpha=0.3)

    os.makedirs(f"logs/v5_recon_plots_{ts}", exist_ok=True)
    plt.savefig(f"logs/v5_recon_plots_{ts}/epoch{epoch}_batch{batch_idx}.png")
    plt.close()


def train(protein_dict, epochs=10):
    #--- 准备数据 ---
    dataset = ProteinDataset(protein_dict)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1,
                            persistent_workers=False,
                            pin_memory=False,
                            collate_fn=collate_fn)

    #--- 模型 & 优化器 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = V5TransformerMAE().to(device)
    
    # Enable gradient checkpointing to save memory
    if hasattr(model.encoder, 'gradient_checkpointing_enable'):
        model.encoder.gradient_checkpointing_enable()
    
    # V5 style optimizer settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                T_0=50,
                                                T_mult=1,
                                                eta_min=1e-6
                                            )
    
    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    #--- 日志 & 保存目录 ---
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_path = f'logs/v5_mae_train_{ts}.json'
    best_path = f'models/v5_mae_best_{ts}.pth'

    best_loss = float('inf')
    history = []
    all_epoch_losses = []

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"V5 MAE Model - Total parameters: {total_params:,}")

    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        for batch_idx, (batch, lengths) in enumerate(dataloader):
            batch = batch.to(device, non_blocking=True).float()  # (B, L, 960)
            lengths = lengths.to(device, non_blocking=True)

            optimizer.zero_grad()

            # Use mixed precision and gradient accumulation
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    recon, compressed, mask_bool = model(batch, lengths)
                    loss = mae_loss(recon, batch, mask_bool)
                
                # Scale loss and backward
                scaler.scale(loss).backward()
                
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                scaler.step(optimizer)
                scaler.update()
            else:
                recon, compressed, mask_bool = model(batch, lengths)
                loss = mae_loss(recon, batch, mask_bool)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            scheduler.step()
            
            # Cleanup immediately
            loss_item = loss.item()
            del recon, compressed, mask_bool, loss
            torch.cuda.empty_cache()
            
            losses.append(loss_item)

            if batch_idx % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f'Epoch {epoch}/{epochs}  Batch {batch_idx}  Loss {loss_item:.4f}  LR {current_lr:.2e}')
                
            if batch_idx == 1:  # 可视化一次
                with torch.no_grad():
                    sample = batch[:1]
                    sample_len = lengths[:1]
                    recon, _, mask_bool_vis = model(sample, sample_len)
                    plot_reconstruction(sample.cpu(), recon.cpu(), mask_bool_vis.cpu(), epoch, batch_idx, ts)

        epoch_loss = np.mean(losses)
        history.append(epoch_loss)

        # 保存最佳模型
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_loss,
                'config': {
                    'embed_dim': 512,
                    'num_layers': 8,
                    'nhead': 8,
                    'ff_dim': 2048,
                    'mask_ratio': 0.5
                }
            }, best_path)
            print(f'>>> 新最佳V5 MAE模型，Epoch={epoch}  Loss={best_loss:.4f}')

        # 写入日志
        with open(log_path, 'a') as f:
            log = {
                'epoch': epoch,
                'loss': epoch_loss,
                'best_loss': best_loss,
                'lr': optimizer.param_groups[0]['lr'],
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            f.write(json.dumps(log) + '\n')

        print(f'--- V5 MAE Epoch {epoch} 完成，Avg Loss={epoch_loss:.4f}，Best Loss={best_loss:.4f} ---')
        all_epoch_losses.append(epoch_loss)
        
        # Aggressive memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    # 损失曲线可视化
    plt.figure(figsize=(10, 6))
    plt.plot(all_epoch_losses, label='V5 MAE Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'V5 MAE Training Loss Curve (Final: {epoch_loss:.4f})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'logs/v5_mae_loss_curve_{ts}.png')
    plt.close()

    # 训练结束保存最终模型
    final_path = f'models/v5_mae_final_{ts}.pth'
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'loss': history[-1],
        'config': {
            'embed_dim': 512,
            'num_layers': 8,
            'nhead': 8,
            'ff_dim': 2048,
            'mask_ratio': 0.5
        }
    }, final_path)
    print(f'V5 MAE训练结束，最终模型已保存到 {final_path}')

    return history


if __name__=='__main__':
    protein_dict = pd.read_pickle('data/full_dataset/embeddings/embeddings_standardized.pkl')
    print("Starting V5 MAE training...")
    history = train(protein_dict, epochs=60) 