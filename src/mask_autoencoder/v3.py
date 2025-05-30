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
import pickle

class ProteinPairDataset(Dataset):
    def __init__(self, pairs_df, embeddings_dict, max_len=1502):
        """
        Args:
            pairs_df: DataFrame with columns ['uniprotID_A', 'uniprotID_B', ...]
            embeddings_dict: Dict with uniprotID -> numpy array of shape (seq_len, 960)
            max_len: Maximum sequence length after padding
        """
        self.pairs_df = pairs_df.reset_index(drop=True)
        self.embeddings_dict = embeddings_dict
        self.max_len = max_len
        
        # Filter pairs where both proteins have embeddings
        valid_indices = []
        for idx in range(len(self.pairs_df)):
            row = self.pairs_df.iloc[idx]
            if row['uniprotID_A'] in self.embeddings_dict and row['uniprotID_B'] in self.embeddings_dict:
                valid_indices.append(idx)
        
        self.valid_indices = valid_indices
        print(f"Dataset: {len(valid_indices)} valid pairs out of {len(self.pairs_df)} total pairs")
        
    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]
        row = self.pairs_df.iloc[actual_idx]
        
        # Get embeddings for both proteins
        emb_A = torch.from_numpy(self.embeddings_dict[row['uniprotID_A']])  # (seq_len_A, 960)
        emb_B = torch.from_numpy(self.embeddings_dict[row['uniprotID_B']])  # (seq_len_B, 960)
        
        # Concatenate the embeddings along the sequence dimension
        combined_emb = torch.cat([emb_A, emb_B], dim=0)  # (seq_len_A + seq_len_B, 960)
        seq_len = combined_emb.shape[0]

        need_cleanup = False
        # Padding to max_len
        if seq_len < self.max_len:
            pad_size = (self.max_len - seq_len, 960)
            padding = torch.zeros(pad_size, dtype=combined_emb.dtype)
            combined_emb = torch.cat([combined_emb, padding], dim=0)
            need_cleanup = True
        else:
            combined_emb = combined_emb[:self.max_len]  # truncate if needed
            seq_len = self.max_len
            
        if need_cleanup:
            del padding
            
        return {
            "seq": combined_emb.clone(),        # (max_len, 960)
            "padding_start": seq_len,           # int
            "uniprotID_A": row['uniprotID_A'],  # string
            "uniprotID_B": row['uniprotID_B'],  # string
            "isInteraction": row['isInteraction'] if 'isInteraction' in row else -1  # int
        }

def collate_fn(batch):
    seqs = torch.stack([item["seq"] for item in batch], dim=0)              # (B, L, 960)
    lengths = torch.tensor([item["padding_start"] for item in batch])        # (B,)
    interactions = torch.tensor([item["isInteraction"] for item in batch])   # (B,)
    return seqs, lengths, interactions

# Masked Autoencoder with Transformer (same as v2)
class TransformerMAE(nn.Module):
    def __init__(self,
                 input_dim=960,
                 embed_dim=512,
                 mask_ratio=0.5,
                 num_layers=4,
                 nhead=16,
                 ff_dim=2048,
                 max_len=1502):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.max_len = max_len

        # ---- embed & mask token & pos embed ----
        self.embed = nn.Linear(input_dim, embed_dim) #嵌入层 (B,L,960)-->(B, L, 512)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) #掩码标记(1,1,512)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, embed_dim)) #位置编码(1,1502,512)

        # ---- Transformer encoder ----
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ---- decoder head (MLP) ----
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Linear(ff_dim, input_dim)
        )

        # ---- 压缩 head: (embed_dim -> input_dim) ----
        self.compress_head = nn.Linear(embed_dim, input_dim)

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

        # 5) 替换成 mask_token
        x_emb = self.embed(x)  # (B, L, E)
        # Ensure mask_token has the same dtype as x_emb
        mask_token_expanded = self.mask_token.expand(B, L, -1).to(x_emb.dtype)
        x_emb = x_emb.masked_scatter(mask_bool.unsqueeze(-1), mask_token_expanded[mask_bool.unsqueeze(-1).expand(-1, -1, x_emb.size(-1))])

        # 6) 位置编码 & Transformer 编码器
        x_emb = x_emb + self.pos_embed[:, :L].to(x_emb.dtype)
        enc_out = self.encoder(x_emb, src_key_padding_mask=mask_pad)

        # 7) 解码器重建
        recon = self.decoder(enc_out)  # (B, L, 960)

        # 8) 生成压缩向量
        compressed = self.compress_head(enc_out.mean(dim=1))  # (B, 960)

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
    plt.title(f"Epoch {epoch} Batch {batch_idx} - Reconstruction (Protein Pairs)")
    plt.xlabel("Sequence Position")
    plt.ylabel("Feature Value")
    plt.grid(True, alpha=0.3)

    os.makedirs(f"logs/recon_plots_{ts}", exist_ok=True)
    plt.savefig(f"logs/recon_plots_{ts}/epoch{epoch}_batch{batch_idx}.png")
    plt.close()


def load_data():
    """Load training data, validation data, and embeddings"""
    print("Loading training data...")
    with open('data/full_dataset/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    print("Loading validation data...")
    with open('data/full_dataset/validation_data.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    print("Loading embeddings (this might take a while)...")
    with open('data/full_dataset/embeddings/embeddings_standardized.pkl', 'rb') as f:
        embeddings_dict = pickle.load(f)
    
    print(f"Loaded {len(train_data)} training pairs, {len(val_data)} validation pairs")
    print(f"Loaded embeddings for {len(embeddings_dict)} proteins")
    
    return train_data, val_data, embeddings_dict


def train(train_data, val_data, embeddings_dict, epochs=10, max_len=1000): # TODO: max_len=?
    """Train the MAE on protein pairs"""
    
    # --- 准备数据 ---
    train_dataset = ProteinPairDataset(train_data, embeddings_dict, max_len=max_len)
    val_dataset = ProteinPairDataset(val_data, embeddings_dict, max_len=max_len)
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2,
                                persistent_workers=True, pin_memory=False, collate_fn=collate_fn)
    
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=2,
                              persistent_workers=True, pin_memory=False, collate_fn=collate_fn)

    # --- 模型 & 优化器 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = TransformerMAE(max_len=max_len, embed_dim=256, num_layers=2, nhead=8, ff_dim=512).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer,
                                                T_0=50,        # 初始周期长度
                                                T_mult=1,       # 周期长度倍增系数
                                                eta_min=1e-5    # 最小学习率
                                            )

    # --- 日志 & 保存目录 ---
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_path = f'logs/mae_pairs_train_{ts}.json'
    best_path = f'models/mae_pairs_best_{ts}.pth'

    best_loss = float('inf')
    history = {'train': [], 'val': []}

    for epoch in range(1, epochs+1):
        # Training phase
        model.train()
        train_losses = []
        for batch_idx, (batch, lengths, interactions) in enumerate(train_dataloader):
            batch = batch.to(device).float()  # (B, L, 960)
            lengths = lengths.to(device)

            recon, compressed, mask_bool = model(batch, lengths)
            loss = mae_loss(recon, batch, mask_bool)
            del recon, compressed

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            train_losses.append(loss.item())

            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}/{epochs}  Batch {batch_idx}  Loss {loss.item():.4f}')
            
            if batch_idx == 1:  # 可视化一次
                with torch.no_grad():
                    sample = batch[:1]
                    sample_len = lengths[:1]
                    recon, _, mask_bool_vis = model(sample, sample_len)
                    plot_reconstruction(sample.cpu(), recon.cpu(), mask_bool_vis.cpu(), epoch, batch_idx, ts)

        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_idx, (batch, lengths, interactions) in enumerate(val_dataloader):
                batch = batch.to(device).float()
                lengths = lengths.to(device)
                
                recon, compressed, mask_bool = model(batch, lengths)
                loss = mae_loss(recon, batch, mask_bool)
                val_losses.append(loss.item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        history['train'].append(train_loss)
        history['val'].append(val_loss)

        # 保存最佳模型
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss
            }, best_path)
            print(f'>>> 新最佳模型，Epoch={epoch}  Val Loss={best_loss:.4f}')

        # 写入日志
        with open(log_path, 'a') as f:
            log = {
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'best_val_loss': best_loss,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            f.write(json.dumps(log) + '\n')

        print(f'--- Epoch {epoch} 完成，Train Loss={train_loss:.4f}，Val Loss={val_loss:.4f}，Best Val Loss={best_loss:.4f} ---')
        
        # 绘制损失曲线
        plt.figure(figsize=(10, 6))
        plt.plot(history['train'], label='Training Loss', color='blue')
        plt.plot(history['val'], label='Validation Loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Training and Validation Loss Curves (Epoch {epoch})')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'logs/loss_curve_pairs_{ts}.png')
        plt.close()

        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        
    # 训练结束保存最终模型
    final_path = f'models/mae_pairs_final_{ts}.pth'
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': history['train'][-1],
        'val_loss': history['val'][-1]
    }, final_path)
    print(f'训练结束，最终模型已保存到 {final_path}')

    # Extract embeddings for both training and validation sets
    print("\n" + "="*60)
    print("EXTRACTING EMBEDDINGS FOR TRAINING AND VALIDATION SETS")
    print("="*60)
    
    print("Extracting embeddings for training set...")
    train_embeddings, train_interactions = extract_embeddings_for_classification(model, train_dataset, device)
    
    print("Extracting embeddings for validation set...")
    val_embeddings, val_interactions = extract_embeddings_for_classification(model, val_dataset, device)
    
    # Create combined DataFrames with original data + MAE embeddings
    print("Creating combined datasets with MAE embeddings...")
    
    # For training set
    train_data_with_embeddings = train_data.copy()
    # Add MAE embeddings as the last column
    train_data_with_embeddings['mae_embeddings'] = [emb for emb in train_embeddings]
    
    # For validation set  
    val_data_with_embeddings = val_data.copy()
    # Add MAE embeddings as the last column
    val_data_with_embeddings['mae_embeddings'] = [emb for emb in val_embeddings]
    
    # Save to pickle files
    train_output_path = f'data/train_data_with_mae_embeddings_{ts}.pkl'
    val_output_path = f'data/val_data_with_mae_embeddings_{ts}.pkl'
    
    print(f"Saving training data with embeddings to {train_output_path}")
    with open(train_output_path, 'wb') as f:
        pickle.dump(train_data_with_embeddings, f)
    
    print(f"Saving validation data with embeddings to {val_output_path}")
    with open(val_output_path, 'wb') as f:
        pickle.dump(val_data_with_embeddings, f)
    
    # Save embeddings separately as numpy arrays for convenience
    embeddings_output_path = f'data/mae_embeddings_{ts}.npz'
    print(f"Saving embeddings as numpy arrays to {embeddings_output_path}")
    np.savez(embeddings_output_path,
             train_embeddings=train_embeddings,
             train_labels=train_interactions,
             val_embeddings=val_embeddings,
             val_labels=val_interactions)
    
    print(f"\n=== EMBEDDING EXTRACTION SUMMARY ===")
    print(f"Training set: {len(train_embeddings)} samples, embedding shape: {train_embeddings.shape}")
    print(f"Validation set: {len(val_embeddings)} samples, embedding shape: {val_embeddings.shape}")
    print(f"Embedding dimension: {train_embeddings.shape[1]}")
    print(f"Files saved:")
    print(f"  - {train_output_path}")
    print(f"  - {val_output_path}")
    print(f"  - {embeddings_output_path}")

    return history


def extract_embeddings_for_classification(model, dataset, device, batch_size=8):
    """Extract embeddings for all protein pairs in the dataset"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=2, collate_fn=collate_fn)
    
    model.eval()
    all_embeddings = []
    all_interactions = []
    
    with torch.no_grad():
        for batch_idx, (batch, lengths, interactions) in enumerate(dataloader):
            batch = batch.to(device).float()
            lengths = lengths.to(device)
            
            # Get compressed embeddings from the MAE
            _, compressed, _ = model(batch, lengths)
            
            all_embeddings.append(compressed.cpu().numpy())
            all_interactions.append(interactions.numpy())
            
            if batch_idx % 100 == 0:
                print(f"  Processed {batch_idx} batches...")
    
    all_embeddings = np.vstack(all_embeddings)  # Shape: (n_samples, 960)
    all_interactions = np.concatenate(all_interactions)  # Shape: (n_samples,)
    
    return all_embeddings, all_interactions


if __name__ == '__main__':
    # Load data
    train_data, val_data, embeddings_dict = load_data()
    
    # Train the model
    history = train(train_data, val_data, embeddings_dict, epochs=60, max_len=1000)
    
    print("Training completed successfully!") 