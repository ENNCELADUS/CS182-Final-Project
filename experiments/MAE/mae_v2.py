'''
mae v2.py
MAE train and save model
using data of mean=0, std=1
adding "padding_start", only mask meaningful positions  
50% mask ratio
'''

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
        #并记录每条序列的有效长度 (padding_start)
        
    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        #seq = self.protein_dict[key]
        seq = torch.from_numpy(self.protein_dict[key]) # shape: (seq_len, 960)
        seq_len = seq.shape[0]

        # padding to max_len
        if seq_len < self.max_len:
            pad_size = (self.max_len - seq_len, 960)
            padding = torch.zeros(pad_size, dtype=seq.dtype)
            seq = torch.cat([seq, padding], dim=0)
        else:
            seq = seq[:self.max_len]  # truncate if needed
            seq_len = self.max_len       

        return {
            "seq": seq.clone(),                 # (max_len, 960)
            "padding_start": seq_len            # int
        }

def collate_fn(batch):
    seqs = torch.stack([item["seq"] for item in batch], dim=0)              # (B, L, 960)
    lengths = torch.tensor([item["padding_start"] for item in batch])        # (B,)
    return seqs, lengths

#  Masked Autoencoder with Transformer
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
        self.max_len = 1502

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
        arange = torch.arange(L, device=device).unsqueeze(0).expand(B, L)  # (B, L)
        mask_pad = arange >= lengths.unsqueeze(1)                          # (B, L)

        len_per_sample = lengths
        num_mask_per_sample = (len_per_sample.float() * self.mask_ratio).long()  # (B,)

        noise = torch.rand(B, L, device=device)
        noise = noise.masked_fill(mask_pad, float("inf"))  # padding 位置永远不会被选中
        sorted_indices = torch.argsort(noise, dim=1)        # (B, L)

        mask_bool = torch.zeros(B, L, dtype=torch.bool, device=device)  # False = 不 mask
        for i in range(B):
            k = num_mask_per_sample[i].item()
            mask_bool[i, sorted_indices[i, :k]] = True

        x_emb = self.embed(x).float()   # (B, L, E)
        x_emb = x_emb.masked_scatter(mask_bool.unsqueeze(-1), self.mask_token.expand(B, L, -1)[mask_bool.unsqueeze(-1).expand(-1, -1, x_emb.size(-1))])

        x_emb = x_emb + self.pos_embed[:, :L]
        enc_out = self.encoder(x_emb, src_key_padding_mask=mask_pad)

        recon = self.decoder(enc_out)  # (B, L, 960)
        compressed = self.compress_head(enc_out.mean(dim=1))  # (B, 960)

        return recon, compressed, mask_bool


# 3. loss only calulate on masked positions
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
    plt.title(f"Epoch {epoch} Batch {batch_idx} - Reconstruction")
    plt.xlabel("Sequence Position")
    plt.ylabel("Feature Value")
    plt.grid(True, alpha=0.3)

    os.makedirs(f"logs/recon_plots_{ts}", exist_ok=True)
    plt.savefig(f"logs/recon_plots_{ts}/epoch{epoch}_batch{batch_idx}.png")
    plt.close()


def train(protein_dict,epochs=10):
    dataset = ProteinDataset(protein_dict)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2,\
                            persistent_workers=True,  
                            pin_memory=False,     # for our cpu limitations
                            collate_fn=collate_fn)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerMAE().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=50,T_mult=1, eta_min=1e-5)

    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_path = f'logs/mae_train_{ts}.json'
    best_path = f'models/mae_best_{ts}.pth'

    best_loss = float('inf')
    history = []
    all_epoch_losses=[]

    for epoch in range(1, epochs+1):
        model.train()
        losses = []
        for batch_idx, (batch, lengths) in enumerate(dataloader):
            batch = batch.to(device).float()  # (B, L, 960)
            lengths = lengths.to(device)

            with torch.amp.autocast("cuda",enabled=False):  # for cpu
                recon, compressed, mask_bool = model(batch, lengths)
            loss = mae_loss(recon, batch, mask_bool)
            del recon, compressed

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  
            losses.append(loss.item())

            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}/{epochs}  Batch {batch_idx}  Loss {loss.item():.4f}')
            if batch_idx == 1 :  # visualize every epoch
                with torch.no_grad():
                    # sample
                    sample = batch[:1]
                    sample_len = lengths[:1]
                    recon, _, mask_bool_vis = model(sample, sample_len)
                    plot_reconstruction(sample.cpu(), recon.cpu(), mask_bool_vis.cpu(), epoch, batch_idx, ts)

        epoch_loss = np.mean(losses)
        history.append(epoch_loss)

        # save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss
            }, best_path)
            print(f'>>> 新最佳模型，Epoch={epoch}  Loss={best_loss:.4f}')

        with open(log_path, 'a') as f:
            log = {
                'epoch': epoch,
                'loss': epoch_loss,
                'best_loss': best_loss,
                'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            f.write(json.dumps(log) + '\n')

        print(f'--- Epoch {epoch} 完成，Avg Loss={epoch_loss:.4f}，Best Loss={best_loss:.4f} ---')
        all_epoch_losses.append(epoch_loss)
        gc.collect()        
        torch.cuda.empty_cache()  
        
    
    plt.figure(figsize=(10, 6))
    plt.plot(all_epoch_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve (Current: {epoch_loss:.4f})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'logs/loss_curve_{ts}.png') 
    plt.close() 

    final_path = f'models/mae_final_{ts}.pth'
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': history[-1]
    }, final_path)
    print(f'训练结束，最终模型已保存到 {final_path}')

    return history


if __name__=='__main__':
    protein_dict = pd.read_pickle('./embeddings_standardized.pkl') # this data is very large
    history = train(protein_dict,epochs=60)