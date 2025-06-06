'''
use the model from mae_v2 to generate compressed feature --> dict[key] = compressed_vector

'''
import torch
import pickle
import torch.nn as nn
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


import os
import pickle
import torch
import pandas as pd
from torch.utils.data import DataLoader

class ProteinDataset(torch.utils.data.Dataset):
    def __init__(self, protein_dict, max_len=1502):
        self.keys = list(protein_dict.keys())
        self.protein_dict = protein_dict
        self.max_len = max_len

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        key = self.keys[idx]
        seq_np = self.protein_dict[key]             # numpy array, shape=(L, 960)
        seq = torch.tensor(seq_np, dtype=torch.float32)
        seq_len = seq.shape[0]

        if seq_len < self.max_len:
            pad = torch.zeros((self.max_len - seq_len, 960), dtype=torch.float32)
            seq = torch.cat([seq, pad], dim=0)
        else:
            seq = seq[:self.max_len]
            seq_len = self.max_len

        return {
            "seq": seq,                 # (max_len, 960)
            "padding_start": seq_len,   # int
            "key": key
        }

def collate_fn(batch):
    seqs = torch.stack([item["seq"] for item in batch], dim=0)                # (B, L, 960)
    lengths = torch.tensor([item["padding_start"] for item in batch])         # (B,)
    keys    = [item["key"] for item in batch]
    return seqs, lengths, keys

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

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = TransformerMAE().to(device).float()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    def load_model(model, optimizer, model_path):
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer, checkpoint['epoch'], checkpoint['loss']

    model_path = 'models/mae_final_20250528-174157.pth'
    model, optimizer, start_epoch, best_loss = load_model(model, optimizer, model_path)
    print(f"Loaded model {model_path} (from epoch {start_epoch}, loss {best_loss:.4f})")

    model.eval()
    model.mask_ratio = 0.0

    protein_dict = pd.read_pickle('./embeddings_standardized.pkl')
    dataset = ProteinDataset(protein_dict, max_len=1502)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=2,
        pin_memory=False,
        collate_fn=collate_fn
    )

    compressed_dict = {}
    with torch.no_grad():
        for seqs, lengths, keys in dataloader:
            seqs = seqs.to(device)      # (B, L, 960)
            lengths = lengths.to(device)
            _, compressed, _ = model(seqs, lengths)
            compressed = compressed.cpu().numpy()  # (B, 960)

            for k, vec in zip(keys, compressed):
                compressed_dict[k] = vec

    out_path = 'compressed_protein_features2.pkl'
    with open(out_path, 'wb') as f:
        pickle.dump(compressed_dict, f)

    print(f"Saved compressed features to {out_path}, total items: {len(compressed_dict)}")


    i=0
    for key, value in compressed_dict.items():
        print(f"key:{key}, type={type(key)}, value type={type(value)}")  
        print(f"value:{value}")
        i+=1
        if i > 3:  # print first 3 items
            break