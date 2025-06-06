"""
same structure as v3
this file is used to abstract compressed feature of protein pairs

differences:
1.  import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
2. dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=0, collate_fn=collate_fn) #num_workers=0 (due to cpu limitation)
3. set up "ts" by hand
"""
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
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from MAE.mae_v3 import TransformerMAE, collate_fn, extract_embeddings_for_classification
from MAE.mae_v3 import ProteinPairDataset, load_data

class ProteinPairDataset(Dataset):
    def __init__(self, pairs_df, embeddings_dict, max_len=2000):
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
        seq_len_A = emb_A.shape[0]
        seq_len_B = emb_B.shape[0]
        single_len=self.max_len//2

        if seq_len_A >= single_len and seq_len_B >= single_len:
            emb_A=emb_A[:single_len]
            emb_B=emb_B[:single_len]
            combined_emb = torch.cat([emb_A, emb_B], dim=0)  # (seq_len_A + seq_len_B, 960)    
            seq_len=self.max_len
        elif seq_len_A < single_len and seq_len_B >= single_len:
            combined_emb = torch.cat([emb_A, emb_B], dim=0)
            seq_len=combined_emb.shape[0]
            if seq_len < self.max_len:
                pad_size = (self.max_len - seq_len, 960)
                padding = torch.zeros(pad_size, dtype=combined_emb.dtype)
                combined_emb = torch.cat([combined_emb, padding], dim=0)
            else:
                combined_emb = combined_emb[:self.max_len]
                seq_len=self.max_len
        elif seq_len_A >= single_len and seq_len_B < single_len:
            if (seq_len_A +seq_len_B) >= self.max_len:
                emb_A=emb_A[:self.max_len-seq_len_B]
                combined_emb = torch.cat([emb_A, emb_B], dim=0)
                seq_len=self.max_len
            else:
                pad_size = (self.max_len - seq_len_A-seq_len_B, 960)
                padding = torch.zeros(pad_size, dtype=emb_A.dtype)
                combined_emb = torch.cat([emb_A, emb_B,padding], dim=0)
                seq_len=seq_len_A+seq_len_B
        else: # both short
            pad_size = (self.max_len - seq_len_A-seq_len_B, 960)
            padding = torch.zeros(pad_size, dtype=emb_A.dtype)
            combined_emb = torch.cat([emb_A, emb_B,padding], dim=0)
            seq_len=seq_len_A+seq_len_B

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

def extract_embeddings_for_classification(model, dataset, device, batch_size=8):
    """Extract embeddings for all protein pairs in the dataset"""
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, 
                          num_workers=0, collate_fn=collate_fn)
    
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

# Masked Autoencoder with Transformer (same as v2)
class TransformerMAE(nn.Module):
    def __init__(self,
                 input_dim=960,
                 embed_dim=512,
                 mask_ratio=0.5,
                 num_layers=4,
                 nhead=16,
                 ff_dim=2048,
                 max_len=2000):
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

        x_emb = self.embed(x)  # (B, L, E)
        # Ensure mask_token has the same dtype as x_emb
        mask_token_expanded = self.mask_token.expand(B, L, -1).to(x_emb.dtype)
        x_emb = x_emb.masked_scatter(mask_bool.unsqueeze(-1), mask_token_expanded[mask_bool.unsqueeze(-1).expand(-1, -1, x_emb.size(-1))])

        x_emb = x_emb + self.pos_embed[:, :L].to(x_emb.dtype)
        enc_out = self.encoder(x_emb, src_key_padding_mask=mask_pad)

        recon = self.decoder(enc_out)  # (B, L, 960)
        compressed = self.compress_head(enc_out.mean(dim=1))  # (B, 960)

        return recon, compressed, mask_bool

def load_data():
    """Load training data, validation data, and embeddings"""
    print("Loading training data...")
    with open('data/train_data.pkl', 'rb') as f:
        train_data = pickle.load(f)
    
    print("Loading validation data...")
    with open('data/validation_data.pkl', 'rb') as f:
        val_data = pickle.load(f)
    
    print("Loading embeddings (this might take a while)...")
    with open('../ESM/embeddings_standardized.pkl', 'rb') as f:
        embeddings_dict = pickle.load(f)
    
    print(f"Loaded {len(train_data)} training pairs, {len(val_data)} validation pairs")
    print(f"Loaded embeddings for {len(embeddings_dict)} proteins")
    
    return train_data, val_data, embeddings_dict


ts="20250530-151013"
model_path = 'models/mae_pairs_best_20250530-151013.pth'
device="cuda"
checkpoint = torch.load(model_path, map_location=device, weights_only=False)
model = TransformerMAE(max_len=2000, embed_dim=256, num_layers=2, nhead=8, ff_dim=512).to(device).float()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# For training set
train_data_with_embeddings,val_data_with_embeddings,embeddings_dict=load_data()

train_dataset = ProteinPairDataset(train_data_with_embeddings, embeddings_dict, max_len=2000)
val_dataset = ProteinPairDataset(val_data_with_embeddings, embeddings_dict, max_len=2000)
    
print("Extracting embeddings for training set...")
train_embeddings, train_interactions = extract_embeddings_for_classification(model, train_dataset, device)

print("Extracting embeddings for validation set...")
val_embeddings, val_interactions = extract_embeddings_for_classification(model, val_dataset, device)

# Create combined DataFrames with original data + MAE embeddings
print("Creating combined datasets with MAE embeddings...")


# Add MAE embeddings as the last column
train_data_with_embeddings['mae_embeddings'] = [emb for emb in train_embeddings]
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