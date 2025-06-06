import numpy as np
import pandas as pd
import os
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from datetime import datetime
import json
import matplotlib
matplotlib.use('Agg')
import gc
import pickle
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import math


# First, let's examine the data structure to identify column names
def examine_dataframe(df):
    """Print the structure of the dataframe to identify column names"""
    print("DataFrame columns:", df.columns.tolist())
    print("First row sample:", df.iloc[0].to_dict())
    return df.columns.tolist()


def load_data():
    """Load the actual data from the project structure"""
    # Get the current script directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Navigate to the project root (CS182-Final-Project)
    # From src/mask_autoencoder/ go up two levels to reach project root
    project_root = os.path.dirname(os.path.dirname(current_dir))
    
    print(f"Script directory: {current_dir}")
    print(f"Project root: {project_root}")
    print("Loading data...")
    
    # Construct full paths to data files
    data_dir = os.path.join(project_root, 'data', 'full_dataset')
    embeddings_dir = os.path.join(data_dir, 'embeddings')
    
    print(f"Looking for data in: {data_dir}")
    
    # Check if data directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Load data files with full paths
    train_path = os.path.join(data_dir, 'train_data.pkl')
    cv_path = os.path.join(data_dir, 'validation_data.pkl')
    test1_path = os.path.join(data_dir, 'test1_data.pkl')
    test2_path = os.path.join(data_dir, 'test2_data.pkl')
    embeddings_path = os.path.join(embeddings_dir, 'embeddings_standardized.pkl')
    
    # Check if all files exist
    for path, name in [(train_path, 'train_data.pkl'), 
                       (cv_path, 'validation_data.pkl'),
                       (test1_path, 'test1_data.pkl'), 
                       (test2_path, 'test2_data.pkl'),
                       (embeddings_path, 'embeddings_standardized.pkl')]:
        if not os.path.exists(path):
            print(f"❌ Missing: {path}")
        else:
            print(f"✅ Found: {name}")
    
    # Load the data
    train_data = pickle.load(open(train_path, 'rb'))
    cv_data = pickle.load(open(cv_path, 'rb'))
    test1_data = pickle.load(open(test1_path, 'rb'))
    test2_data = pickle.load(open(test2_path, 'rb'))

    # Examine structure of the first dataframe to understand its format
    print("\nExamining training data structure:")
    examine_dataframe(train_data)

    print("\nLoading protein embeddings...")
    protein_embeddings = pickle.load(open(embeddings_path, 'rb'))
    print(f"Loaded {len(protein_embeddings)} protein embeddings")

    print(train_data.head())
    for i, (key, value) in enumerate(protein_embeddings.items()):
        if i >= 5:
            break
        print(f"Protein ID: {key}, Embedding shape: {value.shape}")
    
    return train_data, cv_data, test1_data, test2_data, protein_embeddings


def detect_column_names(data_df, embeddings_dict):
    """Automatically detect column names for protein IDs and interaction labels"""
    columns = data_df.columns.tolist()
    
    # Determine protein ID and interaction columns
    protein_a_col = None
    protein_b_col = None
    interaction_col = None
    
    # Common column name patterns
    protein_a_patterns = ['protein_a', 'protein_id_a', 'proteinA', 'proteinIDA', 'protein_A', 'protein_id_A']
    protein_b_patterns = ['protein_b', 'protein_id_b', 'proteinB', 'proteinIDB', 'protein_B', 'protein_id_B']
    interaction_patterns = ['isInteraction', 'is_interaction', 'interaction', 'label']
    
    # Find protein ID columns
    for col in columns:
        col_lower = col.lower()
        if any(pattern.lower() in col_lower for pattern in protein_a_patterns):
            protein_a_col = col
        elif any(pattern.lower() in col_lower for pattern in protein_b_patterns):
            protein_b_col = col
        elif any(pattern.lower() in col_lower for pattern in interaction_patterns):
            interaction_col = col
    
    # If we still can't find the columns, look for any that might contain protein IDs
    if protein_a_col is None or protein_b_col is None:
        # Check the first row to see if any column contains values that match keys in embeddings_dict
        first_row = data_df.iloc[0].to_dict()
        for col, val in first_row.items():
            if isinstance(val, str) and val in embeddings_dict:
                if protein_a_col is None:
                    protein_a_col = col
                elif protein_b_col is None and col != protein_a_col:
                    protein_b_col = col
    
    if protein_a_col is None or protein_b_col is None or interaction_col is None:
        print("Column detection failed. Please specify column names manually.")
        print("Available columns:", columns)
        raise ValueError("Could not detect required columns")
    
    print(f"Using columns: Protein A = '{protein_a_col}', Protein B = '{protein_b_col}', Interaction = '{interaction_col}'")
    return protein_a_col, protein_b_col, interaction_col


class ProteinPairDataset(Dataset):
    def __init__(self, pairs_df, embeddings_dict, protein_a_col=None, protein_b_col=None, interaction_col=None):
        """
        Dataset for protein pair interaction prediction
        
        Args:
            pairs_df: DataFrame with protein pair data
            embeddings_dict: Dict mapping uniprotID -> embedding tensor (seq_len, 960)
            protein_a_col: Column name for protein A IDs (auto-detected if None)
            protein_b_col: Column name for protein B IDs (auto-detected if None) 
            interaction_col: Column name for interaction labels (auto-detected if None)
        """
        self.pairs_df = pairs_df.reset_index(drop=True)
        self.embeddings_dict = embeddings_dict
        
        # Auto-detect column names if not provided
        if protein_a_col is None or protein_b_col is None or interaction_col is None:
            self.protein_a_col, self.protein_b_col, self.interaction_col = detect_column_names(pairs_df, embeddings_dict)
        else:
            self.protein_a_col = protein_a_col
            self.protein_b_col = protein_b_col
            self.interaction_col = interaction_col
        
        # Filter valid pairs
        valid_indices = []
        for idx in range(len(self.pairs_df)):
            row = self.pairs_df.iloc[idx]
            if (row[self.protein_a_col] in self.embeddings_dict and 
                row[self.protein_b_col] in self.embeddings_dict):
                valid_indices.append(idx)
        
        self.valid_indices = valid_indices
        print(f"Dataset: {len(valid_indices)} valid pairs out of {len(self.pairs_df)} total pairs")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        data_idx = self.valid_indices[idx]
        row = self.pairs_df.iloc[data_idx]
        
        # Get embeddings
        emb_a = self.embeddings_dict[row[self.protein_a_col]]
        emb_b = self.embeddings_dict[row[self.protein_b_col]]
        
        # Convert to tensors if needed
        if not isinstance(emb_a, torch.Tensor):
            emb_a = torch.from_numpy(emb_a).float()
        if not isinstance(emb_b, torch.Tensor):
            emb_b = torch.from_numpy(emb_b).float()
        
        # Get interaction label
        interaction = int(row[self.interaction_col])
        
        return {
            'emb_a': emb_a,           # (seq_len_a, 960)
            'emb_b': emb_b,           # (seq_len_b, 960)
            'interaction': interaction,
            'id_a': row[self.protein_a_col],
            'id_b': row[self.protein_b_col]
        }


def collate_fn(batch):
    """
    Collate function for protein pair batches with padding
    """
    # Extract components
    embs_a = [item['emb_a'] for item in batch]
    embs_b = [item['emb_b'] for item in batch]
    interactions = torch.tensor([item['interaction'] for item in batch], dtype=torch.long)
    
    # Pad sequences
    max_len_a = max(emb.shape[0] for emb in embs_a)
    max_len_b = max(emb.shape[0] for emb in embs_b)
    
    # Create padded tensors and length masks
    batch_size = len(batch)
    padded_a = torch.zeros(batch_size, max_len_a, 960)
    padded_b = torch.zeros(batch_size, max_len_b, 960)
    lengths_a = torch.zeros(batch_size, dtype=torch.long)
    lengths_b = torch.zeros(batch_size, dtype=torch.long)
    
    for i, (emb_a, emb_b) in enumerate(zip(embs_a, embs_b)):
        len_a, len_b = emb_a.shape[0], emb_b.shape[0]
        padded_a[i, :len_a] = emb_a
        padded_b[i, :len_b] = emb_b
        lengths_a[i] = len_a
        lengths_b[i] = len_b
    
    return padded_a, padded_b, lengths_a, lengths_b, interactions


class RoPEPositionalEncoding(nn.Module):
    """
    Rotary Position Embedding (RoPE) as recommended by latest research
    Better than learnable positional encoding for protein sequences
    """
    def __init__(self, dim, max_len=5000, theta=10000.0):
        super().__init__()
        self.dim = dim
        self.max_len = max_len
        self.theta = theta
        
        # Precompute inverse frequencies
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
    def forward(self, x):
        """
        Args:
            x: (B, L, dim) input embeddings
        Returns:
            (B, L, dim) embeddings with rotary position encoding applied
        """
        B, L, _ = x.shape
        device = x.device
        
        # Generate position indices
        positions = torch.arange(L, device=device, dtype=torch.float32)
        
        # Compute angles
        freqs = torch.outer(positions, self.inv_freq)  # (L, dim//2)
        emb = torch.cat((freqs, freqs), dim=-1)  # (L, dim)
        
        # Apply rotary embedding
        cos_emb = emb.cos().unsqueeze(0)  # (1, L, dim)
        sin_emb = emb.sin().unsqueeze(0)  # (1, L, dim)
        
        # Rotate x
        x_rot = self.rotate_half(x)
        output = x * cos_emb + x_rot * sin_emb
        
        return output
    
    def rotate_half(self, x):
        """Rotate the second half of the last dimension"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)


class EnhancedTransformerLayer(nn.Module):
    """
    Enhanced Transformer layer with RoPE and improved architecture
    """
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        
        # Multi-head attention with RoPE
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.rope = RoPEPositionalEncoding(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization and residual connections
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, src_key_padding_mask=None):
        """
        Args:
            x: (B, L, d_model) input embeddings
            src_key_padding_mask: (B, L) padding mask
        """
        # Apply RoPE to input
        x_rope = self.rope(x)
        
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(
            x_rope, x_rope, x_rope, 
            key_padding_mask=src_key_padding_mask
        )
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_out))
        
        return x


class ProteinEncoder(nn.Module):
    """
    Enhanced protein encoder based on latest research findings
    Uses 12-24 layers with RoPE positional encoding and variable-length support
    """
    def __init__(self, input_dim=960, embed_dim=256, num_layers=6, nhead=8, ff_dim=1024, 
                 use_variable_length=True, max_fixed_length=512):
        super().__init__()
        
        # Embedding layer
        self.embed = nn.Linear(input_dim, embed_dim)
        self.use_variable_length = use_variable_length
        self.max_fixed_length = max_fixed_length
        
        # Enhanced Transformer encoder layers with RoPE
        self.layers = nn.ModuleList([
            EnhancedTransformerLayer(embed_dim, nhead, ff_dim, dropout=0.1)
            for _ in range(num_layers)
        ])
        
        # Enhanced hierarchical pooling with both local and global attention
        self.global_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        self.local_attn = nn.MultiheadAttention(embed_dim, num_heads=4, batch_first=True)
        
        # Compression head with residual connection
        self.compress_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),  # 2x for local+global features
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, input_dim)
        )
        
        # Learnable queries for attention pooling
        self.global_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.local_query = nn.Parameter(torch.randn(1, 1, embed_dim))
        
    def forward(self, x, lengths):
        """
        Args:
            x: (B, L, 960) padded protein embeddings
            lengths: (B,) actual sequence lengths
        Returns:
            refined_emb: (B, 960) refined protein embeddings
        """
        B, L, _ = x.shape
        device = x.device
        
        # Create padding mask
        mask = torch.arange(L, device=device).unsqueeze(0) >= lengths.unsqueeze(1)  # True for padding
        
        # Embed input
        x_emb = self.embed(x)  # (B, L, embed_dim)
        
        # Apply enhanced transformer layers
        for layer in self.layers:
            x_emb = layer(x_emb, src_key_padding_mask=mask)
        
        # Enhanced hierarchical pooling
        if self.use_variable_length:
            # Global attention pooling
            global_query = self.global_query.expand(B, -1, -1)  # (B, 1, embed_dim)
            global_feat, _ = self.global_attn(
                global_query, x_emb, x_emb,
                key_padding_mask=mask
            )  # (B, 1, embed_dim)
            global_feat = global_feat.squeeze(1)  # (B, embed_dim)
            
            # Local attention pooling (focus on important regions)
            local_query = self.local_query.expand(B, -1, -1)  # (B, 1, embed_dim)
            local_feat, _ = self.local_attn(
                local_query, x_emb, x_emb,
                key_padding_mask=mask
            )  # (B, 1, embed_dim)
            local_feat = local_feat.squeeze(1)  # (B, embed_dim)
            
        else:
            # Fixed-length option for efficiency
            # Truncate or pad to fixed length
            if L > self.max_fixed_length:
                x_emb = x_emb[:, :self.max_fixed_length]
                mask = mask[:, :self.max_fixed_length]
            
            # Simple average pooling for fixed-length
            attn_weights = torch.softmax(
                torch.where(mask, torch.tensor(-1e9, device=device), torch.zeros_like(mask, dtype=torch.float)), 
                dim=1
            )  # (B, L)
            global_feat = torch.sum(x_emb * attn_weights.unsqueeze(-1), dim=1)  # (B, embed_dim)
            local_feat = global_feat  # Same for fixed-length case
        
        # Combine features
        combined_feat = torch.cat([global_feat, local_feat], dim=-1)  # (B, 2*embed_dim)
        
        # Compress to original dimension with residual connection
        refined_emb = self.compress_head(combined_feat)  # (B, 960)
        
        return refined_emb


class CrossAttentionInteraction(nn.Module):
    """
    Cross-attention between protein embeddings as recommended by research
    Better than simple concatenation or pooling for interaction prediction
    """
    def __init__(self, embed_dim=960, num_heads=8, ff_dim=512):
        super().__init__()
        
        # Bidirectional cross-attention
        self.cross_attn_ab = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.cross_attn_ba = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        
        # Layer normalization
        self.norm_ab = nn.LayerNorm(embed_dim)
        self.norm_ba = nn.LayerNorm(embed_dim)
        
        # Feed-forward for interaction processing
        self.interaction_ffn = nn.Sequential(
            nn.Linear(embed_dim * 2, ff_dim),
            nn.LayerNorm(ff_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(ff_dim, embed_dim)
        )
        
        # Residual connection
        self.residual_proj = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, emb_a, emb_b):
        """
        Args:
            emb_a: (B, 960) protein A embedding
            emb_b: (B, 960) protein B embedding
        Returns:
            (B, 960) interaction embedding
        """
        # Add sequence dimension for attention
        emb_a_seq = emb_a.unsqueeze(1)  # (B, 1, 960)
        emb_b_seq = emb_b.unsqueeze(1)  # (B, 1, 960)
        
        # Cross-attention: A attends to B
        attended_ab, _ = self.cross_attn_ab(emb_a_seq, emb_b_seq, emb_b_seq)
        attended_ab = self.norm_ab(attended_ab.squeeze(1) + emb_a)  # (B, 960)
        
        # Cross-attention: B attends to A
        attended_ba, _ = self.cross_attn_ba(emb_b_seq, emb_a_seq, emb_a_seq)
        attended_ba = self.norm_ba(attended_ba.squeeze(1) + emb_b)  # (B, 960)
        
        # Combine attended features
        combined = torch.cat([attended_ab, attended_ba], dim=-1)  # (B, 1920)
        
        # Process interaction with residual connection
        interaction_feat = self.interaction_ffn(combined)  # (B, 960)
        residual = self.residual_proj(combined)  # (B, 960)
        
        return interaction_feat + residual


class EnhancedMLPDecoder(nn.Module):
    """
    Enhanced MLP decoder based on research recommendations:
    - 3-4 layers with [512, 256, 128, 1] dimensions
    - Residual connections
    - Layer normalization and dropout
    """
    def __init__(self, input_dim=960, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        # Build hidden layers with residual connections
        for i, hidden_dim in enumerate(hidden_dims):
            layer = nn.ModuleDict({
                'linear': nn.Linear(prev_dim, hidden_dim),
                'norm': nn.LayerNorm(hidden_dim),
                'activation': nn.GELU(),
                'dropout': nn.Dropout(dropout)
            })
            
            # Add residual connection if dimensions match
            if prev_dim == hidden_dim:
                layer['residual'] = nn.Identity()
            else:
                layer['residual'] = nn.Linear(prev_dim, hidden_dim)
                
            self.layers.append(layer)
            prev_dim = hidden_dim
        
        # Final classification layer
        self.final_layer = nn.Linear(prev_dim, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
    
    def forward(self, x):
        """
        Args:
            x: (B, input_dim) interaction embedding
        Returns:
            (B, 1) classification logits
        """
        for layer in self.layers:
            # Forward pass
            out = layer['linear'](x)
            out = layer['norm'](out)
            out = layer['activation'](out)
            out = layer['dropout'](out)
            
            # Residual connection
            residual = layer['residual'](x)
            x = out + residual
        
        # Final classification
        logits = self.final_layer(x)
        return logits


class ProteinInteractionClassifier(nn.Module):
    """
    Enhanced protein-protein interaction prediction model (v4) based on latest research
    
    Architecture:
    1. Protein A/B ESMC Embedding -> Enhanced ProteinEncoder (6 layers, RoPE) -> Refined Embedding A/B  
    2. Cross-attention between protein embeddings
    3. Enhanced MLP decoder with residual connections for binary classification
    """
    def __init__(self, 
                 encoder_embed_dim=256,
                 encoder_layers=6,  # Reduced from 16 to 6 for memory efficiency
                 encoder_heads=8,   # Reduced from 16 to 8 for memory efficiency
                 use_variable_length=True,
                 decoder_hidden_dims=[256, 128],  # Reduced dimensions for memory efficiency
                 dropout=0.2):
        super().__init__()
        
        # Enhanced shared protein encoder with fewer layers for memory efficiency
        self.protein_encoder = ProteinEncoder(
            input_dim=960,
            embed_dim=encoder_embed_dim,
            num_layers=encoder_layers,
            nhead=encoder_heads,
            ff_dim=encoder_embed_dim * 4,
            use_variable_length=use_variable_length
        )
        
        # Cross-attention interaction layer
        self.interaction_layer = CrossAttentionInteraction(
            embed_dim=960, 
            num_heads=8,  # Reduced from 16 to 8
            ff_dim=512    # Reduced from 1024 to 512
        )
        
        # Enhanced MLP decoder with residual connections
        self.decoder = EnhancedMLPDecoder(
            input_dim=960,
            hidden_dims=decoder_hidden_dims,
            dropout=dropout
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    def forward(self, emb_a, emb_b, lengths_a, lengths_b):
        """
        Args:
            emb_a: (B, L_a, 960) protein A embeddings
            emb_b: (B, L_b, 960) protein B embeddings  
            lengths_a: (B,) actual lengths for protein A
            lengths_b: (B,) actual lengths for protein B
        Returns:
            (B, 1) interaction prediction logits
        """
        # Encode both proteins with shared enhanced encoder
        refined_a = self.protein_encoder(emb_a, lengths_a)  # (B, 960)
        refined_b = self.protein_encoder(emb_b, lengths_b)  # (B, 960)
        
        # Cross-attention interaction
        interaction_emb = self.interaction_layer(refined_a, refined_b)  # (B, 960)
        
        # Enhanced MLP decoder with residual connections
        logits = self.decoder(interaction_emb)  # (B, 1)
        
        return logits


def train_model(train_data, val_data, embeddings_dict, 
                epochs=50, batch_size=8, learning_rate=1e-4,  # Reduced from 16 to 8
                use_variable_length=True,
                save_every_epochs=10,  # Save checkpoint every N epochs
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Train the enhanced protein interaction prediction model
    """
    # Create datasets
    train_dataset = ProteinPairDataset(train_data, embeddings_dict)
    val_dataset = ProteinPairDataset(val_data, embeddings_dict)
    
    # Create data loaders with smaller batch size for memory efficiency
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch to avoid dimension issues
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=False  # Keep all validation data
    )
    
    # Calculate training statistics
    num_train_batches = len(train_loader)
    num_val_batches = len(val_loader)
    total_train_steps = num_train_batches * epochs
    
    print(f"📊 TRAINING STATISTICS:")
    print(f"Training samples: {len(train_dataset):,}")
    print(f"Validation samples: {len(val_dataset):,}")
    print(f"Batch size: {batch_size}")
    print(f"Training batches per epoch: {num_train_batches}")
    print(f"Validation batches per epoch: {num_val_batches}")
    print(f"Total epochs: {epochs}")
    print(f"Total training steps: {total_train_steps:,}")
    print(f"Progress reports every 50 batches")
    print(f"Checkpoints saved every {save_every_epochs} epochs")
    
    # GPU memory monitoring
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"GPU Memory before training: {initial_memory:.2f} GB")
        
        # Enable memory-efficient settings
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
    
    # Create enhanced model with reduced size
    model = ProteinInteractionClassifier(
        encoder_layers=6,  # Reduced for memory efficiency
        encoder_embed_dim=256,  # Reduced for memory efficiency
        encoder_heads=8,  # Reduced for memory efficiency
        use_variable_length=use_variable_length,
        decoder_hidden_dims=[256, 128]  # Reduced for memory efficiency
    ).to(device)
    
    # Use DataParallel if multiple GPUs available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)
    
    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.01
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    # Loss function
    criterion = nn.BCEWithLogitsLoss()
    
    # Training setup - use AUC as primary metric
    best_val_auc = 0
    best_val_loss = float('inf')
    history = []
    
    # Create log directory
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_path = f'logs/ppi_v4_enhanced_{ts}.json'
    best_path = f'models/ppi_v4_enhanced_best_{ts}.pth'
    checkpoint_dir = f'models/checkpoints_{ts}'
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Training enhanced model with variable_length={use_variable_length}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Best model will be saved to: {best_path}")
    print(f"Checkpoints will be saved to: {checkpoint_dir}/")
    
    for epoch in range(1, epochs + 1):
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_probs = []
        train_labels = []
        
        for batch_idx, (emb_a, emb_b, lengths_a, lengths_b, interactions) in enumerate(train_loader):
            # Move to device
            emb_a = emb_a.to(device).float()
            emb_b = emb_b.to(device).float()
            lengths_a = lengths_a.to(device)
            lengths_b = lengths_b.to(device)
            interactions = interactions.to(device).float()
            
            # Forward pass
            logits = model(emb_a, emb_b, lengths_a, lengths_b)  # (B, 1)
            
            # Fix dimension mismatch: ensure both have same shape
            if logits.dim() > 1:
                logits = logits.squeeze(-1)  # (B,) - only squeeze last dimension
            if interactions.dim() == 0:
                interactions = interactions.unsqueeze(0)  # Make sure it's at least 1D
                
            loss = criterion(logits, interactions)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Track metrics
            train_losses.append(loss.item())
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                train_preds.extend(preds.cpu().numpy())
                train_probs.extend(probs.cpu().numpy())
                train_labels.extend(interactions.cpu().numpy())
            
            if batch_idx % 50 == 0:
                progress = (epoch - 1) * num_train_batches + batch_idx
                current_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
                print(f'Epoch {epoch}/{epochs} Batch {batch_idx}/{num_train_batches} '
                      f'({progress}/{total_train_steps} total) Loss: {loss.item():.4f}, GPU: {current_memory:.2f}GB')
        
        # Validation phase
        model.eval()
        val_losses = []
        val_preds = []
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for emb_a, emb_b, lengths_a, lengths_b, interactions in val_loader:
                # Move to device
                emb_a = emb_a.to(device).float()
                emb_b = emb_b.to(device).float()
                lengths_a = lengths_a.to(device)
                lengths_b = lengths_b.to(device)
                interactions = interactions.to(device).float()
                
                # Forward pass
                logits = model(emb_a, emb_b, lengths_a, lengths_b)  # (B, 1)
                
                # Fix dimension mismatch: ensure both have same shape
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)  # (B,) - only squeeze last dimension
                if interactions.dim() == 0:
                    interactions = interactions.unsqueeze(0)  # Make sure it's at least 1D
                    
                loss = criterion(logits, interactions)
                
                # Track metrics
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(interactions.cpu().numpy())
        
        # Calculate metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        
        # Calculate AUC scores (primary metric)
        train_auc = roc_auc_score(train_labels, train_probs) if len(set(train_labels)) > 1 else 0
        val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0
        
        # Update learning rate
        scheduler.step()
        
        # Save best model based on AUC
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_auc': train_auc,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'config': {
                    'use_variable_length': use_variable_length,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'encoder_layers': 6,
                    'encoder_embed_dim': 256,
                    'encoder_heads': 8
                },
                'history': history
            }, best_path)
            print(f'>>> Saved BEST model: Epoch {epoch}, Val AUC: {val_auc:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}')
        
        # Save regular checkpoints
        if epoch % save_every_epochs == 0 or epoch == epochs:
            checkpoint_path = f'{checkpoint_dir}/checkpoint_epoch_{epoch:03d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_auc': train_auc,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'config': {
                    'use_variable_length': use_variable_length,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'encoder_layers': 6,
                    'encoder_embed_dim': 256,
                    'encoder_heads': 8
                },
                'history': history
            }, checkpoint_path)
            print(f'>>> Saved checkpoint: {checkpoint_path}')
        
        # Log progress
        epoch_log = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'val_f1': val_f1,
            'lr': scheduler.get_last_lr()[0],
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        history.append(epoch_log)
        
        with open(log_path, 'a') as f:
            f.write(json.dumps(epoch_log) + '\n')
        
        print(f'Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
              f'Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}, Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}, Val F1={val_f1:.4f}')
        
        # Memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f'\n✅ Training completed! Best validation AUC: {best_val_auc:.4f}')
    print(f'📁 Best model saved to: {best_path}')
    print(f'📁 Checkpoints saved to: {checkpoint_dir}/')
    print(f'📊 Total training steps completed: {total_train_steps:,}')
    
    return history, best_path


def evaluate_model(model_path, test_data, embeddings_dict, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Evaluate the trained enhanced model on test data
    """
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint['config']
    
    model = ProteinInteractionClassifier(
        encoder_layers=config.get('encoder_layers', 6),
        encoder_embed_dim=config.get('encoder_embed_dim', 256),
        encoder_heads=config.get('encoder_heads', 8),
        use_variable_length=config.get('use_variable_length', True),
        decoder_hidden_dims=[256, 128]
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create test dataset and loader
    test_dataset = ProteinPairDataset(test_data, embeddings_dict)
    test_loader = DataLoader(
        test_dataset, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    # Evaluation
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for emb_a, emb_b, lengths_a, lengths_b, interactions in test_loader:
            # Move to device
            emb_a = emb_a.to(device).float()
            emb_b = emb_b.to(device).float()
            lengths_a = lengths_a.to(device)
            lengths_b = lengths_b.to(device)
            
            # Forward pass
            logits = model(emb_a, emb_b, lengths_a, lengths_b)  # (B, 1)
            
            # Fix dimension mismatch
            if logits.dim() > 1:
                logits = logits.squeeze(-1)  # (B,)
                
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(interactions.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    
    print(f"\nTest Results:")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'predictions': all_preds,
        'probabilities': all_probs,
        'labels': all_labels
    }


def resume_training_from_checkpoint(checkpoint_path, train_data, val_data, embeddings_dict,
                                   additional_epochs=20, batch_size=8, learning_rate=1e-4,  # Reduced from 16 to 8
                                   save_every_epochs=10,
                                   device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Resume training from a saved checkpoint
    
    Args:
        checkpoint_path: Path to the checkpoint file
        train_data, val_data, embeddings_dict: Training data
        additional_epochs: Number of additional epochs to train
        Other parameters: Same as train_model
    """
    print(f"🔄 RESUMING TRAINING FROM CHECKPOINT")
    print(f"Checkpoint: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    start_epoch = checkpoint['epoch'] + 1
    
    print(f"Previous training stopped at epoch {checkpoint['epoch']}")
    print(f"Resuming from epoch {start_epoch}")
    print(f"Will train for {additional_epochs} more epochs (until epoch {start_epoch + additional_epochs - 1})")
    
    # Create datasets and loaders
    train_dataset = ProteinPairDataset(train_data, embeddings_dict)
    val_dataset = ProteinPairDataset(val_data, embeddings_dict)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch to avoid dimension issues
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=True,
        drop_last=False  # Keep all validation data
    )
    
    # Create model and load state
    model = ProteinInteractionClassifier(
        encoder_layers=config.get('encoder_layers', 6),
        encoder_embed_dim=config.get('encoder_embed_dim', 256),
        encoder_heads=config.get('encoder_heads', 8),
        use_variable_length=config.get('use_variable_length', True),
        decoder_hidden_dims=[256, 128]
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Create optimizer and scheduler, load states
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=0.01
    )
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    # Initialize from checkpoint
    best_val_auc = checkpoint.get('val_auc', 0)
    history = checkpoint.get('history', [])
    
    # Setup directories
    ts = datetime.now().strftime('%Y%m%d-%H%M%S')
    log_path = f'logs/ppi_v4_resumed_{ts}.json'
    best_path = f'models/ppi_v4_resumed_best_{ts}.pth'
    checkpoint_dir = f'models/checkpoints_resumed_{ts}'
    os.makedirs('logs', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Previous best validation AUC: {best_val_auc:.4f}")
    print(f"New checkpoints will be saved to: {checkpoint_dir}/")
    
    # Continue training (same logic as train_model but starting from start_epoch)
    criterion = nn.BCEWithLogitsLoss()
    
    for epoch in range(start_epoch, start_epoch + additional_epochs):
        # Training phase
        model.train()
        train_losses = []
        train_preds = []
        train_probs = []
        train_labels = []
        
        for batch_idx, (emb_a, emb_b, lengths_a, lengths_b, interactions) in enumerate(train_loader):
            emb_a = emb_a.to(device).float()
            emb_b = emb_b.to(device).float()
            lengths_a = lengths_a.to(device)
            lengths_b = lengths_b.to(device)
            interactions = interactions.to(device).float()
            
            logits = model(emb_a, emb_b, lengths_a, lengths_b)  # (B, 1)
            
            # Fix dimension mismatch: ensure both have same shape
            if logits.dim() > 1:
                logits = logits.squeeze(-1)  # (B,) - only squeeze last dimension
            if interactions.dim() == 0:
                interactions = interactions.unsqueeze(0)  # Make sure it's at least 1D
                
            loss = criterion(logits, interactions)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_losses.append(loss.item())
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                train_preds.extend(preds.cpu().numpy())
                train_probs.extend(probs.cpu().numpy())
                train_labels.extend(interactions.cpu().numpy())
            
            if batch_idx % 50 == 0:
                print(f'Resumed Epoch {epoch} Batch {batch_idx} Loss: {loss.item():.4f}')
        
        # Validation phase (same as train_model)
        model.eval()
        val_losses = []
        val_preds = []
        val_probs = []
        val_labels = []
        
        with torch.no_grad():
            for emb_a, emb_b, lengths_a, lengths_b, interactions in val_loader:
                emb_a = emb_a.to(device).float()
                emb_b = emb_b.to(device).float()
                lengths_a = lengths_a.to(device)
                lengths_b = lengths_b.to(device)
                interactions = interactions.to(device).float()
                
                logits = model(emb_a, emb_b, lengths_a, lengths_b)  # (B, 1)
                
                # Fix dimension mismatch: ensure both have same shape
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)  # (B,) - only squeeze last dimension
                if interactions.dim() == 0:
                    interactions = interactions.unsqueeze(0)  # Make sure it's at least 1D
                    
                loss = criterion(logits, interactions)
                
                val_losses.append(loss.item())
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                val_preds.extend(preds.cpu().numpy())
                val_probs.extend(probs.cpu().numpy())
                val_labels.extend(interactions.cpu().numpy())
        
        # Calculate metrics
        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        train_acc = accuracy_score(train_labels, train_preds)
        val_acc = accuracy_score(val_labels, val_preds)
        val_f1 = f1_score(val_labels, val_preds)
        
        train_auc = roc_auc_score(train_labels, train_probs) if len(set(train_labels)) > 1 else 0
        val_auc = roc_auc_score(val_labels, val_probs) if len(set(val_labels)) > 1 else 0
        
        scheduler.step()
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_auc': train_auc,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'config': config,
                'history': history,
                'resumed_from': checkpoint_path
            }, best_path)
            print(f'>>> Saved BEST resumed model: Epoch {epoch}, Val AUC: {val_auc:.4f}')
        
        # Save checkpoints
        if (epoch - start_epoch + 1) % save_every_epochs == 0 or epoch == start_epoch + additional_epochs - 1:
            checkpoint_path_new = f'{checkpoint_dir}/checkpoint_epoch_{epoch:03d}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_auc': val_auc,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_auc': train_auc,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'config': config,
                'history': history,
                'resumed_from': checkpoint_path
            }, checkpoint_path_new)
            print(f'>>> Saved resumed checkpoint: {checkpoint_path_new}')
        
        # Log progress
        epoch_log = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'val_f1': val_f1,
            'lr': scheduler.get_last_lr()[0],
            'time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'resumed': True
        }
        history.append(epoch_log)
        
        with open(log_path, 'a') as f:
            f.write(json.dumps(epoch_log) + '\n')
        
        print(f'Resumed Epoch {epoch:3d}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, '
              f'Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}, Val F1={val_f1:.4f}')
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f'\n✅ Resumed training completed! Best validation AUC: {best_val_auc:.4f}')
    print(f'📁 Best resumed model saved to: {best_path}')
    
    return history, best_path


if __name__ == '__main__':
    # Load actual data
    print("Enhanced Protein-Protein Interaction Prediction v4")
    print("=" * 60)
    print("Features: RoPE encoding, 6-layer transformers, cross-attention, enhanced MLP")
    print("=" * 60)
    
    try:
        # Load data using the new function
        train_data, cv_data, test1_data, test2_data, protein_embeddings = load_data()
        
        print(f"\nData loaded successfully:")
        print(f"Training data: {len(train_data)} pairs")
        print(f"Validation data: {len(cv_data)} pairs") 
        print(f"Test1 data: {len(test1_data)} pairs")
        print(f"Test2 data: {len(test2_data)} pairs")
        print(f"Protein embeddings: {len(protein_embeddings)} proteins")
        
        # Train models with different configurations
        # Compare variable-length vs fixed-length embeddings
        configurations = [
            {'use_variable_length': True, 'name': 'variable_length'},
            {'use_variable_length': False, 'name': 'fixed_length'}
        ]
        
        best_models = {}
        
        for config in configurations:
            config_name = config['name']
            print(f"\n" + "="*60)
            print(f"Training enhanced model with {config_name} embeddings...")
            print("="*60)
            
            try:
                history, model_path = train_model(
                    train_data, cv_data, protein_embeddings,
                    use_variable_length=config['use_variable_length'],
                    epochs=30,  # Reduced from 50 to 30 for faster training
                    batch_size=8,  # Reduced from 16 to 8 for memory efficiency
                    learning_rate=1e-4,
                    save_every_epochs=10  # Save checkpoints every 10 epochs
                )
                
                # Load the best model and get its validation AUC
                checkpoint = torch.load(model_path, map_location='cpu')
                best_auc = checkpoint['val_auc']
                
                best_models[config_name] = {
                    'model_path': model_path,
                    'val_auc': best_auc,
                    'history': history,
                    'config': config
                }
                
                print(f"Completed training {config_name}: Best Val AUC = {best_auc:.4f}")
                
            except Exception as e:
                print(f"Error training {config_name}: {str(e)}")
                continue
        
        # Find the best model overall (highest validation AUC)
        if best_models:
            best_config = max(best_models.keys(), key=lambda x: best_models[x]['val_auc'])
            best_model_info = best_models[best_config]
            
            print(f"\n" + "="*60)
            print(f"ENHANCED MODEL SUMMARY")
            print("="*60)
            print(f"Architecture Features:")
            print(f"  - RoPE positional encoding")
            print(f"  - 6-layer enhanced transformers")
            print(f"  - Cross-attention interaction")
            print(f"  - Enhanced MLP decoder [256→128→1]")
            print(f"  - Residual connections throughout")
            print(f"Best configuration: {best_config}")
            print(f"Best validation AUC: {best_model_info['val_auc']:.4f}")
            print(f"Model path: {best_model_info['model_path']}")
            
            # Evaluate the best model on both test sets
            print(f"\nEvaluating best enhanced model on test sets...")
            
            # Test1 evaluation
            try:
                print(f"\nTest1 Results (Enhanced {best_config}):")
                results_test1 = evaluate_model(
                    best_model_info['model_path'], 
                    test1_data, 
                    protein_embeddings
                )
            except Exception as e:
                print(f"Error evaluating on test1: {str(e)}")
            
            # Test2 evaluation  
            try:
                print(f"\nTest2 Results (Enhanced {best_config}):")
                results_test2 = evaluate_model(
                    best_model_info['model_path'], 
                    test2_data, 
                    protein_embeddings
                )
            except Exception as e:
                print(f"Error evaluating on test2: {str(e)}")
                
            # Save enhanced summary results
            summary = {
                'architecture': 'enhanced_v4',
                'features': [
                    'RoPE positional encoding',
                    '6-layer enhanced transformers', 
                    'Cross-attention interaction',
                    'Enhanced MLP decoder [256→128→1]',
                    'Residual connections',
                    'Variable/fixed length embedding support'
                ],
                'best_configuration': best_config,
                'best_val_auc': best_model_info['val_auc'],
                'model_path': best_model_info['model_path'],
                'all_models': {k: {'val_auc': v['val_auc'], 'model_path': v['model_path'], 'config': v['config']} 
                              for k, v in best_models.items()},
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            with open(f'logs/enhanced_model_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
                json.dump(summary, f, indent=2)
                
            print(f"\nEnhanced model summary saved to logs/enhanced_model_summary_*.json")
            
            # Print comparison of all configurations
            print(f"\n" + "="*60)
            print("CONFIGURATION COMPARISON")
            print("="*60)
            for config_name, model_info in best_models.items():
                print(f"{config_name:15}: Val AUC = {model_info['val_auc']:.4f}")
                
        else:
            print("No models were successfully trained!")
            
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        import traceback
        traceback.print_exc()
        # Fallback to example usage message
        print("\nPlease ensure your data files are in the correct location:")
        print("- data/full_dataset/train_data.pkl")
        print("- data/full_dataset/validation_data.pkl") 
        print("- data/full_dataset/test1_data.pkl")
        print("- data/full_dataset/test2_data.pkl")
        print("- data/full_dataset/embeddings/embeddings_standardized.pkl") 