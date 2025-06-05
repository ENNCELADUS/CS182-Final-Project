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
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import math

def sinusoid_encoding_table(max_len, d_model):
    """Create sinusoidal positional encoding table"""
    pe = torch.zeros(max_len, d_model).float()
    position = torch.arange(0, max_len).unsqueeze(1).float()
    
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                        -(math.log(10000.0) / d_model))
    
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    
    return pe.unsqueeze(0)  # (1, max_len, d_model)

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


class TransformerEnhancedProteinClassifier(nn.Module):
    """
    Fixed version of TransformerEnhancedProteinClassifier with:
    1. Learnable CLS token and sinusoidal positional encoding
    2. Standard initialization + norm_first=False
    3. Larger feedforward dimension (4x)
    4. No manual gradient scaling (handled by optimizer)
    """
    def __init__(self, input_dim=960, hidden_dim=256, num_transformer_layers=2, 
                 num_heads=8, dropout=0.3):
        super().__init__()
        self.debug_forward = False
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        
        # ① Add learnable CLS token and sinusoidal position encoding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, input_dim))
        self.max_position_embeddings = 2048  # Support longer sequences
        self.register_buffer('pos_encoding',
                             sinusoid_encoding_table(self.max_position_embeddings, input_dim),
                             persistent=False)
        
        # Initial projection to hidden dimension
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # ② Use standard init + no norm_first + larger feedforward
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,  # Increased from 2x to 4x
            dropout=dropout,
            activation='gelu',  # Changed from relu to gelu
            batch_first=True,
            norm_first=False  # Reverted from True to False
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        # Keep the proven protein encoder structure after transformer
        self.protein_encoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Keep the proven interaction layer exactly the same
        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Standard weight initialization
        self.apply(self._init_weights)
        
        # Store architecture details for debugging
        self._architecture_info = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'num_transformer_layers': num_transformer_layers,
            'num_heads': num_heads,
            'dropout': dropout,
            'feedforward_dim': hidden_dim * 4,
            'norm_first': False,
            'activation': 'gelu',
            'has_cls_token': True,
            'has_positional_encoding': True
        }
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use standard Xavier initialization instead of tiny std
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
        """
        B, La, _ = emb_a.shape
        B, Lb, _ = emb_b.shape
        device = emb_a.device
        
        # Prepend CLS token and add positional encoding
        emb_a = torch.cat([self.cls_token.expand(B, -1, -1), emb_a], dim=1)  # (B, La+1, 960)
        emb_b = torch.cat([self.cls_token.expand(B, -1, -1), emb_b], dim=1)  # (B, Lb+1, 960)
        
        # Add positional encoding with bounds checking
        max_len_a = min(La+1, self.max_position_embeddings)
        max_len_b = min(Lb+1, self.max_position_embeddings)
        
        # Warn if sequences exceed positional encoding length (only during debug)
        if self.debug_forward and (La+1 > self.max_position_embeddings or Lb+1 > self.max_position_embeddings):
            print(f"  WARNING: Sequence length exceeds max_position_embeddings ({self.max_position_embeddings})")
            print(f"    Protein A: {La+1}, Protein B: {Lb+1}")
        
        pos_slice_a = self.pos_encoding[:, :max_len_a]  # (1, max_len_a, 960)
        pos_slice_b = self.pos_encoding[:, :max_len_b]  # (1, max_len_b, 960)
        
        # Only add positional encoding to the parts we have encoding for
        emb_a[:, :max_len_a] = emb_a[:, :max_len_a] + pos_slice_a
        emb_b[:, :max_len_b] = emb_b[:, :max_len_b] + pos_slice_b
        
        # If sequences are longer than max_position_embeddings, the remaining positions
        # will not have positional encoding (which is still better than crashing)
        
        # Project to hidden dimension
        emb_a_proj = self.input_projection(emb_a)  # (B, La+1, hidden_dim)
        emb_b_proj = self.input_projection(emb_b)  # (B, Lb+1, hidden_dim)
        
        if self.debug_forward:
            print(f"  DEBUG: emb_a_proj - mean: {emb_a_proj.mean():.4f}, std: {emb_a_proj.std():.4f}, min: {emb_a_proj.min():.4f}, max: {emb_a_proj.max():.4f}")
            print(f"  DEBUG: emb_b_proj - mean: {emb_b_proj.mean():.4f}, std: {emb_b_proj.std():.4f}, min: {emb_b_proj.min():.4f}, max: {emb_b_proj.max():.4f}")
        
        # Build padding masks (shifted by 1 because of CLS token)
        mask_a = torch.arange(La+1, device=device).unsqueeze(0) >= (lengths_a+1).unsqueeze(1)
        mask_b = torch.arange(Lb+1, device=device).unsqueeze(0) >= (lengths_b+1).unsqueeze(1)
        
        # Apply transformer encoder with attention masks
        z_a = self.transformer_encoder(emb_a_proj, src_key_padding_mask=mask_a)
        z_b = self.transformer_encoder(emb_b_proj, src_key_padding_mask=mask_b)

        if self.debug_forward:
            print(f"  DEBUG: emb_a_transformed - mean: {z_a.mean():.4f}, std: {z_a.std():.4f}, min: {z_a.min():.4f}, max: {z_a.max():.4f}")
            print(f"  DEBUG: emb_b_transformed - mean: {z_b.mean():.4f}, std: {z_b.std():.4f}, min: {z_b.min():.4f}, max: {z_b.max():.4f}")
            # Additional statistics for transformer effectiveness
            std_change_a = z_a.std() / emb_a_proj.std()
            std_change_b = z_b.std() / emb_b_proj.std()
            print(f"  DEBUG: Transformer std amplification - A: {std_change_a:.4f}, B: {std_change_b:.4f}")
        
        # Take CLS token only (index 0)
        emb_a_avg = z_a[:, 0]  # (B, hidden_dim)
        emb_b_avg = z_b[:, 0]  # (B, hidden_dim)

        if self.debug_forward:
            print(f"  DEBUG: emb_a_avg (CLS) - mean: {emb_a_avg.mean():.4f}, std: {emb_a_avg.std():.4f}, min: {emb_a_avg.min():.4f}, max: {emb_a_avg.max():.4f}")
            print(f"  DEBUG: emb_b_avg (CLS) - mean: {emb_b_avg.mean():.4f}, std: {emb_b_avg.std():.4f}, min: {emb_b_avg.min():.4f}, max: {emb_b_avg.max():.4f}")
        
        # Use the proven encoder and interaction layers
        enc_a = self.protein_encoder(emb_a_avg)  # (B, hidden_dim)
        enc_b = self.protein_encoder(emb_b_avg)  # (B, hidden_dim)

        if self.debug_forward:
            print(f"  DEBUG: enc_a - mean: {enc_a.mean():.4f}, std: {enc_a.std():.4f}, min: {enc_a.min():.4f}, max: {enc_a.max():.4f}")
            print(f"  DEBUG: enc_b - mean: {enc_b.mean():.4f}, std: {enc_b.std():.4f}, min: {enc_b.min():.4f}, max: {enc_b.max():.4f}")
            # Check for ReLU saturation
            zero_frac_a = (enc_a == 0).float().mean()
            zero_frac_b = (enc_b == 0).float().mean()
            print(f"  DEBUG: ReLU zero fraction - A: {zero_frac_a:.4f}, B: {zero_frac_b:.4f}")
        
        # Combine and predict interaction (exactly as before)
        combined = torch.cat([enc_a, enc_b], dim=-1)  # (B, 2*hidden_dim)
        logits = self.interaction_layer(combined)  # (B, 1)

        if self.debug_forward:
            print(f"  DEBUG: logits - mean: {logits.mean():.4f}, std: {logits.std():.4f}, min: {logits.min():.4f}, max: {logits.max():.4f}")
            # Check logits magnitude for learning signal
            logits_magnitude = logits.abs().mean()
            print(f"  DEBUG: logits magnitude: {logits_magnitude:.4f} (should be > 0.1 for good learning)")
            # Expected sigmoid range
            sigmoid_mean = torch.sigmoid(logits).mean()
            print(f"  DEBUG: sigmoid(logits) mean: {sigmoid_mean:.4f} (should deviate from 0.5 for learning)")
        
        return logits


class SimplifiedProteinClassifier(nn.Module):
    """
    Proven simplified model architecture - keep as fallback
    """
    def __init__(self, input_dim=960, hidden_dim=256, dropout=0.3):
        super().__init__()
        
        # Simple encoder for each protein
        self.protein_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Simple interaction layer
        self.interaction_layer = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Better weight initialization
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
        """
        # Simple average pooling for variable length sequences
        device = emb_a.device
        
        # Create masks for averaging
        mask_a = torch.arange(emb_a.size(1), device=device).unsqueeze(0) < lengths_a.unsqueeze(1)
        mask_b = torch.arange(emb_b.size(1), device=device).unsqueeze(0) < lengths_b.unsqueeze(1)
        
        # Average pooling with mask
        emb_a_avg = (emb_a * mask_a.unsqueeze(-1).float()).sum(dim=1) / lengths_a.unsqueeze(-1).float()
        emb_b_avg = (emb_b * mask_b.unsqueeze(-1).float()).sum(dim=1) / lengths_b.unsqueeze(-1).float()
        
        # Encode proteins
        enc_a = self.protein_encoder(emb_a_avg)  # (B, hidden_dim)
        enc_b = self.protein_encoder(emb_b_avg)  # (B, hidden_dim)
        
        # Combine and predict interaction
        combined = torch.cat([enc_a, enc_b], dim=-1)  # (B, 2*hidden_dim)
        logits = self.interaction_layer(combined)  # (B, 1)
        
        return logits


def create_model(model_type='enhanced', **kwargs):
    """
    Factory function to create models
    
    Args:
        model_type: 'simple' or 'enhanced'
        **kwargs: model parameters
    """
    if model_type == 'simple':
        return SimplifiedProteinClassifier(**kwargs)
    elif model_type == 'enhanced':
        return TransformerEnhancedProteinClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


# Training utilities
def train_model(model, train_loader, val_loader, num_epochs=20, lr=5e-3, device='cuda', debug_mode=False):
    """
    Train a protein interaction model with the proven training setup
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of epochs to train
        lr: Learning rate
        device: Device to train on
        debug_mode: Whether to enable debug mode
    """
    print(f"Training model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    model = model.to(device)
    
    # ③ Use different learning rates for transformer vs other parameters
    transformer_params = [p for n, p in model.named_parameters() if 'transformer_encoder' in n]
    other_params = [p for n, p in model.named_parameters() if 'transformer_encoder' not in n]
    
    if transformer_params:
        optimizer = torch.optim.AdamW([
            {'params': other_params, 'lr': lr},
            {'params': transformer_params, 'lr': lr}  # Same LR initially
        ], betas=(0.9, 0.98), weight_decay=0.01)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.98), weight_decay=0.01)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr, 
        epochs=num_epochs,
        steps_per_epoch=len(train_loader),
        pct_start=0.1
    )
    criterion = nn.BCEWithLogitsLoss()
    
    best_val_auc = 0
    history = []
    
    # Enable debug for the model if applicable
    if debug_mode and hasattr(model, 'debug_forward'):
        print("  ⚠️ DEBUG MODE: Enabling comprehensive debugging for the model.")
        model.debug_forward = True
        
        # Print detailed architecture information
        if hasattr(model, '_architecture_info'):
            print("  📋 Model Architecture Details:")
            arch = model._architecture_info
            print(f"    Input dimension: {arch['input_dim']}")
            print(f"    Hidden dimension: {arch['hidden_dim']}")
            print(f"    Transformer layers: {arch['num_transformer_layers']}")
            print(f"    Attention heads: {arch['num_heads']}")
            print(f"    Feedforward dimension: {arch['feedforward_dim']} ({arch['feedforward_dim']//arch['hidden_dim']}x hidden)")
            print(f"    Activation: {arch['activation']}")
            print(f"    Norm first: {arch['norm_first']}")
            print(f"    Dropout: {arch['dropout']}")
            print(f"    Has CLS token: {arch['has_cls_token']}")
            print(f"    Has positional encoding: {arch['has_positional_encoding']}")
            
        # Print parameter counts by component
        total_params = sum(p.numel() for p in model.parameters())
        transformer_params = sum(p.numel() for n, p in model.named_parameters() if 'transformer_encoder' in n)
        other_params = total_params - transformer_params
        
        print(f"  📊 Parameter Distribution:")
        print(f"    Total parameters: {total_params:,}")
        print(f"    Transformer parameters: {transformer_params:,} ({100*transformer_params/total_params:.1f}%)")
        print(f"    Other parameters: {other_params:,} ({100*other_params/total_params:.1f}%)")
        print(f"    Expected improvements:")
        print(f"      - Transformer std amplification: ~2-3x (was ~1x)")
        print(f"      - Transformer grad norms: ~0.3-0.5 (was ≤0.16)")  
        print(f"      - Logits magnitude: >0.1 (was ~0.1)")
        print(f"      - AUC should jump off 0.5 floor within 2-3 epochs")
    
    # Counter for debug batches
    debug_batch_count = 0
    max_debug_batches = 2  # Log for first N batches
    
    for epoch in range(1, num_epochs + 1):
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
            
            optimizer.zero_grad()
            logits = model(emb_a, emb_b, lengths_a, lengths_b)
            if logits.dim() > 1:
                logits = logits.squeeze(-1)
            
            loss = criterion(logits, interactions)
            loss.backward()
            
            # Enhanced debug information (only for TransformerEnhancedProteinClassifier in debug_mode)
            if debug_mode and hasattr(model, 'transformer_encoder') and epoch == 1 and debug_batch_count < max_debug_batches:
                print(f"  DEBUG Batch {batch_idx+1}, Epoch {epoch}: Comprehensive Statistics")
                print("  " + "="*50)
                
                # Gradient norms with more detail
                print(f"  Gradient Norms:")
                transformer_grads = []
                other_grads = []
                
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        grad_norm = param.grad.norm().item()
                        # Log key layers with more detail
                        if 'input_projection' in name or \
                           'transformer_encoder.layers.0.self_attn.out_proj' in name or \
                           'transformer_encoder.layers.0.linear1' in name or \
                           'transformer_encoder.layers.0.norm1' in name or \
                           'protein_encoder.0.weight' in name or \
                           'interaction_layer.0.weight' in name:
                            print(f"    {name}: {grad_norm:.4e}")
                        
                        # Collect transformer vs other gradients for analysis
                        if 'transformer_encoder' in name:
                            transformer_grads.append(grad_norm)
                        else:
                            other_grads.append(grad_norm)
                    else:
                        if 'input_projection' in name or \
                           'transformer_encoder.layers.0.self_attn.out_proj' in name or \
                           'transformer_encoder.layers.0.linear1' in name or \
                           'transformer_encoder.layers.0.norm1' in name or \
                           'protein_encoder.0.weight' in name or \
                           'interaction_layer.0.weight' in name:
                            print(f"    {name}: No gradient")
                
                # Gradient analysis
                if transformer_grads and other_grads:
                    avg_transformer = sum(transformer_grads) / len(transformer_grads)
                    avg_other = sum(other_grads) / len(other_grads)
                    print(f"  Gradient Analysis:")
                    print(f"    Avg transformer grad norm: {avg_transformer:.4e} ({len(transformer_grads)} params)")
                    print(f"    Avg other grad norm: {avg_other:.4e} ({len(other_grads)} params)")
                    print(f"    Transformer/Other ratio: {avg_transformer/avg_other:.4f}")
                
                print("  ---- End of Gradient Analysis ----")
            
            # ③ Remove manual gradient scaling - use simple gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            train_losses.append(loss.item())
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
                train_preds.extend(preds.cpu().numpy())
                train_probs.extend(probs.cpu().numpy())
                train_labels.extend(interactions.cpu().numpy())
            
            if debug_mode and hasattr(model, 'debug_forward') and epoch == 1:
                debug_batch_count += 1
                if debug_batch_count >= max_debug_batches:
                    print("  ⚠️ DEBUG MODE: Disabling forward pass statistics after first few batches.")
                    model.debug_forward = False # Turn off after a few batches
        
        # Validation phase
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
                
                logits = model(emb_a, emb_b, lengths_a, lengths_b)
                if logits.dim() > 1:
                    logits = logits.squeeze(-1)
                
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
        train_auc = roc_auc_score(train_labels, train_probs)
        val_auc = roc_auc_score(val_labels, val_probs)
        val_f1 = f1_score(val_labels, val_preds)
        
        # Save best model
        if val_auc > best_val_auc:
            best_val_auc = val_auc
        
        # Log
        epoch_log = {
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'train_auc': train_auc,
            'val_auc': val_auc,
            'val_f1': val_f1,
            'lr': scheduler.get_last_lr()[0]
        }
        history.append(epoch_log)
        
        if epoch % 5 == 0 or epoch <= 3:
            print(f'Epoch {epoch:2d}: Train AUC={train_auc:.4f}, Val AUC={val_auc:.4f}, '
                  f'Val F1={val_f1:.4f}, LR={scheduler.get_last_lr()[0]:.2e}')
    
    print(f'Training completed! Best validation AUC: {best_val_auc:.4f}')
    return history, best_val_auc


def main():
    """
    Main function to train protein interaction models
    """
    print("🚀 Starting Protein Interaction Model Training")
    print("=" * 60)
    
    # Configuration
    config = {
        'batch_size': 32,
        'num_epochs': 5,  # Reduced for faster debugging
        'learning_rate': 3e-3,
        'hidden_dim': 256,
        'dropout': 0.3,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Configuration: {config}")
    print(f"Using device: {config['device']}")
    
    # Load data
    print("\n📂 Loading data...")
    try:
        train_data, cv_data, test1_data, test2_data, protein_embeddings = load_data()
        print(f"✅ Data loaded successfully!")
        print(f"   Train samples: {len(train_data)}")
        print(f"   Validation samples: {len(cv_data)}")
        print(f"   Test1 samples: {len(test1_data)}")
        print(f"   Test2 samples: {len(test2_data)}")
        print(f"   Protein embeddings: {len(protein_embeddings)}")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return
    
    # Create datasets
    print("\n📊 Creating datasets...")
    try:
        train_dataset = ProteinPairDataset(train_data, protein_embeddings)
        val_dataset = ProteinPairDataset(cv_data, protein_embeddings)
        test1_dataset = ProteinPairDataset(test1_data, protein_embeddings)
        test2_dataset = ProteinPairDataset(test2_data, protein_embeddings)
        
        print(f"✅ Datasets created successfully!")
        print(f"   Train dataset: {len(train_dataset)} valid pairs")
        print(f"   Validation dataset: {len(val_dataset)} valid pairs")
        print(f"   Test1 dataset: {len(test1_dataset)} valid pairs")
        print(f"   Test2 dataset: {len(test2_dataset)} valid pairs")
    except Exception as e:
        print(f"❌ Error creating datasets: {e}")
        return
    
    # Create data loaders
    print("\n🔄 Creating data loaders...")
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['batch_size'], 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=False  # Disabled pin_memory to avoid CUDA invalid argument errors
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=False
    )
    test1_loader = DataLoader(
        test1_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=False
    )
    test2_loader = DataLoader(
        test2_dataset, 
        batch_size=config['batch_size'], 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=False
    )
    
    print(f"✅ Data loaders created!")
    print(f"   Train batches: {len(train_loader)}")
    print(f"   Validation batches: {len(val_loader)}")
    
    # Models to train
    models_to_train = {
        # 'SimplifiedProteinClassifier': {
        #     'type': 'simple',
        #     'params': {
        #         'hidden_dim': config['hidden_dim'],
        #         'dropout': config['dropout']
        #     }
        # },
        'TransformerEnhancedProteinClassifier': {
            'type': 'enhanced',
            'params': {
                'hidden_dim': config['hidden_dim'],
                'num_transformer_layers': 1,
                'num_heads': 8,
                'dropout': config['dropout']
            }
        }
    }
    
    # Train each model
    results = {}
    
    for model_name, model_config in models_to_train.items():
        print(f"\n🧠 Training {model_name}")
        print("=" * (10 + len(model_name)))
        
        try:
            # Create model
            model_debug_mode = False
            # if model_name == 'TransformerEnhancedProteinClassifier':
            #     print("  🔬 ENABLING DEBUG MODE FOR TRANSFORMER MODEL")
            #     model_debug_mode = True
            
            model = create_model(model_config['type'], **model_config['params'])
            print(f"📋 Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
            
            # Train model
            history, best_val_auc = train_model(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=config['num_epochs'],
                lr=config['learning_rate'],
                device=config['device'],
                debug_mode=model_debug_mode # Pass the debug_mode flag
            )
            
            # Store results
            results[model_name] = {
                'model': model,
                'history': history,
                'best_val_auc': best_val_auc
            }
            
            # Save model
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_path = f"models/{model_name}_{timestamp}_auc{best_val_auc:.4f}.pt"
            os.makedirs("models", exist_ok=True)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': model_config,
                'best_val_auc': best_val_auc,
                'history': history,
                'training_config': config
            }, model_path)
            
            print(f"💾 Model saved to: {model_path}")
            
        except Exception as e:
            print(f"❌ Error training {model_name}: {e}")
            continue
    
    # Compare results
    print(f"\n📊 Training Results Summary")
    print("=" * 50)
    
    for model_name, result in results.items():
        print(f"{model_name}:")
        print(f"  Best Validation AUC: {result['best_val_auc']:.4f}")
        
        # Show final epoch metrics
        final_metrics = result['history'][-1]
        print(f"  Final Train AUC: {final_metrics['train_auc']:.4f}")
        print(f"  Final Val F1: {final_metrics['val_f1']:.4f}")
        print(f"  Final Val Accuracy: {final_metrics['val_acc']:.4f}")
        print()
    
    # Evaluate on test sets
    print(f"\n🧪 Evaluating on Test Sets")
    print("=" * 40)
    
    test_results = {}
    
    for model_name, result in results.items():
        print(f"\n{model_name} Test Results:")
        print("-" * (len(model_name) + 14))
        
        model = result['model']
        model.eval()
        
        test_results[model_name] = {}
        
        # Test on both test sets
        for test_name, test_loader in [('Test1', test1_loader), ('Test2', test2_loader)]:
            test_preds = []
            test_probs = []
            test_labels = []
            
            with torch.no_grad():
                for emb_a, emb_b, lengths_a, lengths_b, interactions in test_loader:
                    emb_a = emb_a.to(config['device']).float()
                    emb_b = emb_b.to(config['device']).float()
                    lengths_a = lengths_a.to(config['device'])
                    lengths_b = lengths_b.to(config['device'])
                    interactions = interactions.to(config['device']).float()
                    
                    logits = model(emb_a, emb_b, lengths_a, lengths_b)
                    if logits.dim() > 1:
                        logits = logits.squeeze(-1)
                    
                    probs = torch.sigmoid(logits)
                    preds = (probs > 0.5).float()
                    
                    test_preds.extend(preds.cpu().numpy())
                    test_probs.extend(probs.cpu().numpy())
                    test_labels.extend(interactions.cpu().numpy())
            
            # Calculate metrics
            test_acc = accuracy_score(test_labels, test_preds)
            test_auroc = roc_auc_score(test_labels, test_probs)
            test_auprc = average_precision_score(test_labels, test_probs)
            test_f1 = f1_score(test_labels, test_preds)
            test_precision = precision_score(test_labels, test_preds)
            test_recall = recall_score(test_labels, test_preds)
            
            # Store results for plotting
            test_results[model_name][test_name] = {
                'labels': test_labels,
                'probs': test_probs,
                'preds': test_preds,
                'auroc': test_auroc,
                'auprc': test_auprc,
                'accuracy': test_acc,
                'f1': test_f1,
                'precision': test_precision,
                'recall': test_recall
            }
            
            print(f"  {test_name} - AUROC: {test_auroc:.4f}, AUPRC: {test_auprc:.4f}, Acc: {test_acc:.4f}, F1: {test_f1:.4f}, "
                  f"Prec: {test_precision:.4f}, Rec: {test_recall:.4f}")
    
    # Plot training curves
    print(f"\n📈 Generating training plots...")
    try:
        plot_training_curves(results)
        print("✅ Training plots saved!")
    except Exception as e:
        print(f"⚠️ Could not generate plots: {e}")
    
    # Plot test set AUROC and AUPRC curves
    print(f"\n📊 Generating test set ROC and PR curves...")
    try:
        plot_test_curves(test_results)
        print("✅ Test set curves saved!")
    except Exception as e:
        print(f"⚠️ Could not generate test plots: {e}")
    
    print(f"\n🎉 Training completed successfully!")
    return results


def plot_training_curves(results):
    """Plot training curves for comparison"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training Curves Comparison', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, (model_name, result) in enumerate(results.items()):
        history = result['history']
        epochs = [h['epoch'] for h in history]
        
        color = colors[i % len(colors)]
        
        # Training and validation loss
        train_losses = [h['train_loss'] for h in history]
        val_losses = [h['val_loss'] for h in history]
        axes[0, 0].plot(epochs, train_losses, f'{color}-', label=f'{model_name} Train', alpha=0.7)
        axes[0, 0].plot(epochs, val_losses, f'{color}--', label=f'{model_name} Val', alpha=0.7)
        
        # Training and validation AUC
        train_aucs = [h['train_auc'] for h in history]
        val_aucs = [h['val_auc'] for h in history]
        axes[0, 1].plot(epochs, train_aucs, f'{color}-', label=f'{model_name} Train', alpha=0.7)
        axes[0, 1].plot(epochs, val_aucs, f'{color}--', label=f'{model_name} Val', alpha=0.7)
        
        # Validation F1 and Accuracy
        val_f1s = [h['val_f1'] for h in history]
        val_accs = [h['val_acc'] for h in history]
        axes[1, 0].plot(epochs, val_f1s, f'{color}-', label=f'{model_name} F1', alpha=0.7)
        axes[1, 1].plot(epochs, val_accs, f'{color}-', label=f'{model_name} Acc', alpha=0.7)
    
    # Customize plots
    axes[0, 0].set_title('Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].set_title('AUC')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('AUC')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].set_title('Validation F1')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1 Score')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    axes[1, 1].set_title('Validation Accuracy')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"training_curves_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training curves saved to: {plot_path}")


def plot_test_curves(test_results):
    """Plot ROC and Precision-Recall curves for test sets"""
    
    # Create subplots for ROC and PR curves
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Test Set Performance: ROC and Precision-Recall Curves', fontsize=16)
    
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    line_styles = ['-', '--', '-.', ':']
    
    # Plot ROC curves
    axes[0, 0].set_title('ROC Curves - Test1')
    axes[0, 1].set_title('ROC Curves - Test2')
    axes[1, 0].set_title('Precision-Recall Curves - Test1')
    axes[1, 1].set_title('Precision-Recall Curves - Test2')
    
    model_idx = 0
    for model_name, model_results in test_results.items():
        color = colors[model_idx % len(colors)]
        
        for test_idx, (test_name, test_data) in enumerate(model_results.items()):
            line_style = line_styles[test_idx % len(line_styles)]
            
            # Get test data
            y_true = np.array(test_data['labels'])
            y_scores = np.array(test_data['probs'])
            auroc = test_data['auroc']
            auprc = test_data['auprc']
            
            # Calculate ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            
            # Calculate PR curve
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            
            # Plot ROC curves
            if test_name == 'Test1':
                axes[0, 0].plot(fpr, tpr, color=color, linestyle=line_style, linewidth=2,
                               label=f'{model_name} (AUROC={auroc:.4f})')
            else:  # Test2
                axes[0, 1].plot(fpr, tpr, color=color, linestyle=line_style, linewidth=2,
                               label=f'{model_name} (AUROC={auroc:.4f})')
            
            # Plot PR curves
            if test_name == 'Test1':
                axes[1, 0].plot(recall, precision, color=color, linestyle=line_style, linewidth=2,
                               label=f'{model_name} (AUPRC={auprc:.4f})')
            else:  # Test2
                axes[1, 1].plot(recall, precision, color=color, linestyle=line_style, linewidth=2,
                               label=f'{model_name} (AUPRC={auprc:.4f})')
        
        model_idx += 1
    
    # Add diagonal lines for ROC plots (random classifier)
    for i in range(2):
        axes[0, i].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier')
        axes[0, i].set_xlim([0.0, 1.0])
        axes[0, i].set_ylim([0.0, 1.05])
        axes[0, i].set_xlabel('False Positive Rate')
        axes[0, i].set_ylabel('True Positive Rate')
        axes[0, i].legend(loc="lower right")
        axes[0, i].grid(True, alpha=0.3)
    
    # Add baseline for PR plots (random classifier based on class distribution)
    for i, test_name in enumerate(['Test1', 'Test2']):
        # Get class distribution for baseline
        for model_results in test_results.values():
            if test_name in model_results:
                y_true = np.array(model_results[test_name]['labels'])
                baseline_precision = np.mean(y_true)
                axes[1, i].axhline(y=baseline_precision, color='k', linestyle='--', alpha=0.5, 
                                  label=f'Random Classifier ({baseline_precision:.3f})')
                break
        
        axes[1, i].set_xlim([0.0, 1.0])
        axes[1, i].set_ylim([0.0, 1.05])
        axes[1, i].set_xlabel('Recall')
        axes[1, i].set_ylabel('Precision')
        axes[1, i].legend(loc="lower left")
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"test_roc_pr_curves_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Test ROC and PR curves saved to: {plot_path}")
    
    # Also create a summary metrics bar plot
    plot_test_metrics_summary(test_results)


def plot_test_metrics_summary(test_results):
    """Create a summary bar plot of test metrics"""
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('Test Set Metrics Summary', fontsize=16)
    
    # Prepare data for plotting
    models = list(test_results.keys())
    test_sets = ['Test1', 'Test2']
    
    metrics = ['auroc', 'auprc', 'f1', 'accuracy']
    metric_names = ['AUROC', 'AUPRC', 'F1 Score', 'Accuracy']
    
    x = np.arange(len(models))
    width = 0.35
    
    for i, (metric, metric_name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i // 2, i % 2]
        
        test1_values = [test_results[model]['Test1'][metric] for model in models]
        test2_values = [test_results[model]['Test2'][metric] for model in models]
        
        bars1 = ax.bar(x - width/2, test1_values, width, label='Test1', alpha=0.8, color='skyblue')
        bars2 = ax.bar(x + width/2, test2_values, width, label='Test2', alpha=0.8, color='lightcoral')
        
        ax.set_xlabel('Model')
        ax.set_ylabel(metric_name)
        ax.set_title(f'{metric_name} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels([name.replace('ProteinClassifier', '') for name in models], rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),  # 3 points vertical offset
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    
    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = f"test_metrics_summary_{timestamp}.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Test metrics summary saved to: {plot_path}")


if __name__ == "__main__":
    main()


